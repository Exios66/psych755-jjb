"""Factor analysis and predictive feature-importance utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ca_personas.load import load_and_prepare, load_qualtrics
from ca_personas.ml_baseline import (
    TARGETS,
    available_feature_columns,
    make_preprocessor,
    prepare_modeling_frame,
)
from ca_personas.scoring import (
    GROUP_REVERSE,
    GROUP_STRAIGHT,
    INTERPERSONAL_REVERSE,
    INTERPERSONAL_STRAIGHT,
    likert_to_int,
    reverse_score,
)

CA_ITEM_ORDER = (
    list(GROUP_STRAIGHT)
    + list(GROUP_REVERSE)
    + list(INTERPERSONAL_STRAIGHT)
    + list(INTERPERSONAL_REVERSE)
)

CA_ITEM_LABELS = {
    "Q1": "Group: dislike discussions",
    "Q2": "Group: comfortable (rev)",
    "Q3": "Group: tense/nervous",
    "Q4": "Group: like involvement (rev)",
    "Q5": "Group: tense with new people",
    "Q6": "Group: calm/relaxed (rev)",
    "Q13": "Interpersonal: nervous w/ acquaintance",
    "Q14": "Interpersonal: no fear speaking (rev)",
    "Q15": "Interpersonal: tense in conversations",
    "Q16": "Interpersonal: calm in conversations (rev)",
    "Q17": "Interpersonal: relaxed w/ acquaintance (rev)",
    "Q18": "Interpersonal: afraid to speak up",
}


def _resolve_item_column(df: pd.DataFrame, item: str) -> str | None:
    if item in df.columns:
        return item
    if item == "Q18" and "Q18_ca" in df.columns:
        return "Q18_ca"
    return None


def likert_item_matrix(
    qualtrics: pd.DataFrame,
    *,
    scored_direction: bool = True,
) -> pd.DataFrame:
    """
    Build a numeric matrix of PRCA Likert items (1–5).

    When ``scored_direction`` is True, reverse-coded comfort items are flipped
    so higher values always mean higher apprehension (factor-analytic friendly).
    """
    reverse = set(GROUP_REVERSE) | set(INTERPERSONAL_REVERSE)
    columns: dict[str, list[float | None]] = {}
    n = len(qualtrics)
    for item in CA_ITEM_ORDER:
        src = _resolve_item_column(qualtrics, item)
        values: list[float | None] = []
        for i in range(n):
            if src is None:
                values.append(None)
                continue
            mapped = likert_to_int(qualtrics.iloc[i][src])
            if mapped is None:
                values.append(None)
            elif scored_direction and item in reverse:
                values.append(float(reverse_score(mapped)))
            else:
                values.append(float(mapped))
        columns[item] = values
    frame = pd.DataFrame(columns)
    frame.index = qualtrics.index
    return frame.dropna(how="any").astype(float)


def run_item_factor_analysis(
    item_matrix: pd.DataFrame,
    *,
    n_factors: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Fit PCA + maximum-likelihood FactorAnalysis on PRCA items.

    Returns loadings, variance explained, and suggested factor count.
    """
    if item_matrix.shape[0] < 3:
        raise ValueError("Need at least 3 complete Likert rows for factor analysis")

    max_factors = min(item_matrix.shape[0] - 1, item_matrix.shape[1], 4)
    k = n_factors or max_factors
    k = max(1, min(k, max_factors))

    scaler = StandardScaler()
    X = scaler.fit_transform(item_matrix.to_numpy())

    pca = PCA(n_components=k, random_state=random_state)
    pca.fit(X)
    pca_loadings = pd.DataFrame(
        pca.components_.T,
        index=[CA_ITEM_LABELS.get(c, c) for c in item_matrix.columns],
        columns=[f"PC{i+1}" for i in range(k)],
    )
    pca_var = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(k)],
            "variance_explained": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    fa = FactorAnalysis(n_components=k, random_state=random_state, max_iter=2000)
    fa.fit(X)
    fa_loadings = pd.DataFrame(
        fa.components_.T,
        index=[CA_ITEM_LABELS.get(c, c) for c in item_matrix.columns],
        columns=[f"Factor{i+1}" for i in range(k)],
    )

    # Communality proxy from PCA: row sum of squared loadings on retained components.
    communalities = pd.DataFrame(
        {
            "item": pca_loadings.index,
            "communality_pca": (pca_loadings.to_numpy() ** 2).sum(axis=1),
        }
    ).sort_values("communality_pca", ascending=False)

    return {
        "n_rows": int(item_matrix.shape[0]),
        "n_factors": k,
        "pca_loadings": pca_loadings,
        "pca_variance": pca_var,
        "fa_loadings": fa_loadings,
        "communalities": communalities,
        "pca": pca,
        "fa": fa,
    }


def predictor_feature_importance(
    participants: pd.DataFrame,
    *,
    tier: str = "transit",
    targets: Iterable[str] = TARGETS,
    random_state: int = 42,
    n_repeats: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Random-forest impurity importance + permutation importance for each CA target.

    Returns a dict keyed by target name with long-form importance tables.
    """
    model_df = prepare_modeling_frame(participants, tier=tier, targets=targets)
    feature_cols = available_feature_columns(model_df, tier)
    if len(model_df) < 2:
        raise ValueError("Need at least 2 complete rows for feature importance")

    out: dict[str, pd.DataFrame] = {}
    for target in targets:
        pipe = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(feature_cols)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        random_state=random_state,
                        min_samples_leaf=1,
                    ),
                ),
            ]
        )
        X_raw = model_df[feature_cols]
        y = model_df[target].astype(float)
        pipe.fit(X_raw, y)
        pre = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]
        names = list(pre.get_feature_names_out())
        impurity = pd.Series(model.feature_importances_, index=names, name="impurity_importance")

        # Permutation importance on the fitted pipeline (small-N: interpret cautiously).
        perm = permutation_importance(
            pipe,
            X_raw,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="neg_mean_absolute_error",
        )
        # Map permutation scores back to raw input columns (pre-one-hot).
        raw_perm = pd.DataFrame(
            {
                "feature": feature_cols,
                "permutation_importance_mean": perm.importances_mean,
                "permutation_importance_std": perm.importances_std,
            }
        ).sort_values("permutation_importance_mean", ascending=False)

        encoded = (
            impurity.reset_index()
            .rename(columns={"index": "encoded_feature", "impurity_importance": "importance"})
            .sort_values("importance", ascending=False)
        )
        encoded["target"] = target
        encoded["tier"] = tier
        raw_perm["target"] = target
        raw_perm["tier"] = tier
        out[f"{target}__encoded_impurity"] = encoded.reset_index(drop=True)
        out[f"{target}__raw_permutation"] = raw_perm.reset_index(drop=True)

    return out


def predictor_pca(
    participants: pd.DataFrame,
    *,
    tier: str = "transit",
    n_components: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """PCA over the one-hot/scaled predictor matrix used by the ML baselines."""
    model_df = prepare_modeling_frame(participants, tier=tier)
    feature_cols = available_feature_columns(model_df, tier)
    pre = make_preprocessor(feature_cols)
    X = pre.fit_transform(model_df[feature_cols])
    names = list(pre.get_feature_names_out())
    max_k = min(X.shape[0], X.shape[1], 6)
    if max_k < 1:
        raise ValueError("Not enough features/rows for predictor PCA")
    k = n_components or max_k
    k = max(1, min(k, max_k))
    pca = PCA(n_components=k, random_state=random_state)
    pca.fit(X)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=names,
        columns=[f"PC{i+1}" for i in range(k)],
    )
    variance = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(k)],
            "variance_explained": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    # Top contributing encoded features per component by absolute loading.
    top_rows = []
    for comp in loadings.columns:
        ranked = loadings[comp].abs().sort_values(ascending=False).head(8)
        for feat, abs_load in ranked.items():
            top_rows.append(
                {
                    "component": comp,
                    "encoded_feature": feat,
                    "loading": float(loadings.loc[feat, comp]),
                    "abs_loading": float(abs_load),
                }
            )
    return {
        "n_rows": int(model_df.shape[0]),
        "n_components": k,
        "loadings": loadings,
        "variance": variance,
        "top_loadings": pd.DataFrame(top_rows),
    }


def run_factor_and_importance_bundle(
    prolific_path: str | Path,
    qualtrics_path: str | Path,
    output_dir: str | Path,
    *,
    join_how: str = "inner",
    tier: str = "transit",
) -> dict[str, Path]:
    """Compute factor/importance artifacts and write CSVs under ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    qualtrics = load_qualtrics(qualtrics_path)
    items = likert_item_matrix(qualtrics, scored_direction=True)
    item_fa = run_item_factor_analysis(items)

    participants = load_and_prepare(prolific_path, qualtrics_path, how=join_how)
    importances = predictor_feature_importance(participants, tier=tier)
    pred_pca = predictor_pca(participants, tier=tier)

    paths: dict[str, Path] = {}
    paths["item_pca_loadings"] = out / "ca_item_pca_loadings.csv"
    paths["item_pca_variance"] = out / "ca_item_pca_variance.csv"
    paths["item_fa_loadings"] = out / "ca_item_fa_loadings.csv"
    paths["item_communalities"] = out / "ca_item_communalities.csv"
    item_fa["pca_loadings"].to_csv(paths["item_pca_loadings"])
    item_fa["pca_variance"].to_csv(paths["item_pca_variance"], index=False)
    item_fa["fa_loadings"].to_csv(paths["item_fa_loadings"])
    item_fa["communalities"].to_csv(paths["item_communalities"], index=False)

    for key, frame in importances.items():
        path = out / f"importance_{key}.csv"
        frame.to_csv(path, index=False)
        paths[key] = path

    paths["predictor_pca_variance"] = out / "predictor_pca_variance.csv"
    paths["predictor_pca_top_loadings"] = out / "predictor_pca_top_loadings.csv"
    pred_pca["variance"].to_csv(paths["predictor_pca_variance"], index=False)
    pred_pca["top_loadings"].to_csv(paths["predictor_pca_top_loadings"], index=False)

    # Compact ranked table of top raw features by mean permutation importance across targets.
    perm_frames = [
        frame for name, frame in importances.items() if name.endswith("__raw_permutation")
    ]
    if perm_frames:
        stacked = pd.concat(perm_frames, ignore_index=True)
        ranked = (
            stacked.groupby("feature", as_index=False)["permutation_importance_mean"]
            .mean()
            .sort_values("permutation_importance_mean", ascending=False)
        )
        paths["top_predictive_features"] = out / "top_predictive_features.csv"
        ranked.to_csv(paths["top_predictive_features"], index=False)

    return paths
