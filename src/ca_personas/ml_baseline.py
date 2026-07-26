"""Stage-one classical / modern ML baselines for CA score prediction.

Mirrors the LLM persona information tiers and predicts the same targets:
ground-truth PRCA group and interpersonal subscale scores (6–30).

Default suite: Ridge, Elastic Net, k-NN, Random Forest, HistGradientBoosting,
XGBoost, and a small MLP neural net — a mix of linear, instance-based,
tree-ensemble, boosting, and neural baselines for LLM comparison.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ca_personas.load import load_and_prepare
from ca_personas.personas import BASE_DEMO_FIELDS, RESEARCH_TIERS
from ca_personas.scoring import (
    band_distance,
    ca_band,
    normalized_band_distance,
    normalized_score_distance,
)

TARGETS = ("gt_group_ca", "gt_interpersonal_ca")

# Mirror personas.demos_block / BASE_DEMO_FIELDS. Core File A/B demos are
# Age, Sex, Country of residence, and Student status; optional Prolific
# ethnicity / nationality / language / birth-country columns are used when present.
DEMO_FEATURES = [
    "Age",
    "Sex",
    "Ethnicity simplified",
    "Country of birth",
    "Country of residence",
    "Nationality",
    "Language",
    "Student status",
]
assert set(BASE_DEMO_FIELDS).issubset(DEMO_FEATURES)
EMPLOYMENT_FEATURES = ["Employment status"]
GEO_FEATURES = ["LocationLatitude", "LocationLongitude"]
TRANSIT_FEATURES = ["Q26", "Q27", "Q28", "Q29", "Q20", "Q21"]

_GEO_BASE = DEMO_FEATURES + EMPLOYMENT_FEATURES + GEO_FEATURES

TIER_FEATURES: dict[str, list[str]] = {
    "demos": DEMO_FEATURES,
    "employment": DEMO_FEATURES + EMPLOYMENT_FEATURES,
    "geo": _GEO_BASE,
    "transit": _GEO_BASE + TRANSIT_FEATURES,
    # v3 ablations (tabular). Voice open-text is LLM-only — ML uses geo-base features.
    "v3_rideshare": _GEO_BASE + ["Q28", "Q29"],
    "v3_public_transit": _GEO_BASE + ["Q26", "Q27"],
    "v3_voice": list(_GEO_BASE),
}

NUMERIC_CANDIDATES = {"Age", "LocationLatitude", "LocationLongitude"}

# Canonical suite order for tables / plots.
DEFAULT_MODEL_SUITE: tuple[str, ...] = (
    "ridge",
    "elastic_net",
    "knn",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
    "mlp",
)

# Kept for SHAP / lighter compare paths that expect tree + instance baselines.
CLASSIC_MODEL_SUITE: tuple[str, ...] = ("random_forest", "knn")

MODEL_LABELS: dict[str, str] = {
    "ridge": "Ridge",
    "elastic_net": "Elastic Net",
    "knn": "k-NN",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Hist. Gradient Boosting",
    "xgboost": "XGBoost",
    "mlp": "Neural net (MLP)",
}


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    estimator: Any
    family: str = "other"


def available_feature_columns(df: pd.DataFrame, tier: str) -> list[str]:
    if tier not in TIER_FEATURES:
        raise ValueError(f"Unknown tier {tier!r}; expected one of {tuple(TIER_FEATURES)}")
    return [c for c in TIER_FEATURES[tier] if c in df.columns]


def prepare_modeling_frame(
    df: pd.DataFrame,
    *,
    tier: str,
    targets: Iterable[str] = TARGETS,
) -> pd.DataFrame:
    """Keep rows with demographics + complete target scores for a tier."""
    features = available_feature_columns(df, tier)
    needed = ["participant_id", *features, *targets]
    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        raise ValueError(f"Data frame missing target columns: {missing_targets}")

    out = df[needed].copy()
    # Require Age (core demographic) and both CA targets.
    out = out.dropna(subset=["Age", *targets])
    out = out.reset_index(drop=True)
    return out


def split_feature_types(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    numeric = [c for c in feature_cols if c in NUMERIC_CANDIDATES]
    categorical = [c for c in feature_cols if c not in NUMERIC_CANDIDATES]
    return numeric, categorical


def make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    numeric, categorical = split_feature_types(feature_cols)

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns for preprocessor")
    return ColumnTransformer(transformers=transformers)


def _make_xgboost(*, random_state: int) -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - exercised when dep missing
        raise ImportError(
            "xgboost is required for the stage-one ML suite. "
            "Install with: pip install 'xgboost>=2.0'"
        ) from exc
    return XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=1,
        verbosity=0,
    )


def baseline_models(
    *,
    n_neighbors: int = 3,
    random_state: int = 42,
    include: Sequence[str] | None = None,
) -> list[BaselineSpec]:
    """Return stage-one estimators used as LLM comparison baselines.

    Parameters
    ----------
    include :
        Optional subset of model names. ``None`` selects the full default suite.
    """
    wanted = list(DEFAULT_MODEL_SUITE if include is None else include)
    unknown = sorted(set(wanted) - set(DEFAULT_MODEL_SUITE))
    if unknown:
        raise ValueError(
            f"Unknown model name(s) {unknown}; expected subset of {DEFAULT_MODEL_SUITE}"
        )

    builders: dict[str, BaselineSpec] = {
        "ridge": BaselineSpec(
            name="ridge",
            family="linear",
            estimator=Ridge(alpha=1.0, random_state=random_state),
        ),
        "elastic_net": BaselineSpec(
            name="elastic_net",
            family="linear",
            estimator=ElasticNet(
                alpha=0.05,
                l1_ratio=0.5,
                max_iter=5000,
                random_state=random_state,
            ),
        ),
        "knn": BaselineSpec(
            name="knn",
            family="instance",
            estimator=KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance"),
        ),
        "random_forest": BaselineSpec(
            name="random_forest",
            family="tree_ensemble",
            estimator=RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                min_samples_leaf=1,
            ),
        ),
        "hist_gradient_boosting": BaselineSpec(
            name="hist_gradient_boosting",
            family="boosting",
            estimator=HistGradientBoostingRegressor(
                max_depth=4,
                learning_rate=0.08,
                max_iter=200,
                random_state=random_state,
            ),
        ),
        "mlp": BaselineSpec(
            name="mlp",
            family="neural",
            # Adam + modest width is stabler under 5-fold CV than lbfgs on
            # one-hot demographic matrices (avoids iteration-limit warnings).
            estimator=MLPRegressor(
                hidden_layer_sizes=(32,),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=400,
                tol=1e-3,
                random_state=random_state,
            ),
        ),
    }
    if "xgboost" in wanted:
        builders["xgboost"] = BaselineSpec(
            name="xgboost",
            family="boosting",
            estimator=_make_xgboost(random_state=random_state),
        )
    return [builders[name] for name in wanted]


def choose_cv(n_samples: int, *, random_state: int = 42):
    """Leave-one-out for tiny excerpt samples; 5-fold otherwise."""
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for cross-validated baselines")
    if n_samples < 8:
        return LeaveOneOut()
    n_splits = min(5, n_samples)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _safe_n_neighbors(n_samples: int, requested: int) -> int:
    # In LOOCV each fold trains on n-1 rows.
    return max(1, min(requested, n_samples - 1))


def run_baselines_for_tier(
    df: pd.DataFrame,
    *,
    tier: str,
    targets: Iterable[str] = TARGETS,
    n_neighbors: int = 3,
    random_state: int = 42,
    models: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit the ML suite under cross-validation for one information tier.

    Returns
    -------
    predictions : long-form predictions with absolute errors
    metrics : one row per model × target with MAE / RMSE / R²
    """
    model_df = prepare_modeling_frame(df, tier=tier, targets=targets)
    feature_cols = available_feature_columns(model_df, tier)
    if model_df.empty:
        raise ValueError(f"No complete rows available for tier={tier}")

    n = len(model_df)
    cv = choose_cv(n, random_state=random_state)
    knn_k = _safe_n_neighbors(n, n_neighbors)

    pred_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for spec in baseline_models(
        n_neighbors=knn_k, random_state=random_state, include=models
    ):
        pipe = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(feature_cols)),
                ("model", spec.estimator),
            ]
        )
        X = model_df[feature_cols].copy()
        # SimpleImputer on pandas object columns with NA can fail on some
        # sklearn builds; stringify categoricals so imputation is stable.
        _, categorical = split_feature_types(feature_cols)
        for col in categorical:
            X[col] = X[col].astype("string").fillna("<NA>").astype(str)
        for target in targets:
            y = model_df[target].astype(float)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                y_hat = cross_val_predict(pipe, X, y, cv=cv)
            # Clamp to the legal PRCA subscale range for fair comparison with LLMs.
            y_hat = np.clip(y_hat, 6, 30)

            mae = float(mean_absolute_error(y, y_hat))
            rmse = float(np.sqrt(mean_squared_error(y, y_hat)))
            r2 = float(r2_score(y, y_hat)) if n >= 3 else float("nan")
            exact = y.to_numpy().round() == np.rint(y_hat)
            exact_acc = float(exact.mean())
            true_bands = [ca_band(int(v)) for v in y.to_numpy().round().astype(int)]
            pred_bands = [ca_band(int(v)) for v in np.rint(y_hat).astype(int)]
            band_match = np.array(
                [tb == pb and tb is not None for tb, pb in zip(true_bands, pred_bands, strict=True)]
            )
            band_acc = float(band_match.mean()) if len(band_match) else float("nan")
            band_distances = [
                band_distance(pb, tb) for pb, tb in zip(pred_bands, true_bands, strict=True)
            ]
            mean_band_dist = float(np.mean([d for d in band_distances if d is not None]))
            mean_norm_score_dist = float(
                np.mean([normalized_score_distance(abs(float(p - t))) for t, p in zip(y, y_hat)])
            )
            mean_norm_band_dist = float(
                np.mean(
                    [
                        normalized_band_distance(d)
                        for d in band_distances
                        if d is not None
                    ]
                )
            )

            metric_rows.append(
                {
                    "tier": tier,
                    "model": spec.name,
                    "model_label": MODEL_LABELS.get(spec.name, spec.name),
                    "model_family": spec.family,
                    "target": target,
                    "n_samples": n,
                    "n_features": len(feature_cols),
                    "cv": type(cv).__name__,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "exact_acc": exact_acc,
                    "band_acc": band_acc,
                    "mean_band_distance": mean_band_dist,
                    "mean_norm_score_distance": mean_norm_score_dist,
                    "mean_norm_band_distance": mean_norm_band_dist,
                }
            )

            side = "group" if "group" in target else "interpersonal"
            for pid, truth, pred, tb, pb, em, bm, bd in zip(
                model_df["participant_id"],
                y.to_numpy(),
                y_hat,
                true_bands,
                pred_bands,
                exact,
                band_match,
                band_distances,
                strict=True,
            ):
                abs_err = float(abs(pred - truth))
                pred_rows.append(
                    {
                        "participant_id": pid,
                        "tier": tier,
                        "model": spec.name,
                        "model_label": MODEL_LABELS.get(spec.name, spec.name),
                        "model_family": spec.family,
                        "target": target,
                        "side": side,
                        "y_true": float(truth),
                        "y_pred": float(pred),
                        "error": float(pred - truth),
                        "abs_error": abs_err,
                        "score_distance": abs_err,
                        "norm_score_distance": normalized_score_distance(abs_err),
                        "gt_band": tb,
                        "pred_band": pb,
                        "exact_match": bool(em),
                        "band_match": bool(bm),
                        "band_distance": bd,
                        "norm_band_distance": normalized_band_distance(bd),
                    }
                )

    return pd.DataFrame(pred_rows), pd.DataFrame(metric_rows)


def run_stage_one_baselines(
    prolific_path: str | Path | list[Path],
    qualtrics_path: str | Path,
    *,
    tiers: Iterable[str] = RESEARCH_TIERS,
    join_how: str = "inner",
    n_neighbors: int = 3,
    random_state: int = 42,
    models: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data and evaluate the ML suite across research tiers."""
    participants = load_and_prepare(
        prolific_path,
        qualtrics_path,
        how=join_how,
        clean=True,
    )
    all_preds: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []

    # ML baselines use tabular tiers only (not free-text "full" persona tier).
    selected = [t for t in tiers if t != "full"]
    for tier in selected:
        preds, metrics = run_baselines_for_tier(
            participants,
            tier=tier,
            n_neighbors=n_neighbors,
            random_state=random_state,
            models=models,
        )
        all_preds.append(preds)
        all_metrics.append(metrics)

    predictions = pd.concat(all_preds, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    return participants, predictions, metrics


def metrics_wide(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pivot MAE into a model × tier table for group and interpersonal targets."""
    rows: list[dict[str, Any]] = []
    for (model, tier), frame in metrics.groupby(["model", "tier"]):
        row: dict[str, Any] = {
            "model": model,
            "model_label": MODEL_LABELS.get(str(model), str(model)),
            "tier": tier,
            "n_samples": int(frame["n_samples"].iloc[0]),
        }
        if "model_family" in frame.columns:
            row["model_family"] = frame["model_family"].iloc[0]
        for _, r in frame.iterrows():
            side = "group" if "group" in r["target"] else "interpersonal"
            row[f"mae_{side}"] = r["mae"]
            row[f"rmse_{side}"] = r["rmse"]
            row[f"r2_{side}"] = r["r2"]
            row[f"exact_acc_{side}"] = r.get("exact_acc")
            row[f"band_acc_{side}"] = r.get("band_acc")
            row[f"mean_band_distance_{side}"] = r.get("mean_band_distance")
            row[f"mean_norm_score_distance_{side}"] = r.get("mean_norm_score_distance")
            row[f"mean_norm_band_distance_{side}"] = r.get("mean_norm_band_distance")
        rows.append(row)
    wide = pd.DataFrame(rows)
    if wide.empty:
        return wide
    model_order = {m: i for i, m in enumerate(DEFAULT_MODEL_SUITE)}
    tier_order = {t: i for i, t in enumerate(("demos", "employment", "geo", "transit"))}
    wide["_m"] = wide["model"].map(model_order).fillna(99)
    wide["_t"] = wide["tier"].map(tier_order).fillna(99)
    return (
        wide.sort_values(["_m", "_t"])
        .drop(columns=["_m", "_t"])
        .reset_index(drop=True)
    )


def leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    """Best model (lowest MAE) per tier × target, plus runner-up delta."""
    rows: list[dict[str, Any]] = []
    for (tier, target), frame in metrics.groupby(["tier", "target"]):
        ordered = frame.sort_values(["mae", "rmse", "model"]).reset_index(drop=True)
        best = ordered.iloc[0]
        second_mae = float(ordered.iloc[1]["mae"]) if len(ordered) > 1 else float("nan")
        rows.append(
            {
                "tier": tier,
                "target": target,
                "best_model": best["model"],
                "best_model_label": MODEL_LABELS.get(str(best["model"]), str(best["model"])),
                "best_mae": float(best["mae"]),
                "best_rmse": float(best["rmse"]),
                "best_r2": float(best["r2"]),
                "best_band_acc": float(best["band_acc"]),
                "runner_up_mae": second_mae,
                "mae_gap_to_second": float(second_mae - best["mae"])
                if np.isfinite(second_mae)
                else float("nan"),
                "n_models": int(len(ordered)),
                "n_samples": int(best["n_samples"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    tier_order = {t: i for i, t in enumerate(("demos", "employment", "geo", "transit"))}
    out["_t"] = out["tier"].map(tier_order).fillna(99)
    return out.sort_values(["_t", "target"]).drop(columns=["_t"]).reset_index(drop=True)


def mae_pivot(
    metrics: pd.DataFrame,
    *,
    target: str = "gt_group_ca",
    value: str = "mae",
) -> pd.DataFrame:
    """Model × tier pivot of a metric for one target (for docs/tables)."""
    frame = metrics[metrics["target"] == target].copy()
    if frame.empty:
        return frame
    pivot = frame.pivot(index="model", columns="tier", values=value)
    ordered_models = [m for m in DEFAULT_MODEL_SUITE if m in pivot.index] + [
        m for m in pivot.index if m not in DEFAULT_MODEL_SUITE
    ]
    ordered_tiers = [t for t in ("demos", "employment", "geo", "transit") if t in pivot.columns]
    pivot = pivot.reindex(index=ordered_models, columns=ordered_tiers)
    pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]
    pivot.index.name = "model"
    return pivot.reset_index()


def save_baseline_artifacts(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": out / "ml_baseline_predictions.csv",
        "metrics": out / "ml_baseline_metrics.csv",
        "metrics_wide": out / "ml_baseline_metrics_wide.csv",
        "leaderboard": out / "ml_baseline_leaderboard.csv",
        "mae_pivot_group": out / "ml_baseline_mae_pivot_group.csv",
        "mae_pivot_interpersonal": out / "ml_baseline_mae_pivot_interpersonal.csv",
    }
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    metrics_wide(metrics).to_csv(paths["metrics_wide"], index=False)
    leaderboard(metrics).to_csv(paths["leaderboard"], index=False)
    mae_pivot(metrics, target="gt_group_ca").to_csv(paths["mae_pivot_group"], index=False)
    mae_pivot(metrics, target="gt_interpersonal_ca").to_csv(
        paths["mae_pivot_interpersonal"], index=False
    )
    return paths
