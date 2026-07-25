"""Secondary RQ: which available predictors best forecast regular transit use?

Builds a mixed-type Random Forest over demographics, employment, geolocation,
car access, ride-share behavior, and PRCA scores; compares feature-group
ablations; tunes hyperparameters for ROC-AUC; and reports permutation /
impurity feature importance. An optional upper-bound model adds Q27
(public-transit rides on a typical use day), which is proximal to the Q26
outcome and is therefore labeled separately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ca_personas.geo_transit_rf import classification_metrics, majority_baseline_probs
from ca_personas.load import load_full_cohort
from ca_personas.transit_ca import PRIMARY_REGULAR_LABELS, label_regular_riders

TARGET = "regular_transit"

FEATURE_GROUPS: dict[str, list[str]] = {
    "demographics": ["Age", "Sex", "Country of residence", "Student status"],
    "employment": ["Employment status"],
    "geography": ["LocationLatitude", "LocationLongitude"],
    "car_access": ["Q20", "Q21"],
    "rideshare": ["Q28", "Q29"],
    "ca_scores": ["gt_group_ca", "gt_interpersonal_ca"],
}

# Proximal to the outcome (public-transit intensity on a use day).
AUX_TRANSIT_FEATURES = ["Q27"]

NUMERIC_CANDIDATES = {
    "Age",
    "LocationLatitude",
    "LocationLongitude",
    "gt_group_ca",
    "gt_interpersonal_ca",
}

CATEGORICAL_CANDIDATES = {
    "Sex",
    "Country of residence",
    "Student status",
    "Employment status",
    "Q20",
    "Q21",
    "Q27",
    "Q28",
    "Q29",
}

# Compact search space suitable for n≈240 and nested CV.
PARAM_DISTRIBUTIONS: dict[str, list[Any]] = {
    "rf__n_estimators": [300, 500, 800],
    "rf__max_depth": [None, 4, 6, 10, 16],
    "rf__min_samples_leaf": [1, 2, 3, 5, 8],
    "rf__max_features": ["sqrt", "log2", 0.5, None],
    "rf__class_weight": ["balanced", "balanced_subsample"],
}


def primary_features(available: Sequence[str] | None = None) -> list[str]:
    """Defensible kitchen-sink predictors (excludes Q26 and Q27)."""
    feats: list[str] = []
    for group in FEATURE_GROUPS.values():
        feats.extend(group)
    if available is not None:
        avail = set(available)
        feats = [f for f in feats if f in avail]
    return feats


def upper_bound_features(available: Sequence[str] | None = None) -> list[str]:
    feats = primary_features(available) + list(AUX_TRANSIT_FEATURES)
    if available is not None:
        avail = set(available)
        feats = [f for f in feats if f in avail]
    return feats


def prepare_comprehensive_transit_frame(
    df: pd.DataFrame,
    *,
    regular_labels: Sequence[str] = tuple(PRIMARY_REGULAR_LABELS),
    require_geo: bool = True,
) -> pd.DataFrame:
    """Label regular riders and retain rows with outcome + CA scores.

    Geography is required by default so primary vs geo-only comparisons share
    the same analytic sample used elsewhere (n≈241). Car/rideshare items may
    be missing and are imputed inside the model pipeline.
    """
    labeled = label_regular_riders(df, regular_labels=regular_labels)
    id_cols = ["participant_id", "Q26", TARGET, "transit_group"]
    feature_cols = [
        c
        for c in primary_features(labeled.columns) + AUX_TRANSIT_FEATURES
        if c in labeled.columns
    ]
    cols = [c for c in id_cols + feature_cols if c in labeled.columns]
    out = labeled[cols].copy()

    must_have = [TARGET, "gt_group_ca", "gt_interpersonal_ca"]
    if require_geo:
        must_have.extend(["LocationLatitude", "LocationLongitude"])
    out = out.dropna(subset=[c for c in must_have if c in out.columns])
    out[TARGET] = out[TARGET].astype(bool)
    out["y"] = out[TARGET].astype(int)
    return out.reset_index(drop=True)


def _split_feature_types(features: Sequence[str]) -> tuple[list[str], list[str]]:
    numeric = [f for f in features if f in NUMERIC_CANDIDATES]
    categorical = [f for f in features if f in CATEGORICAL_CANDIDATES]
    unknown = [f for f in features if f not in numeric and f not in categorical]
    if unknown:
        raise ValueError(f"Unhandled feature types: {unknown}")
    return numeric, categorical


def make_rf_pipeline(
    features: Sequence[str],
    *,
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 3,
    max_features: str | float | None = "sqrt",
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> Pipeline:
    numeric, categorical = _split_feature_types(features)
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
                list(numeric),
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
                list(categorical),
            )
        )
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("prep", pre), ("rf", rf)])


def _cv(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    n_splits: int,
    random_state: int,
    model_name: str,
    pipeline: Pipeline | None = None,
) -> dict[str, Any]:
    X = frame[list(features)]
    y = frame["y"].to_numpy()
    n_splits = min(n_splits, int(np.min(np.bincount(y))))
    if n_splits < 2:
        raise ValueError("Need at least 2 examples per class for stratified CV")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipe = pipeline or make_rf_pipeline(features, random_state=random_state)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    metrics = classification_metrics(y, proba)
    metrics["model"] = model_name
    metrics["n_splits"] = int(n_splits)
    metrics["n_features"] = int(len(features))
    metrics["features"] = list(features)
    fpr, tpr, thr = roc_curve(y, proba)
    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
    return {
        "metrics": metrics,
        "proba": proba,
        "roc": roc,
        "cv": cv,
        "pipeline_template": pipe,
        "features": list(features),
    }


def null_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    y = frame["y"].to_numpy()
    prev = majority_baseline_probs(y)
    mode = int(np.round(y.mean()))
    hard = np.full(len(y), mode, dtype=float)
    return pd.DataFrame(
        [
            {"model": "prevalence_prob", **classification_metrics(y, prev)},
            {"model": "majority_class", **classification_metrics(y, hard)},
        ]
    )


def univariate_auc_table(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Single-feature RF ROC-AUC for each candidate predictor."""
    rows: list[dict[str, Any]] = []
    for feat in features:
        try:
            result = _cv(
                frame,
                [feat],
                n_splits=n_splits,
                random_state=random_state,
                model_name=f"univariate_{feat}",
            )
            auc = float(result["metrics"]["roc_auc"])
        except Exception as exc:  # pragma: no cover - rare single-feature CV failures
            import warnings

            warnings.warn(
                f"Univariate AUC failed for feature={feat!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            auc = float("nan")
        group = next(
            (g for g, cols in FEATURE_GROUPS.items() if feat in cols),
            "aux_transit" if feat in AUX_TRANSIT_FEATURES else "other",
        )
        rows.append(
            {
                "feature": feat,
                "feature_group": group,
                "univariate_roc_auc": auc,
                "n_non_missing": int(frame[feat].notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "univariate_roc_auc", ascending=False, na_position="last"
    ).reset_index(drop=True)


def group_ablation_table(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    """Cumulative and leave-one-group-out ROC-AUC comparisons."""
    available = [c for c in primary_features(frame.columns) if c in frame.columns]
    rows: list[dict[str, Any]] = []

    # Single groups
    for group_name, cols in FEATURE_GROUPS.items():
        feats = [c for c in cols if c in available]
        if not feats:
            continue
        res = _cv(
            frame,
            feats,
            n_splits=n_splits,
            random_state=random_state,
            model_name=f"group_only_{group_name}",
        )
        rows.append(
            {
                "comparison": "group_only",
                "group": group_name,
                "n_features": len(feats),
                "roc_auc": res["metrics"]["roc_auc"],
                "average_precision": res["metrics"]["average_precision"],
                "balanced_accuracy": res["metrics"]["balanced_accuracy"],
            }
        )

    # Cumulative kitchen sink
    cumulative: list[str] = []
    for group_name, cols in FEATURE_GROUPS.items():
        cumulative.extend([c for c in cols if c in available])
        res = _cv(
            frame,
            cumulative,
            n_splits=n_splits,
            random_state=random_state,
            model_name=f"cumulative_through_{group_name}",
        )
        rows.append(
            {
                "comparison": "cumulative",
                "group": group_name,
                "n_features": len(cumulative),
                "roc_auc": res["metrics"]["roc_auc"],
                "average_precision": res["metrics"]["average_precision"],
                "balanced_accuracy": res["metrics"]["balanced_accuracy"],
            }
        )

    # Leave-one-group-out from full primary set
    full = list(available)
    full_res = _cv(
        frame,
        full,
        n_splits=n_splits,
        random_state=random_state,
        model_name="kitchen_sink_default",
    )
    full_auc = float(full_res["metrics"]["roc_auc"])
    rows.append(
        {
            "comparison": "kitchen_sink",
            "group": "all_primary",
            "n_features": len(full),
            "roc_auc": full_auc,
            "average_precision": full_res["metrics"]["average_precision"],
            "balanced_accuracy": full_res["metrics"]["balanced_accuracy"],
        }
    )
    for group_name, cols in FEATURE_GROUPS.items():
        reduced = [c for c in full if c not in cols]
        if len(reduced) == len(full):
            continue
        res = _cv(
            frame,
            reduced,
            n_splits=n_splits,
            random_state=random_state,
            model_name=f"drop_{group_name}",
        )
        rows.append(
            {
                "comparison": "leave_one_group_out",
                "group": group_name,
                "n_features": len(reduced),
                "roc_auc": res["metrics"]["roc_auc"],
                "delta_auc_vs_full": float(res["metrics"]["roc_auc"] - full_auc),
                "average_precision": res["metrics"]["average_precision"],
                "balanced_accuracy": res["metrics"]["balanced_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def tune_random_forest(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    n_splits: int = 5,
    n_iter: int = 24,
    random_state: int = 42,
) -> dict[str, Any]:
    """RandomizedSearchCV maximizing ROC-AUC on the modeling frame."""
    X = frame[list(features)]
    y = frame["y"].to_numpy()
    n_splits = min(n_splits, int(np.min(np.bincount(y))))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipe = make_rf_pipeline(features, random_state=random_state)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    # Honest OOF probabilities with the selected hyperparameters (same outer folds).
    best_pipe = clone(search.best_estimator_)
    proba = cross_val_predict(
        best_pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    metrics = classification_metrics(y, proba)
    metrics["model"] = "kitchen_sink_tuned"
    metrics["n_splits"] = int(n_splits)
    metrics["n_features"] = int(len(features))
    metrics["features"] = list(features)
    fpr, tpr, thr = roc_curve(y, proba)
    cv_results = pd.DataFrame(search.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )
    return {
        "search": search,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "metrics": metrics,
        "proba": proba,
        "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}),
        "cv_results": cv_results,
        "pipeline": search.best_estimator_,
        "features": list(features),
    }


def original_feature_permutation_importance(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance on original columns (handles OHE via the pipeline)."""
    X = frame[list(features)]
    y = frame["y"].to_numpy()
    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )
    rows = []
    for feat, mean, std in zip(
        features, result.importances_mean, result.importances_std, strict=True
    ):
        group = next(
            (g for g, cols in FEATURE_GROUPS.items() if feat in cols),
            "aux_transit" if feat in AUX_TRANSIT_FEATURES else "other",
        )
        rows.append(
            {
                "feature": feat,
                "feature_group": group,
                "importance_perm_mean": float(mean),
                "importance_perm_std": float(std),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("importance_perm_mean", ascending=False)
        .reset_index(drop=True)
    )


def encoded_gini_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Impurity importance on the transformed (one-hot) feature space."""
    prep: ColumnTransformer = pipeline.named_steps["prep"]
    rf: RandomForestClassifier = pipeline.named_steps["rf"]
    names = list(prep.get_feature_names_out())
    return (
        pd.DataFrame(
            {"feature_encoded": names, "importance_gini": rf.feature_importances_}
        )
        .sort_values("importance_gini", ascending=False)
        .reset_index(drop=True)
    )


def _interpret(
    tuned_auc: float,
    default_auc: float,
    upper_auc: float | None,
    top_features: Sequence[str],
    strongest_group: str | None,
) -> str:
    if np.isnan(tuned_auc):
        return "Unable to estimate ROC-AUC (degenerate class labels)."
    strength = (
        "strong"
        if tuned_auc >= 0.75
        else (
            "moderate"
            if tuned_auc >= 0.65
            else ("modest" if tuned_auc > 0.55 else "weak")
        )
    )
    top = ", ".join(top_features[:5]) if top_features else "n/a"
    upper_bit = (
        f" Including proximal Q27 raises CV ROC-AUC to {upper_auc:.3f} (upper-bound)."
        if upper_auc is not None and not np.isnan(upper_auc)
        else ""
    )
    group_bit = (
        f" Strongest single feature group: {strongest_group}."
        if strongest_group
        else ""
    )
    return (
        f"Tuned kitchen-sink RF CV ROC-AUC = {tuned_auc:.3f} ({strength} discrimination); "
        f"default RF = {default_auc:.3f}.{group_bit} "
        f"Top permutation-importance features: {top}.{upper_bit}"
    )


def run_comprehensive_transit_rf_analysis(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    n_tune_iter: int = 24,
    random_state: int = 42,
    include_upper_bound: bool = True,
) -> dict[str, Any]:
    frame = prepare_comprehensive_transit_frame(df)
    if len(frame) < 20:
        raise ValueError(f"Analytic frame too small (n={len(frame)})")

    primary = primary_features(frame.columns)
    univariate = univariate_auc_table(
        frame, primary + [c for c in AUX_TRANSIT_FEATURES if c in frame.columns],
        n_splits=n_splits,
        random_state=random_state,
    )
    ablations = group_ablation_table(
        frame, n_splits=n_splits, random_state=random_state
    )

    default = _cv(
        frame,
        primary,
        n_splits=n_splits,
        random_state=random_state,
        model_name="kitchen_sink_default",
    )
    tuned = tune_random_forest(
        frame,
        primary,
        n_splits=n_splits,
        n_iter=n_tune_iter,
        random_state=random_state,
    )

    upper: dict[str, Any] | None = None
    if include_upper_bound and all(c in frame.columns for c in AUX_TRANSIT_FEATURES):
        ub_feats = upper_bound_features(frame.columns)
        upper = _cv(
            frame,
            ub_feats,
            n_splits=n_splits,
            random_state=random_state,
            model_name="kitchen_sink_plus_q27",
            pipeline=make_rf_pipeline(
                ub_feats,
                random_state=random_state,
                **{
                    k.replace("rf__", ""): v
                    for k, v in tuned["best_params"].items()
                },
            ),
        )

    pipe = tuned["pipeline"]
    perm = original_feature_permutation_importance(
        pipe,
        frame,
        primary,
        n_repeats=n_perm_repeats,
        random_state=random_state,
    )
    gini = encoded_gini_importance(pipe)

    nulls = null_baselines(frame)
    metric_rows = [
        default["metrics"],
        tuned["metrics"],
        *nulls.to_dict(orient="records"),
    ]
    if upper is not None:
        metric_rows.insert(2, upper["metrics"])
    # Prior single-RQ baselines for context
    for group_name in ("geography", "ca_scores", "car_access", "rideshare"):
        row = ablations[
            (ablations["comparison"] == "group_only") & (ablations["group"] == group_name)
        ]
        if not row.empty:
            m = {
                "model": f"group_only_{group_name}",
                "n": int(len(frame)),
                "roc_auc": float(row.iloc[0]["roc_auc"]),
                "average_precision": float(row.iloc[0]["average_precision"]),
                "balanced_accuracy": float(row.iloc[0]["balanced_accuracy"]),
                "n_features": int(row.iloc[0]["n_features"]),
            }
            metric_rows.append(m)
    metrics_table = pd.DataFrame(metric_rows)

    oof = frame[["participant_id", "Q26", TARGET, "y"]].copy()
    oof["y_prob_default"] = default["proba"]
    oof["y_prob_tuned"] = tuned["proba"]
    if upper is not None:
        oof["y_prob_plus_q27"] = upper["proba"]
    oof["y_pred_tuned"] = (tuned["proba"] >= 0.5).astype(int)

    group_only = ablations[ablations["comparison"] == "group_only"]
    strongest_group = (
        str(group_only.sort_values("roc_auc", ascending=False).iloc[0]["group"])
        if not group_only.empty
        else None
    )
    top_features = perm["feature"].head(8).tolist()
    tuned_auc = float(tuned["metrics"]["roc_auc"])
    default_auc = float(default["metrics"]["roc_auc"])
    upper_auc = float(upper["metrics"]["roc_auc"]) if upper is not None else None

    summary = {
        "secondary_rq": (
            "Among available Prolific/Qualtrics fields (excluding direct Q26 "
            "leakage), which predictors most powerfully discriminate regular "
            "public-transit use in a Random Forest maximized for ROC-AUC?"
        ),
        "outcome": {
            "name": "regular_transit",
            "definition": (
                "Q26 in {4-8 days a month, 8 or more days a month} "
                "(weekly-or-more public transit)"
            ),
        },
        "feature_groups": FEATURE_GROUPS,
        "primary_features": primary,
        "sample": {
            "n": int(len(frame)),
            "n_regular": int(frame["y"].sum()),
            "n_not_regular": int((frame["y"] == 0).sum()),
            "prevalence": float(frame["y"].mean()),
        },
        "tuning": {
            "n_iter": int(n_tune_iter),
            "best_params": {
                k: (None if v is None else v) for k, v in tuned["best_params"].items()
            },
            "best_cv_score_refit_search": float(tuned["best_cv_score"]),
        },
        "cv_metrics": {
            "default": default["metrics"],
            "tuned": tuned["metrics"],
            "upper_bound_plus_q27": upper["metrics"] if upper is not None else None,
        },
        "top_permutation_features": perm.head(10).to_dict(orient="records"),
        "strongest_feature_group": strongest_group,
        "verdict": {
            "tuned_roc_auc": tuned_auc,
            "default_roc_auc": default_auc,
            "upper_bound_roc_auc": upper_auc,
            "beats_chance": bool(tuned_auc > 0.5),
            "top_features": top_features,
            "interpretation": _interpret(
                tuned_auc, default_auc, upper_auc, top_features, strongest_group
            ),
        },
        "caveats": [
            "Q26 is the outcome source and is never used as a predictor.",
            "Q27 (rides on a typical public-transit day) is only in the labeled upper-bound model.",
            "Car/rideshare items have substantial missingness and are imputed in-pipeline.",
            "Tuning uses the same outer folds for selection and OOF scoring; treat tuned AUC as mildly optimistic.",
            "Lat/lon are approximate Qualtrics survey geolocation, not verified home addresses.",
            "Associations are same-wave and do not establish causality.",
        ],
    }

    return {
        "frame": frame,
        "metrics_table": metrics_table,
        "ablations": ablations,
        "univariate": univariate,
        "oof": oof,
        "roc_default": default["roc"],
        "roc_tuned": tuned["roc"],
        "roc_upper": upper["roc"] if upper is not None else None,
        "pipeline": pipe,
        "best_params": tuned["best_params"],
        "tune_cv_results": tuned["cv_results"][
            [
                c
                for c in tuned["cv_results"].columns
                if c.startswith("param_") or c in {"mean_test_score", "std_test_score", "rank_test_score"}
            ]
        ].head(20),
        "gini_importance": gini,
        "permutation_importance": perm,
        "null_baselines": nulls,
        "summary": summary,
    }


def save_comprehensive_transit_rf_artifacts(
    analysis: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    mapping = {
        "frame": "comprehensive_transit_modeling_frame.csv",
        "metrics_table": "comprehensive_transit_rf_metrics.csv",
        "ablations": "comprehensive_transit_rf_ablations.csv",
        "univariate": "comprehensive_transit_rf_univariate_auc.csv",
        "oof": "comprehensive_transit_rf_oof_predictions.csv",
        "roc_tuned": "comprehensive_transit_rf_roc_curve.csv",
        "gini_importance": "comprehensive_transit_rf_gini_importance.csv",
        "permutation_importance": "comprehensive_transit_rf_permutation_importance.csv",
        "null_baselines": "comprehensive_transit_rf_null_baselines.csv",
        "tune_cv_results": "comprehensive_transit_rf_tune_results.csv",
    }
    for key, filename in mapping.items():
        frame = analysis.get(key)
        if isinstance(frame, pd.DataFrame):
            path = out / filename
            frame.to_csv(path, index=False)
            paths[key] = path

    if isinstance(analysis.get("roc_default"), pd.DataFrame):
        path = out / "comprehensive_transit_rf_roc_default.csv"
        analysis["roc_default"].to_csv(path, index=False)
        paths["roc_default"] = path
    if isinstance(analysis.get("roc_upper"), pd.DataFrame):
        path = out / "comprehensive_transit_rf_roc_upper.csv"
        analysis["roc_upper"].to_csv(path, index=False)
        paths["roc_upper"] = path

    summary_path = out / "comprehensive_transit_rf_summary.json"
    summary_path.write_text(json.dumps(analysis["summary"], indent=2), encoding="utf-8")
    paths["summary"] = summary_path

    card = {
        "secondary_rq": analysis["summary"]["secondary_rq"],
        "sample": analysis["summary"]["sample"],
        "tuning": analysis["summary"]["tuning"],
        "cv_metrics": {
            "tuned_roc_auc": analysis["summary"]["verdict"]["tuned_roc_auc"],
            "default_roc_auc": analysis["summary"]["verdict"]["default_roc_auc"],
            "upper_bound_roc_auc": analysis["summary"]["verdict"]["upper_bound_roc_auc"],
        },
        "strongest_feature_group": analysis["summary"]["strongest_feature_group"],
        "top_permutation_features": analysis["summary"]["top_permutation_features"],
        "verdict": analysis["summary"]["verdict"],
        "caveats": analysis["summary"]["caveats"],
    }
    card_path = out / "comprehensive_transit_rf_results_card.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    paths["results_card"] = card_path
    return paths


def run_comprehensive_transit_rf_pipeline(
    *,
    prolific_paths: Sequence[str | Path] | None = None,
    qualtrics_path: str | Path | None = None,
    join_how: str = "inner",
    output_dir: str | Path = "outputs/comprehensive_transit_rf",
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    n_tune_iter: int = 24,
    random_state: int = 42,
    include_upper_bound: bool = True,
) -> dict[str, Path]:
    participants, _report = load_full_cohort(
        prolific_paths=prolific_paths,
        qualtrics_path=qualtrics_path,
        join_how=join_how,
    )
    analysis = run_comprehensive_transit_rf_analysis(
        participants,
        n_splits=n_splits,
        n_perm_repeats=n_perm_repeats,
        n_tune_iter=n_tune_iter,
        random_state=random_state,
        include_upper_bound=include_upper_bound,
    )
    return save_comprehensive_transit_rf_artifacts(analysis, output_dir)
