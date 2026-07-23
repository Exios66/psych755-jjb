"""Stage-one classical ML baselines for CA score prediction.

Mirrors the LLM persona information tiers and predicts the same targets:
ground-truth PRCA group and interpersonal subscale scores (6–30).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ca_personas.load import load_and_prepare
from ca_personas.personas import TIERS

TARGETS = ("gt_group_ca", "gt_interpersonal_ca")

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
EMPLOYMENT_FEATURES = ["Employment status"]
GEO_FEATURES = ["LocationLatitude", "LocationLongitude"]
TRANSIT_FEATURES = ["Q26", "Q27", "Q28", "Q29", "Q20", "Q21"]

TIER_FEATURES: dict[str, list[str]] = {
    "demos": DEMO_FEATURES,
    "employment": DEMO_FEATURES + EMPLOYMENT_FEATURES,
    "geo": DEMO_FEATURES + EMPLOYMENT_FEATURES + GEO_FEATURES,
    "transit": DEMO_FEATURES + EMPLOYMENT_FEATURES + GEO_FEATURES + TRANSIT_FEATURES,
}

NUMERIC_CANDIDATES = {"Age", "LocationLatitude", "LocationLongitude"}


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    estimator: Any


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


def baseline_models(*, n_neighbors: int = 3, random_state: int = 42) -> list[BaselineSpec]:
    """Return untuned stage-one estimators used as LLM comparison baselines."""
    return [
        BaselineSpec(
            name="random_forest",
            estimator=RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                min_samples_leaf=1,
            ),
        ),
        BaselineSpec(
            name="knn",
            estimator=KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance"),
        ),
    ]


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit RF/KNN under cross-validation for one information tier.

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

    for spec in baseline_models(n_neighbors=knn_k, random_state=random_state):
        pipe = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(feature_cols)),
                ("model", spec.estimator),
            ]
        )
        X = model_df[feature_cols]
        for target in targets:
            y = model_df[target].astype(float)
            y_hat = cross_val_predict(pipe, X, y, cv=cv)
            # Clamp to the legal PRCA subscale range for fair comparison with LLMs.
            y_hat = np.clip(y_hat, 6, 30)

            mae = float(mean_absolute_error(y, y_hat))
            rmse = float(np.sqrt(mean_squared_error(y, y_hat)))
            r2 = float(r2_score(y, y_hat)) if n >= 3 else float("nan")

            metric_rows.append(
                {
                    "tier": tier,
                    "model": spec.name,
                    "target": target,
                    "n_samples": n,
                    "n_features": len(feature_cols),
                    "cv": type(cv).__name__,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                }
            )

            side = "group" if "group" in target else "interpersonal"
            for pid, truth, pred in zip(
                model_df["participant_id"],
                y.to_numpy(),
                y_hat,
                strict=True,
            ):
                pred_rows.append(
                    {
                        "participant_id": pid,
                        "tier": tier,
                        "model": spec.name,
                        "target": target,
                        "side": side,
                        "y_true": float(truth),
                        "y_pred": float(pred),
                        "error": float(pred - truth),
                        "abs_error": float(abs(pred - truth)),
                    }
                )

    return pd.DataFrame(pred_rows), pd.DataFrame(metric_rows)


def run_stage_one_baselines(
    prolific_path: str | Path,
    qualtrics_path: str | Path,
    *,
    tiers: Iterable[str] = TIERS,
    join_how: str = "inner",
    n_neighbors: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data and evaluate RF/KNN baselines across all tiers."""
    participants = load_and_prepare(prolific_path, qualtrics_path, how=join_how)
    all_preds: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []

    for tier in tiers:
        preds, metrics = run_baselines_for_tier(
            participants,
            tier=tier,
            n_neighbors=n_neighbors,
            random_state=random_state,
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
        row: dict[str, Any] = {"model": model, "tier": tier, "n_samples": int(frame["n_samples"].iloc[0])}
        for _, r in frame.iterrows():
            side = "group" if "group" in r["target"] else "interpersonal"
            row[f"mae_{side}"] = r["mae"]
            row[f"rmse_{side}"] = r["rmse"]
            row[f"r2_{side}"] = r["r2"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "tier"]).reset_index(drop=True)


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
    }
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    metrics_wide(metrics).to_csv(paths["metrics_wide"], index=False)
    return paths
