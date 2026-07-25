"""Secondary RQ: does geography (lat/long) predict regular public-transit use?

Uses Qualtrics ``LocationLatitude`` / ``LocationLongitude`` (approximate survey
geolocation) to classify the Q26-based ``regular_transit`` outcome with a
Random Forest, then reports cross-validated predictive performance, null
baselines, and feature importances.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ca_personas.load import load_full_cohort
from ca_personas.transit_ca import PRIMARY_REGULAR_LABELS, label_regular_riders

GEO_FEATURES = ["LocationLatitude", "LocationLongitude"]
TARGET = "regular_transit"


def prepare_geo_transit_frame(
    df: pd.DataFrame,
    *,
    regular_labels: Sequence[str] = tuple(PRIMARY_REGULAR_LABELS),
) -> pd.DataFrame:
    """Label regular riders and keep rows with complete geo + outcome."""
    labeled = label_regular_riders(df, regular_labels=regular_labels)
    needed = ["participant_id", *GEO_FEATURES, TARGET, "transit_group", "Q26"]
    optional = [
        c
        for c in (
            "Country of residence",
            "Age",
            "Sex",
            "Employment status",
            "Student status",
        )
        if c in labeled.columns
    ]
    cols = [c for c in needed + optional if c in labeled.columns]
    out = labeled[cols].copy()
    out = out.dropna(subset=[*GEO_FEATURES, TARGET])
    out[TARGET] = out[TARGET].astype(bool)
    out["y"] = out[TARGET].astype(int)
    return out.reset_index(drop=True)


def make_rf_pipeline(
    *,
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 3,
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> Pipeline:
    """Standardize lat/long, then fit a Random Forest classifier."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rf", clf),
        ]
    )


def majority_baseline_probs(y: np.ndarray) -> np.ndarray:
    """Predict the majority class with constant probability = prevalence."""
    prev = float(np.mean(y))
    # Probability of class 1 for every row under a prevalence prior.
    return np.full(len(y), prev, dtype=float)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_auc(y_true, y_prob),
        "average_precision": (
            float(average_precision_score(y_true, y_prob))
            if len(np.unique(y_true)) > 1
            else float("nan")
        ),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def cross_validated_rf(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    pipeline: Pipeline | None = None,
) -> dict[str, Any]:
    """Stratified CV predictions for the geo-only Random Forest."""
    X = frame[GEO_FEATURES]
    y = frame["y"].to_numpy()
    pipe = pipeline or make_rf_pipeline(random_state=random_state)
    n_splits = min(n_splits, int(np.min(np.bincount(y))))
    if n_splits < 2:
        raise ValueError("Need at least 2 examples per class for stratified CV")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    proba = cross_val_predict(
        pipe,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    metrics = classification_metrics(y, proba)
    metrics["model"] = "random_forest_lat_lon"
    metrics["n_splits"] = int(n_splits)
    metrics["features"] = list(GEO_FEATURES)

    fpr, tpr, thr = roc_curve(y, proba)
    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})

    oof = frame[["participant_id", *GEO_FEATURES, TARGET, "y"]].copy()
    oof["y_prob_rf"] = proba
    oof["y_pred_rf"] = (proba >= 0.5).astype(int)
    return {"metrics": metrics, "oof": oof, "roc": roc, "cv": cv, "pipeline": pipe}


def fit_full_model(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
    pipeline: Pipeline | None = None,
) -> Pipeline:
    pipe = clone(pipeline) if pipeline is not None else make_rf_pipeline(random_state=random_state)
    pipe.fit(frame[GEO_FEATURES], frame["y"])
    return pipe


def impurity_importance(pipe: Pipeline) -> pd.DataFrame:
    rf: RandomForestClassifier = pipe.named_steps["rf"]
    return pd.DataFrame(
        {
            "feature": GEO_FEATURES,
            "importance_gini": rf.feature_importances_,
        }
    ).sort_values("importance_gini", ascending=False)


def permutation_importance_table(
    pipe: Pipeline,
    frame: pd.DataFrame,
    *,
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    result = permutation_importance(
        pipe,
        frame[GEO_FEATURES],
        frame["y"],
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": GEO_FEATURES,
                "importance_perm_mean": result.importances_mean,
                "importance_perm_std": result.importances_std,
            }
        )
        .sort_values("importance_perm_mean", ascending=False)
        .reset_index(drop=True)
    )


def null_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Majority-class / prevalence baselines for context."""
    y = frame["y"].to_numpy()
    prev = majority_baseline_probs(y)
    # Majority hard prediction: always predict mode.
    mode = 1 if float(y.mean()) >= 0.5 else 0
    hard = np.full(len(y), mode, dtype=int)
    hard_prob = hard.astype(float)  # degenerate scores
    rows = [
        {"model": "prevalence_prob", **classification_metrics(y, prev)},
        {"model": "majority_class", **classification_metrics(y, hard_prob)},
    ]
    return pd.DataFrame(rows)


def country_only_baseline(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any] | None:
    """
    Stratified CV logistic-style baseline using country of residence only.

    Implemented as a Random Forest on one-hot country for a fair tree comparison
    against the lat/long RF (same estimator family, different features).
    """
    if "Country of residence" not in frame.columns:
        return None
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    y = frame["y"].to_numpy()
    n_splits = min(n_splits, int(np.min(np.bincount(y))))
    if n_splits < 2:
        return None
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pre = ColumnTransformer(
        [
            (
                "country",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Country of residence"],
            )
        ]
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    proba = cross_val_predict(
        pipe,
        frame[["Country of residence"]],
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    metrics = classification_metrics(y, proba)
    metrics["model"] = "random_forest_country_only"
    metrics["n_splits"] = int(n_splits)
    metrics["features"] = ["Country of residence"]
    return {"metrics": metrics, "y_prob": proba}


def prediction_grid(
    pipe: Pipeline,
    frame: pd.DataFrame,
    *,
    grid_size: int = 80,
    pad: float = 0.05,
) -> pd.DataFrame:
    """Lat/lon grid of predicted P(regular) for decision-surface plots."""
    lat = frame["LocationLatitude"].to_numpy()
    lon = frame["LocationLongitude"].to_numpy()
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()
    lat_pad = (lat_max - lat_min) * pad or 1.0
    lon_pad = (lon_max - lon_min) * pad or 1.0
    lat_lin = np.linspace(lat_min - lat_pad, lat_max + lat_pad, grid_size)
    lon_lin = np.linspace(lon_min - lon_pad, lon_max + lon_pad, grid_size)
    xx, yy = np.meshgrid(lon_lin, lat_lin)  # xx=lon, yy=lat
    grid = pd.DataFrame(
        {
            "LocationLatitude": yy.ravel(),
            "LocationLongitude": xx.ravel(),
        }
    )
    proba = pipe.predict_proba(grid[GEO_FEATURES])[:, 1]
    grid["p_regular"] = proba
    return grid


def descriptive_geo_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_group = (
        frame.groupby("transit_group")
        .agg(
            n=("participant_id", "count"),
            mean_lat=("LocationLatitude", "mean"),
            std_lat=("LocationLatitude", "std"),
            mean_lon=("LocationLongitude", "mean"),
            std_lon=("LocationLongitude", "std"),
            median_lat=("LocationLatitude", "median"),
            median_lon=("LocationLongitude", "median"),
        )
        .reset_index()
    )
    if "Country of residence" in frame.columns:
        by_country = (
            frame.groupby(["Country of residence", "transit_group"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        if "regular" in by_country.columns and "not_regular" in by_country.columns:
            by_country["n"] = by_country["regular"] + by_country["not_regular"]
            by_country["pct_regular"] = by_country["regular"] / by_country["n"]
            by_country = by_country.sort_values("n", ascending=False)
    else:
        by_country = pd.DataFrame()
    return {"by_group": by_group, "by_country": by_country}


def run_geo_transit_rf_analysis(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    random_state: int = 42,
    grid_size: int = 80,
) -> dict[str, Any]:
    """End-to-end analysis bundle for the geography → regular-transit RQ."""
    frame = prepare_geo_transit_frame(df)
    if len(frame) < 20:
        raise ValueError(f"Analytic geo frame too small for RF CV (n={len(frame)})")

    cv_result = cross_validated_rf(frame, n_splits=n_splits, random_state=random_state)
    pipe = fit_full_model(frame, random_state=random_state)
    gini = impurity_importance(pipe)
    perm = permutation_importance_table(
        pipe, frame, n_repeats=n_perm_repeats, random_state=random_state
    )
    nulls = null_baselines(frame)
    country = country_only_baseline(frame, n_splits=n_splits, random_state=random_state)
    grid = prediction_grid(pipe, frame, grid_size=grid_size)
    desc = descriptive_geo_tables(frame)

    metrics_rows = [cv_result["metrics"]]
    metrics_rows.extend(nulls.to_dict(orient="records"))
    if country is not None:
        metrics_rows.append(country["metrics"])
    metrics_table = pd.DataFrame(metrics_rows)

    rf_auc = cv_result["metrics"]["roc_auc"]
    maj_auc = float(nulls.loc[nulls["model"] == "prevalence_prob", "roc_auc"].iloc[0])
    country_auc = (
        country["metrics"]["roc_auc"] if country is not None else float("nan")
    )
    beats_null = bool(rf_auc > maj_auc) if not np.isnan(rf_auc) else False
    # Prevalence baseline ROC-AUC is 0.5 by construction for constant scores;
    # interpret "predictive power" via AUC > 0.5 and lift vs country-only.
    summary = {
        "secondary_rq": (
            "Does Qualtrics survey geolocation (latitude & longitude) predict "
            "whether a matched respondent takes public transportation regularly?"
        ),
        "outcome": {
            "name": "regular_transit",
            "definition": (
                "Q26 in {4-8 days a month, 8 or more days a month} "
                "(weekly-or-more public transit)"
            ),
        },
        "features": GEO_FEATURES,
        "sample": {
            "n": int(len(frame)),
            "n_regular": int(frame["y"].sum()),
            "n_not_regular": int((frame["y"] == 0).sum()),
            "prevalence": float(frame["y"].mean()),
        },
        "cv_metrics": cv_result["metrics"],
        "baselines": {
            "prevalence_roc_auc": maj_auc,
            "country_only_roc_auc": country_auc,
        },
        "verdict": {
            "roc_auc": rf_auc,
            "beats_chance_auc_0_5": bool(rf_auc > 0.5) if not np.isnan(rf_auc) else False,
            "beats_prevalence_baseline": beats_null,
            "delta_auc_vs_country": (
                float(rf_auc - country_auc) if not np.isnan(country_auc) else None
            ),
            "interpretation": (
                f"Lat/long Random Forest CV ROC-AUC = {rf_auc:.3f} "
                f"(country-only AUC = {country_auc:.3f}). "
                + (
                    "Geography carries usable signal for regular transit use."
                    if (not np.isnan(rf_auc) and rf_auc >= 0.60)
                    else (
                        "Geography shows modest signal above chance; interpret cautiously."
                        if (not np.isnan(rf_auc) and rf_auc > 0.5)
                        else "Little evidence that lat/long alone predicts regular transit use."
                    )
                )
            ),
        },
        "caveats": [
            "Qualtrics LocationLatitude/Longitude are approximate IP/browser geolocation, not verified home addresses.",
            "Observational association ≠ causal effect of place on transit use.",
            "Country composition and urbanicity may confound continuous lat/long effects.",
        ],
    }

    return {
        "frame": frame,
        "metrics_table": metrics_table,
        "oof": cv_result["oof"],
        "roc": cv_result["roc"],
        "pipeline": pipe,
        "gini_importance": gini,
        "permutation_importance": perm,
        "null_baselines": nulls,
        "country_baseline": country,
        "grid": grid,
        "descriptives_by_group": desc["by_group"],
        "descriptives_by_country": desc["by_country"],
        "summary": summary,
    }


def save_geo_transit_rf_artifacts(
    analysis: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    mapping = {
        "frame": "geo_transit_modeling_frame.csv",
        "metrics_table": "geo_transit_rf_metrics.csv",
        "oof": "geo_transit_rf_oof_predictions.csv",
        "roc": "geo_transit_rf_roc_curve.csv",
        "gini_importance": "geo_transit_rf_gini_importance.csv",
        "permutation_importance": "geo_transit_rf_permutation_importance.csv",
        "null_baselines": "geo_transit_rf_null_baselines.csv",
        "descriptives_by_group": "geo_transit_descriptives_by_group.csv",
        "descriptives_by_country": "geo_transit_descriptives_by_country.csv",
        "grid": "geo_transit_rf_prediction_grid.csv",
    }
    for key, filename in mapping.items():
        frame = analysis.get(key)
        if isinstance(frame, pd.DataFrame):
            path = out / filename
            frame.to_csv(path, index=False)
            paths[key] = path

    summary_path = out / "geo_transit_rf_summary.json"
    summary_path.write_text(json.dumps(analysis["summary"], indent=2), encoding="utf-8")
    paths["summary"] = summary_path

    card = {
        "secondary_rq": analysis["summary"]["secondary_rq"],
        "sample": analysis["summary"]["sample"],
        "cv_metrics": analysis["summary"]["cv_metrics"],
        "baselines": analysis["summary"]["baselines"],
        "verdict": analysis["summary"]["verdict"],
        "caveats": analysis["summary"]["caveats"],
    }
    card_path = out / "geo_transit_rf_results_card.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    paths["results_card"] = card_path
    return paths


def run_geo_transit_rf_pipeline(
    *,
    prolific_paths: Sequence[str | Path] | None = None,
    qualtrics_path: str | Path | None = None,
    join_how: str = "inner",
    output_dir: str | Path = "outputs/geo_transit_rf",
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    random_state: int = 42,
) -> dict[str, Path]:
    participants, _report = load_full_cohort(
        prolific_paths=prolific_paths,
        qualtrics_path=qualtrics_path,
        join_how=join_how,
    )
    analysis = run_geo_transit_rf_analysis(
        participants,
        n_splits=n_splits,
        n_perm_repeats=n_perm_repeats,
        random_state=random_state,
    )
    return save_geo_transit_rf_artifacts(analysis, output_dir)
