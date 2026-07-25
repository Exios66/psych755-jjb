"""Secondary RQ: do PRCA CA scores predict regular public-transit use?

Trains a Random Forest classifier with ground-truth **group** and
**interpersonal** communication-apprehension scores (6–30) as features and the
Q26-based ``regular_transit`` flag as the outcome. Reports stratified CV
performance, single-subscale baselines, null baselines, and feature
importances.
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
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ca_personas.geo_transit_rf import (
    classification_metrics,
    majority_baseline_probs,
)
from ca_personas.load import load_full_cohort
from ca_personas.transit_ca import PRIMARY_REGULAR_LABELS, label_regular_riders

CA_FEATURES = ["gt_group_ca", "gt_interpersonal_ca"]
TARGET = "regular_transit"


def prepare_ca_transit_frame(
    df: pd.DataFrame,
    *,
    regular_labels: Sequence[str] = tuple(PRIMARY_REGULAR_LABELS),
) -> pd.DataFrame:
    """Label regular riders and keep rows with complete CA scores + outcome."""
    labeled = label_regular_riders(df, regular_labels=regular_labels)
    needed = [
        "participant_id",
        *CA_FEATURES,
        TARGET,
        "transit_group",
        "Q26",
        "gt_group_band",
        "gt_interpersonal_band",
    ]
    optional = [
        c
        for c in (
            "Age",
            "Sex",
            "Country of residence",
            "Employment status",
            "Student status",
        )
        if c in labeled.columns
    ]
    cols = [c for c in needed + optional if c in labeled.columns]
    out = labeled[cols].copy()
    out = out.dropna(subset=[*CA_FEATURES, TARGET])
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
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("scaler", StandardScaler()), ("rf", clf)])


def _cv_proba(
    frame: pd.DataFrame,
    features: list[str],
    *,
    n_splits: int,
    random_state: int,
    model_name: str,
) -> dict[str, Any]:
    X = frame[features]
    y = frame["y"].to_numpy()
    n_splits = min(n_splits, int(np.min(np.bincount(y))))
    if n_splits < 2:
        raise ValueError("Need at least 2 examples per class for stratified CV")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipe = make_rf_pipeline(random_state=random_state)
    proba = cross_val_predict(
        pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    metrics = classification_metrics(y, proba)
    metrics["model"] = model_name
    metrics["n_splits"] = int(n_splits)
    metrics["features"] = list(features)
    fpr, tpr, thr = roc_curve(y, proba)
    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
    return {"metrics": metrics, "proba": proba, "roc": roc, "cv": cv, "pipeline_template": pipe}


def null_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    y = frame["y"].to_numpy()
    prev = majority_baseline_probs(y)
    mode = 1 if float(y.mean()) >= 0.5 else 0
    hard = np.full(len(y), mode, dtype=float)
    return pd.DataFrame(
        [
            {"model": "prevalence_prob", **classification_metrics(y, prev)},
            {"model": "majority_class", **classification_metrics(y, hard)},
        ]
    )


def point_biserial_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Association of each CA subscale with the binary regular-transit outcome."""
    from scipy import stats

    rows = []
    y = frame["y"].to_numpy()
    for col in CA_FEATURES:
        x = frame[col].astype(float).to_numpy()
        r, p = stats.pointbiserialr(y, x)
        rows.append(
            {
                "feature": col,
                "point_biserial_r": float(r),
                "p_value": float(p),
                "mean_regular": float(frame.loc[frame["y"] == 1, col].mean()),
                "mean_not_regular": float(frame.loc[frame["y"] == 0, col].mean()),
                "diff_regular_minus_not": float(
                    frame.loc[frame["y"] == 1, col].mean()
                    - frame.loc[frame["y"] == 0, col].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def descriptive_by_transit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, mask in [
        ("overall", slice(None)),
        ("regular", frame["y"] == 1),
        ("not_regular", frame["y"] == 0),
    ]:
        sub = frame.loc[mask]
        row: dict[str, Any] = {"group": group_name, "n": int(len(sub))}
        for col in CA_FEATURES:
            row[f"mean_{col}"] = float(sub[col].mean())
            row[f"std_{col}"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else float("nan")
            row[f"median_{col}"] = float(sub[col].median())
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_grid(
    pipe: Pipeline,
    frame: pd.DataFrame,
    *,
    grid_size: int = 60,
) -> pd.DataFrame:
    """Group × interpersonal CA grid of predicted P(regular)."""
    g = frame["gt_group_ca"].to_numpy()
    i = frame["gt_interpersonal_ca"].to_numpy()
    g_lin = np.linspace(max(6, g.min() - 1), min(30, g.max() + 1), grid_size)
    i_lin = np.linspace(max(6, i.min() - 1), min(30, i.max() + 1), grid_size)
    gg, ii = np.meshgrid(g_lin, i_lin)
    grid = pd.DataFrame(
        {"gt_group_ca": gg.ravel(), "gt_interpersonal_ca": ii.ravel()}
    )
    grid["p_regular"] = pipe.predict_proba(grid[CA_FEATURES])[:, 1]
    return grid


def run_ca_transit_rf_analysis(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    random_state: int = 42,
    grid_size: int = 60,
) -> dict[str, Any]:
    frame = prepare_ca_transit_frame(df)
    if len(frame) < 20:
        raise ValueError(f"Analytic CA→transit frame too small (n={len(frame)})")

    # Primary: both subscales
    both = _cv_proba(
        frame,
        CA_FEATURES,
        n_splits=n_splits,
        random_state=random_state,
        model_name="random_forest_group_and_interpersonal",
    )
    group_only = _cv_proba(
        frame,
        ["gt_group_ca"],
        n_splits=n_splits,
        random_state=random_state,
        model_name="random_forest_group_only",
    )
    interp_only = _cv_proba(
        frame,
        ["gt_interpersonal_ca"],
        n_splits=n_splits,
        random_state=random_state,
        model_name="random_forest_interpersonal_only",
    )

    pipe = clone(both["pipeline_template"])
    pipe.fit(frame[CA_FEATURES], frame["y"])
    rf: RandomForestClassifier = pipe.named_steps["rf"]
    gini = pd.DataFrame(
        {"feature": CA_FEATURES, "importance_gini": rf.feature_importances_}
    ).sort_values("importance_gini", ascending=False)
    perm_res = permutation_importance(
        pipe,
        frame[CA_FEATURES],
        frame["y"],
        n_repeats=n_perm_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )
    perm = (
        pd.DataFrame(
            {
                "feature": CA_FEATURES,
                "importance_perm_mean": perm_res.importances_mean,
                "importance_perm_std": perm_res.importances_std,
            }
        )
        .sort_values("importance_perm_mean", ascending=False)
        .reset_index(drop=True)
    )

    nulls = null_baselines(frame)
    associations = point_biserial_table(frame)
    descriptives = descriptive_by_transit(frame)
    grid = prediction_grid(pipe, frame, grid_size=grid_size)

    oof = frame[
        ["participant_id", *CA_FEATURES, TARGET, "y", "gt_group_band", "gt_interpersonal_band"]
        if "gt_group_band" in frame.columns
        else ["participant_id", *CA_FEATURES, TARGET, "y"]
    ].copy()
    oof["y_prob_rf"] = both["proba"]
    oof["y_pred_rf"] = (both["proba"] >= 0.5).astype(int)

    metrics_table = pd.DataFrame(
        [
            both["metrics"],
            group_only["metrics"],
            interp_only["metrics"],
            *nulls.to_dict(orient="records"),
        ]
    )

    auc = both["metrics"]["roc_auc"]
    group_auc = group_only["metrics"]["roc_auc"]
    interp_auc = interp_only["metrics"]["roc_auc"]
    summary = {
        "secondary_rq": (
            "Do ground-truth group and interpersonal communication-apprehension "
            "(PRCA) scores predict whether a matched respondent takes public "
            "transportation regularly?"
        ),
        "outcome": {
            "name": "regular_transit",
            "definition": (
                "Q26 in {4-8 days a month, 8 or more days a month} "
                "(weekly-or-more public transit)"
            ),
        },
        "features": CA_FEATURES,
        "sample": {
            "n": int(len(frame)),
            "n_regular": int(frame["y"].sum()),
            "n_not_regular": int((frame["y"] == 0).sum()),
            "prevalence": float(frame["y"].mean()),
        },
        "associations": associations.to_dict(orient="records"),
        "cv_metrics": both["metrics"],
        "single_feature_auc": {
            "group_only": group_auc,
            "interpersonal_only": interp_auc,
        },
        "baselines": {
            "prevalence_roc_auc": float(
                nulls.loc[nulls["model"] == "prevalence_prob", "roc_auc"].iloc[0]
            ),
            "majority_accuracy": float(
                nulls.loc[nulls["model"] == "majority_class", "accuracy"].iloc[0]
            ),
        },
        "verdict": {
            "roc_auc": auc,
            "beats_chance_auc_0_5": bool(auc > 0.5) if not np.isnan(auc) else False,
            "stronger_subscale": (
                "gt_group_ca" if group_auc >= interp_auc else "gt_interpersonal_ca"
            ),
            "delta_auc_both_minus_best_single": float(
                auc - max(group_auc, interp_auc)
            ),
            "interpretation": _interpret(auc, group_auc, interp_auc, associations),
        },
        "caveats": [
            "Directionality is associative: CA and transit were measured in the same survey wave.",
            "Regular transit is a thresholded Q26 label (weekly+), not continuous ridership.",
            "Self-reported PRCA and transit may share common method variance.",
            "Results characterize this Prolific↔Qualtrics matched cohort, not a census.",
        ],
    }

    return {
        "frame": frame,
        "metrics_table": metrics_table,
        "oof": oof,
        "roc": both["roc"],
        "roc_group_only": group_only["roc"],
        "roc_interpersonal_only": interp_only["roc"],
        "pipeline": pipe,
        "gini_importance": gini,
        "permutation_importance": perm,
        "null_baselines": nulls,
        "associations": associations,
        "descriptives": descriptives,
        "grid": grid,
        "summary": summary,
    }


def _interpret(
    auc: float,
    group_auc: float,
    interp_auc: float,
    associations: pd.DataFrame,
) -> str:
    if np.isnan(auc):
        return "Unable to estimate ROC-AUC (degenerate class labels)."
    assoc_bits = []
    for _, row in associations.iterrows():
        direction = "lower" if row["diff_regular_minus_not"] < 0 else "higher"
        assoc_bits.append(
            f"{row['feature']} is {direction} among regular riders "
            f"(Δ={row['diff_regular_minus_not']:+.2f}, r={row['point_biserial_r']:+.3f})"
        )
    assoc_txt = "; ".join(assoc_bits)
    strength = (
        "usable predictive signal"
        if auc >= 0.60
        else ("modest signal above chance" if auc > 0.5 else "little predictive signal")
    )
    return (
        f"CA→transit Random Forest CV ROC-AUC = {auc:.3f} ({strength}). "
        f"Group-only AUC = {group_auc:.3f}; interpersonal-only AUC = {interp_auc:.3f}. "
        f"{assoc_txt}."
    )


def save_ca_transit_rf_artifacts(
    analysis: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    mapping = {
        "frame": "ca_transit_modeling_frame.csv",
        "metrics_table": "ca_transit_rf_metrics.csv",
        "oof": "ca_transit_rf_oof_predictions.csv",
        "roc": "ca_transit_rf_roc_curve.csv",
        "gini_importance": "ca_transit_rf_gini_importance.csv",
        "permutation_importance": "ca_transit_rf_permutation_importance.csv",
        "null_baselines": "ca_transit_rf_null_baselines.csv",
        "associations": "ca_transit_associations.csv",
        "descriptives": "ca_transit_descriptives.csv",
        "grid": "ca_transit_rf_prediction_grid.csv",
    }
    for key, filename in mapping.items():
        frame = analysis.get(key)
        if isinstance(frame, pd.DataFrame):
            path = out / filename
            frame.to_csv(path, index=False)
            paths[key] = path

    summary_path = out / "ca_transit_rf_summary.json"
    summary_path.write_text(json.dumps(analysis["summary"], indent=2), encoding="utf-8")
    paths["summary"] = summary_path

    card = {
        "secondary_rq": analysis["summary"]["secondary_rq"],
        "sample": analysis["summary"]["sample"],
        "associations": analysis["summary"]["associations"],
        "cv_metrics": analysis["summary"]["cv_metrics"],
        "single_feature_auc": analysis["summary"]["single_feature_auc"],
        "verdict": analysis["summary"]["verdict"],
        "caveats": analysis["summary"]["caveats"],
    }
    card_path = out / "ca_transit_rf_results_card.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    paths["results_card"] = card_path
    return paths


def run_ca_transit_rf_pipeline(
    *,
    prolific_paths: Sequence[str | Path] | None = None,
    qualtrics_path: str | Path | None = None,
    join_how: str = "inner",
    output_dir: str | Path = "outputs/ca_transit_rf",
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    random_state: int = 42,
) -> dict[str, Path]:
    participants, _report = load_full_cohort(
        prolific_paths=prolific_paths,
        qualtrics_path=qualtrics_path,
        join_how=join_how,
    )
    analysis = run_ca_transit_rf_analysis(
        participants,
        n_splits=n_splits,
        n_perm_repeats=n_perm_repeats,
        random_state=random_state,
    )
    return save_ca_transit_rf_artifacts(analysis, output_dir)
