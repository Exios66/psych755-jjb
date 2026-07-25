"""Extended secondary RQs answering open questions from existing transit memos.

Experiments (offline; matched File A/B/C cohort; same Q26 weekly+ outcome unless noted):

1. ``demographics`` — Age + Sex + Student status → regular transit
2. ``country`` — Country of residence → regular transit
3. ``nested_q28_car`` — Does Q28 retain lift after conditioning on car access?
4. ``ca_q28_car`` — Joint CA + Q28 + car access RF
5. ``country_car`` — Country × car access joint RF
6. ``q27_among_regular`` — Among weekly+ riders, does anything predict Q27 intensity?
7. ``common_n`` — Head-to-head AUCs on one overlapping complete-case subset
8. ``residual_ca_q28`` — Does group CA still separate regular riders after Q28 strata?
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ca_personas.geo_transit_rf import classification_metrics, majority_baseline_probs
from ca_personas.load import load_full_cohort
from ca_personas.transit_ca import (
    PRIMARY_REGULAR_LABELS,
    compare_regular_vs_rest,
    label_regular_riders,
)
from ca_personas.transit_covariate_rf import (
    association_table,
    make_categorical_rf_pipeline,
    null_baselines,
    prepare_covariate_frame,
)

TARGET = "regular_transit"

# High transit-day intensity among regular riders (above the modal 1–2 rides).
HIGH_INTENSITY_LABELS = (
    "3-4 rides in a typical day",
    "5-6 rides in a typical day",
    "7 or more rides in a typical day",
)

COMMON_N_FAMILIES: dict[str, list[str]] = {
    "q28_days": ["Q28"],
    "car_access": ["Q20", "Q21"],
    "employment": ["Employment status"],
    "demographics_cat": ["Sex", "Student status"],
    "country": ["Country of residence"],
    "ca_scores": ["gt_group_ca", "gt_interpersonal_ca"],
    "geo": ["LocationLatitude", "LocationLongitude"],
}


def _clean_string_col(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    bad = text.eq("") | text.str.lower().isin({"nan", "none", "<na>", "data_expired"})
    out = text.mask(bad)
    return out.astype("string")


def prepare_labeled_cohort(
    df: pd.DataFrame,
    *,
    regular_labels: Sequence[str] = tuple(PRIMARY_REGULAR_LABELS),
) -> pd.DataFrame:
    """Label regular riders and normalize key covariates."""
    out = label_regular_riders(df, regular_labels=regular_labels).copy()
    for col in (
        "Sex",
        "Student status",
        "Country of residence",
        "Employment status",
        "Q20",
        "Q21",
        "Q27",
        "Q28",
        "Q29",
    ):
        if col in out.columns:
            out[col] = _clean_string_col(out[col])
    if "Age" in out.columns:
        out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    for col in ("gt_group_ca", "gt_interpersonal_ca", "LocationLatitude", "LocationLongitude"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out[TARGET] = out[TARGET].astype(bool)
    out["y"] = out[TARGET].astype(int)
    return out


def make_mixed_rf_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    n_estimators: int = 500,
    min_samples_leaf: int = 3,
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> Pipeline:
    """ColumnTransformer RF for mixed numeric + categorical predictors."""
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            )
        )
    if categorical_features:
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
                list(categorical_features),
            )
        )
    if not transformers:
        raise ValueError("Need at least one numeric or categorical feature")
    pre = ColumnTransformer(transformers)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("pre", pre), ("rf", clf)])


def _cv_mixed(
    frame: pd.DataFrame,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    n_splits: int,
    random_state: int,
    model_name: str,
) -> dict[str, Any]:
    feats = list(numeric_features) + list(categorical_features)
    X = frame[feats].copy()
    for col in categorical_features:
        X[col] = X[col].astype("string").fillna("<NA>").astype(str)
    y = frame["y"].to_numpy()
    n_splits = min(n_splits, int(np.min(np.bincount(y))))
    if n_splits < 2:
        raise ValueError("Need at least 2 examples per class for stratified CV")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipe = make_mixed_rf_pipeline(
        numeric_features,
        categorical_features,
        random_state=random_state,
    )
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    metrics = classification_metrics(y, proba)
    metrics["model"] = model_name
    metrics["n_splits"] = int(n_splits)
    metrics["features"] = feats
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, proba)
    oof = frame[["participant_id", "y", "transit_group"]].copy()
    oof["y_prob"] = proba
    oof["y_pred"] = (proba >= 0.5).astype(int)
    return {
        "metrics": metrics,
        "proba": proba,
        "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}),
        "oof": oof,
        "pipeline_template": pipe,
        "X": X,
        "y": y,
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
    }


def _fit_importances(
    cv_result: dict[str, Any],
    *,
    n_perm_repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pipe = make_mixed_rf_pipeline(
        cv_result["numeric_features"],
        cv_result["categorical_features"],
        random_state=random_state,
    )
    pipe.fit(cv_result["X"], cv_result["y"])
    enc_names = list(pipe.named_steps["pre"].get_feature_names_out())
    gini = pd.DataFrame(
        {
            "encoded_feature": enc_names,
            "importance_gini": pipe.named_steps["rf"].feature_importances_,
        }
    ).sort_values("importance_gini", ascending=False)
    raw_feats = cv_result["numeric_features"] + cv_result["categorical_features"]
    raw_rows = []
    for feat in raw_feats:
        mask = gini["encoded_feature"].str.contains(feat, regex=False)
        raw_rows.append(
            {
                "feature": feat,
                "importance_gini_sum": float(gini.loc[mask, "importance_gini"].sum()),
            }
        )
    gini_raw = pd.DataFrame(raw_rows).sort_values("importance_gini_sum", ascending=False)
    perm = permutation_importance(
        pipe,
        cv_result["X"],
        cv_result["y"],
        n_repeats=n_perm_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )
    perm_table = pd.DataFrame(
        {
            "feature": raw_feats,
            "importance_perm_mean": perm.importances_mean,
            "importance_perm_std": perm.importances_std,
        }
    ).sort_values("importance_perm_mean", ascending=False)
    return gini_raw.reset_index(drop=True), perm_table.reset_index(drop=True)


_LABELS = {
    "q28_only": "Q28 only",
    "q28_q21": "Q28 + Q21",
    "q28_q20_q21": "Q28 + Q20 + Q21",
    "q21_only": "Q21 only",
    "ca_only": "CA only",
    "ca_q28": "CA + Q28",
    "ca_q28_car": "CA + Q28 + Q21",
    "country_only": "Country only",
    "car_only": "Car access only",
    "country_car": "Country + car",
    "country_x_car": "Country × car (interaction)",
    "q28_days": "Q28 days",
    "car_access": "Car license & access",
    "employment": "Employment status",
    "demographics": "Age + Sex + Student",
    "country": "Country of residence",
    "ca_scores": "Group + interpersonal CA",
    "geo": "Lat/long geo",
    "chance": "Chance / prevalence",
}


def _pretty_label(key: str) -> str:
    return _LABELS.get(key, key.replace("_", " "))


def _metric_row(label: str, key: str, metrics: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "spec_key": key,
        "label": label if label != key else _pretty_label(key),
        "n": int(len(frame)),
        "n_regular": int(frame["y"].sum()) if "y" in frame.columns else None,
        "prevalence": float(frame["y"].mean()) if "y" in frame.columns else None,
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "f1": metrics.get("f1"),
        "brier": metrics.get("brier"),
    }


# ---------------------------------------------------------------------------
# Individual experiments
# ---------------------------------------------------------------------------


def run_demographics_experiment(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    random_state: int = 42,
) -> dict[str, Any]:
    """Age + Sex + Student status → regular transit (mixed RF)."""
    labeled = prepare_labeled_cohort(df)
    frame = labeled.dropna(subset=["Age", "Sex", "Student status", TARGET]).copy()
    cv = _cv_mixed(
        frame,
        numeric_features=["Age"],
        categorical_features=["Sex", "Student status"],
        n_splits=n_splits,
        random_state=random_state,
        model_name="random_forest_demographics",
    )
    gini_raw, perm = _fit_importances(cv, n_perm_repeats=n_perm_repeats, random_state=random_state)
    assoc = association_table(frame, ["Sex", "Student status"])
    # Age tertile prevalence for the memo table.
    frame = frame.copy()
    frame["Age_tertile"] = pd.qcut(frame["Age"], 3, labels=["younger", "middle", "older"])
    age_assoc = association_table(frame.rename(columns={"Age_tertile": "Age_bin"}), ["Age_bin"])
    age_assoc["feature"] = "Age (tertile)"
    nulls = null_baselines(frame)
    return {
        "spec_key": "demographics",
        "label": "Age + Sex + Student status",
        "research_question": (
            "Do Prolific demographics (Age, Sex, Student status) predict regular "
            "public-transit use in the matched cohort?"
        ),
        "frame": frame,
        "metrics": cv["metrics"],
        "metrics_table": pd.DataFrame([cv["metrics"], *nulls.to_dict(orient="records")]),
        "oof": cv["oof"],
        "roc": cv["roc"],
        "gini_raw": gini_raw,
        "permutation_importance": perm,
        "associations": pd.concat([assoc, age_assoc], ignore_index=True),
        "null_baselines": nulls,
        "numeric_features": ["Age"],
        "categorical_features": ["Sex", "Student status"],
    }


def run_country_experiment(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    random_state: int = 42,
) -> dict[str, Any]:
    """Dedicated Country of residence → regular transit RF."""
    frame = prepare_covariate_frame(df, ["Country of residence"])
    X = frame[["Country of residence"]].astype("string").fillna("<NA>").astype(str)
    y = frame["y"].to_numpy()
    n_splits_use = min(n_splits, int(np.min(np.bincount(y))))
    cv = StratifiedKFold(n_splits=n_splits_use, shuffle=True, random_state=random_state)
    pipe = make_categorical_rf_pipeline(["Country of residence"], random_state=random_state)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    metrics = classification_metrics(y, proba)
    metrics["model"] = "random_forest_country"
    metrics["n_splits"] = int(n_splits_use)
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, proba)
    pipe.fit(X, y)
    perm = permutation_importance(
        pipe, X, y, n_repeats=n_perm_repeats, random_state=random_state, scoring="roc_auc", n_jobs=-1
    )
    return {
        "spec_key": "country",
        "label": "Country of residence",
        "research_question": (
            "Does country of residence alone predict regular public-transit use, "
            "and how does it compare to lat/long geography?"
        ),
        "frame": frame,
        "metrics": metrics,
        "metrics_table": pd.DataFrame([metrics, *null_baselines(frame).to_dict(orient="records")]),
        "oof": pd.DataFrame(
            {
                "participant_id": frame["participant_id"],
                "y": y,
                "transit_group": frame["transit_group"],
                "y_prob": proba,
                "y_pred": (proba >= 0.5).astype(int),
            }
        ),
        "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}),
        "gini_raw": pd.DataFrame(
            [{"feature": "Country of residence", "importance_gini_sum": 1.0}]
        ),
        "permutation_importance": pd.DataFrame(
            {
                "feature": ["Country of residence"],
                "importance_perm_mean": perm.importances_mean,
                "importance_perm_std": perm.importances_std,
            }
        ),
        "associations": association_table(frame, ["Country of residence"]),
        "null_baselines": null_baselines(frame),
        "numeric_features": [],
        "categorical_features": ["Country of residence"],
    }


def run_nested_q28_car_experiment(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    random_state: int = 42,
) -> dict[str, Any]:
    """Q28 alone vs Q28+Q21 vs Q28+Q20+Q21 on the same complete-case rows."""
    labeled = prepare_labeled_cohort(df)
    base = labeled.dropna(subset=["Q28", "Q20", "Q21", TARGET]).copy()
    specs = {
        "q28_only": ["Q28"],
        "q28_q21": ["Q28", "Q21"],
        "q28_q20_q21": ["Q28", "Q20", "Q21"],
        "q21_only": ["Q21"],
    }
    analyses: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for key, feats in specs.items():
        X = base[feats].astype("string").fillna("<NA>").astype(str)
        y = base["y"].to_numpy()
        n_splits_use = min(n_splits, int(np.min(np.bincount(y))))
        cv = StratifiedKFold(n_splits=n_splits_use, shuffle=True, random_state=random_state)
        pipe = make_categorical_rf_pipeline(feats, random_state=random_state)
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        metrics = classification_metrics(y, proba)
        metrics["model"] = f"random_forest_{key}"
        metrics["n_splits"] = int(n_splits_use)
        pipe.fit(X, y)
        perm = permutation_importance(
            pipe,
            X,
            y,
            n_repeats=n_perm_repeats,
            random_state=random_state,
            scoring="roc_auc",
            n_jobs=-1,
        )
        from sklearn.metrics import roc_curve

        fpr, tpr, thr = roc_curve(y, proba)
        analyses[key] = {
            "features": feats,
            "metrics": metrics,
            "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}),
            "permutation_importance": pd.DataFrame(
                {
                    "feature": feats,
                    "importance_perm_mean": perm.importances_mean,
                    "importance_perm_std": perm.importances_std,
                }
            ).sort_values("importance_perm_mean", ascending=False),
            "associations": association_table(base, feats),
        }
        rows.append(_metric_row(_pretty_label(key), key, metrics, base))
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    q28_auc = float(analyses["q28_only"]["metrics"]["roc_auc"])
    joint_auc = float(analyses["q28_q21"]["metrics"]["roc_auc"])
    return {
        "spec_key": "nested_q28_car",
        "label": "Q28 conditioned on car access (nested)",
        "research_question": (
            "Does ride-share days (Q28) retain predictive lift for regular transit "
            "after conditioning on car access (Q21 / Q20)?"
        ),
        "frame": base,
        "metrics": analyses["q28_q21"]["metrics"],
        "comparison": comparison,
        "analyses": analyses,
        "associations": association_table(base, ["Q28", "Q21", "Q20"]),
        "summary_delta": {
            "n_common": int(len(base)),
            "q28_only_auc": q28_auc,
            "q28_q21_auc": joint_auc,
            "q21_only_auc": float(analyses["q21_only"]["metrics"]["roc_auc"]),
            "q28_q20_q21_auc": float(analyses["q28_q20_q21"]["metrics"]["roc_auc"]),
            "delta_q28_to_q28_q21": joint_auc - q28_auc,
            "q28_retains_lift": bool(q28_auc > 0.65),
        },
        "roc": analyses["q28_q21"]["roc"],
        "permutation_importance": analyses["q28_q21"]["permutation_importance"],
        "gini_raw": analyses["q28_q21"]["permutation_importance"].rename(
            columns={"importance_perm_mean": "importance_gini_sum"}
        )[["feature", "importance_gini_sum"]],
        "null_baselines": null_baselines(base),
        "metrics_table": comparison,
        "oof": pd.DataFrame(
            {
                "participant_id": base["participant_id"],
                "y": base["y"],
                "transit_group": base["transit_group"],
            }
        ),
        "numeric_features": [],
        "categorical_features": ["Q28", "Q21"],
    }


def run_ca_q28_car_experiment(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    random_state: int = 42,
) -> dict[str, Any]:
    """Joint CA scores + Q28 + car access (complete-case, no imputation)."""
    labeled = prepare_labeled_cohort(df)
    needed = ["gt_group_ca", "gt_interpersonal_ca", "Q28", "Q21", TARGET]
    frame = labeled.dropna(subset=needed).copy()
    # Nested ablations on the same rows.
    ablations = {
        "ca_only": {"num": ["gt_group_ca", "gt_interpersonal_ca"], "cat": []},
        "q28_only": {"num": [], "cat": ["Q28"]},
        "q21_only": {"num": [], "cat": ["Q21"]},
        "ca_q28": {"num": ["gt_group_ca", "gt_interpersonal_ca"], "cat": ["Q28"]},
        "ca_q28_car": {
            "num": ["gt_group_ca", "gt_interpersonal_ca"],
            "cat": ["Q28", "Q21"],
        },
    }
    analyses: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for key, spec in ablations.items():
        cv = _cv_mixed(
            frame,
            numeric_features=spec["num"],
            categorical_features=spec["cat"],
            n_splits=n_splits,
            random_state=random_state,
            model_name=f"random_forest_{key}",
        )
        gini_raw, perm = _fit_importances(
            cv, n_perm_repeats=n_perm_repeats, random_state=random_state
        )
        analyses[key] = {
            "metrics": cv["metrics"],
            "roc": cv["roc"],
            "permutation_importance": perm,
            "gini_raw": gini_raw,
            "oof": cv["oof"],
        }
        rows.append(_metric_row(_pretty_label(key), key, cv["metrics"], frame))
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    primary = analyses["ca_q28_car"]
    return {
        "spec_key": "ca_q28_car",
        "label": "CA + Q28 + car access",
        "research_question": (
            "On a complete-case frame, how do group/interpersonal CA, ride-share days "
            "(Q28), and car access (Q21) jointly and separately predict regular transit?"
        ),
        "frame": frame,
        "metrics": primary["metrics"],
        "metrics_table": comparison,
        "comparison": comparison,
        "analyses": analyses,
        "oof": primary["oof"],
        "roc": primary["roc"],
        "gini_raw": primary["gini_raw"],
        "permutation_importance": primary["permutation_importance"],
        "associations": association_table(frame, ["Q28", "Q21"]),
        "null_baselines": null_baselines(frame),
        "numeric_features": ["gt_group_ca", "gt_interpersonal_ca"],
        "categorical_features": ["Q28", "Q21"],
    }


def run_country_car_experiment(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    random_state: int = 42,
) -> dict[str, Any]:
    """Country of residence + car access joint / nested RFs."""
    labeled = prepare_labeled_cohort(df)
    frame = labeled.dropna(subset=["Country of residence", "Q21", TARGET]).copy()
    # Interaction-style feature: country × car concatenated level.
    frame["country_x_car"] = (
        frame["Country of residence"].astype(str) + " | car=" + frame["Q21"].astype(str)
    )
    specs = {
        "country_only": ["Country of residence"],
        "car_only": ["Q21"],
        "country_car": ["Country of residence", "Q21"],
        "country_x_car": ["country_x_car"],
    }
    analyses: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for key, feats in specs.items():
        X = frame[feats].astype("string").fillna("<NA>").astype(str)
        y = frame["y"].to_numpy()
        n_splits_use = min(n_splits, int(np.min(np.bincount(y))))
        cv = StratifiedKFold(n_splits=n_splits_use, shuffle=True, random_state=random_state)
        pipe = make_categorical_rf_pipeline(feats, random_state=random_state)
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        metrics = classification_metrics(y, proba)
        metrics["model"] = f"random_forest_{key}"
        from sklearn.metrics import roc_curve

        fpr, tpr, thr = roc_curve(y, proba)
        pipe.fit(X, y)
        perm = permutation_importance(
            pipe,
            X,
            y,
            n_repeats=n_perm_repeats,
            random_state=random_state,
            scoring="roc_auc",
            n_jobs=-1,
        )
        analyses[key] = {
            "features": feats,
            "metrics": metrics,
            "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}),
            "permutation_importance": pd.DataFrame(
                {
                    "feature": feats,
                    "importance_perm_mean": perm.importances_mean,
                    "importance_perm_std": perm.importances_std,
                }
            ).sort_values("importance_perm_mean", ascending=False),
        }
        rows.append(_metric_row(_pretty_label(key), key, metrics, frame))
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    primary = analyses["country_car"]
    return {
        "spec_key": "country_car",
        "label": "Country × car access",
        "research_question": (
            "Do country of residence and car access jointly (or as an interaction) "
            "predict regular public-transit use better than either alone?"
        ),
        "frame": frame,
        "metrics": primary["metrics"],
        "metrics_table": comparison,
        "comparison": comparison,
        "analyses": analyses,
        "oof": pd.DataFrame(
            {
                "participant_id": frame["participant_id"],
                "y": frame["y"],
                "transit_group": frame["transit_group"],
            }
        ),
        "roc": primary["roc"],
        "gini_raw": primary["permutation_importance"].rename(
            columns={"importance_perm_mean": "importance_gini_sum"}
        )[["feature", "importance_gini_sum"]],
        "permutation_importance": primary["permutation_importance"],
        "associations": association_table(
            frame, ["Country of residence", "Q21", "country_x_car"]
        ),
        "null_baselines": null_baselines(frame),
        "numeric_features": [],
        "categorical_features": ["Country of residence", "Q21"],
    }


def run_q27_among_regular_experiment(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    random_state: int = 42,
) -> dict[str, Any]:
    """Among regular riders, predict high vs low Q27 intensity."""
    labeled = prepare_labeled_cohort(df)
    riders = labeled.loc[labeled[TARGET]].dropna(subset=["Q27"]).copy()
    riders["high_intensity"] = riders["Q27"].isin(HIGH_INTENSITY_LABELS)
    riders["y"] = riders["high_intensity"].astype(int)
    if riders["y"].nunique() < 2 or riders["y"].sum() < 5:
        raise ValueError("Insufficient high-intensity regular riders for CV")

    # Candidate predictors available within the rider subgroup.
    candidate_specs = {
        "ca_scores": {"num": ["gt_group_ca", "gt_interpersonal_ca"], "cat": []},
        "q28_days": {"num": [], "cat": ["Q28"]},
        "car_access": {"num": [], "cat": ["Q21"]},
        "demographics": {"num": ["Age"], "cat": ["Sex", "Student status"]},
        "employment": {"num": [], "cat": ["Employment status"]},
    }
    analyses: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for key, spec in candidate_specs.items():
        feats = spec["num"] + spec["cat"]
        sub = riders.dropna(subset=feats).copy()
        if sub["y"].nunique() < 2 or int(np.min(np.bincount(sub["y"].to_numpy()))) < 2:
            continue
        cv = _cv_mixed(
            sub,
            numeric_features=spec["num"],
            categorical_features=spec["cat"],
            n_splits=min(n_splits, int(np.min(np.bincount(sub["y"].to_numpy())))),
            random_state=random_state,
            model_name=f"q27_intensity_{key}",
        )
        gini_raw, perm = _fit_importances(
            cv, n_perm_repeats=n_perm_repeats, random_state=random_state
        )
        analyses[key] = {
            "metrics": cv["metrics"],
            "roc": cv["roc"],
            "permutation_importance": perm,
            "gini_raw": gini_raw,
            "n": int(len(sub)),
            "n_high": int(sub["y"].sum()),
        }
        rows.append(
            {
                "spec_key": key,
                "label": _pretty_label(key),
                "n": int(len(sub)),
                "n_high_intensity": int(sub["y"].sum()),
                "prevalence_high": float(sub["y"].mean()),
                "roc_auc": cv["metrics"]["roc_auc"],
                "average_precision": cv["metrics"]["average_precision"],
                "balanced_accuracy": cv["metrics"]["balanced_accuracy"],
                "f1": cv["metrics"]["f1"],
            }
        )
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    intensity_counts = (
        riders.groupby("Q27", dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    best_key = comparison.iloc[0]["spec_key"] if len(comparison) else "ca_scores"
    primary = analyses.get(best_key, next(iter(analyses.values())))
    return {
        "spec_key": "q27_among_regular",
        "label": "Q27 intensity among regular riders",
        "research_question": (
            "Among weekly+ public-transit riders, which covariates predict high "
            "transit-day intensity (Q27 ≥ 3–4 rides)?"
        ),
        "frame": riders,
        "metrics": primary["metrics"],
        "metrics_table": comparison,
        "comparison": comparison,
        "analyses": analyses,
        "intensity_counts": intensity_counts,
        "oof": pd.DataFrame(
            {
                "participant_id": riders["participant_id"],
                "y": riders["y"],
                "Q27": riders["Q27"],
            }
        ),
        "roc": primary["roc"],
        "gini_raw": primary["gini_raw"],
        "permutation_importance": primary["permutation_importance"],
        "associations": association_table(
            riders.assign(regular_transit=riders["high_intensity"], y=riders["y"]),
            ["Q28"] if "Q28" in riders.columns else [],
        )
        if "Q28" in riders.columns
        else pd.DataFrame(),
        "null_baselines": null_baselines(riders),
        "sample_summary": {
            "n_regular_riders": int(len(riders)),
            "n_high_intensity": int(riders["y"].sum()),
            "pct_high_intensity": float(riders["y"].mean()),
        },
        "numeric_features": [],
        "categorical_features": [],
    }


def run_common_n_experiment(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate major feature families on one overlapping complete-case subset."""
    labeled = prepare_labeled_cohort(df)
    # Student status often DATA_EXPIRED → drop from required overlap if too sparse.
    base_cols = [
        "Q28",
        "Q20",
        "Q21",
        "Employment status",
        "Sex",
        "Country of residence",
        "gt_group_ca",
        "gt_interpersonal_ca",
        "LocationLatitude",
        "LocationLongitude",
        TARGET,
    ]
    optional_student = "Student status" in labeled.columns
    frame = labeled.dropna(subset=base_cols).copy()

    # Use intersection that maximizes fair comparison: require student when available.
    if optional_student and frame["Student status"].notna().mean() > 0.5:
        frame = frame.dropna(subset=["Student status"]).copy()
        demo_cat = ["Sex", "Student status"]
    else:
        demo_cat = ["Sex"]

    rows: list[dict[str, Any]] = []
    analyses: dict[str, Any] = {}
    family_specs: dict[str, dict[str, list[str]]] = {
        "q28_days": {"num": [], "cat": ["Q28"]},
        "car_access": {"num": [], "cat": ["Q20", "Q21"]},
        "employment": {"num": [], "cat": ["Employment status"]},
        "demographics": {"num": ["Age"] if frame["Age"].notna().all() else [], "cat": demo_cat},
        "country": {"num": [], "cat": ["Country of residence"]},
        "ca_scores": {"num": ["gt_group_ca", "gt_interpersonal_ca"], "cat": []},
        "geo": {"num": ["LocationLatitude", "LocationLongitude"], "cat": []},
    }
    # Age may still be missing on some rows.
    if "Age" in family_specs["demographics"]["num"]:
        frame = frame.dropna(subset=["Age"]).copy()

    for key, spec in family_specs.items():
        if not spec["num"] and not spec["cat"]:
            continue
        cv = _cv_mixed(
            frame,
            numeric_features=spec["num"],
            categorical_features=spec["cat"],
            n_splits=n_splits,
            random_state=random_state,
            model_name=f"common_n_{key}",
        )
        analyses[key] = {"metrics": cv["metrics"], "roc": cv["roc"]}
        rows.append(_metric_row(_pretty_label(key), key, cv["metrics"], frame))

    rows.extend(
        [
            {
                "spec_key": "chance",
                "label": "Chance / prevalence",
                "n": None,
                "n_regular": None,
                "prevalence": None,
                "roc_auc": 0.5,
                "average_precision": None,
                "balanced_accuracy": 0.5,
                "f1": None,
                "brier": None,
            }
        ]
    )
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    best = comparison.loc[comparison["spec_key"] != "chance"].iloc[0]
    return {
        "spec_key": "common_n",
        "label": "Common-N head-to-head",
        "research_question": (
            "When every major predictor family is evaluated on the same complete-case "
            "subset, which features still rank highest for regular-transit discrimination?"
        ),
        "frame": frame,
        "metrics": analyses[str(best["spec_key"])]["metrics"],
        "metrics_table": comparison,
        "comparison": comparison,
        "analyses": analyses,
        "oof": pd.DataFrame(
            {
                "participant_id": frame["participant_id"],
                "y": frame["y"],
                "transit_group": frame["transit_group"],
            }
        ),
        "roc": analyses[str(best["spec_key"])]["roc"],
        "gini_raw": pd.DataFrame(),
        "permutation_importance": pd.DataFrame(),
        "associations": pd.DataFrame(),
        "null_baselines": null_baselines(frame),
        "sample_summary": {
            "n_common": int(len(frame)),
            "n_regular": int(frame["y"].sum()),
            "prevalence": float(frame["y"].mean()),
            "required_columns": base_cols + (["Student status", "Age"] if "Age" in frame else []),
        },
        "numeric_features": [],
        "categorical_features": [],
    }


def run_residual_ca_q28_experiment(
    df: pd.DataFrame,
    *,
    n_boot: int = 2000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Does group CA still differ by regular transit within Q28 strata?"""
    labeled = prepare_labeled_cohort(df)
    frame = labeled.dropna(subset=["gt_group_ca", "gt_interpersonal_ca", "Q28", TARGET]).copy()

    # Overall contrast (replication of transit→CA on Q28-complete rows).
    overall = pd.DataFrame(
        [
            compare_regular_vs_rest(
                frame,
                score_col=col,
                n_boot=n_boot,
                random_state=random_state,
            )
            for col in ("gt_group_ca", "gt_interpersonal_ca")
        ]
    )

    strata_rows: list[dict[str, Any]] = []
    for level, sub in frame.groupby(frame["Q28"].astype(str), dropna=False):
        if sub["y"].nunique() < 2 or len(sub) < 12:
            strata_rows.append(
                {
                    "Q28": level,
                    "n": int(len(sub)),
                    "n_regular": int(sub["y"].sum()),
                    "skipped": True,
                    "reason": "too few cases or single class",
                }
            )
            continue
        for outcome in ("gt_group_ca", "gt_interpersonal_ca"):
            a = sub.loc[sub["y"] == 1, outcome].dropna().to_numpy(dtype=float)
            b = sub.loc[sub["y"] == 0, outcome].dropna().to_numpy(dtype=float)
            if len(a) < 3 or len(b) < 3:
                strata_rows.append(
                    {
                        "Q28": level,
                        "outcome": outcome,
                        "n": int(len(sub)),
                        "n_regular": int(len(a)),
                        "n_not_regular": int(len(b)),
                        "skipped": True,
                        "reason": "stratum too small",
                    }
                )
                continue
            t_res = stats.ttest_ind(a, b, equal_var=False)
            u_res = stats.mannwhitneyu(a, b, alternative="two-sided")
            d = (a.mean() - b.mean()) / np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2)
            strata_rows.append(
                {
                    "Q28": level,
                    "outcome": outcome,
                    "n": int(len(sub)),
                    "n_regular": int(len(a)),
                    "n_not_regular": int(len(b)),
                    "mean_regular": float(a.mean()),
                    "mean_not_regular": float(b.mean()),
                    "mean_diff": float(a.mean() - b.mean()),
                    "cohens_d": float(d),
                    "welch_p": float(t_res.pvalue),
                    "mannwhitney_p": float(u_res.pvalue),
                    "skipped": False,
                }
            )
    strata = pd.DataFrame(strata_rows)

    # Residualized RF: predict residual of regular_transit after Q28 one-hot...
    # Practically: compare CA-only AUC vs CA|Q28 nested lift on same rows.
    ca_cv = _cv_mixed(
        frame,
        numeric_features=["gt_group_ca", "gt_interpersonal_ca"],
        categorical_features=[],
        n_splits=5,
        random_state=random_state,
        model_name="ca_only_on_q28_frame",
    )
    joint_cv = _cv_mixed(
        frame,
        numeric_features=["gt_group_ca", "gt_interpersonal_ca"],
        categorical_features=["Q28"],
        n_splits=5,
        random_state=random_state,
        model_name="ca_plus_q28",
    )
    q28_cv = _cv_mixed(
        frame,
        numeric_features=[],
        categorical_features=["Q28"],
        n_splits=5,
        random_state=random_state,
        model_name="q28_only_on_frame",
    )
    comparison = pd.DataFrame(
        [
            _metric_row("Q28 only", "q28_only", q28_cv["metrics"], frame),
            _metric_row("CA + Q28", "ca_q28", joint_cv["metrics"], frame),
            _metric_row("CA only", "ca_only", ca_cv["metrics"], frame),
            {
                "spec_key": "chance",
                "label": "Chance",
                "n": None,
                "n_regular": None,
                "prevalence": None,
                "roc_auc": 0.5,
                "average_precision": None,
                "balanced_accuracy": 0.5,
                "f1": None,
                "brier": None,
            },
        ]
    ).sort_values("roc_auc", ascending=False)

    return {
        "spec_key": "residual_ca_q28",
        "label": "CA contrast residual to Q28",
        "research_question": (
            "After accounting for ride-share days (Q28), does group/interpersonal CA "
            "still separate regular from non-regular public-transit riders?"
        ),
        "frame": frame,
        "metrics": ca_cv["metrics"],
        "metrics_table": comparison,
        "comparison": comparison,
        "overall_contrast": overall,
        "strata": strata,
        "analyses": {
            "ca_only": {"metrics": ca_cv["metrics"], "roc": ca_cv["roc"]},
            "q28_only": {"metrics": q28_cv["metrics"], "roc": q28_cv["roc"]},
            "ca_q28": {"metrics": joint_cv["metrics"], "roc": joint_cv["roc"]},
        },
        "oof": ca_cv["oof"],
        "roc": ca_cv["roc"],
        "gini_raw": pd.DataFrame(),
        "permutation_importance": pd.DataFrame(),
        "associations": association_table(frame, ["Q28"]),
        "null_baselines": null_baselines(frame),
        "summary_delta": {
            "n": int(len(frame)),
            "ca_only_auc": float(ca_cv["metrics"]["roc_auc"]),
            "q28_only_auc": float(q28_cv["metrics"]["roc_auc"]),
            "ca_q28_auc": float(joint_cv["metrics"]["roc_auc"]),
            "ca_incremental_over_q28": float(
                joint_cv["metrics"]["roc_auc"] - q28_cv["metrics"]["roc_auc"]
            ),
        },
        "numeric_features": ["gt_group_ca", "gt_interpersonal_ca"],
        "categorical_features": ["Q28"],
    }


EXPERIMENT_RUNNERS = {
    "demographics": run_demographics_experiment,
    "country": run_country_experiment,
    "nested_q28_car": run_nested_q28_car_experiment,
    "ca_q28_car": run_ca_q28_car_experiment,
    "country_car": run_country_car_experiment,
    "q27_among_regular": run_q27_among_regular_experiment,
    "common_n": run_common_n_experiment,
    "residual_ca_q28": run_residual_ca_q28_experiment,
}


def run_all_followup_experiments(
    df: pd.DataFrame,
    *,
    experiment_keys: Sequence[str] | None = None,
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    n_boot: int = 2000,
    random_state: int = 42,
) -> dict[str, Any]:
    keys = list(experiment_keys) if experiment_keys is not None else list(EXPERIMENT_RUNNERS)
    unknown = [k for k in keys if k not in EXPERIMENT_RUNNERS]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}; choose from {sorted(EXPERIMENT_RUNNERS)}")
    analyses: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for key in keys:
        runner = EXPERIMENT_RUNNERS[key]
        kwargs: dict[str, Any] = {"random_state": random_state}
        if key == "residual_ca_q28":
            kwargs["n_boot"] = n_boot
        elif key == "common_n":
            kwargs["n_splits"] = n_splits
        else:
            kwargs["n_splits"] = n_splits
            kwargs["n_perm_repeats"] = n_perm_repeats
        result = runner(df, **kwargs)
        analyses[key] = result
        m = result.get("metrics", {})
        summary_rows.append(
            {
                "spec_key": key,
                "label": result.get("label", key),
                "n": int(len(result["frame"])) if "frame" in result else None,
                "roc_auc": m.get("roc_auc"),
                "average_precision": m.get("average_precision"),
                "balanced_accuracy": m.get("balanced_accuracy"),
                "f1": m.get("f1"),
            }
        )
    overview = pd.DataFrame(summary_rows).sort_values("roc_auc", ascending=False)
    card = {
        "secondary_rq": (
            "Extended follow-up experiments answering open questions from the geo, "
            "CA, Q27/Q28, and covariate-followup memos."
        ),
        "experiments": list(keys),
        "overview": overview.to_dict(orient="records"),
        "benchmarks": {"geo_auc": 0.551, "ca_auc": 0.590, "chance_auc": 0.500, "q28_auc": 0.762},
    }
    return {"analyses": analyses, "overview": overview, "results_card": card}


def save_experiment_artifacts(analysis: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = analysis["spec_key"]
    paths: dict[str, Path] = {}
    for name in (
        "frame",
        "metrics_table",
        "comparison",
        "oof",
        "roc",
        "gini_raw",
        "permutation_importance",
        "associations",
        "null_baselines",
        "strata",
        "intensity_counts",
        "overall_contrast",
    ):
        obj = analysis.get(name)
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            path = out / f"{key}_{name}.csv"
            obj.to_csv(path, index=False)
            paths[name] = path
    # Nested comparison tables from analyses dict.
    if "analyses" in analysis and isinstance(analysis["analyses"], dict):
        nested_rows = []
        for nested_key, nested in analysis["analyses"].items():
            if isinstance(nested, dict) and "metrics" in nested:
                nested_rows.append({"nested_key": nested_key, **nested["metrics"]})
        if nested_rows:
            nested_path = out / f"{key}_nested_metrics.csv"
            pd.DataFrame(nested_rows).to_csv(nested_path, index=False)
            paths["nested_metrics"] = nested_path
    card = {
        "secondary_rq": analysis.get("research_question"),
        "spec_key": key,
        "label": analysis.get("label"),
        "sample_n": int(len(analysis["frame"])) if "frame" in analysis else None,
        "cv_metrics": analysis.get("metrics"),
        "summary_delta": analysis.get("summary_delta"),
        "sample_summary": analysis.get("sample_summary"),
    }
    card_path = out / f"{key}_results_card.json"
    card_path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    paths["results_card"] = card_path
    return paths


def save_followup_experiment_bundle(
    bundle: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, analysis in bundle["analyses"].items():
        sub = save_experiment_artifacts(analysis, out / key)
        for name, path in sub.items():
            paths[f"{key}_{name}"] = path
    overview_path = out / "followup_experiments_overview.csv"
    bundle["overview"].to_csv(overview_path, index=False)
    paths["overview"] = overview_path
    card_path = out / "followup_experiments_results_card.json"
    card_path.write_text(json.dumps(bundle["results_card"], indent=2, default=str), encoding="utf-8")
    paths["results_card"] = card_path
    return paths


def _assoc_for_feature(assoc: pd.DataFrame, feature: str) -> pd.DataFrame:
    sub = assoc.loc[assoc["feature"] == feature].copy()
    return sub.sort_values("pct_regular", ascending=True)


def plot_experiment_memo_figure(
    analysis: dict[str, Any],
    *,
    output_path: str | Path,
) -> Path:
    """Publication-quality, experiment-specific memo figure."""
    import matplotlib.pyplot as plt

    from ca_personas.viz_style import (
        PRIMARY,
        SUCCESS,
        WARN,
        add_title_block,
        apply_memo_style,
        family_color,
        plot_auc_bars,
        plot_forest_diffs,
        plot_prevalence_bars,
        plot_roc_curve,
        save_figure,
        short_level,
        style_axes,
    )

    apply_memo_style()
    out = Path(output_path)
    key = analysis["spec_key"]
    label = analysis.get("label", key)
    n = int(len(analysis["frame"])) if "frame" in analysis else 0
    auc = float(analysis.get("metrics", {}).get("roc_auc", float("nan")))
    assoc = analysis.get("associations")
    roc = analysis.get("roc")

    # ---- Specialized layouts -------------------------------------------------
    if key == "demographics":
        fig = plt.figure(figsize=(12.2, 5.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.05], wspace=0.32)
        ax0, ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
        age = _assoc_for_feature(assoc, "Age (tertile)") if isinstance(assoc, pd.DataFrame) else pd.DataFrame()
        if not age.empty:
            plot_prevalence_bars(
                ax0,
                [short_level(x, max_len=16) for x in age["level"]],
                age["pct_regular"],
                ns=age["n"].astype(int).tolist(),
                sample_prevalence=float(analysis["frame"]["y"].mean()),
                title="Age tertile → weekly+ transit",
            )
        sex = _assoc_for_feature(assoc, "Sex") if isinstance(assoc, pd.DataFrame) else pd.DataFrame()
        stu = _assoc_for_feature(assoc, "Student status") if isinstance(assoc, pd.DataFrame) else pd.DataFrame()
        combo = pd.concat([sex.assign(feature="Sex"), stu.assign(feature="Student")], ignore_index=True)
        if not combo.empty:
            labels = [f"{r.feature}: {r.level}" for r in combo.itertuples()]
            plot_prevalence_bars(
                ax1,
                labels,
                combo["pct_regular"],
                ns=combo["n"].astype(int).tolist(),
                sample_prevalence=float(analysis["frame"]["y"].mean()),
                title="Sex & student status",
                cmap_mode="flat",
            )
        if isinstance(roc, pd.DataFrame) and not roc.empty:
            plot_roc_curve(ax2, roc, auc=auc, title="Mixed RF discrimination")
        add_title_block(
            fig,
            "Demographics as predictors of regular transit",
            f"Age + Sex + Student status  ·  n={n}  ·  CV ROC-AUC={auc:.3f}  ·  Age dominates permutation importance",
        )
        fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.12)
        return save_figure(fig, out)

    if key == "country":
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
        sub = _assoc_for_feature(assoc, "Country of residence") if isinstance(assoc, pd.DataFrame) else pd.DataFrame()
        if not sub.empty:
            keep = sub.loc[sub["n"] >= 5].copy()
            plot_prevalence_bars(
                axes[0],
                [short_level(x, max_len=20) for x in keep["level"]],
                keep["pct_regular"],
                ns=keep["n"].astype(int).tolist(),
                sample_prevalence=float(analysis["frame"]["y"].mean()),
                title="Countries with n ≥ 5",
            )
        if isinstance(roc, pd.DataFrame) and not roc.empty:
            plot_roc_curve(axes[1], roc, auc=auc, title="Country-only Random Forest")
        add_title_block(
            fig,
            "Country of residence vs regular transit",
            f"Dedicated country RF  ·  n={n}  ·  AUC={auc:.3f}  ·  essentially ties lat/long geo (≈0.551)",
        )
        fig.subplots_adjust(top=0.80, left=0.10, right=0.98, bottom=0.12, wspace=0.28)
        return save_figure(fig, out)

    if key == "residual_ca_q28":
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), gridspec_kw={"width_ratios": [1.05, 1.15]})
        comp = analysis.get("comparison")
        if isinstance(comp, pd.DataFrame) and not comp.empty:
            plot_df = comp.dropna(subset=["roc_auc"]).copy()
            plot_auc_bars(
                axes[0],
                plot_df["label"].tolist(),
                plot_df["roc_auc"].tolist(),
                colors=[family_color(k) for k in plot_df["spec_key"]],
                ns=plot_df["n"].tolist() if "n" in plot_df else None,
                benchmarks=("chance", "geo", "ca"),
                xlim=(0.45, 0.88),
            )
            axes[0].set_title("Nested predictive models", loc="left", fontsize=11, pad=8)
        strata = analysis.get("strata")
        if isinstance(strata, pd.DataFrame):
            plot_forest_diffs(axes[1], strata, outcome="gt_group_ca")
        delta = analysis.get("summary_delta") or {}
        add_title_block(
            fig,
            "Does CA still separate riders after accounting for Q28?",
            (
                f"n={n}  ·  CA-only AUC={delta.get('ca_only_auc', auc):.3f}  ·  "
                f"Q28-only={delta.get('q28_only_auc', float('nan')):.3f}  ·  "
                f"CA+Q28={delta.get('ca_q28_auc', float('nan')):.3f}  ·  "
                f"CA incremental ≈ {delta.get('ca_incremental_over_q28', float('nan')):+.3f}"
            ),
        )
        fig.subplots_adjust(top=0.80, left=0.10, right=0.98, bottom=0.12, wspace=0.30)
        return save_figure(fig, out)

    if key == "q27_among_regular":
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.3), gridspec_kw={"width_ratios": [1.0, 1.15]})
        counts = analysis.get("intensity_counts")
        if isinstance(counts, pd.DataFrame) and not counts.empty:
            c = counts.sort_values("n", ascending=True)
            axes[0].barh(
                [short_level(x, max_len=26) for x in c["Q27"]],
                c["n"],
                color=PRIMARY,
                edgecolor="white",
                height=0.72,
            )
            for i, (n_i, _) in enumerate(zip(c["n"], c["Q27"])):
                axes[0].text(n_i + 0.4, i, str(int(n_i)), va="center", fontsize=9, color="#1B1F24")
            axes[0].set_xlabel("Regular riders (n)")
            axes[0].set_title("Q27 among weekly+ riders", loc="left", fontsize=11, pad=8)
            style_axes(axes[0], grid="x")
        comp = analysis.get("comparison")
        if isinstance(comp, pd.DataFrame) and not comp.empty:
            plot_auc_bars(
                axes[1],
                comp["label"].tolist(),
                comp["roc_auc"].tolist(),
                colors=[family_color(k) for k in comp["spec_key"]],
                ns=comp["n"].tolist() if "n" in comp else None,
                xlabel="CV ROC-AUC for high intensity",
                benchmarks=("chance",),
                xlim=(0.40, 0.70),
            )
            axes[1].set_title("Predictors of high intensity (≥3–4 rides)", loc="left", fontsize=11, pad=8)
        summary = analysis.get("sample_summary") or {}
        add_title_block(
            fig,
            "Transit-day intensity among regular riders",
            (
                f"n_riders={summary.get('n_regular_riders', n)}  ·  "
                f"high intensity={summary.get('n_high_intensity', '—')} "
                f"({summary.get('pct_high_intensity', float('nan')):.0%})  ·  "
                f"best AUC={auc:.3f} (near chance)"
            ),
        )
        fig.subplots_adjust(top=0.80, left=0.12, right=0.98, bottom=0.12, wspace=0.30)
        return save_figure(fig, out)

    # Nested / comparison experiments: rich AUC ranking
    if "comparison" in analysis and isinstance(analysis["comparison"], pd.DataFrame):
        frame = analysis["comparison"].dropna(subset=["roc_auc"]).copy()
        if len(frame):
            fig, ax = plt.subplots(figsize=(10.2, max(4.6, 0.55 * len(frame) + 2.2)))
            colors = [family_color(k) for k in frame["spec_key"]]
            # Soften chance bars
            colors = [
                WARN if k in {"chance", "geo_benchmark", "ca_benchmark"} else c
                for k, c in zip(frame["spec_key"], colors)
            ]
            plot_auc_bars(
                ax,
                frame["label"].tolist(),
                frame["roc_auc"].tolist(),
                colors=colors,
                ns=frame["n"].tolist() if "n" in frame.columns else None,
                benchmarks=("chance", "geo", "ca"),
                xlim=(0.45, min(0.90, float(frame["roc_auc"].max()) + 0.10)),
            )
            subtitle_bits = [f"n={n}", f"primary AUC={auc:.3f}"]
            if analysis.get("summary_delta"):
                d = analysis["summary_delta"]
                if "delta_q28_to_q28_q21" in d:
                    subtitle_bits.append(f"Δ(Q28→Q28+Q21)={d['delta_q28_to_q28_q21']:+.3f}")
                if "ca_incremental_over_q28" in d:
                    subtitle_bits.append(f"CA incremental={d['ca_incremental_over_q28']:+.3f}")
            if key == "common_n":
                subtitle_bits.append("equal complete-case frame")
            add_title_block(fig, label, "  ·  ".join(subtitle_bits))
            fig.subplots_adjust(top=0.82, left=0.28, right=0.96, bottom=0.14)
            return save_figure(fig, out)

    # Default two-panel: prevalence + ROC
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), gridspec_kw={"width_ratios": [1.1, 1.0]})
    if isinstance(assoc, pd.DataFrame) and not assoc.empty and "pct_regular" in assoc.columns:
        primary = assoc["feature"].iloc[0]
        sub = _assoc_for_feature(assoc, primary)
        plot_prevalence_bars(
            axes[0],
            [short_level(x) for x in sub["level"]],
            sub["pct_regular"],
            ns=sub["n"].astype(int).tolist(),
            sample_prevalence=float(analysis["frame"]["y"].mean()) if "frame" in analysis else None,
            title=f"{primary} prevalence",
        )
    else:
        axes[0].axis("off")
    if isinstance(roc, pd.DataFrame) and not roc.empty:
        plot_roc_curve(axes[1], roc, auc=auc, color=SUCCESS if auc >= 0.70 else PRIMARY)
    else:
        axes[1].axis("off")
    add_title_block(fig, label, f"Follow-up experiment  ·  n={n}  ·  CV ROC-AUC={auc:.3f}")
    fig.subplots_adjust(top=0.80, left=0.10, right=0.98, bottom=0.12, wspace=0.28)
    return save_figure(fig, out)


def plot_overview_figure(overview: pd.DataFrame, *, output_path: str | Path) -> Path:
    """Flagship overview figure for the wave-2 memo agenda."""
    import matplotlib.pyplot as plt

    from ca_personas.viz_style import (
        SUCCESS,
        WARN,
        add_title_block,
        apply_memo_style,
        color_by_auc,
        family_color,
        plot_auc_bars,
        save_figure,
    )

    apply_memo_style()
    out = Path(output_path)
    frame = overview.dropna(subset=["roc_auc"]).copy()
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    # Recolor by strength for overview readability
    colors = [color_by_auc(a) for a in frame["roc_auc"]]
    plot_auc_bars(
        ax,
        frame["label"].tolist(),
        frame["roc_auc"].tolist(),
        colors=colors,
        ns=frame["n"].tolist() if "n" in frame.columns else None,
        xlabel="Primary reported stratified CV ROC-AUC",
        benchmarks=("chance", "geo", "ca", "q28"),
        xlim=(0.48, 0.86),
        highlight_best=True,
    )
    best = frame.sort_values("roc_auc", ascending=False).iloc[0]
    add_title_block(
        fig,
        "Wave-2 follow-up experiments — discrimination overview",
        (
            f"Best: {best['label']} (AUC={best['roc_auc']:.3f}, n={int(best['n'])})  ·  "
            "Green ≥0.70  ·  Teal ≥0.60  ·  Slate ≥0.55  ·  Tan <0.55  ·  "
            "Reference lines: chance / geo / CA / full-cohort Q28"
        ),
    )
    # Dual legends: benchmarks stay on-axes; AUC bands sit below the figure
    from matplotlib.patches import Patch

    band_handles = [
        Patch(facecolor=SUCCESS, edgecolor="none", label="Strong (≥0.70)"),
        Patch(facecolor="#1F4E5F", edgecolor="none", label="Useful (≥0.60)"),
        Patch(facecolor="#2F6F7E", edgecolor="none", label="Modest (≥0.55)"),
        Patch(facecolor=WARN, edgecolor="none", label="Weak (<0.55)"),
    ]
    fig.legend(
        handles=band_handles,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.63, 0.01),
        title="AUC band",
        title_fontsize=8.5,
    )
    fig.subplots_adjust(top=0.82, left=0.30, right=0.96, bottom=0.16)
    return save_figure(fig, out)


def run_followup_experiments_pipeline(
    *,
    prolific_paths: Sequence[str | Path] | None = None,
    qualtrics_path: str | Path | None = None,
    join_how: str = "inner",
    output_dir: str | Path = "outputs/followup_experiments",
    figures_dir: str | Path | None = "memos/figures",
    experiment_keys: Sequence[str] | None = None,
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    n_boot: int = 2000,
    random_state: int = 42,
) -> dict[str, Path]:
    """Load cohort, run extended follow-up experiments, write artifacts + figures."""
    participants, _report = load_full_cohort(
        prolific_paths=prolific_paths,
        qualtrics_path=qualtrics_path,
        join_how=join_how,
    )
    bundle = run_all_followup_experiments(
        participants,
        experiment_keys=experiment_keys,
        n_splits=n_splits,
        n_perm_repeats=n_perm_repeats,
        n_boot=n_boot,
        random_state=random_state,
    )
    paths = save_followup_experiment_bundle(bundle, output_dir)
    if figures_dir is not None:
        fig_root = Path(figures_dir)
        fig_root.mkdir(parents=True, exist_ok=True)
        for key, analysis in bundle["analyses"].items():
            paths[f"figure_{key}"] = plot_experiment_memo_figure(
                analysis,
                output_path=fig_root / f"{key}_followup_memo.png",
            )
        paths["figure_overview"] = plot_overview_figure(
            bundle["overview"],
            output_path=fig_root / "followup_experiments_overview.png",
        )
    return paths
