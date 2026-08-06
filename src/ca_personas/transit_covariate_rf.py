"""Follow-up RQs: do car access, employment, or ride-share predict regular transit?

Companion analyses to the geography → transit memo. Each feature family is
evaluated with a balanced Random Forest (stratified CV) predicting the same
Q26-based ``regular_transit`` outcome, with chance / prevalence baselines and
head-to-head comparison against the published geo (AUC≈0.55) and CA (AUC≈0.59)
benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ca_personas.geo_transit_rf import classification_metrics, majority_baseline_probs
from ca_personas.load import load_full_cohort
from ca_personas.transit_ca import PRIMARY_REGULAR_LABELS, label_regular_riders

TARGET = "regular_transit"

# Named feature families from the geo memo follow-up paragraph.
FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "car_access": {
        "features": ["Q20", "Q21"],
        "label": "Car license & access (Q20/Q21)",
        "research_question": (
            "Do driver's license (Q20) and car access (Q21) predict whether a "
            "matched respondent takes public transportation regularly?"
        ),
    },
    "employment": {
        "features": ["Employment status"],
        "label": "Employment status",
        "research_question": (
            "Does employment status predict whether a matched respondent takes "
            "public transportation regularly?"
        ),
    },
    "rideshare": {
        "features": ["Q28", "Q29"],
        "label": "Ride-share frequency (Q28/Q29)",
        "research_question": (
            "Does ride-share use (Q28 days; Q29 typical rides) predict whether a "
            "matched respondent takes public transportation regularly?"
        ),
    },
    "q27_intensity": {
        "features": ["Q27"],
        "label": "Transit intensity on use days (Q27)",
        "research_question": (
            "Does rides-per-typical-public-transit-day (Q27) predict whether a "
            "matched respondent takes public transportation regularly?"
        ),
    },
    "q28_days": {
        "features": ["Q28"],
        "label": "Ride-share days (Q28)",
        "research_question": (
            "Do ride-share days in the last three months (Q28) predict whether a "
            "matched respondent takes public transportation regularly?"
        ),
    },
    "q27_q28": {
        "features": ["Q27", "Q28"],
        "label": "Transit intensity + ride-share days (Q27/Q28)",
        "research_question": (
            "Do public-transit intensity (Q27) and ride-share days (Q28) jointly "
            "predict regular public-transit use in a traditional ML classifier?"
        ),
    },
    "mobility_bundle": {
        "features": ["Q20", "Q21", "Q28", "Q29", "Employment status"],
        "label": "Car + ride-share + employment bundle",
        "research_question": (
            "Do car access, ride-share frequency, and employment status jointly "
            "predict regular public-transit use better than any family alone?"
        ),
    },
}


def prepare_covariate_frame(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    regular_labels: Sequence[str] = tuple(PRIMARY_REGULAR_LABELS),
) -> pd.DataFrame:
    """Label regular riders and keep complete cases for the requested features."""
    labeled = label_regular_riders(df, regular_labels=regular_labels)
    missing = [c for c in features if c not in labeled.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    keep = [
        "participant_id",
        *features,
        TARGET,
        "transit_group",
        "Q26",
    ]
    optional = [
        c
        for c in (
            "Age",
            "Sex",
            "Country of residence",
            "Student status",
            "gt_group_ca",
            "gt_interpersonal_ca",
            "LocationLatitude",
            "LocationLongitude",
            "Employment status",
            "Q20",
            "Q21",
            "Q27",
            "Q28",
            "Q29",
        )
        if c in labeled.columns and c not in keep
    ]
    out = labeled[[c for c in keep + optional if c in labeled.columns]].copy()
    # Treat blank / sentinel strings as missing before dropna.
    for col in features:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            text = out[col].astype(str).str.strip()
            out.loc[text.eq("") | text.str.lower().isin({"nan", "none", "<na>"}), col] = pd.NA
            out[col] = out[col].astype("string")
    out = out.dropna(subset=[*features, TARGET]).copy()
    out[TARGET] = out[TARGET].astype(bool)
    out["y"] = out[TARGET].astype(int)
    return out.reset_index(drop=True)


def make_categorical_rf_pipeline(
    features: Sequence[str],
    *,
    n_estimators: int = 500,
    min_samples_leaf: int = 3,
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> Pipeline:
    pre = ColumnTransformer(
        [
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
                list(features),
            )
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("pre", pre), ("rf", clf)])


def _cv_proba(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    n_splits: int,
    random_state: int,
    model_name: str,
) -> dict[str, Any]:
    X = frame[list(features)].copy()
    for col in features:
        X[col] = X[col].astype("string").fillna("<NA>").astype(str)
    y = frame["y"].to_numpy()
    n_splits = min(n_splits, int(np.min(np.bincount(y))))
    if n_splits < 2:
        raise ValueError("Need at least 2 examples per class for stratified CV")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipe = make_categorical_rf_pipeline(features, random_state=random_state)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    metrics = classification_metrics(y, proba)
    metrics["model"] = model_name
    metrics["n_splits"] = int(n_splits)
    metrics["features"] = list(features)
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, proba)
    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
    oof = frame[["participant_id", "y", "transit_group"]].copy()
    oof["y_prob"] = proba
    oof["y_pred"] = (proba >= 0.5).astype(int)
    return {
        "metrics": metrics,
        "proba": proba,
        "roc": roc,
        "oof": oof,
        "cv": cv,
        "pipeline_template": pipe,
        "X": X,
        "y": y,
    }


def association_table(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Regular-transit prevalence by level of each categorical feature."""
    rows: list[dict[str, Any]] = []
    for col in features:
        for level, sub in frame.groupby(frame[col].astype(str), dropna=False):
            n = int(len(sub))
            n_reg = int(sub["y"].sum())
            rows.append(
                {
                    "feature": col,
                    "level": level,
                    "n": n,
                    "n_regular": n_reg,
                    "n_not_regular": n - n_reg,
                    "pct_regular": float(n_reg / n) if n else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "n"], ascending=[True, False])


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


def run_feature_family_analysis(
    df: pd.DataFrame,
    *,
    spec_key: str,
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run RF CV + importances for one named feature family."""
    if spec_key not in FEATURE_SPECS:
        raise ValueError(f"Unknown spec {spec_key!r}; expected one of {sorted(FEATURE_SPECS)}")
    spec = FEATURE_SPECS[spec_key]
    features = list(spec["features"])
    frame = prepare_covariate_frame(df, features)
    if len(frame) < 20:
        raise ValueError(f"Analytic frame too small for {spec_key} (n={len(frame)})")

    cv_result = _cv_proba(
        frame,
        features,
        n_splits=n_splits,
        random_state=random_state,
        model_name=f"random_forest_{spec_key}",
    )
    # Fit full model for importances.
    pipe = make_categorical_rf_pipeline(features, random_state=random_state)
    pipe.fit(cv_result["X"], cv_result["y"])
    enc_names = list(pipe.named_steps["pre"].get_feature_names_out())
    gini = pd.DataFrame(
        {
            "encoded_feature": enc_names,
            "importance_gini": pipe.named_steps["rf"].feature_importances_,
        }
    ).sort_values("importance_gini", ascending=False)
    # Aggregate one-hot importances back to raw columns.
    raw_rows = []
    for feat in features:
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
            "feature": list(features),
            "importance_perm_mean": perm.importances_mean,
            "importance_perm_std": perm.importances_std,
        }
    ).sort_values("importance_perm_mean", ascending=False)

    nulls = null_baselines(frame)
    assoc = association_table(frame, features)
    auc = float(cv_result["metrics"]["roc_auc"])
    prev_auc = float(nulls.loc[nulls["model"] == "prevalence_prob", "roc_auc"].iloc[0])

    summary = {
        "secondary_rq": spec["research_question"],
        "spec_key": spec_key,
        "label": spec["label"],
        "outcome": {
            "name": "regular_transit",
            "definition": (
                "Q26 in {4-8 days a month, 8 or more days a month} "
                "(weekly-or-more public transit)"
            ),
        },
        "features": features,
        "sample": {
            "n": int(len(frame)),
            "n_regular": int(frame["y"].sum()),
            "n_not_regular": int((frame["y"] == 0).sum()),
            "prevalence": float(frame["y"].mean()),
        },
        "cv_metrics": cv_result["metrics"],
        "baselines": {
            "prevalence_roc_auc": prev_auc,
            "geo_benchmark_auc": 0.551,
            "ca_benchmark_auc": 0.590,
        },
        "verdict": {
            "roc_auc": auc,
            "beats_chance_auc_0_5": bool(auc > 0.5) if not np.isnan(auc) else False,
            "beats_geo_benchmark_0_551": bool(auc > 0.551) if not np.isnan(auc) else False,
            "beats_ca_benchmark_0_590": bool(auc > 0.590) if not np.isnan(auc) else False,
            "interpretation": (
                f"{spec['label']} Random Forest CV ROC-AUC = {auc:.3f}. "
                + (
                    "Stronger discrimination than the geo benchmark (≈0.55)."
                    if (not np.isnan(auc) and auc > 0.551)
                    else "Comparable to or weaker than the geo benchmark (≈0.55)."
                )
                + " "
                + (
                    "Exceeds the CA-score benchmark (≈0.59)."
                    if (not np.isnan(auc) and auc > 0.590)
                    else "Does not exceed the CA-score benchmark (≈0.59)."
                )
            ),
        },
        "caveats": [
            "Same-wave self-reports; association ≠ causal effect on transit use.",
            "Complete-case analysis for the listed features; N may be smaller than the full matched cohort when items are missing.",
            "Regular transit is a thresholded Q26 label (weekly+), not continuous ridership.",
        ],
    }
    return {
        "frame": frame,
        "features": features,
        "spec_key": spec_key,
        "metrics": cv_result["metrics"],
        "metrics_table": pd.DataFrame(
            [cv_result["metrics"], *nulls.to_dict(orient="records")]
        ),
        "oof": cv_result["oof"],
        "roc": cv_result["roc"],
        "gini_encoded": gini.reset_index(drop=True),
        "gini_raw": gini_raw.reset_index(drop=True),
        "permutation_importance": perm_table.reset_index(drop=True),
        "associations": assoc,
        "null_baselines": nulls,
        "summary": summary,
        "pipeline": pipe,
    }


def run_all_followup_analyses(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    random_state: int = 42,
    spec_keys: Sequence[str] | None = None,
    geo_benchmark_auc: float | None = None,
    ca_benchmark_auc: float | None = None,
    benchmark_n: int | None = None,
    benchmark_n_regular: int | None = None,
) -> dict[str, Any]:
    """Run every follow-up feature family and build a comparison table.

    When ``geo_benchmark_auc`` / ``ca_benchmark_auc`` are provided (from live
    companion RF runs on the same cohort), those values are used in the
    comparison table. Otherwise the published seed=42 full-cohort benchmarks
    (0.551 / 0.590) are retained for offline/excerpt smoke tests.
    """
    keys = list(spec_keys) if spec_keys is not None else list(FEATURE_SPECS)
    analyses: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for key in keys:
        analysis = run_feature_family_analysis(
            df,
            spec_key=key,
            n_splits=n_splits,
            n_perm_repeats=n_perm_repeats,
            random_state=random_state,
        )
        analyses[key] = analysis
        m = analysis["metrics"]
        rows.append(
            {
                "spec_key": key,
                "label": FEATURE_SPECS[key]["label"],
                "n": m["n"],
                "n_regular": int(analysis["frame"]["y"].sum()),
                "prevalence": m["prevalence"],
                "roc_auc": m["roc_auc"],
                "average_precision": m["average_precision"],
                "balanced_accuracy": m["balanced_accuracy"],
                "f1": m["f1"],
                "brier": m["brier"],
            }
        )
    geo_auc = 0.551 if geo_benchmark_auc is None else float(geo_benchmark_auc)
    ca_auc = 0.590 if ca_benchmark_auc is None else float(ca_benchmark_auc)
    # Round published-style benchmarks to 3 decimals for table display stability.
    geo_auc_display = round(geo_auc, 3)
    ca_auc_display = round(ca_auc, 3)
    n_bench = 241 if benchmark_n is None else int(benchmark_n)
    n_reg_bench = 101 if benchmark_n_regular is None else int(benchmark_n_regular)
    rows.extend(
        [
            {
                "spec_key": "geo_benchmark",
                "label": "Lat/long (geo memo benchmark)",
                "n": n_bench,
                "n_regular": n_reg_bench,
                "prevalence": n_reg_bench / n_bench if n_bench else None,
                "roc_auc": geo_auc_display,
                "average_precision": None,
                "balanced_accuracy": None,
                "f1": None,
                "brier": None,
            },
            {
                "spec_key": "ca_benchmark",
                "label": "Group + interpersonal CA (CA memo benchmark)",
                "n": n_bench,
                "n_regular": n_reg_bench,
                "prevalence": n_reg_bench / n_bench if n_bench else None,
                "roc_auc": ca_auc_display,
                "average_precision": None,
                "balanced_accuracy": None,
                "f1": None,
                "brier": None,
            },
            {
                "spec_key": "chance",
                "label": "Chance / prevalence",
                "n": None,
                "n_regular": None,
                "prevalence": None,
                "roc_auc": 0.500,
                "average_precision": None,
                "balanced_accuracy": 0.5,
                "f1": None,
                "brier": None,
            },
        ]
    )
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    best_key = max(
        (k for k in keys),
        key=lambda k: analyses[k]["metrics"]["roc_auc"],
    )
    card = {
        "secondary_rq": (
            "Among the geo-memo follow-up candidates (car access/license, employment, "
            "ride-share frequency), which features best predict regular public-transit use?"
        ),
        "benchmarks": {
            "geo_auc": geo_auc_display,
            "ca_auc": ca_auc_display,
            "chance_auc": 0.500,
        },
        "best_spec": best_key,
        "best_auc": analyses[best_key]["metrics"]["roc_auc"],
        "comparison": comparison.to_dict(orient="records"),
    }
    return {"analyses": analyses, "comparison": comparison, "results_card": card}


def save_feature_family_artifacts(
    analysis: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = analysis["spec_key"]
    paths: dict[str, Path] = {}
    mapping = {
        "frame": f"{key}_modeling_frame.csv",
        "metrics_table": f"{key}_metrics.csv",
        "oof": f"{key}_oof_predictions.csv",
        "roc": f"{key}_roc_curve.csv",
        "gini_encoded": f"{key}_gini_encoded.csv",
        "gini_raw": f"{key}_gini_raw.csv",
        "permutation_importance": f"{key}_permutation_importance.csv",
        "associations": f"{key}_associations.csv",
        "null_baselines": f"{key}_null_baselines.csv",
    }
    for name, filename in mapping.items():
        frame = analysis.get(name)
        if isinstance(frame, pd.DataFrame):
            path = out / filename
            frame.to_csv(path, index=False)
            paths[name] = path
    summary_path = out / f"{key}_summary.json"
    summary_path.write_text(json.dumps(analysis["summary"], indent=2), encoding="utf-8")
    paths["summary"] = summary_path
    card_path = out / f"{key}_results_card.json"
    card = {
        "secondary_rq": analysis["summary"]["secondary_rq"],
        "sample": analysis["summary"]["sample"],
        "cv_metrics": analysis["summary"]["cv_metrics"],
        "baselines": analysis["summary"]["baselines"],
        "verdict": analysis["summary"]["verdict"],
        "caveats": analysis["summary"]["caveats"],
        "top_permutation": analysis["permutation_importance"].head(5).to_dict(orient="records"),
    }
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    paths["results_card"] = card_path
    return paths


def save_followup_bundle(
    bundle: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, analysis in bundle["analyses"].items():
        sub = save_feature_family_artifacts(analysis, out / key)
        for name, path in sub.items():
            paths[f"{key}_{name}"] = path
    comp_path = out / "followup_comparison.csv"
    bundle["comparison"].to_csv(comp_path, index=False)
    paths["comparison"] = comp_path
    card_path = out / "followup_results_card.json"
    card_path.write_text(json.dumps(bundle["results_card"], indent=2), encoding="utf-8")
    paths["results_card"] = card_path
    return paths


def plot_family_memo_figure(
    analysis: dict[str, Any],
    *,
    output_path: str | Path,
) -> Path:
    """Two-panel memo figure: prevalence by feature level + ROC curve."""
    import matplotlib.pyplot as plt

    from ca_personas.viz_style import (
        PRIMARY,
        SUCCESS,
        add_title_block,
        apply_memo_style,
        plot_prevalence_bars,
        plot_roc_curve,
        save_figure,
        short_level,
    )

    apply_memo_style()
    assoc = analysis["associations"]
    roc = analysis["roc"]
    label = analysis["summary"]["label"]
    auc = float(analysis["metrics"]["roc_auc"])
    n = int(analysis["summary"]["sample"]["n"])
    primary = analysis["features"][0]
    sub = assoc.loc[assoc["feature"] == primary].copy().sort_values("pct_regular", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    plot_prevalence_bars(
        axes[0],
        [short_level(x) for x in sub["level"]],
        sub["pct_regular"],
        ns=sub["n"].astype(int).tolist(),
        sample_prevalence=float(analysis["summary"]["sample"]["prevalence"]),
        title=f"{primary}: prevalence by level",
    )
    plot_roc_curve(
        axes[1],
        roc,
        auc=auc,
        title="Stratified CV ROC",
        color=SUCCESS if auc >= 0.70 else PRIMARY,
    )
    add_title_block(
        fig,
        f"Predictor of regular transit: {label}",
        f"Balanced Random Forest  ·  n={n}  ·  CV ROC-AUC={auc:.3f}  ·  outcome = Q26 weekly+",
    )
    fig.subplots_adjust(top=0.80, left=0.14, right=0.98, bottom=0.12, wspace=0.28)
    return save_figure(fig, Path(output_path))


def plot_comparison_memo_figure(
    comparison: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str = "Geo-memo follow-ups: predictors of regular public transit",
) -> Path:
    """Bar chart comparing follow-up AUCs against geo/CA/chance benchmarks."""
    import matplotlib.pyplot as plt

    from ca_personas.viz_style import (
        WARN,
        add_title_block,
        apply_memo_style,
        family_color,
        plot_auc_bars,
        save_figure,
    )

    apply_memo_style()
    frame = comparison.dropna(subset=["roc_auc"]).copy()
    colors = [
        WARN if k in {"geo_benchmark", "ca_benchmark", "chance"} else family_color(str(k))
        for k in frame["spec_key"]
    ]
    fig, ax = plt.subplots(figsize=(10.4, max(4.8, 0.48 * len(frame) + 2.0)))
    plot_auc_bars(
        ax,
        frame["label"].tolist(),
        frame["roc_auc"].astype(float).tolist(),
        colors=colors,
        ns=frame["n"].tolist() if "n" in frame.columns else None,
        benchmarks=("chance", "geo", "ca"),
        xlim=(0.45, min(0.88, float(frame["roc_auc"].max()) + 0.08)),
        legend_loc="below",
    )
    best = frame.sort_values("roc_auc", ascending=False).iloc[0]
    add_title_block(
        fig,
        title,
        f"Best family: {best['label']} (AUC={best['roc_auc']:.3f})  ·  seeded stratified CV  ·  unequal complete-case N noted in memo",
    )
    fig.subplots_adjust(top=0.82, left=0.32, right=0.96, bottom=0.12)
    return save_figure(fig, Path(output_path))


def run_transit_covariate_pipeline(
    *,
    prolific_paths: Sequence[str | Path] | None = None,
    qualtrics_path: str | Path | None = None,
    join_how: str = "inner",
    output_dir: str | Path = "outputs/transit_covariate_rf",
    n_splits: int = 5,
    n_perm_repeats: int = 30,
    random_state: int = 42,
    spec_keys: Sequence[str] | None = None,
    figures_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Load cohort, run all follow-up RFs, write artifacts (and optional figures)."""
    participants, _report = load_full_cohort(
        prolific_paths=prolific_paths,
        qualtrics_path=qualtrics_path,
        join_how=join_how,
        allow_excerpt_fallback=False,
    )
    bundle = run_all_followup_analyses(
        participants,
        n_splits=n_splits,
        n_perm_repeats=n_perm_repeats,
        random_state=random_state,
        spec_keys=spec_keys,
    )
    paths = save_followup_bundle(bundle, output_dir)
    if figures_dir is not None:
        fig_root = Path(figures_dir)
        fig_root.mkdir(parents=True, exist_ok=True)
        for key, analysis in bundle["analyses"].items():
            paths[f"figure_{key}"] = plot_family_memo_figure(
                analysis,
                output_path=fig_root / f"{key}_predicts_transit_memo.png",
            )
        paths["figure_comparison"] = plot_comparison_memo_figure(
            bundle["comparison"],
            output_path=fig_root / "transit_covariate_followups_memo.png",
        )
    return paths
