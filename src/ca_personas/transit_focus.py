"""Secondary *focus* RQs: predict transit use with mobility items held out.

Two complementary tasks on the Prolific↔Qualtrics matched cohort:

**TF1 — Regular rider (binary).** Given demographics, employment, and geography
(and optionally CA scores), estimate whether the respondent is a weekly+ public
transit rider (``regular_transit`` from Q26). All transportation self-reports
(Q26–Q29, Q20, Q21) and mobility free-text (Q19) are **held out** of the feature
set / persona prompt.

**TF2 — Transit intensity (ordinal).** Given demographics, employment,
geography, and CA scores, estimate Q26 day-frequency (and optionally Q27 rides
among users). Again mobility self-reports are held out as predictors.

This module also builds LLM persona prompts that ask for transit predictions
instead of CA scores, reusing the narrative demos/employment/geo blocks from
``personas.py`` without the transit / voice layers.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ca_personas.geo_transit_rf import classification_metrics, majority_baseline_probs
from ca_personas.load import load_full_cohort
from ca_personas.personas import (
    PersonaPrompt,
    demos_sentences,
    employment_sentences,
    geo_sentences,
)
from ca_personas.transit_ca import (
    PRIMARY_REGULAR_LABELS,
    Q26_ORDER,
    RIDES_PER_DAY_ORDER,
    label_regular_riders,
    normalize_q26,
)

# ---------------------------------------------------------------------------
# Feature specs (transit / car / rideshare held out)
# ---------------------------------------------------------------------------

DEMO_FEATURES = [
    "Age",
    "Sex",
    "Country of residence",
    "Student status",
]
EMPLOYMENT_FEATURES = ["Employment status"]
GEO_FEATURES = ["LocationLatitude", "LocationLongitude"]
CA_FEATURES = ["gt_group_ca", "gt_interpersonal_ca"]

# Explicit hold-out list — never used as predictors in this focus set.
HELD_OUT_MOBILITY = ["Q26", "Q27", "Q28", "Q29", "Q20", "Q21", "Q19"]

PROFILE_FEATURES = DEMO_FEATURES + EMPLOYMENT_FEATURES + GEO_FEATURES
PROFILE_CA_FEATURES = PROFILE_FEATURES + CA_FEATURES

NUMERIC_FEATURES = {
    "Age",
    "LocationLatitude",
    "LocationLongitude",
    "gt_group_ca",
    "gt_interpersonal_ca",
}

FOCUS_SPECS: dict[str, dict[str, Any]] = {
    "tf1_profile_regular": {
        "task": "regular",
        "features": PROFILE_FEATURES,
        "label": "Profile (demos + employment + geo) → regular transit",
        "research_question": (
            "Given demographics, employment, and geography — with all transit / "
            "ride-share / car self-reports held out — can a model estimate whether "
            "the respondent is a regular (weekly+) public-transit user?"
        ),
    },
    "tf1_profile_ca_regular": {
        "task": "regular",
        "features": PROFILE_CA_FEATURES,
        "label": "Profile + CA → regular transit",
        "research_question": (
            "Given demographics, employment, geography, and PRCA CA scores — with "
            "transit self-reports held out — can a model estimate regular "
            "public-transit use?"
        ),
    },
    "tf2_intensity_q26": {
        "task": "q26",
        "features": PROFILE_CA_FEATURES,
        "label": "Profile + CA → Q26 day-frequency",
        "research_question": (
            "Given demographics, employment, geography, and CA scores — with "
            "transit self-reports held out — can a model estimate public-transit "
            "day-frequency (Q26)?"
        ),
    },
    "tf2_intensity_q27": {
        "task": "q27",
        "features": PROFILE_CA_FEATURES,
        "label": "Profile + CA → Q27 rides/day (among users with Q27)",
        "research_question": (
            "Given demographics, employment, geography, and CA scores — with "
            "transit self-reports held out — can a model estimate rides on a "
            "typical transit day (Q27) among respondents who reported Q27?"
        ),
    },
}

# ---------------------------------------------------------------------------
# LLM transit-focus prompts (separate from CA SYSTEM_PROMPT)
# ---------------------------------------------------------------------------

TRANSIT_FOCUS_SYSTEM_PROMPT = """You inhabit the identity described to you. Answer as that
person would, in first person, using only the facts in the profile.

You will estimate this person's public-transportation habits. The profile
intentionally omits their own transit, ride-share, and car-access survey
answers — infer from demographics, work situation, place, and (when given)
communication-apprehension scores only. Do not invent specific trip histories
that contradict the listed facts.

Respond with ONLY a JSON object, no other text:
{
  "regular_transit": true | false,
  "q26_days": "Never" | "0-1 days a month" | "2-4 days a month" | "4-8 days a month" | "8 or more days a month",
  "confidence": "low" | "moderate" | "high"
}

Use weekly-or-more ("4-8 days a month" or "8 or more days a month") for
regular_transit = true.
"""

TRANSIT_FOCUS_TIERS = ("tf_demos", "tf_employment", "tf_geo", "tf_geo_ca")


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "na"}


def _ca_sentences(row: pd.Series) -> list[str]:
    lines: list[str] = []
    if _present(row.get("gt_group_ca")):
        lines.append(
            f"Your self-reported group-discussion communication apprehension "
            f"score is {_fmt_num(row.get('gt_group_ca'))} on a 6–30 scale."
        )
    if _present(row.get("gt_interpersonal_ca")):
        lines.append(
            f"Your self-reported one-on-one communication apprehension score "
            f"is {_fmt_num(row.get('gt_interpersonal_ca'))} on a 6–30 scale."
        )
    return lines


def _fmt_num(value: Any) -> str:
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return str(value)


def build_transit_focus_narrative(row: pd.Series, tier: str) -> str:
    """Narrative profile for transit-focus prompts (no transit / car / voice)."""
    if tier not in TRANSIT_FOCUS_TIERS:
        raise ValueError(f"Unknown transit-focus tier {tier!r}; expected {TRANSIT_FOCUS_TIERS}")
    parts: list[str] = []
    demos = demos_sentences(row)
    if demos:
        parts.extend(demos)
    if tier in {"tf_employment", "tf_geo", "tf_geo_ca"}:
        parts.extend(employment_sentences(row))
    if tier in {"tf_geo", "tf_geo_ca"}:
        parts.extend(geo_sentences(row))
    if tier == "tf_geo_ca":
        parts.extend(_ca_sentences(row))
    return " ".join(parts) if parts else "You are a survey respondent."


def build_transit_focus_user_prompt(row: pd.Series, tier: str) -> str:
    persona = build_transit_focus_narrative(row, tier)
    ask = (
        "From this profile alone — without knowing your own transit, ride-share, "
        "or car-access answers — estimate:\n"
        "(1) whether you are a regular (weekly+) public-transit user, and\n"
        "(2) how many days a month you use public transit (Q26 scale).\n\n"
        "Return ONLY the JSON object specified in the system instructions."
    )
    return f"{persona}\n\n{ask}"


def build_transit_focus_prompt(row: pd.Series, tier: str) -> PersonaPrompt:
    pid = str(row.get("participant_id", "")).strip()
    if not pid:
        raise ValueError("Row is missing participant_id")
    return PersonaPrompt(
        participant_id=pid,
        tier=tier,
        system_prompt=TRANSIT_FOCUS_SYSTEM_PROMPT.strip(),
        user_prompt=build_transit_focus_user_prompt(row, tier),
    )


def build_transit_focus_prompts(
    df: pd.DataFrame,
    tiers: Sequence[str] = TRANSIT_FOCUS_TIERS,
) -> list[PersonaPrompt]:
    prompts: list[PersonaPrompt] = []
    for _, row in df.iterrows():
        if not _present(row.get("participant_id")):
            continue
        for tier in tiers:
            prompts.append(build_transit_focus_prompt(row, tier))
    return prompts


def assert_no_mobility_leak(text: str) -> None:
    """Raise if a held-out mobility stem/answer pattern appears in a prompt."""
    lower = text.lower()
    banned = (
        "public transportation (bus, train",
        "ride share platforms",
        "license to drive",
        "access to a car",
        "ideal way to get around",
    )
    for phrase in banned:
        if phrase in lower:
            raise AssertionError(f"Transit-focus prompt leaks mobility cue: {phrase!r}")


# ---------------------------------------------------------------------------
# Tabular ML
# ---------------------------------------------------------------------------


def _available_features(df: pd.DataFrame, features: Sequence[str]) -> list[str]:
    return [c for c in features if c in df.columns]


def prepare_regular_frame(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    regular_labels: Sequence[str] = tuple(PRIMARY_REGULAR_LABELS),
) -> pd.DataFrame:
    """Complete-case frame for binary regular_transit with held-out mobility."""
    labeled = label_regular_riders(df, regular_labels=regular_labels)
    feats = _available_features(labeled, features)
    if not feats:
        raise ValueError("No requested features present in frame")
    leak = [c for c in HELD_OUT_MOBILITY if c in feats]
    if leak:
        raise ValueError(f"Held-out mobility features requested as predictors: {leak}")
    keep: list[str] = []
    for c in ["participant_id", *feats, "regular_transit", "transit_group", "Q26", *CA_FEATURES]:
        if c in labeled.columns and c not in keep:
            keep.append(c)
    out = labeled[keep].copy()
    for col in feats:
        if col in NUMERIC_FEATURES:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            text = out[col].astype(str).str.strip()
            out.loc[text.eq("") | text.str.lower().isin({"nan", "none", "<na>"}), col] = pd.NA
    out = out.dropna(subset=[*feats, "regular_transit"]).copy()
    out["regular_transit"] = out["regular_transit"].astype(bool)
    out["y"] = out["regular_transit"].astype(int)
    return out.reset_index(drop=True)


def prepare_q26_frame(
    df: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    """Complete-case frame for ordinal Q26 prediction."""
    work = df.copy()
    if "Q26" not in work.columns:
        raise ValueError("Q26 missing from frame")
    work["q26_normalized"] = work["Q26"].map(normalize_q26)
    order_map = {lab: i for i, lab in enumerate(Q26_ORDER)}
    work["y"] = work["q26_normalized"].map(order_map)
    feats = _available_features(work, features)
    leak = [c for c in HELD_OUT_MOBILITY if c in feats]
    if leak:
        raise ValueError(f"Held-out mobility features requested as predictors: {leak}")
    for col in feats:
        if col in NUMERIC_FEATURES:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    out = work.dropna(subset=[*feats, "y"]).copy()
    out["y"] = out["y"].astype(int)
    return out.reset_index(drop=True)


def prepare_q27_frame(
    df: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    """Complete-case frame for Q27 multiclass among respondents with Q27."""
    work = df.copy()
    if "Q27" not in work.columns:
        raise ValueError("Q27 missing from frame")
    order_map = {lab: i for i, lab in enumerate(RIDES_PER_DAY_ORDER)}
    work["q27_normalized"] = work["Q27"].astype(str).str.strip()
    work["y"] = work["q27_normalized"].map(order_map)
    feats = _available_features(work, features)
    leak = [c for c in HELD_OUT_MOBILITY if c in feats]
    if leak:
        raise ValueError(f"Held-out mobility features requested as predictors: {leak}")
    for col in feats:
        if col in NUMERIC_FEATURES:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    out = work.dropna(subset=[*feats, "y"]).copy()
    out["y"] = out["y"].astype(int)
    return out.reset_index(drop=True)


def make_mixed_rf_pipeline(
    features: Sequence[str],
    *,
    n_estimators: int = 500,
    min_samples_leaf: int = 3,
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> Pipeline:
    numeric = [c for c in features if c in NUMERIC_FEATURES]
    categorical = [c for c in features if c not in NUMERIC_FEATURES]
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
    pre = ColumnTransformer(transformers)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("pre", pre), ("rf", clf)])


def _cv_binary(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    n_splits: int,
    random_state: int,
    model_name: str,
) -> dict[str, Any]:
    feats = list(features)
    X = frame[feats]
    y = frame["y"].to_numpy()
    n_splits = min(n_splits, int(np.min(np.bincount(y))))
    if n_splits < 2:
        raise ValueError("Need at least 2 examples per class for stratified CV")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipe = make_mixed_rf_pipeline(feats, random_state=random_state)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    metrics = classification_metrics(y, proba)
    metrics["model"] = model_name
    metrics["n_splits"] = int(n_splits)
    metrics["features"] = feats
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, proba)
    return {
        "metrics": metrics,
        "proba": proba,
        "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}),
        "pipeline_template": pipe,
        "X": X,
        "y": y,
    }


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ordinal_mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _cv_multiclass(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    n_splits: int,
    random_state: int,
    model_name: str,
) -> dict[str, Any]:
    feats = list(features)
    work = frame.copy()
    # Drop singleton classes — StratifiedKFold cannot place them in every fold.
    counts = work["y"].value_counts()
    keep_levels = counts[counts >= 2].index
    dropped = sorted(set(counts.index) - set(keep_levels))
    if dropped:
        warnings.warn(
            f"{model_name}: dropping singleton ordinal classes {dropped} before CV",
            stacklevel=2,
        )
        work = work[work["y"].isin(keep_levels)].copy()
    X = work[feats]
    y = work["y"].to_numpy()
    counts_arr = np.bincount(y)
    positive_counts = counts_arr[counts_arr > 0]
    n_splits = min(n_splits, int(positive_counts.min()) if len(positive_counts) else 0)
    if n_splits < 2 or len(work) < 20:
        raise ValueError(
            f"Need ≥2 examples per observed class and n≥20 for stratified CV "
            f"(got n={len(work)}, min_class={int(positive_counts.min()) if len(positive_counts) else 0})"
        )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipe = make_mixed_rf_pipeline(feats, random_state=random_state, class_weight="balanced")
    pred = cross_val_predict(pipe, X, y, cv=cv, method="predict", n_jobs=-1)
    metrics = _multiclass_metrics(y, pred)
    metrics["model"] = model_name
    metrics["n_splits"] = int(n_splits)
    metrics["features"] = feats
    metrics["n_classes_observed"] = int(len(np.unique(y)))
    metrics["dropped_singleton_classes"] = [int(x) for x in dropped]
    # Chance = majority class accuracy
    mode = int(np.bincount(y).argmax())
    metrics["majority_accuracy"] = float(np.mean(y == mode))
    return {
        "metrics": metrics,
        "pred": pred,
        "pipeline_template": pipe,
        "frame_used": work,
        "X": X,
        "y": y,
    }


def run_focus_spec_analysis(
    df: pd.DataFrame,
    *,
    spec_key: str,
    n_splits: int = 5,
    n_perm_repeats: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    if spec_key not in FOCUS_SPECS:
        raise ValueError(f"Unknown spec {spec_key!r}; expected one of {sorted(FOCUS_SPECS)}")
    spec = FOCUS_SPECS[spec_key]
    features = _available_features(df, spec["features"])
    task = spec["task"]

    if task == "regular":
        frame = prepare_regular_frame(df, features)
        if len(frame) < 20:
            raise ValueError(f"Analytic frame too small for {spec_key} (n={len(frame)})")
        features = _available_features(frame, features)
        cv_out = _cv_binary(
            frame,
            features,
            n_splits=n_splits,
            random_state=random_state,
            model_name=spec_key,
        )
        pipe = clone(cv_out["pipeline_template"])
        pipe.fit(frame[features], frame["y"])
        nulls = pd.DataFrame(
            [
                {
                    "model": "prevalence_prob",
                    **classification_metrics(frame["y"].to_numpy(), majority_baseline_probs(frame["y"].to_numpy())),
                }
            ]
        )
        try:
            perm_res = permutation_importance(
                pipe,
                frame[features],
                frame["y"],
                n_repeats=n_perm_repeats,
                random_state=random_state,
                scoring="roc_auc",
                n_jobs=-1,
            )
            perm = pd.DataFrame(
                {
                    "feature": features,
                    "importance_perm_mean": perm_res.importances_mean,
                    "importance_perm_std": perm_res.importances_std,
                }
            ).sort_values("importance_perm_mean", ascending=False)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Permutation importance failed for {spec_key}: {exc}", stacklevel=2)
            perm = pd.DataFrame({"feature": features, "importance_perm_mean": np.nan})

        oof = frame[["participant_id", "y", "regular_transit"]].copy()
        oof["y_prob"] = cv_out["proba"]
        oof["y_pred"] = (cv_out["proba"] >= 0.5).astype(int)
        summary = {
            "focus_id": "TF1" if "regular" in spec_key else "TF",
            "spec_key": spec_key,
            "research_question": spec["research_question"],
            "label": spec["label"],
            "task": task,
            "held_out": HELD_OUT_MOBILITY,
            "features": features,
            "sample": {
                "n": int(len(frame)),
                "n_regular": int(frame["y"].sum()),
                "n_not_regular": int((frame["y"] == 0).sum()),
                "prevalence": float(frame["y"].mean()),
            },
            "cv_metrics": cv_out["metrics"],
            "baselines": {
                "prevalence_roc_auc": float(nulls.iloc[0]["roc_auc"]),
            },
            "verdict": {
                "roc_auc": cv_out["metrics"]["roc_auc"],
                "beats_chance": bool(cv_out["metrics"]["roc_auc"] > 0.5),
            },
        }
        return {
            "frame": frame,
            "metrics": cv_out["metrics"],
            "roc": cv_out["roc"],
            "oof": oof,
            "permutation_importance": perm,
            "null_baselines": nulls,
            "summary": summary,
            "pipeline": pipe,
        }

    # Intensity tasks
    frame = prepare_q26_frame(df, features) if task == "q26" else prepare_q27_frame(df, features)
    if len(frame) < 20:
        raise ValueError(f"Analytic frame too small for {spec_key} (n={len(frame)})")
    features = _available_features(frame, features)
    cv_out = _cv_multiclass(
        frame,
        features,
        n_splits=n_splits,
        random_state=random_state,
        model_name=spec_key,
    )
    used = cv_out["frame_used"]
    pipe = clone(cv_out["pipeline_template"])
    pipe.fit(used[features], used["y"])
    oof = used[["participant_id", "y"]].copy()
    oof["y_pred"] = cv_out["pred"]
    labels = Q26_ORDER if task == "q26" else RIDES_PER_DAY_ORDER
    summary = {
        "focus_id": "TF2",
        "spec_key": spec_key,
        "research_question": spec["research_question"],
        "label": spec["label"],
        "task": task,
        "held_out": HELD_OUT_MOBILITY,
        "features": features,
        "outcome_labels": labels,
        "sample": {
            "n": int(len(used)),
            "n_before_singleton_drop": int(len(frame)),
            "class_counts": {
                labels[i] if i < len(labels) else str(i): int((used["y"] == i).sum())
                for i in sorted(used["y"].unique())
            },
        },
        "cv_metrics": cv_out["metrics"],
        "verdict": {
            "balanced_accuracy": cv_out["metrics"]["balanced_accuracy"],
            "ordinal_mae": cv_out["metrics"]["ordinal_mae"],
            "beats_majority_accuracy": bool(
                cv_out["metrics"]["accuracy"] > cv_out["metrics"]["majority_accuracy"]
            ),
        },
    }
    return {
        "frame": used,
        "metrics": cv_out["metrics"],
        "oof": oof,
        "summary": summary,
        "pipeline": pipe,
    }


def run_transit_focus_bundle(
    df: pd.DataFrame,
    *,
    spec_keys: Sequence[str] | None = None,
    n_splits: int = 5,
    n_perm_repeats: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    keys = list(spec_keys) if spec_keys else list(FOCUS_SPECS)
    analyses: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for key in keys:
        analysis = run_focus_spec_analysis(
            df,
            spec_key=key,
            n_splits=n_splits,
            n_perm_repeats=n_perm_repeats,
            random_state=random_state,
        )
        analyses[key] = analysis
        m = analysis["metrics"]
        row = {
            "spec_key": key,
            "label": FOCUS_SPECS[key]["label"],
            "task": FOCUS_SPECS[key]["task"],
            "n": analysis["summary"]["sample"]["n"],
        }
        if FOCUS_SPECS[key]["task"] == "regular":
            row.update(
                {
                    "roc_auc": m.get("roc_auc"),
                    "average_precision": m.get("average_precision"),
                    "balanced_accuracy": m.get("balanced_accuracy"),
                    "f1": m.get("f1"),
                }
            )
        else:
            row.update(
                {
                    "accuracy": m.get("accuracy"),
                    "balanced_accuracy": m.get("balanced_accuracy"),
                    "macro_f1": m.get("macro_f1"),
                    "ordinal_mae": m.get("ordinal_mae"),
                    "majority_accuracy": m.get("majority_accuracy"),
                }
            )
        rows.append(row)
    comparison = pd.DataFrame(rows)
    card = {
        "secondary_focus": "Transit prediction with mobility items held out (TF1/TF2)",
        "held_out": HELD_OUT_MOBILITY,
        "specs": keys,
        "comparison": comparison.to_dict(orient="records"),
        "seed": random_state,
        "n_splits": n_splits,
    }
    return {"analyses": analyses, "comparison": comparison, "results_card": card}


def save_transit_focus_artifacts(
    bundle: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["comparison"] = out / "transit_focus_comparison.csv"
    bundle["comparison"].to_csv(paths["comparison"], index=False)
    paths["results_card"] = out / "transit_focus_results_card.json"
    paths["results_card"].write_text(
        json.dumps(bundle["results_card"], indent=2), encoding="utf-8"
    )
    for key, analysis in bundle["analyses"].items():
        sub = out / key
        sub.mkdir(parents=True, exist_ok=True)
        paths[f"{key}_summary"] = sub / f"{key}_summary.json"
        paths[f"{key}_summary"].write_text(
            json.dumps(analysis["summary"], indent=2), encoding="utf-8"
        )
        paths[f"{key}_oof"] = sub / f"{key}_oof.csv"
        analysis["oof"].to_csv(paths[f"{key}_oof"], index=False)
        if "roc" in analysis:
            paths[f"{key}_roc"] = sub / f"{key}_roc.csv"
            analysis["roc"].to_csv(paths[f"{key}_roc"], index=False)
        if "permutation_importance" in analysis:
            paths[f"{key}_perm"] = sub / f"{key}_permutation_importance.csv"
            analysis["permutation_importance"].to_csv(paths[f"{key}_perm"], index=False)
        paths[f"{key}_results_card"] = sub / f"{key}_results_card.json"
        paths[f"{key}_results_card"].write_text(
            json.dumps(analysis["summary"], indent=2), encoding="utf-8"
        )
    return paths


def run_transit_focus_pipeline(
    *,
    prolific_paths: Sequence[str | Path] | None = None,
    qualtrics_path: str | Path | None = None,
    join_how: str = "inner",
    output_dir: str | Path = "outputs/transit_focus",
    spec_keys: Sequence[str] | None = None,
    n_splits: int = 5,
    n_perm_repeats: int = 20,
    random_state: int = 42,
    write_prompts: bool = True,
    prompts_dir: str | Path | None = None,
) -> dict[str, Path]:
    participants, _report = load_full_cohort(
        prolific_paths=prolific_paths,
        qualtrics_path=qualtrics_path,
        join_how=join_how,
    )
    bundle = run_transit_focus_bundle(
        participants,
        spec_keys=spec_keys,
        n_splits=n_splits,
        n_perm_repeats=n_perm_repeats,
        random_state=random_state,
    )
    paths = save_transit_focus_artifacts(bundle, output_dir)
    if write_prompts:
        pdir = Path(prompts_dir) if prompts_dir else Path(output_dir) / "persona_prompts"
        pdir.mkdir(parents=True, exist_ok=True)
        prompts = build_transit_focus_prompts(participants)
        frame = pd.DataFrame([p.to_dict() for p in prompts])
        # Sanity: no mobility leak in user prompts
        for text in frame["user_prompt"].head(20):
            assert_no_mobility_leak(text)
        csv_path = pdir / "transit_focus_persona_prompts.csv"
        frame.to_csv(csv_path, index=False)
        paths["persona_prompts"] = csv_path
        # Write a few markdown examples
        for prompt in prompts[:8]:
            md = pdir / f"{prompt.participant_id}__{prompt.tier}.md"
            md.write_text(prompt.to_markdown(), encoding="utf-8")
    return paths
