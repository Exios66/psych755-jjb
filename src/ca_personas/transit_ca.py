"""Secondary RQ: regular public-transit use and communication apprehension.

Question
--------
Among Prolific↔Qualtrics matched respondents (complete PRCA ground truth), do
individuals who take public transportation **regularly** have CA scores that
differ statistically from the larger cohort — and by how much?

Primary exposure (Q26 — public transportation days in the last 3 months)
------------------------------------------------------------------------
``regular`` = ``4-8 days a month`` **or** ``8 or more days a month``
(roughly weekly-or-more ridership).

Alternate cutoffs are evaluated for sensitivity (see ``CUTOFF_DEFINITIONS``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ca_personas.load import load_full_cohort

CA_TARGETS = ("gt_group_ca", "gt_interpersonal_ca")
BAND_TARGETS = ("gt_group_band", "gt_interpersonal_band")

# Official Qualtrics stem (from the 3-row export question-label row).
Q26_STEM = (
    "In the last three months on how many days did you use public transportation "
    "(bus, train, tram, etc.)?"
)

# Canonical Q26 choice labels — identical in the testing excerpt, repo excerpt,
# and full File C sibling export. Note: the stem asks about days in the last
# three months, but the closed choices are worded as "days a month".
Q26_ORDER = [
    "Never",
    "0-1 days a month",
    "2-4 days a month",
    "4-8 days a month",
    "8 or more days a month",
]

# Parallel day-frequency scale used by ride-share item Q28.
Q28_ORDER = list(Q26_ORDER)

# Rides-per-typical-day scale for Q27 (public transit) and Q29 (ride share).
RIDES_PER_DAY_ORDER = [
    "1-2 rides in a typical day",
    "3-4 rides in a typical day",
    "5-6 rides in a typical day",
    "7 or more rides in a typical day",
]

Q20_CHOICES = ["Yes", "No"]
Q21_CHOICES = ["Yes", "No", "Not Sure"]  # "Not Sure" appears in full File C

# Primary definition: weekly-or-more public transit (per Q26 choice wording).
PRIMARY_REGULAR_LABELS = frozenset({"4-8 days a month", "8 or more days a month"})

CUTOFF_DEFINITIONS: dict[str, frozenset[str]] = {
    "weekly_plus": PRIMARY_REGULAR_LABELS,
    "twice_monthly_plus": frozenset(
        {"2-4 days a month", "4-8 days a month", "8 or more days a month"}
    ),
    "any_use": frozenset(
        {
            "0-1 days a month",
            "2-4 days a month",
            "4-8 days a month",
            "8 or more days a month",
        }
    ),
    "heavy_only": frozenset({"8 or more days a month"}),
}


def normalize_q26(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na"}:
        return None
    return text


def label_regular_riders(
    df: pd.DataFrame,
    *,
    regular_labels: Iterable[str] = PRIMARY_REGULAR_LABELS,
    q26_col: str = "Q26",
    out_col: str = "regular_transit",
) -> pd.DataFrame:
    """
    Attach a boolean ``regular_transit`` flag from Q26.

    Rows with missing Q26 receive ``regular_transit = NA`` and are excluded from
    inferential tests that require the exposure.
    """
    if q26_col not in df.columns:
        raise ValueError(f"Expected column {q26_col!r} for transit exposure")
    out = df.copy()
    labels = {str(x).strip() for x in regular_labels}
    q26 = out[q26_col].map(normalize_q26)
    out["q26_normalized"] = q26
    # Use pandas nullable boolean so missing Q26 stays <NA>, not False.
    flagged = pd.Series(pd.NA, index=out.index, dtype="boolean")
    known = q26.notna()
    flagged.loc[known] = q26.loc[known].isin(labels)
    out[out_col] = flagged
    out["transit_group"] = "unknown"
    out.loc[out[out_col] == True, "transit_group"] = "regular"
    out.loc[out[out_col] == False, "transit_group"] = "not_regular"
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for independent samples (pooled SD)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Small-sample corrected Cohen's d."""
    d = cohens_d(a, b)
    n1, n2 = len(a), len(b)
    if n1 + n2 < 3 or np.isnan(d):
        return float("nan")
    correction = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0))
    return float(d * correction)


def bootstrap_mean_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean(a) - mean(b). Returns (diff, lo, hi)."""
    rng = np.random.default_rng(random_state)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = aa.mean() - bb.mean()
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def describe_groups(df: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    """Descriptive CA summaries for regular / not_regular / overall."""
    rows: list[dict[str, Any]] = []
    usable = df.dropna(subset=[score_col]).copy()

    def _row(name: str, frame: pd.DataFrame) -> dict[str, Any]:
        vals = frame[score_col].astype(float)
        return {
            "score": score_col,
            "group": name,
            "n": int(len(vals)),
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "median": float(vals.median()) if len(vals) else float("nan"),
            "min": float(vals.min()) if len(vals) else float("nan"),
            "max": float(vals.max()) if len(vals) else float("nan"),
            "q25": float(vals.quantile(0.25)) if len(vals) else float("nan"),
            "q75": float(vals.quantile(0.75)) if len(vals) else float("nan"),
        }

    rows.append(_row("overall", usable))
    if "transit_group" in usable.columns:
        for name in ("regular", "not_regular"):
            rows.append(_row(name, usable.loc[usable["transit_group"] == name]))
    return pd.DataFrame(rows)


def compare_regular_vs_rest(
    df: pd.DataFrame,
    *,
    score_col: str,
    n_boot: int = 5000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Inferential comparison for one CA subscale.

    Primary contrast: regular riders vs non-regular riders (Welch t-test).
    Also reports regular mean vs overall cohort mean (nested contrast) with
    the mean difference in PRCA points for results tables.
    """
    work = df.dropna(subset=[score_col, "regular_transit"]).copy()
    regular = work.loc[work["regular_transit"] == True, score_col].astype(float).to_numpy()
    rest = work.loc[work["regular_transit"] == False, score_col].astype(float).to_numpy()
    overall = work[score_col].astype(float).to_numpy()

    result: dict[str, Any] = {
        "score": score_col,
        "n_regular": int(len(regular)),
        "n_not_regular": int(len(rest)),
        "n_overall": int(len(overall)),
        "mean_regular": float(regular.mean()) if len(regular) else float("nan"),
        "mean_not_regular": float(rest.mean()) if len(rest) else float("nan"),
        "mean_overall": float(overall.mean()) if len(overall) else float("nan"),
        "diff_regular_minus_not_regular": float("nan"),
        "diff_regular_minus_overall": float("nan"),
        "pct_diff_vs_overall": float("nan"),
        "welch_t": float("nan"),
        "welch_df": float("nan"),
        "welch_p": float("nan"),
        "mannwhitney_u": float("nan"),
        "mannwhitney_p": float("nan"),
        "cohens_d": float("nan"),
        "hedges_g": float("nan"),
        "boot_diff_vs_not_regular": float("nan"),
        "boot_ci_low": float("nan"),
        "boot_ci_high": float("nan"),
        "significant_at_05": False,
    }

    if len(regular) < 2 or len(rest) < 2:
        return result

    t_res = stats.ttest_ind(regular, rest, equal_var=False, alternative="two-sided")
    u_res = stats.mannwhitneyu(regular, rest, alternative="two-sided")
    diff_rest = float(regular.mean() - rest.mean())
    diff_overall = float(regular.mean() - overall.mean())
    boot_diff, boot_lo, boot_hi = bootstrap_mean_diff_ci(
        regular, rest, n_boot=n_boot, random_state=random_state
    )

    result.update(
        {
            "diff_regular_minus_not_regular": diff_rest,
            "diff_regular_minus_overall": diff_overall,
            "pct_diff_vs_overall": (
                float(100.0 * diff_overall / overall.mean())
                if overall.mean() != 0
                else float("nan")
            ),
            "welch_t": float(t_res.statistic),
            "welch_df": float(t_res.df) if hasattr(t_res, "df") else float("nan"),
            "welch_p": float(t_res.pvalue),
            "mannwhitney_u": float(u_res.statistic),
            "mannwhitney_p": float(u_res.pvalue),
            "cohens_d": cohens_d(regular, rest),
            "hedges_g": hedges_g(regular, rest),
            "boot_diff_vs_not_regular": boot_diff,
            "boot_ci_low": boot_lo,
            "boot_ci_high": boot_hi,
            "significant_at_05": bool(t_res.pvalue < 0.05),
        }
    )
    return result


def band_prevalence(df: pd.DataFrame, *, band_col: str) -> pd.DataFrame:
    """Low/moderate/high band shares for regular vs not_regular vs overall."""
    if band_col not in df.columns:
        return pd.DataFrame()
    rows = []
    work = df.dropna(subset=[band_col]).copy()
    frames = [("overall", work)]
    if "transit_group" in work.columns:
        frames.extend(
            [
                ("regular", work.loc[work["transit_group"] == "regular"]),
                ("not_regular", work.loc[work["transit_group"] == "not_regular"]),
            ]
        )
    for name, frame in frames:
        if frame.empty:
            continue
        counts = frame[band_col].astype(str).str.lower().value_counts()
        total = float(counts.sum())
        row = {"band_col": band_col, "group": name, "n": int(total)}
        for band in ("low", "moderate", "high"):
            n_band = int(counts.get(band, 0))
            row[f"{band}_n"] = n_band
            row[f"{band}_pct"] = n_band / total if total else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def ca_by_q26_level(df: pd.DataFrame) -> pd.DataFrame:
    """Mean CA at each Q26 response level (distribution table)."""
    if "Q26" not in df.columns:
        return pd.DataFrame()
    work = df.dropna(subset=["gt_group_ca", "gt_interpersonal_ca"]).copy()
    work["q26_normalized"] = work["Q26"].map(normalize_q26)
    work = work.dropna(subset=["q26_normalized"])
    out = (
        work.groupby("q26_normalized", dropna=False)
        .agg(
            n=("participant_id", "count"),
            mean_group_ca=("gt_group_ca", "mean"),
            std_group_ca=("gt_group_ca", "std"),
            mean_interpersonal_ca=("gt_interpersonal_ca", "mean"),
            std_interpersonal_ca=("gt_interpersonal_ca", "std"),
            median_group_ca=("gt_group_ca", "median"),
            median_interpersonal_ca=("gt_interpersonal_ca", "median"),
        )
        .reset_index()
        .rename(columns={"q26_normalized": "Q26"})
    )
    out["_order"] = out["Q26"].map({lab: i for i, lab in enumerate(Q26_ORDER)})
    out = out.sort_values(["_order", "Q26"]).drop(columns=["_order"])
    return out.reset_index(drop=True)


def sensitivity_by_cutoff(
    df: pd.DataFrame,
    *,
    cutoffs: dict[str, frozenset[str]] | None = None,
    n_boot: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Re-run regular-vs-rest tests under alternate Q26 cutoffs."""
    cutoffs = cutoffs or CUTOFF_DEFINITIONS
    rows: list[dict[str, Any]] = []
    for name, labels in cutoffs.items():
        labeled = label_regular_riders(df, regular_labels=labels)
        for score in CA_TARGETS:
            if score not in labeled.columns:
                continue
            stats_row = compare_regular_vs_rest(
                labeled,
                score_col=score,
                n_boot=n_boot,
                random_state=random_state,
            )
            stats_row["cutoff"] = name
            stats_row["regular_labels"] = "; ".join(sorted(labels))
            rows.append(stats_row)
    return pd.DataFrame(rows)


def distribution_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Long-form score table for plotting / exporting distributions."""
    work = label_regular_riders(df)
    work = work.dropna(subset=["regular_transit"])
    rows = []
    for score in CA_TARGETS:
        if score not in work.columns:
            continue
        piece = work[["participant_id", "transit_group", "q26_normalized", score]].copy()
        piece = piece.rename(columns={score: "score_value", "q26_normalized": "Q26"})
        piece["score"] = score
        rows.append(piece)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def run_transit_ca_analysis(
    df: pd.DataFrame,
    *,
    regular_labels: Iterable[str] = PRIMARY_REGULAR_LABELS,
    n_boot: int = 5000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    End-to-end secondary-RQ analysis on an already-cleaned analytic sample.

    Returns in-memory tables + a JSON-serializable summary verdict.
    """
    labeled = label_regular_riders(df, regular_labels=regular_labels)
    exposure = labeled.dropna(subset=["regular_transit"])
    n_regular = int((exposure["regular_transit"] == True).sum())
    n_not = int((exposure["regular_transit"] == False).sum())

    comparisons = [
        compare_regular_vs_rest(
            labeled, score_col=score, n_boot=n_boot, random_state=random_state
        )
        for score in CA_TARGETS
        if score in labeled.columns
    ]
    comparisons_df = pd.DataFrame(comparisons)

    descriptives = pd.concat(
        [describe_groups(labeled, score_col=score) for score in CA_TARGETS if score in labeled.columns],
        ignore_index=True,
    )
    bands = pd.concat(
        [band_prevalence(labeled, band_col=band) for band in BAND_TARGETS if band in labeled.columns],
        ignore_index=True,
    )
    by_q26 = ca_by_q26_level(labeled)
    sensitivity = sensitivity_by_cutoff(
        labeled, n_boot=min(n_boot, 2000), random_state=random_state
    )
    dist = distribution_frame(labeled)

    any_sig = bool(comparisons_df["significant_at_05"].fillna(False).any()) if not comparisons_df.empty else False
    summary = {
        "secondary_rq": (
            "Do regular public-transit riders have CA scores that differ "
            "statistically from the larger matched cohort, and by how much?"
        ),
        "sample": {
            "n_analytic_input": int(len(df)),
            "n_with_q26": int(len(exposure)),
            "n_regular": n_regular,
            "n_not_regular": n_not,
            "regular_definition": (
                "Q26 in {4-8 days a month, 8 or more days a month} "
                "(weekly-or-more public transit)"
            ),
            "regular_labels": sorted(regular_labels),
        },
        "primary_tests": comparisons,
        "verdict": {
            "any_subscale_significant_at_05": any_sig,
            "interpretation": (
                "At least one PRCA subscale differs significantly (Welch t, α=.05) "
                "between regular and non-regular riders."
                if any_sig
                else "No statistically significant CA difference (Welch t, α=.05) "
                "between regular and non-regular riders under the primary cutoff."
            ),
        },
    }

    return {
        "labeled": labeled,
        "descriptives": descriptives,
        "comparisons": comparisons_df,
        "bands": bands,
        "by_q26": by_q26,
        "sensitivity": sensitivity,
        "distribution": dist,
        "summary": summary,
    }


def save_transit_ca_artifacts(
    analysis: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write CSV/JSON artifacts suitable for manuscript tables and distribution."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    mapping = {
        "descriptives": "transit_ca_descriptives.csv",
        "comparisons": "transit_ca_comparisons.csv",
        "bands": "transit_ca_band_prevalence.csv",
        "by_q26": "transit_ca_by_q26.csv",
        "sensitivity": "transit_ca_sensitivity.csv",
        "distribution": "transit_ca_score_distribution.csv",
    }
    for key, filename in mapping.items():
        frame = analysis[key]
        path = out / filename
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(path, index=False)
        paths[key] = path

    labeled_path = out / "transit_ca_labeled_sample.csv"
    keep = [
        c
        for c in (
            "participant_id",
            "Age",
            "Sex",
            "Country of residence",
            "Employment status",
            "Student status",
            "Q26",
            "Q27",
            "q26_normalized",
            "regular_transit",
            "transit_group",
            "gt_group_ca",
            "gt_interpersonal_ca",
            "gt_group_band",
            "gt_interpersonal_band",
        )
        if c in analysis["labeled"].columns
    ]
    analysis["labeled"][keep].to_csv(labeled_path, index=False)
    paths["labeled"] = labeled_path

    summary_path = out / "transit_ca_summary.json"
    summary_path.write_text(json.dumps(analysis["summary"], indent=2), encoding="utf-8")
    paths["summary"] = summary_path

    # Compact results card for quick distribution.
    card = {
        "secondary_rq": analysis["summary"]["secondary_rq"],
        "n_regular": analysis["summary"]["sample"]["n_regular"],
        "n_not_regular": analysis["summary"]["sample"]["n_not_regular"],
        "regular_definition": analysis["summary"]["sample"]["regular_definition"],
        "verdict": analysis["summary"]["verdict"],
        "effects": analysis["summary"]["primary_tests"],
    }
    card_path = out / "transit_ca_results_card.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    paths["results_card"] = card_path
    return paths


def run_transit_ca_pipeline(
    *,
    prolific_paths: Sequence[str | Path] | None = None,
    qualtrics_path: str | Path | None = None,
    join_how: str = "inner",
    output_dir: str | Path = "outputs/transit_ca",
    n_boot: int = 5000,
    random_state: int = 42,
) -> dict[str, Path]:
    """Load matched cohort → secondary RQ analysis → write artifacts."""
    participants, _report = load_full_cohort(
        prolific_paths=prolific_paths,
        qualtrics_path=qualtrics_path,
        join_how=join_how,
        allow_excerpt_fallback=False,
    )
    analysis = run_transit_ca_analysis(
        participants,
        n_boot=n_boot,
        random_state=random_state,
    )
    return save_transit_ca_artifacts(analysis, output_dir)
