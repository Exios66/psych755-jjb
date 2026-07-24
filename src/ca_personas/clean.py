"""Cleaning rules for Prolific + Qualtrics PRCA exports.

Aligned to the research questions:

1. Employment vs demographics-only prediction of CA
2. Transportation-use contribution to prediction / stereotype patterns
3. Combined employment + transit signal vs redundancy

Cleaning therefore preserves employment, transit (Q26–Q29, Q20, Q21),
core demographics available in the full cohort, PRCA items for ground
truth, and open-text fields used by the ``full`` persona tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ca_personas.scoring import ALL_CA_ITEMS, add_ground_truth_scores

CA_ITEM_COLS = list(ALL_CA_ITEMS)  # includes Q18 (interpersonal Likert)
TRANSIT_COLS = ["Q26", "Q27", "Q28", "Q29", "Q20", "Q21"]


def _with_q18_alias(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the interpersonal Likert column is named Q18 for scoring checks."""
    if "Q18" not in df.columns and "Q18_ca" in df.columns:
        out = df.copy()
        out["Q18"] = out["Q18_ca"]
        return out
    return df
DEMO_COLS_AVAILABLE = [
    "Age",
    "Sex",
    "Country of residence",
    "Student status",
]
EMPLOYMENT_COL = "Employment status"

# Optional richer demographics present in excerpt fixtures / fuller Prolific exports.
OPTIONAL_DEMO_COLS = [
    "Ethnicity simplified",
    "Country of birth",
    "Nationality",
    "Language",
]


@dataclass
class CleaningReport:
    """Audit trail for the cleaning decisions applied to the analytic sample."""

    n_prolific_raw: int = 0
    n_prolific_unique: int = 0
    n_qualtrics_raw: int = 0
    n_qualtrics_with_pid: int = 0
    n_qualtrics_complete_ca: int = 0
    n_joined: int = 0
    n_matched_both: int = 0
    n_analytic: int = 0
    n_dropped_missing_pid: int = 0
    n_dropped_incomplete_ca: int = 0
    n_dropped_unjoined: int = 0
    n_prolific_only: int = 0
    n_qualtrics_only: int = 0
    n_qualtrics_missing_pid: int = 0
    waves: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_prolific_raw": self.n_prolific_raw,
            "n_prolific_unique": self.n_prolific_unique,
            "n_qualtrics_raw": self.n_qualtrics_raw,
            "n_qualtrics_with_pid": self.n_qualtrics_with_pid,
            "n_qualtrics_complete_ca": self.n_qualtrics_complete_ca,
            "n_joined": self.n_joined,
            "n_matched_both": self.n_matched_both,
            "n_analytic": self.n_analytic,
            "n_dropped_missing_pid": self.n_dropped_missing_pid,
            "n_dropped_incomplete_ca": self.n_dropped_incomplete_ca,
            "n_dropped_unjoined": self.n_dropped_unjoined,
            "n_prolific_only": self.n_prolific_only,
            "n_qualtrics_only": self.n_qualtrics_only,
            "n_qualtrics_missing_pid": self.n_qualtrics_missing_pid,
            "waves": self.waves,
            "notes": self.notes,
        }


# Prolific sometimes emits sentinel strings instead of blank cells.
PROLIFIC_SENTINELS = {
    "",
    "nan",
    "none",
    "na",
    "n/a",
    "data_expired",
    "consented_not_submitted",
    "not applicable",
}


def _nonempty(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return series.notna() & ~text.str.lower().isin(PROLIFIC_SENTINELS)


def normalize_prolific_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Prolific sentinel strings with NA so personas skip them."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            mask = out[col].astype(str).str.strip().str.lower().isin(PROLIFIC_SENTINELS)
            out.loc[mask, col] = pd.NA
    return out


def ca_item_completeness(df: pd.DataFrame) -> pd.Series:
    """Count non-empty CA Likert responses among the 12 scored items."""
    work = _with_q18_alias(df)
    cols = [c for c in CA_ITEM_COLS if c in work.columns]
    if not cols:
        return pd.Series(0, index=work.index)
    return work[cols].apply(lambda r: int(_nonempty(r).sum()), axis=1)


def complete_ca_mask(df: pd.DataFrame) -> pd.Series:
    work = _with_q18_alias(df)
    cols = [c for c in CA_ITEM_COLS if c in work.columns]
    if len(cols) < 12:
        # Incomplete schema → nothing is "complete" for scoring.
        return pd.Series(False, index=work.index)
    return ca_item_completeness(work).eq(12)


def flag_research_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach boolean flags used by EDA and tiered modeling.

    These flags map directly onto the research contrasts:
    - has_employment_info → RQ1 (employment tier vs demos)
    - has_transit_info → RQ2 (transit tier)
    - has_employment_and_transit → RQ3 (combined / redundancy)
    """
    out = df.copy()
    out["has_employment_info"] = (
        _nonempty(out[EMPLOYMENT_COL]) if EMPLOYMENT_COL in out.columns else False
    )
    transit_present = [c for c in TRANSIT_COLS if c in out.columns]
    if transit_present:
        # Require at least one of the core frequency items (Q26–Q29).
        core = [c for c in ("Q26", "Q27", "Q28", "Q29") if c in out.columns]
        out["has_transit_info"] = (
            out[core].apply(lambda r: bool(_nonempty(r).any()), axis=1)
            if core
            else False
        )
    else:
        out["has_transit_info"] = False
    out["has_employment_and_transit"] = out["has_employment_info"] & out["has_transit_info"]
    demo_cols = [c for c in DEMO_COLS_AVAILABLE if c in out.columns]
    out["has_core_demos"] = (
        out[demo_cols].apply(lambda r: bool(_nonempty(r).any()), axis=1)
        if demo_cols
        else False
    )
    return out


def clean_joined_participants(
    joined: pd.DataFrame,
    *,
    require_complete_ca: bool = True,
    require_employment: bool = False,
    require_transit: bool = False,
    low_max: int = 13,
    high_min: int = 20,
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Apply analytic-sample filters to an already-joined Prolific↔Qualtrics frame.

    Parameters mirror the research design: by default keep respondents with
    scorable PRCA ground truth; optionally restrict to rows that can support
    the employment / transit information tiers.
    """
    report = CleaningReport()
    report.n_joined = int(len(joined))
    if "prolific_wave" in joined.columns:
        report.waves = (
            joined["prolific_wave"].astype(str).value_counts(dropna=False).to_dict()
        )

    work = normalize_prolific_sentinels(joined)
    if "Q18_ca" in work.columns and "Q18" not in work.columns:
        work = work.rename(columns={"Q18_ca": "Q18"})

    before_pid = len(work)
    work = work.loc[_nonempty(work["participant_id"])].copy()
    report.n_dropped_missing_pid = before_pid - len(work)
    if "Student status" in joined.columns:
        n_expired = int(
            joined["Student status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("data_expired")
            .sum()
        )
        if n_expired:
            report.notes.append(
                f"Normalized {n_expired} DATA_EXPIRED Student status values to missing."
            )

    if require_complete_ca:
        complete = complete_ca_mask(work)
        report.n_dropped_incomplete_ca = int((~complete).sum())
        work = work.loc[complete].copy()

    work = flag_research_covariates(work)

    if require_employment:
        work = work.loc[work["has_employment_info"]].copy()
        report.notes.append("Filtered to rows with Employment status (RQ1).")
    if require_transit:
        work = work.loc[work["has_transit_info"]].copy()
        report.notes.append("Filtered to rows with transit items Q26–Q29 (RQ2).")

    scored = add_ground_truth_scores(work, low_max=low_max, high_min=high_min)
    # Drop rows that still failed scoring (e.g. unmapped Likert labels).
    scored = scored.dropna(subset=["gt_group_ca", "gt_interpersonal_ca"], how="any")
    report.n_analytic = int(len(scored))

    if "Ethnicity simplified" not in scored.columns:
        report.notes.append(
            "Full Prolific waves (File A/B) omit Ethnicity / Nationality / Language; "
            "demos tier uses Age, Sex, Country of residence, and Student status."
        )
    report.notes.append(
        "Analytic sample = Prolific∩Qualtrics with complete PRCA group + interpersonal items."
    )
    return scored.reset_index(drop=True), report
