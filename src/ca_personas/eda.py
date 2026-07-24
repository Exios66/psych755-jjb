"""Exploratory data analysis aligned to the PRCA persona research questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ca_personas.clean import DEMO_COLS_AVAILABLE, EMPLOYMENT_COL, TRANSIT_COLS


def _safe_value_counts(series: pd.Series, *, top: int | None = None) -> dict[str, int]:
    counts = series.fillna("(missing)").astype(str).str.strip().value_counts()
    if top is not None:
        counts = counts.head(top)
    return {str(k): int(v) for k, v in counts.items()}


def describe_ca_scores(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("gt_group_ca", "gt_interpersonal_ca") if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return df[cols].describe().T.reset_index(names="measure")


def ca_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Mean CA scores by a demographic / behavioral grouping (RQ bias lens)."""
    if group_col not in df.columns:
        return pd.DataFrame()
    usable = df.dropna(subset=["gt_group_ca", "gt_interpersonal_ca"], how="any")
    if usable.empty:
        return pd.DataFrame()
    out = (
        usable.groupby(usable[group_col].fillna("(missing)").astype(str), dropna=False)
        .agg(
            n=("participant_id", "count"),
            mean_group_ca=("gt_group_ca", "mean"),
            mean_interpersonal_ca=("gt_interpersonal_ca", "mean"),
            pct_high_group=("gt_group_band", lambda s: float((s == "high").mean())),
            pct_high_interpersonal=(
                "gt_interpersonal_band",
                lambda s: float((s == "high").mean()),
            ),
        )
        .reset_index()
        .rename(columns={group_col: "group"})
    )
    out.insert(0, "group_col", group_col)
    return out


def employment_transit_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """
    Employment × transit contingency — directly informs RQ3 (redundant signal).

    Uses Q26 (public-transit days) when available as the transit summary.
    """
    if EMPLOYMENT_COL not in df.columns or "Q26" not in df.columns:
        return pd.DataFrame()
    work = df[[EMPLOYMENT_COL, "Q26"]].copy()
    work[EMPLOYMENT_COL] = work[EMPLOYMENT_COL].fillna("(missing)").astype(str)
    work["Q26"] = work["Q26"].fillna("(missing)").astype(str)
    table = pd.crosstab(work[EMPLOYMENT_COL], work["Q26"], margins=True)
    return table.reset_index()


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in (
            *DEMO_COLS_AVAILABLE,
            EMPLOYMENT_COL,
            *TRANSIT_COLS,
            "LocationLatitude",
            "LocationLongitude",
            "Q18_advice",
            "Q19",
            "gt_group_ca",
            "gt_interpersonal_ca",
        )
        if c in df.columns
    ]
    rows = []
    n = len(df)
    for col in cols:
        series = df[col]
        if series.dtype == object:
            missing = series.isna() | series.astype(str).str.strip().isin(
                {"", "nan", "None", "NA"}
            )
        else:
            missing = series.isna()
        rows.append(
            {
                "column": col,
                "n_missing": int(missing.sum()),
                "pct_missing": float(missing.mean()) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def research_alignment_summary(df: pd.DataFrame, cleaning_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Compact JSON summary tying the cleaned sample to each research question.
    """
    n = len(df)
    summary: dict[str, Any] = {
        "n_analytic": n,
        "research_questions": {
            "rq_main": (
                "LLM persona recovery of PRCA scores; does error track demographics "
                "(stereotyping) rather than random noise?"
            ),
            "rq1_employment": (
                "Does employment status improve prediction over demographics alone?"
            ),
            "rq2_transit": (
                "Does transportation-use data improve prediction, and is it used sensibly?"
            ),
            "rq3_combined": (
                "Does combining employment + transit help beyond either alone, "
                "or are the cues redundant?"
            ),
        },
        "sample_support": {
            "has_core_demos": int(df["has_core_demos"].sum())
            if "has_core_demos" in df.columns
            else None,
            "has_employment_info": int(df["has_employment_info"].sum())
            if "has_employment_info" in df.columns
            else None,
            "has_transit_info": int(df["has_transit_info"].sum())
            if "has_transit_info" in df.columns
            else None,
            "has_employment_and_transit": int(df["has_employment_and_transit"].sum())
            if "has_employment_and_transit" in df.columns
            else None,
        },
        "distributions": {
            "sex": _safe_value_counts(df["Sex"]) if "Sex" in df.columns else {},
            "employment": _safe_value_counts(df[EMPLOYMENT_COL])
            if EMPLOYMENT_COL in df.columns
            else {},
            "student_status": _safe_value_counts(df["Student status"])
            if "Student status" in df.columns
            else {},
            "country": _safe_value_counts(df["Country of residence"], top=10)
            if "Country of residence" in df.columns
            else {},
            "transit_q26": _safe_value_counts(df["Q26"]) if "Q26" in df.columns else {},
            "group_band": _safe_value_counts(df["gt_group_band"])
            if "gt_group_band" in df.columns
            else {},
            "interpersonal_band": _safe_value_counts(df["gt_interpersonal_band"])
            if "gt_interpersonal_band" in df.columns
            else {},
        },
        "available_demo_fields": [c for c in DEMO_COLS_AVAILABLE if c in df.columns],
        "missing_rich_demos": [
            c
            for c in (
                "Ethnicity simplified",
                "Country of birth",
                "Nationality",
                "Language",
            )
            if c not in df.columns
        ],
    }
    if cleaning_report:
        summary["cleaning_report"] = cleaning_report
    return summary


def run_eda(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    cleaning_report: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write EDA tables + research-alignment JSON under ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}

    ca_desc = describe_ca_scores(df)
    path = out / "ca_score_summary.csv"
    ca_desc.to_csv(path, index=False)
    artifacts["ca_score_summary"] = path

    group_frames = []
    for col in (EMPLOYMENT_COL, "Sex", "Student status", "Country of residence", "Q26"):
        frame = ca_by_group(df, col)
        if not frame.empty:
            group_frames.append(frame)
    by_group = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
    path = out / "ca_by_group.csv"
    by_group.to_csv(path, index=False)
    artifacts["ca_by_group"] = path

    xtab = employment_transit_crosstab(df)
    path = out / "employment_by_transit_q26.csv"
    xtab.to_csv(path, index=False)
    artifacts["employment_by_transit"] = path

    miss = missingness_table(df)
    path = out / "missingness.csv"
    miss.to_csv(path, index=False)
    artifacts["missingness"] = path

    summary = research_alignment_summary(df, cleaning_report=cleaning_report)
    path = out / "research_alignment.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    artifacts["research_alignment"] = path

    return artifacts
