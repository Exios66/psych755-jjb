"""Aggregate and export participant ground-truth PRCA scores for evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ca_personas.load import load_and_prepare

GT_EXPORT_COLS = [
    "participant_id",
    "Age",
    "Sex",
    "Ethnicity simplified",
    "Country of birth",
    "Country of residence",
    "Nationality",
    "Language",
    "Student status",
    "Employment status",
    "LocationLatitude",
    "LocationLongitude",
    "Q26",
    "Q27",
    "Q28",
    "Q29",
    "Q20",
    "Q21",
    "Q18_advice",
    "Q19",
    "gt_group_ca",
    "gt_interpersonal_ca",
    "gt_group_band",
    "gt_interpersonal_band",
]


def score_participants(
    prolific_path: str | Path,
    qualtrics_path: str | Path,
    *,
    join_how: str = "inner",
    low_max: int = 13,
    high_min: int = 20,
) -> pd.DataFrame:
    """Load Prolific + Qualtrics and return scored participants."""
    return load_and_prepare(
        prolific_path,
        qualtrics_path,
        how=join_how,
        low_max=low_max,
        high_min=high_min,
    )


def ground_truth_table(participants: pd.DataFrame) -> pd.DataFrame:
    """Compact evaluation table: covariates + scored CA targets."""
    cols = [c for c in GT_EXPORT_COLS if c in participants.columns]
    out = participants[cols].copy()
    # Keep rows that have at least one scored subscale.
    scored = out.dropna(subset=["gt_group_ca", "gt_interpersonal_ca"], how="all")
    return scored.reset_index(drop=True)


def aggregate_ground_truth(participants: pd.DataFrame) -> pd.DataFrame:
    """
    Cohort-level aggregates for reporting and agent benchmarking.

    Includes overall means and band prevalence for both subscales, plus
    breakdowns by sex and employment when those columns exist.
    """
    rows: list[dict[str, Any]] = []

    def _append(scope: str, key: str, frame: pd.DataFrame) -> None:
        usable = frame.dropna(subset=["gt_group_ca", "gt_interpersonal_ca"], how="any")
        if usable.empty:
            return
        row: dict[str, Any] = {
            "scope": scope,
            "group_key": key,
            "n": int(len(usable)),
            "mean_group_ca": float(usable["gt_group_ca"].mean()),
            "mean_interpersonal_ca": float(usable["gt_interpersonal_ca"].mean()),
            "std_group_ca": float(usable["gt_group_ca"].std(ddof=0)),
            "std_interpersonal_ca": float(usable["gt_interpersonal_ca"].std(ddof=0)),
            "min_group_ca": float(usable["gt_group_ca"].min()),
            "max_group_ca": float(usable["gt_group_ca"].max()),
            "min_interpersonal_ca": float(usable["gt_interpersonal_ca"].min()),
            "max_interpersonal_ca": float(usable["gt_interpersonal_ca"].max()),
        }
        for side in ("group", "interpersonal"):
            band_col = f"gt_{side}_band"
            if band_col in usable.columns:
                counts = usable[band_col].value_counts(dropna=True)
                total = float(counts.sum()) if len(counts) else 0.0
                for band in ("low", "moderate", "high"):
                    n_band = int(counts.get(band, 0))
                    row[f"{side}_band_{band}_n"] = n_band
                    row[f"{side}_band_{band}_pct"] = (n_band / total) if total else None
        rows.append(row)

    _append("overall", "all", participants)

    if "Sex" in participants.columns:
        for sex, frame in participants.groupby("Sex", dropna=False):
            _append("sex", str(sex), frame)
    if "Employment status" in participants.columns:
        for emp, frame in participants.groupby("Employment status", dropna=False):
            _append("employment", str(emp), frame)
    if "Country of residence" in participants.columns:
        for country, frame in participants.groupby("Country of residence", dropna=False):
            _append("country", str(country), frame)

    return pd.DataFrame(rows)


def export_ground_truth_bundle(
    prolific_path: str | Path,
    qualtrics_path: str | Path,
    output_dir: str | Path,
    *,
    join_how: str = "inner",
    low_max: int = 13,
    high_min: int = 20,
) -> dict[str, Path]:
    """Score participants and write ground-truth evaluation artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    participants = score_participants(
        prolific_path,
        qualtrics_path,
        join_how=join_how,
        low_max=low_max,
        high_min=high_min,
    )
    gt = ground_truth_table(participants)
    aggregates = aggregate_ground_truth(participants)

    paths = {
        "participants_scored": out / "participants_scored.csv",
        "ground_truth": out / "ground_truth.csv",
        "aggregates": out / "ground_truth_aggregates.csv",
    }
    participants.to_csv(paths["participants_scored"], index=False)
    gt.to_csv(paths["ground_truth"], index=False)
    aggregates.to_csv(paths["aggregates"], index=False)
    return paths
