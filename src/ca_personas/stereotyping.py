"""Stereotyping / discriminatory-error evaluation for persona CA predictions.

Primary RQs ask whether prediction error correlates with demographic (or
mobility) group membership and whether gaps widen as persona tiers accumulate.
This module:

1. Attaches audit covariates (Sex, Age bins, Student, Employment, transit, Q28)
2. Writes per-slice MAE / signed-error tables
3. Summarizes max−min MAE gaps by tier and Δ-gap vs ``demos``
4. Runs within-tier association tests (Kruskal–Wallis / Welch / Spearman)

References for interpretation: Cheng et al. (2023) marked personas; Santurkar
et al. (2023) opinion reflection; Argyle et al. (2023) silicon sampling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ca_personas.evaluate import summarize_errors_by_group
from ca_personas.transit_ca import label_regular_riders

# Canonical audit slices for primary stereotyping RQs.
DEMOGRAPHIC_SLICES: tuple[str, ...] = (
    "Sex",
    "Student status",
    "Employment status",
    "Age_bin",
)
MOBILITY_SLICES: tuple[str, ...] = (
    "regular_transit",
    "Q28",
)
DEFAULT_SLICES: tuple[str, ...] = (*DEMOGRAPHIC_SLICES, *MOBILITY_SLICES)

_BASELINE_TIER = "demos"
_METRICS = ("mae_group", "mae_interpersonal")


def attach_audit_covariates(participants: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Age_bin, regular_transit, and normalized Q28 labels."""
    out = participants.copy()
    if "Age" in out.columns:
        ages = pd.to_numeric(out["Age"], errors="coerce")
        # Tertiles on observed ages; fallback labels if too few unique values.
        try:
            out["Age_bin"] = pd.qcut(ages, q=3, labels=["younger", "mid", "older"], duplicates="drop")
            out["Age_bin"] = out["Age_bin"].astype("string")
        except (ValueError, TypeError):
            out["Age_bin"] = pd.cut(
                ages,
                bins=[-np.inf, 29, 44, np.inf],
                labels=["younger", "mid", "older"],
            ).astype("string")
    if "Q26" in out.columns and "regular_transit" not in out.columns:
        out = label_regular_riders(out)
    if "regular_transit" in out.columns:
        # String keys for groupby CSV stability (True/False/<NA>).
        rt = out["regular_transit"]
        out["regular_transit"] = rt.map(
            {True: "regular", False: "not_regular"},
            na_action="ignore",
        ).astype("string")
    if "Q28" in out.columns:
        out["Q28"] = out["Q28"].map(lambda x: str(x).strip() if pd.notna(x) else pd.NA).astype("string")
    return out


def enrich_evaluation_with_audits(
    evaluation: pd.DataFrame,
    participants: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join missing audit covariates onto an evaluation frame by participant_id."""
    audited = attach_audit_covariates(participants)
    want = [
        c
        for c in (
            "Age_bin",
            "regular_transit",
            "Q28",
            "Q26",
            "Sex",
            "Student status",
            "Employment status",
            "Age",
        )
        if c in audited.columns and c not in evaluation.columns
    ]
    if not want:
        # Still refresh derived audit cols if participants have newer labels.
        refresh = [c for c in ("Age_bin", "regular_transit") if c in audited.columns]
        if not refresh:
            return evaluation.copy()
        extra = audited[["participant_id", *refresh]].drop_duplicates("participant_id")
        out = evaluation.drop(columns=[c for c in refresh if c in evaluation.columns], errors="ignore")
        return out.merge(extra, on="participant_id", how="left")

    extra = audited[["participant_id", *want]].drop_duplicates("participant_id")
    return evaluation.merge(extra, on="participant_id", how="left")


def group_mae_gaps(
    by_group: pd.DataFrame,
    *,
    metrics: Sequence[str] = _METRICS,
) -> pd.DataFrame:
    """Per-tier max−min MAE across group keys (stereotyping spread)."""
    if by_group.empty or "tier" not in by_group.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_col = str(by_group["group_col"].iloc[0]) if "group_col" in by_group.columns else "group"
    for tier, frame in by_group.groupby("tier", dropna=False):
        row: dict[str, Any] = {
            "group_col": group_col,
            "tier": str(tier),
            "n_groups": int(frame["group_key"].nunique()),
        }
        for metric in metrics:
            if metric not in frame.columns:
                row[f"{metric}_min"] = None
                row[f"{metric}_max"] = None
                row[f"{metric}_gap"] = None
                continue
            usable = frame.dropna(subset=[metric])
            if usable.empty:
                row[f"{metric}_min"] = None
                row[f"{metric}_max"] = None
                row[f"{metric}_gap"] = None
                continue
            lo = float(usable[metric].min())
            hi = float(usable[metric].max())
            row[f"{metric}_min"] = lo
            row[f"{metric}_max"] = hi
            row[f"{metric}_gap"] = hi - lo
            # Which groups drive the extremes.
            row[f"{metric}_min_group"] = str(usable.loc[usable[metric].idxmin(), "group_key"])
            row[f"{metric}_max_group"] = str(usable.loc[usable[metric].idxmax(), "group_key"])
        rows.append(row)
    return pd.DataFrame(rows)


def tier_gap_deltas(
    gaps: pd.DataFrame,
    *,
    baseline_tier: str = _BASELINE_TIER,
    metrics: Sequence[str] = _METRICS,
) -> pd.DataFrame:
    """Δ gap vs demos: positive ⇒ stereotyping spread widens with context."""
    if gaps.empty:
        return pd.DataFrame()
    base = gaps.loc[gaps["tier"] == baseline_tier]
    if base.empty:
        return pd.DataFrame()
    base_row = base.iloc[0]
    rows: list[dict[str, Any]] = []
    for _, row in gaps.iterrows():
        out: dict[str, Any] = {
            "group_col": row.get("group_col"),
            "tier": row["tier"],
            "baseline_tier": baseline_tier,
        }
        for metric in metrics:
            gap_key = f"{metric}_gap"
            if gap_key not in gaps.columns:
                continue
            cur = row.get(gap_key)
            ref = base_row.get(gap_key)
            if cur is None or ref is None or (isinstance(cur, float) and np.isnan(cur)):
                out[f"delta_{gap_key}"] = None
            else:
                out[f"delta_{gap_key}"] = float(cur) - float(ref)
        rows.append(out)
    return pd.DataFrame(rows)


def _kruskal_p(groups: list[np.ndarray]) -> float | None:
    usable = [g for g in groups if len(g) >= 2]
    if len(usable) < 2:
        return None
    try:
        return float(stats.kruskal(*usable).pvalue)
    except ValueError:
        return None


def association_tests_by_tier(
    evaluation: pd.DataFrame,
    group_col: str,
    *,
    error_cols: Sequence[str] = ("abs_error_group", "abs_error_interpersonal", "error_group", "error_interpersonal"),
) -> pd.DataFrame:
    """Within-tier association of error with group membership.

    Categorical groups → Kruskal–Wallis on absolute / signed error.
    ``Age`` (if continuous column still present) also gets Spearman vs signed error.
    """
    if group_col not in evaluation.columns or "tier" not in evaluation.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for tier, frame in evaluation.groupby("tier", dropna=False):
        for err_col in error_cols:
            if err_col not in frame.columns:
                continue
            work = frame.dropna(subset=[err_col, group_col])
            if work.empty:
                continue
            groups = [
                work.loc[work[group_col] == key, err_col].astype(float).to_numpy()
                for key in work[group_col].dropna().unique()
            ]
            p = _kruskal_p(groups)
            # Epsilon-squared effect size for Kruskal–Wallis.
            eps2 = None
            if p is not None and len(work) > 1:
                try:
                    h = float(stats.kruskal(*[g for g in groups if len(g) >= 1]).statistic)
                    n = float(len(work))
                    eps2 = float((h - len(groups) + 1) / (n - len(groups))) if n > len(groups) else None
                except ValueError:
                    eps2 = None
            rows.append(
                {
                    "group_col": group_col,
                    "tier": str(tier),
                    "error_col": err_col,
                    "n": int(len(work)),
                    "n_groups": int(work[group_col].nunique()),
                    "test": "kruskal",
                    "p_value": p,
                    "epsilon_squared": eps2,
                }
            )
    return pd.DataFrame(rows)


def age_spearman_by_tier(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of Age with signed/abs error within each tier."""
    if "Age" not in evaluation.columns or "tier" not in evaluation.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for tier, frame in evaluation.groupby("tier", dropna=False):
        for err_col in ("abs_error_group", "abs_error_interpersonal", "error_group", "error_interpersonal"):
            if err_col not in frame.columns:
                continue
            work = frame.dropna(subset=[err_col, "Age"])
            if len(work) < 5:
                continue
            age = pd.to_numeric(work["Age"], errors="coerce")
            mask = age.notna()
            if mask.sum() < 5:
                continue
            rho, p = stats.spearmanr(age[mask], work.loc[mask, err_col].astype(float))
            rows.append(
                {
                    "group_col": "Age",
                    "tier": str(tier),
                    "error_col": err_col,
                    "n": int(mask.sum()),
                    "test": "spearman",
                    "rho": float(rho),
                    "p_value": float(p),
                }
            )
    return pd.DataFrame(rows)


def run_stereotyping_battery(
    evaluation: pd.DataFrame,
    participants: pd.DataFrame | None = None,
    *,
    slices: Iterable[str] = DEFAULT_SLICES,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Full stereotyping battery; optionally write CSVs + results_card.json."""
    eval_frame = evaluation.copy()
    if participants is not None:
        eval_frame = enrich_evaluation_with_audits(eval_frame, participants)

    slice_tables: dict[str, pd.DataFrame] = {}
    gap_tables: list[pd.DataFrame] = []
    delta_tables: list[pd.DataFrame] = []
    assoc_tables: list[pd.DataFrame] = []

    for slice_col in slices:
        if slice_col not in eval_frame.columns:
            continue
        by_g = summarize_errors_by_group(eval_frame, slice_col)
        if by_g.empty:
            continue
        slug = (
            slice_col.lower()
            .replace(" ", "_")
            .replace("/", "_")
        )
        slice_tables[slug] = by_g
        gaps = group_mae_gaps(by_g)
        if not gaps.empty:
            gap_tables.append(gaps)
            deltas = tier_gap_deltas(gaps)
            if not deltas.empty:
                delta_tables.append(deltas)
        assoc = association_tests_by_tier(eval_frame, slice_col)
        if not assoc.empty:
            assoc_tables.append(assoc)

    age_assoc = age_spearman_by_tier(eval_frame)
    gaps_all = pd.concat(gap_tables, ignore_index=True) if gap_tables else pd.DataFrame()
    deltas_all = pd.concat(delta_tables, ignore_index=True) if delta_tables else pd.DataFrame()
    assoc_all = pd.concat(assoc_tables, ignore_index=True) if assoc_tables else pd.DataFrame()
    if not age_assoc.empty:
        assoc_all = pd.concat([assoc_all, age_assoc], ignore_index=True)

    # Headline: which slices show demos-tier gaps and transit widening.
    headlines: list[dict[str, Any]] = []
    if not gaps_all.empty:
        for _, row in gaps_all.iterrows():
            headlines.append(
                {
                    "group_col": row["group_col"],
                    "tier": row["tier"],
                    "mae_group_gap": row.get("mae_group_gap"),
                    "mae_interpersonal_gap": row.get("mae_interpersonal_gap"),
                }
            )

    card = {
        "n_evaluation_rows": int(len(eval_frame)),
        "slices_evaluated": sorted(slice_tables.keys()),
        "baseline_tier": _BASELINE_TIER,
        "interpretation_notes": [
            "mae_*_gap = max−min MAE across group keys within a tier (larger ⇒ more uneven error).",
            "delta_mae_*_gap > 0 vs demos ⇒ stereotyping spread widens when richer context is added.",
            "Kruskal p_value tests whether abs/signed error distributions differ by group within a tier.",
            "Mobility slices (regular_transit, Q28) audit non-demographic stereotype surfaces.",
        ],
        "headlines": headlines[:40],
    }

    artifacts: dict[str, Path] = {}
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for slug, table in slice_tables.items():
            path = out / f"error_by_{slug}.csv"
            table.to_csv(path, index=False)
            artifacts[f"error_by_{slug}"] = path
        if not gaps_all.empty:
            path = out / "mae_gaps_by_tier.csv"
            gaps_all.to_csv(path, index=False)
            artifacts["mae_gaps_by_tier"] = path
        if not deltas_all.empty:
            path = out / "mae_gap_deltas_vs_demos.csv"
            deltas_all.to_csv(path, index=False)
            artifacts["mae_gap_deltas_vs_demos"] = path
        if not assoc_all.empty:
            path = out / "association_tests_by_tier.csv"
            assoc_all.to_csv(path, index=False)
            artifacts["association_tests_by_tier"] = path
        card_path = out / "stereotyping_results_card.json"
        card_path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
        artifacts["results_card"] = card_path
        # Keep enriched evaluation for inspection.
        enrich_path = out / "evaluation_with_audits.csv"
        eval_frame.to_csv(enrich_path, index=False)
        artifacts["evaluation_with_audits"] = enrich_path

    return {
        "evaluation": eval_frame,
        "slice_tables": slice_tables,
        "gaps": gaps_all,
        "gap_deltas": deltas_all,
        "association_tests": assoc_all,
        "card": card,
        "artifacts": artifacts,
    }
