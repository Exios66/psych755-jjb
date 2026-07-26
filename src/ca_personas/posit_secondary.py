"""Build ``artifacts/posit_full_cohort/secondary_results.json`` from seeded RFs.

The Quarto manuscript always loads this JSON for secondary RQ sections. Sync
scripts must regenerate it alongside participants/evaluation/summary so primary
and secondary numbers cannot drift independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ca_personas.ca_transit_rf import run_ca_transit_rf_analysis
from ca_personas.geo_transit_rf import run_geo_transit_rf_analysis
from ca_personas.transit_ca import label_regular_riders, run_transit_ca_analysis
from ca_personas.transit_covariate_rf import run_all_followup_analyses

DEFAULT_COVARIATE_SPECS = (
    "car_access",
    "employment",
    "rideshare",
    "q27_intensity",
    "q28_days",
    "q27_q28",
    "mobility_bundle",
)

MEMO_LINKS = {
    "q27_q28": "memos/q27_q28_predict_transit.md",
    "rideshare": "memos/rideshare_predicts_transit.md",
    "followups": "memos/transit_covariate_followups.md",
    "feature_power": "memos/feature_predictive_power_ml_llm.md",
    "ml_baselines": "docs/ml_baselines.md",
}

_TRANSIT_CA_KEYS = (
    "score",
    "n_regular",
    "n_not_regular",
    "n_overall",
    "mean_regular",
    "mean_not_regular",
    "mean_overall",
    "diff_regular_minus_not_regular",
    "diff_regular_minus_overall",
    "pct_diff_vs_overall",
    "welch_t",
    "welch_df",
    "welch_p",
    "mannwhitney_u",
    "mannwhitney_p",
    "cohens_d",
    "hedges_g",
    "boot_diff_vs_not_regular",
    "boot_ci_low",
    "boot_ci_high",
    "significant_at_05",
)


def _level_prevalence(frame: pd.DataFrame, col: str) -> list[dict[str, Any]]:
    work = frame.dropna(subset=[col, "regular_transit"]).copy()
    rows: list[dict[str, Any]] = []
    for level, grp in work.groupby(col, dropna=False):
        n = int(len(grp))
        n_reg = int(grp["regular_transit"].astype(bool).sum())
        rows.append(
            {
                "level": str(level),
                "n": n,
                "n_regular": n_reg,
                "prevalence": float(n_reg / n) if n else float("nan"),
            }
        )
    return rows


def _oof_confusion(oof: pd.DataFrame) -> dict[str, int]:
    y = oof["y"].astype(int)
    pred_col = "y_pred_rf" if "y_pred_rf" in oof.columns else "y_pred"
    pred = oof[pred_col].astype(int)
    return {
        "tn": int(((y == 0) & (pred == 0)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
        "tp": int(((y == 1) & (pred == 1)).sum()),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _transit_ca_block(analysis: dict[str, Any]) -> dict[str, Any]:
    """Map transit_ca comparisons into the manuscript JSON shape."""
    comparisons = analysis["comparisons"]
    out: dict[str, Any] = {"memo": "memos/transit_riders_ca.md"}
    for side, score in (
        ("group", "gt_group_ca"),
        ("interpersonal", "gt_interpersonal_ca"),
    ):
        row = comparisons.loc[comparisons["score"] == score].iloc[0]
        block = {k: _json_safe(row[k]) for k in _TRANSIT_CA_KEYS if k in row.index}
        block["score"] = score
        out[side] = block
    return out


def build_secondary_results(
    participants: pd.DataFrame,
    *,
    random_state: int = 42,
    n_boot: int = 800,
    n_perm_repeats: int = 8,
    covariate_specs: tuple[str, ...] = DEFAULT_COVARIATE_SPECS,
) -> dict[str, Any]:
    """Run seeded secondary RQs and return the manuscript JSON payload."""
    labeled = label_regular_riders(participants)
    exposure = labeled.dropna(subset=["regular_transit"])
    n_analytic = int(len(exposure))
    n_regular = int((exposure["regular_transit"] == True).sum())  # noqa: E712
    n_not = int((exposure["regular_transit"] == False).sum())  # noqa: E712
    prevalence = float(n_regular / n_analytic) if n_analytic else float("nan")

    transit_ca = run_transit_ca_analysis(
        participants, n_boot=n_boot, random_state=random_state
    )
    geo = run_geo_transit_rf_analysis(
        participants, n_perm_repeats=n_perm_repeats, random_state=random_state
    )
    ca = run_ca_transit_rf_analysis(
        participants, n_perm_repeats=n_perm_repeats, random_state=random_state
    )

    geo_auc = float(geo["summary"]["cv_metrics"]["roc_auc"])
    ca_auc = float(ca["summary"]["cv_metrics"]["roc_auc"])
    country_auc = float(geo["summary"]["baselines"]["country_only_roc_auc"])

    cov = run_all_followup_analyses(
        participants,
        spec_keys=list(covariate_specs),
        n_perm_repeats=n_perm_repeats,
        random_state=random_state,
        geo_benchmark_auc=geo_auc,
        ca_benchmark_auc=ca_auc,
        benchmark_n=n_analytic,
        benchmark_n_regular=n_regular,
    )

    comparison_records = [
        {k: _json_safe(v) for k, v in row.items()}
        for row in cov["comparison"].to_dict(orient="records")
    ]

    return {
        "n_analytic": n_analytic,
        "n_regular": n_regular,
        "n_not_regular": n_not,
        "prevalence": prevalence,
        "transit_ca": _transit_ca_block(transit_ca),
        "geo_rf": {
            "roc_auc": geo_auc,
            "country_only_roc_auc": country_auc,
            "chance": 0.5,
            "memo": "memos/geo_predicts_transit.md",
        },
        "ca_rf": {
            "roc_auc": ca_auc,
            "group_only": float(ca["summary"]["single_feature_auc"]["group_only"]),
            "interpersonal_only": float(
                ca["summary"]["single_feature_auc"]["interpersonal_only"]
            ),
            "confusion": _oof_confusion(ca["oof"]),
            "memo": "memos/ca_scores_predict_transit.md",
        },
        "covariate_comparison": comparison_records,
        "q27_prevalence": _level_prevalence(labeled, "Q27")
        if "Q27" in labeled.columns
        else [],
        "q28_prevalence": _level_prevalence(labeled, "Q28")
        if "Q28" in labeled.columns
        else [],
        "memos": dict(MEMO_LINKS),
        "meta": {
            "seed": random_state,
            "n_boot": n_boot,
            "n_perm_repeats": n_perm_repeats,
            "covariate_specs": list(covariate_specs),
        },
    }


def write_secondary_results(
    participants: pd.DataFrame,
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Build secondary results and write JSON to ``path``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_secondary_results(participants, **kwargs)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out
