"""Tests for the secondary transit ↔ CA research question pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ca_personas.paths import sibling_data_available
from ca_personas.transit_ca import (
    PRIMARY_REGULAR_LABELS,
    compare_regular_vs_rest,
    label_regular_riders,
    run_transit_ca_analysis,
    save_transit_ca_artifacts,
)


def _synthetic_cohort(n_regular: int = 40, n_other: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_regular):
        rows.append(
            {
                "participant_id": f"reg{i}",
                "Q26": "8 or more days a month" if i % 2 == 0 else "4-8 days a month",
                # Shifted lower CA for regular riders.
                "gt_group_ca": float(rng.normal(12.0, 3.0)),
                "gt_interpersonal_ca": float(rng.normal(11.5, 3.0)),
                "gt_group_band": "low",
                "gt_interpersonal_band": "low",
            }
        )
    for i in range(n_other):
        label = ["Never", "0-1 days a month", "2-4 days a month"][i % 3]
        rows.append(
            {
                "participant_id": f"oth{i}",
                "Q26": label,
                "gt_group_ca": float(rng.normal(16.0, 3.0)),
                "gt_interpersonal_ca": float(rng.normal(15.5, 3.0)),
                "gt_group_band": "moderate",
                "gt_interpersonal_band": "moderate",
            }
        )
    return pd.DataFrame(rows)


def test_q26_scale_matches_excerpt_and_file_c_choices():
    """Closed choices used in code must match Qualtrics exports."""
    from ca_personas.load import load_qualtrics
    from ca_personas.transit_ca import Q26_ORDER, RIDES_PER_DAY_ORDER

    root = Path(__file__).resolve().parents[1]
    excerpt = load_qualtrics(root / "data/excerpts/qualtrics_excerpt.csv")
    observed_q26 = set(excerpt["Q26"].dropna().astype(str).str.strip()) - {""}
    assert observed_q26 <= set(Q26_ORDER)

    # Rides-per-day choices seen in excerpt are a subset of the full scale.
    observed_q27 = set(excerpt["Q27"].dropna().astype(str).str.strip()) - {""}
    assert observed_q27 <= set(RIDES_PER_DAY_ORDER)


def test_label_regular_riders_primary_cutoff():
    df = pd.DataFrame(
        {
            "participant_id": ["a", "b", "c", "d"],
            "Q26": [
                "8 or more days a month",
                "2-4 days a month",
                "Never",
                None,
            ],
            "gt_group_ca": [10, 12, 14, 16],
            "gt_interpersonal_ca": [11, 13, 15, 17],
        }
    )
    labeled = label_regular_riders(df, regular_labels=PRIMARY_REGULAR_LABELS)
    assert bool(labeled.loc[0, "regular_transit"]) is True
    assert bool(labeled.loc[1, "regular_transit"]) is False
    assert bool(labeled.loc[2, "regular_transit"]) is False
    assert pd.isna(labeled.loc[3, "regular_transit"])


def test_compare_detects_mean_shift():
    df = _synthetic_cohort()
    labeled = label_regular_riders(df)
    result = compare_regular_vs_rest(labeled, score_col="gt_group_ca", n_boot=500)
    assert result["n_regular"] == 40
    assert result["n_not_regular"] == 60
    assert result["diff_regular_minus_not_regular"] < 0
    assert result["welch_p"] < 0.05
    assert result["significant_at_05"] is True
    assert result["boot_ci_high"] < 0  # CI should exclude 0 given large shift


def test_run_and_save_artifacts(tmp_path: Path):
    df = _synthetic_cohort()
    analysis = run_transit_ca_analysis(df, n_boot=300)
    assert "summary" in analysis
    assert analysis["summary"]["sample"]["n_regular"] == 40
    paths = save_transit_ca_artifacts(analysis, tmp_path / "transit_ca")
    assert paths["comparisons"].exists()
    assert paths["distribution"].exists()
    assert paths["results_card"].exists()
    assert paths["summary"].exists()
    card = paths["results_card"].read_text()
    assert "secondary_rq" in card
    assert "verdict" in card


def test_sibling_integration_optional():
    if not sibling_data_available():
        return
    from ca_personas.transit_ca import run_transit_ca_pipeline

    paths = run_transit_ca_pipeline(
        join_how="inner",
        output_dir=Path("outputs/transit_ca_test"),
        n_boot=500,
    )
    assert paths["comparisons"].exists()
    import json

    payload = json.loads(paths["summary"].read_text())
    assert payload["sample"]["n_with_q26"] >= 1
    assert "primary_tests" in payload
