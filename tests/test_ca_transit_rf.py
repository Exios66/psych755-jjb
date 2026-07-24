"""Tests for the CA → regular-transit Random Forest RQ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ca_personas.ca_transit_rf import (
    prepare_ca_transit_frame,
    run_ca_transit_rf_analysis,
    save_ca_transit_rf_artifacts,
)
from ca_personas.paths import sibling_data_available


def _synthetic(n: int = 140, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.uniform(6, 30, size=n)
    interp = 0.6 * group + rng.normal(0, 3, size=n)
    interp = np.clip(interp, 6, 30)
    # Lower CA → higher chance of regular transit.
    logit = 1.2 - 0.12 * group - 0.04 * interp
    p = 1 / (1 + np.exp(-logit))
    y = rng.random(n) < p
    q26 = np.where(y, "4-8 days a month", "Never")
    return pd.DataFrame(
        {
            "participant_id": [f"p{i}" for i in range(n)],
            "gt_group_ca": group,
            "gt_interpersonal_ca": interp,
            "gt_group_band": np.where(group <= 13, "low", np.where(group >= 20, "high", "moderate")),
            "gt_interpersonal_band": np.where(
                interp <= 13, "low", np.where(interp >= 20, "high", "moderate")
            ),
            "Q26": q26,
        }
    )


def test_prepare_frame():
    frame = prepare_ca_transit_frame(_synthetic())
    assert set(["gt_group_ca", "gt_interpersonal_ca", "y"]).issubset(frame.columns)
    assert frame["y"].isin([0, 1]).all()


def test_rf_recovers_ca_signal(tmp_path: Path):
    analysis = run_ca_transit_rf_analysis(
        _synthetic(n=180, seed=2),
        n_splits=4,
        n_perm_repeats=5,
        random_state=2,
        grid_size=20,
    )
    assert analysis["summary"]["cv_metrics"]["roc_auc"] > 0.65
    paths = save_ca_transit_rf_artifacts(analysis, tmp_path / "ca")
    assert paths["results_card"].exists()
    assert paths["associations"].exists()


def test_sibling_integration_optional():
    if not sibling_data_available():
        return
    from ca_personas.ca_transit_rf import run_ca_transit_rf_pipeline
    import json

    paths = run_ca_transit_rf_pipeline(
        join_how="inner",
        output_dir=Path("outputs/ca_transit_rf_test"),
        n_splits=5,
        n_perm_repeats=5,
    )
    summary = json.loads(paths["summary"].read_text())
    assert summary["sample"]["n"] >= 20
    assert "roc_auc" in summary["cv_metrics"]
