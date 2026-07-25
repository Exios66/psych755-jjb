"""Tests for the comprehensive feature-importance transit Random Forest RQ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ca_personas.comprehensive_transit_rf import (
    prepare_comprehensive_transit_frame,
    primary_features,
    run_comprehensive_transit_rf_analysis,
    save_comprehensive_transit_rf_artifacts,
)
from ca_personas.paths import sibling_data_available


def _synthetic(n: int = 220, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, size=n).astype(float)
    lat = rng.normal(40, 8, size=n)
    lon = rng.normal(-70, 15, size=n)
    group = rng.uniform(6, 30, size=n)
    interp = np.clip(0.5 * group + rng.normal(0, 4, size=n), 6, 30)
    # Strong signal: no car + younger + lower group CA → regular transit.
    has_car = rng.random(n) < 0.55
    logit = (
        1.4
        - 1.8 * has_car.astype(float)
        - 0.03 * age
        - 0.08 * group
        + 0.01 * lon
    )
    p = 1 / (1 + np.exp(-logit))
    y = rng.random(n) < p
    q26 = np.where(y, "8 or more days a month", "Never")
    q28 = np.where(y, "4-8 days a month", "Never")
    countries = rng.choice(["United States", "United Kingdom", "Canada"], size=n)
    return pd.DataFrame(
        {
            "participant_id": [f"p{i}" for i in range(n)],
            "Age": age,
            "Sex": rng.choice(["Male", "Female"], size=n),
            "Country of residence": countries,
            "Student status": rng.choice(["Yes", "No"], size=n),
            "Employment status": rng.choice(["Full-Time", "Part-Time", "Other"], size=n),
            "LocationLatitude": lat,
            "LocationLongitude": lon,
            "Q20": np.where(has_car, "Yes", "No"),
            "Q21": np.where(has_car, "Yes", "No"),
            "Q27": rng.choice(
                ["1-2 rides in a typical day", "3-4 rides in a typical day"], size=n
            ),
            "Q28": q28,
            "Q29": "1-2 rides in a typical day",
            "gt_group_ca": group,
            "gt_interpersonal_ca": interp,
            "Q26": q26,
        }
    )


def test_primary_features_exclude_q26_q27():
    feats = primary_features()
    assert "Q26" not in feats
    assert "Q27" not in feats
    assert "Q20" in feats
    assert "gt_group_ca" in feats


def test_prepare_frame():
    frame = prepare_comprehensive_transit_frame(_synthetic())
    assert "y" in frame.columns
    assert frame["y"].isin([0, 1]).all()
    assert frame[["LocationLatitude", "gt_group_ca"]].notna().all().all()


def test_rf_recovers_signal_and_writes_artifacts(tmp_path: Path):
    analysis = run_comprehensive_transit_rf_analysis(
        _synthetic(n=240, seed=3),
        n_splits=4,
        n_perm_repeats=5,
        n_tune_iter=6,
        random_state=3,
    )
    auc = analysis["summary"]["verdict"]["tuned_roc_auc"]
    assert auc > 0.70
    assert analysis["permutation_importance"]["feature"].notna().all()
    paths = save_comprehensive_transit_rf_artifacts(analysis, tmp_path / "comp")
    assert paths["results_card"].exists()
    assert paths["ablations"].exists()
    assert paths["univariate"].exists()


def test_sibling_integration_optional():
    if not sibling_data_available():
        return
    import json

    from ca_personas.comprehensive_transit_rf import run_comprehensive_transit_rf_pipeline

    paths = run_comprehensive_transit_rf_pipeline(
        join_how="inner",
        output_dir=Path("outputs/comprehensive_transit_rf_test"),
        n_splits=5,
        n_perm_repeats=5,
        n_tune_iter=8,
    )
    summary = json.loads(paths["summary"].read_text())
    assert summary["sample"]["n"] >= 20
    assert summary["verdict"]["tuned_roc_auc"] >= 0.5
