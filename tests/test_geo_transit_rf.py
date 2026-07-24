"""Tests for the geography → regular-transit Random Forest RQ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ca_personas.geo_transit_rf import (
    prepare_geo_transit_frame,
    run_geo_transit_rf_analysis,
    save_geo_transit_rf_artifacts,
)
from ca_personas.paths import sibling_data_available


def _synthetic_geo(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Longitude drives transit: western longitudes → more regular riders.
    lon = rng.uniform(-120, 10, size=n)
    lat = rng.uniform(25, 55, size=n)
    p = 1 / (1 + np.exp(-( -0.04 * (lon + 40))))  # higher p for more negative lon
    y = rng.random(n) < p
    q26 = np.where(y, "8 or more days a month", "Never")
    return pd.DataFrame(
        {
            "participant_id": [f"p{i}" for i in range(n)],
            "LocationLatitude": lat,
            "LocationLongitude": lon,
            "Q26": q26,
            "Country of residence": np.where(lon < -30, "United States", "United Kingdom"),
        }
    )


def test_prepare_geo_transit_frame():
    df = _synthetic_geo()
    frame = prepare_geo_transit_frame(df)
    assert set(["LocationLatitude", "LocationLongitude", "y"]).issubset(frame.columns)
    assert frame["y"].isin([0, 1]).all()
    assert len(frame) == len(df)


def test_rf_recovers_spatial_signal(tmp_path: Path):
    df = _synthetic_geo(n=160, seed=1)
    analysis = run_geo_transit_rf_analysis(
        df, n_splits=4, n_perm_repeats=5, random_state=1, grid_size=20
    )
    auc = analysis["summary"]["cv_metrics"]["roc_auc"]
    assert auc > 0.65  # synthetic lon→transit signal should be learnable
    paths = save_geo_transit_rf_artifacts(analysis, tmp_path / "geo")
    assert paths["metrics_table"].exists()
    assert paths["results_card"].exists()
    assert paths["permutation_importance"].exists()


def test_sibling_integration_optional():
    if not sibling_data_available():
        return
    from ca_personas.geo_transit_rf import run_geo_transit_rf_pipeline

    paths = run_geo_transit_rf_pipeline(
        join_how="inner",
        output_dir=Path("outputs/geo_transit_rf_test"),
        n_splits=5,
        n_perm_repeats=5,
    )
    import json

    summary = json.loads(paths["summary"].read_text())
    assert summary["sample"]["n"] >= 20
    assert "roc_auc" in summary["cv_metrics"]
