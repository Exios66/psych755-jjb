"""Tests for geo-memo follow-up covariate → regular-transit Random Forests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ca_personas.transit_covariate_rf import (
    FEATURE_SPECS,
    prepare_covariate_frame,
    run_all_followup_analyses,
    run_feature_family_analysis,
    save_followup_bundle,
)


def _synthetic(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    license_ = rng.choice(["Yes", "No"], size=n, p=[0.8, 0.2])
    car = np.where(
        license_ == "Yes",
        rng.choice(["Yes", "No", "Not Sure"], size=n, p=[0.75, 0.2, 0.05]),
        rng.choice(["Yes", "No"], size=n, p=[0.15, 0.85]),
    )
    emp = rng.choice(["Full-Time", "Part-Time", "Other"], size=n, p=[0.55, 0.2, 0.25])
    q28 = rng.choice(
        [
            "Never",
            "0-1 days a month",
            "2-4 days a month",
            "4-8 days a month",
            "8 or more days a month",
        ],
        size=n,
        p=[0.3, 0.2, 0.25, 0.15, 0.1],
    )
    q29 = rng.choice(
        [
            "1-2 rides in a typical day",
            "3-4 rides in a typical day",
            "5-6 rides in a typical day",
        ],
        size=n,
        p=[0.85, 0.1, 0.05],
    )
    # Signal: no car + frequent ride-share → more regular transit.
    logit = (
        -0.4
        + 1.1 * (car == "No")
        + 0.7 * np.isin(q28, ["4-8 days a month", "8 or more days a month"])
        + 0.35 * (emp == "Other")
    )
    p = 1 / (1 + np.exp(-logit))
    y = rng.random(n) < p
    q26 = np.where(y, "8 or more days a month", "Never")
    return pd.DataFrame(
        {
            "participant_id": [f"p{i}" for i in range(n)],
            "Q20": license_,
            "Q21": car,
            "Q28": q28,
            "Q29": q29,
            "Employment status": emp,
            "Q26": q26,
        }
    )


def test_feature_specs_cover_geo_memo_candidates():
    assert set(FEATURE_SPECS) >= {"car_access", "employment", "rideshare", "mobility_bundle"}
    assert FEATURE_SPECS["car_access"]["features"] == ["Q20", "Q21"]
    assert FEATURE_SPECS["rideshare"]["features"] == ["Q28", "Q29"]


def test_prepare_drops_missing_car_items():
    df = _synthetic(n=80, seed=1)
    df.loc[:19, "Q20"] = pd.NA
    frame = prepare_covariate_frame(df, ["Q20", "Q21"])
    assert len(frame) == 60
    assert frame[["Q20", "Q21"]].isna().sum().sum() == 0


def test_car_access_recovers_signal(tmp_path: Path):
    analysis = run_feature_family_analysis(
        _synthetic(n=220, seed=3),
        spec_key="car_access",
        n_splits=4,
        n_perm_repeats=5,
        random_state=3,
    )
    assert analysis["metrics"]["roc_auc"] > 0.58
    assert analysis["summary"]["sample"]["n"] == 220


def test_all_followups_bundle(tmp_path: Path):
    bundle = run_all_followup_analyses(
        _synthetic(n=200, seed=4),
        n_splits=4,
        n_perm_repeats=3,
        random_state=4,
    )
    assert set(bundle["analyses"]) == set(FEATURE_SPECS)
    assert "roc_auc" in bundle["comparison"].columns
    paths = save_followup_bundle(bundle, tmp_path / "cov")
    assert paths["results_card"].exists()
    assert paths["comparison"].exists()
    assert paths["car_access_summary"].exists()
