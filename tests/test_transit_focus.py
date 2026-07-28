"""Tests for the transit-focus secondary RQ suite (TF1 / TF2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ca_personas.paths import sibling_data_available
from ca_personas.transit_focus import (
    FOCUS_SPECS,
    HELD_OUT_MOBILITY,
    assert_no_mobility_leak,
    build_transit_focus_prompt,
    build_transit_focus_prompts,
    prepare_regular_frame,
    run_focus_spec_analysis,
    run_transit_focus_bundle,
    save_transit_focus_artifacts,
)


def _synthetic(n: int = 160, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 65, size=n).astype(float)
    lat = rng.normal(40, 5, size=n)
    lon = rng.normal(-90, 10, size=n)
    group = rng.uniform(6, 30, size=n)
    interp = np.clip(0.5 * group + rng.normal(0, 4, size=n), 6, 30)
    # Stronger synthetic signal: younger + higher lat + lower CA → regular
    logit = 2.5 - 0.08 * age + 0.25 * (lat - 40) - 0.12 * group
    p = 1 / (1 + np.exp(-logit))
    y = rng.random(n) < p
    q26 = np.where(
        y,
        rng.choice(["4-8 days a month", "8 or more days a month"], size=n),
        rng.choice(["Never", "0-1 days a month", "2-4 days a month"], size=n),
    )
    q27 = rng.choice(
        [
            "1-2 rides in a typical day",
            "3-4 rides in a typical day",
            "5-6 rides in a typical day",
            "7 or more rides in a typical day",
        ],
        size=n,
    )
    return pd.DataFrame(
        {
            "participant_id": [f"p{i}" for i in range(n)],
            "Age": age,
            "Sex": rng.choice(["Male", "Female"], size=n),
            "Country of residence": rng.choice(["United States", "United Kingdom"], size=n),
            "Student status": rng.choice(["Yes", "No"], size=n),
            "Employment status": rng.choice(["Full-Time", "Part-Time", "Other"], size=n),
            "LocationLatitude": lat,
            "LocationLongitude": lon,
            "gt_group_ca": group,
            "gt_interpersonal_ca": interp,
            "Q26": q26,
            "Q27": q27,
            "Q28": "Never",
            "Q20": "Yes",
            "Q21": "Yes",
            "Q19": "I prefer driving everywhere",
        }
    )


def test_held_out_mobility_rejected():
    df = _synthetic()
    try:
        prepare_regular_frame(df, ["Age", "Q26"])
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_prompts_omit_mobility():
    row = _synthetic(n=5).iloc[0]
    for tier in ("tf_demos", "tf_employment", "tf_geo", "tf_geo_ca"):
        prompt = build_transit_focus_prompt(row, tier)
        assert_no_mobility_leak(prompt.user_prompt)
        assert_no_mobility_leak(prompt.system_prompt)
        assert "regular_transit" in prompt.system_prompt
        if tier == "tf_geo_ca":
            assert "communication apprehension" in prompt.user_prompt.lower()


def test_tf1_recovers_signal(tmp_path: Path):
    analysis = run_focus_spec_analysis(
        _synthetic(n=200, seed=3),
        spec_key="tf1_profile_regular",
        n_splits=4,
        n_perm_repeats=3,
        random_state=3,
    )
    assert analysis["summary"]["cv_metrics"]["roc_auc"] > 0.60
    assert set(HELD_OUT_MOBILITY).isdisjoint(analysis["summary"]["features"])


def test_bundle_and_artifacts(tmp_path: Path):
    bundle = run_transit_focus_bundle(
        _synthetic(n=180, seed=4),
        spec_keys=["tf1_profile_regular", "tf2_intensity_q26"],
        n_splits=3,
        n_perm_repeats=2,
        random_state=4,
    )
    assert len(bundle["comparison"]) == 2
    paths = save_transit_focus_artifacts(bundle, tmp_path / "tf")
    assert paths["results_card"].exists()
    assert paths["tf1_profile_regular_oof"].exists()


def test_build_prompts_count():
    prompts = build_transit_focus_prompts(_synthetic(n=3), tiers=("tf_geo", "tf_geo_ca"))
    assert len(prompts) == 6


def test_sibling_integration_optional(tmp_path: Path):
    if not sibling_data_available():
        return
    from ca_personas.transit_focus import run_transit_focus_pipeline
    import json

    paths = run_transit_focus_pipeline(
        join_how="inner",
        output_dir=tmp_path / "transit_focus_sib",
        spec_keys=["tf1_profile_regular", "tf1_profile_ca_regular"],
        n_splits=5,
        n_perm_repeats=5,
        write_prompts=True,
    )
    card = json.loads(paths["results_card"].read_text())
    assert len(card["comparison"]) == 2
    assert paths["persona_prompts"].exists()
    assert set(FOCUS_SPECS)  # module imported
