"""Tests for stereotyping / discriminatory-error evaluation."""

from __future__ import annotations

import pandas as pd

from ca_personas.evaluate import evaluate_predictions
from ca_personas.stereotyping import (
    attach_audit_covariates,
    group_mae_gaps,
    run_stereotyping_battery,
    tier_gap_deltas,
)
from inference.predict_vllm import load_vllm_preset


def _toy_participants() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "participant_id": "p1",
                "Age": 22,
                "Sex": "Female",
                "Student status": "Yes",
                "Employment status": "Part-Time",
                "Q26": "8 or more days a month",
                "Q28": "Never",
                "gt_group_ca": 12,
                "gt_interpersonal_ca": 18,
                "gt_group_band": "low",
                "gt_interpersonal_band": "moderate",
            },
            {
                "participant_id": "p2",
                "Age": 45,
                "Sex": "Male",
                "Student status": "No",
                "Employment status": "Full-Time",
                "Q26": "Never",
                "Q28": "4-8 days a month",
                "gt_group_ca": 22,
                "gt_interpersonal_ca": 10,
                "gt_group_band": "high",
                "gt_interpersonal_band": "low",
            },
            {
                "participant_id": "p3",
                "Age": 33,
                "Sex": "Female",
                "Student status": "No",
                "Employment status": "Other",
                "Q26": "4-8 days a month",
                "Q28": "1-3 days a month",
                "gt_group_ca": 16,
                "gt_interpersonal_ca": 16,
                "gt_group_band": "moderate",
                "gt_interpersonal_band": "moderate",
            },
        ]
    )


def _toy_predictions() -> pd.DataFrame:
    rows = []
    for pid, g, ip in (("p1", 14, 24), ("p2", 20, 10), ("p3", 16, 22)):
        for tier in ("demos", "transit"):
            rows.append(
                {
                    "participant_id": pid,
                    "tier": tier,
                    "pred_group_ca": g + (2 if tier == "transit" else 0),
                    "pred_interpersonal_ca": ip + (4 if tier == "transit" else 0),
                    "pred_group_band": "moderate",
                    "pred_interpersonal_band": "high",
                }
            )
    return pd.DataFrame(rows)


def test_attach_audit_covariates_adds_bins_and_transit():
    audited = attach_audit_covariates(_toy_participants())
    assert "Age_bin" in audited.columns
    assert "regular_transit" in audited.columns
    assert set(audited["regular_transit"].dropna().astype(str)) <= {
        "regular",
        "not_regular",
    }


def test_evaluate_predictions_joins_mobility_cols():
    evaluated = evaluate_predictions(_toy_participants(), _toy_predictions())
    assert "Q28" in evaluated.columns
    assert "Q26" in evaluated.columns
    assert "Sex" in evaluated.columns


def test_group_mae_gaps_and_tier_deltas():
    evaluated = evaluate_predictions(_toy_participants(), _toy_predictions())
    result = run_stereotyping_battery(evaluated, _toy_participants())
    assert "sex" in result["slice_tables"]
    assert "regular_transit" in result["slice_tables"]
    gaps = result["gaps"]
    assert not gaps.empty
    assert "mae_group_gap" in gaps.columns
    deltas = result["gap_deltas"]
    assert not deltas.empty
    assert any(str(t) == "transit" for t in deltas["tier"])


def test_run_stereotyping_battery_writes_artifacts(tmp_path):
    evaluated = evaluate_predictions(_toy_participants(), _toy_predictions())
    result = run_stereotyping_battery(
        evaluated,
        _toy_participants(),
        output_dir=tmp_path,
    )
    assert (tmp_path / "stereotyping_results_card.json").is_file()
    assert (tmp_path / "mae_gaps_by_tier.csv").is_file()
    assert "results_card" in result["artifacts"]


def test_vllm_presets_load():
    v1 = load_vllm_preset("v1_baseline")
    assert float(v1["temperature"]) == 0.0
    assert v1.get("seed") is None
    v2 = load_vllm_preset("v2_enhanced")
    assert float(v2["temperature"]) == 0.3
    assert int(v2["seed"]) == 42
    assert v2["guided_json"] is True
    v3 = load_vllm_preset("v3_enhanced")
    assert int(v3["max_output_tokens"]) == 512
