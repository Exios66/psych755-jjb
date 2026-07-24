"""Tests for CA digital-twin ↔ vLLM CSV bridge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ca_personas.load import load_and_prepare
from ca_personas.personas import build_persona_prompts
from inference.ca_prompts import (
    export_vllm_prompt_bundle,
    make_caseid,
    parse_caseid,
    personas_to_prompt_frame,
    results_to_predictions,
)

PROLIFIC = Path("data/excerpts/prolific_excerpt.csv")
QUALTRICS = Path("data/excerpts/qualtrics_excerpt.csv")


def test_make_and_parse_caseid():
    cid = make_caseid("abc123", "demos")
    assert cid == "abc123__demos"
    assert parse_caseid(cid) == ("abc123", "demos")


def test_export_prompt_schema(tmp_path: Path):
    participants = load_and_prepare(PROLIFIC, QUALTRICS, how="inner")
    paths = export_vllm_prompt_bundle(
        participants,
        tmp_path,
        tiers=["demos", "full"],
    )
    prompts = pd.read_csv(paths["prompts"])
    assert {"caseid", "prompt"}.issubset(prompts.columns)
    assert prompts["caseid"].is_unique
    assert prompts["caseid"].str.contains("__").all()
    assert len(prompts) == len(participants) * 2
    assert "ground_truth" in paths
    truth = pd.read_csv(paths["ground_truth"])
    assert {"caseid", "answer"}.issubset(truth.columns)


def test_personas_to_prompt_frame_includes_user_prompt():
    participants = load_and_prepare(PROLIFIC, QUALTRICS, how="inner")
    prompts = build_persona_prompts(participants, tiers=["demos"])
    frame = personas_to_prompt_frame(prompts, participants)
    assert "Fully personify" in frame.iloc[0]["prompt"]
    assert "answer" in frame.columns


def test_results_to_predictions_parses_json(tmp_path: Path):
    path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "caseid": "p1__demos",
                "generated_text": (
                    '{"self_reported_group_ca": 12, '
                    '"self_reported_interpersonal_ca": 18, '
                    '"self_reported_band_group": "low", '
                    '"self_reported_band_interpersonal": "moderate"}'
                ),
            },
            {
                "caseid": "p2__employment",
                "generated_text": "not json at all",
            },
        ]
    ).to_csv(path, index=False)

    preds = results_to_predictions(path)
    assert len(preds) == 2
    assert preds.iloc[0]["participant_id"] == "p1"
    assert preds.iloc[0]["tier"] == "demos"
    assert preds.iloc[0]["pred_group_ca"] == 12
    assert preds.iloc[0]["pred_interpersonal_ca"] == 18
    assert preds.iloc[0]["error"] is None or pd.isna(preds.iloc[0]["error"])
    assert preds.iloc[1]["error"] is not None
