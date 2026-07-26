"""Regression tests for previously silent failure modes."""

from pathlib import Path

import pandas as pd
import pytest

from ca_personas.llm.base import validate_prediction
from ca_personas.paths import EXCERPT_PROLIFIC, EXCERPT_QUALTRICS
from ca_personas.pipeline import _resolve_prolific_paths, _resolve_qualtrics_path, load_config
from ca_personas.predict import run_predictions
from ca_personas.personas import PersonaPrompt
from inference.predict_vllm import _attach_ground_truth


ROOT = Path(__file__).resolve().parents[1]


class _FailClient:
    provider = "mock"
    model = "fail"

    def complete(self, system_prompt: str, user_prompt: str):
        raise RuntimeError("endpoint down")


def test_validate_prediction_rejects_non_integral_scores():
    with pytest.raises(ValueError, match="integer"):
        validate_prediction(
            {
                "self_reported_group_ca": 12.9,
                "self_reported_interpersonal_ca": 18,
            }
        )


def test_validate_prediction_rejects_invalid_bands():
    with pytest.raises(ValueError, match="self_reported_band_group"):
        validate_prediction(
            {
                "self_reported_group_ca": 12,
                "self_reported_interpersonal_ca": 18,
                "self_reported_band_group": "anxious",
                "self_reported_band_interpersonal": "low",
            }
        )


def test_run_predictions_fails_when_all_rows_error():
    prompts = [
        PersonaPrompt("p1", "demos", "sys", "user"),
        PersonaPrompt("p2", "demos", "sys", "user"),
    ]
    with pytest.raises(RuntimeError, match="All 2 predictions failed"):
        run_predictions(_FailClient(), prompts)


def test_config_path_resolution_falls_back_to_excerpts(monkeypatch, tmp_path):
    # When no full-cohort staging directory is present, library resolution may
    # fall back to public excerpt fixtures (tests / offline CI). CLI analysis
    # commands separately refuse that fallback via allow_excerpt_fallback=False.
    import ca_personas.paths as paths_mod

    monkeypatch.delenv("CA_SIBLING_DATA", raising=False)
    monkeypatch.setattr(paths_mod, "SIBLING_DATA_DIR", tmp_path / "missing_sibling")
    monkeypatch.setattr(
        paths_mod,
        "_staging_dirs",
        lambda: [tmp_path / "missing_sibling", tmp_path / "also_missing"],
    )
    # Config lists sibling_data paths that are absent in this sandbox.
    config = load_config(ROOT / "config" / "default.yaml")
    prolific = _resolve_prolific_paths(None, config, allow_excerpt_fallback=True)
    qualtrics = _resolve_qualtrics_path(None, config, allow_excerpt_fallback=True)
    assert all(p.is_file() for p in prolific)
    assert qualtrics.is_file()
    assert EXCERPT_PROLIFIC in prolific or any(
        p.name == "prolific_excerpt.csv" for p in prolific
    )
    assert qualtrics == EXCERPT_QUALTRICS or qualtrics.name == "qualtrics_excerpt.csv"


def test_config_path_resolution_can_refuse_excerpt_fallback(monkeypatch, tmp_path):
    import ca_personas.paths as paths_mod

    monkeypatch.delenv("CA_SIBLING_DATA", raising=False)
    monkeypatch.setattr(paths_mod, "SIBLING_DATA_DIR", tmp_path / "missing_sibling")
    monkeypatch.setattr(
        paths_mod,
        "_staging_dirs",
        lambda: [tmp_path / "missing_sibling", tmp_path / "also_missing"],
    )
    config = load_config(ROOT / "config" / "default.yaml")
    with pytest.raises(FileNotFoundError):
        _resolve_prolific_paths(None, config, allow_excerpt_fallback=False)


def test_load_config_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_attach_ground_truth_coalesces_missing_answers(tmp_path: Path):
    prompts = pd.DataFrame(
        {
            "caseid": ["a__demos", "b__demos"],
            "prompt": ["p1", "p2"],
            "answer": ['{"gt_group_ca":10}', None],
        }
    )
    truth = tmp_path / "gt.csv"
    pd.DataFrame(
        {
            "caseid": ["a__demos", "b__demos"],
            "answer": ['{"gt_group_ca":10}', '{"gt_group_ca":20}'],
        }
    ).to_csv(truth, index=False)

    out = _attach_ground_truth(prompts, str(truth))
    assert list(out["answer"]) == ['{"gt_group_ca":10}', '{"gt_group_ca":20}']
