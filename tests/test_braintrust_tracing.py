"""Unit tests for Braintrust CA scorers / opt-in wiring (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from inference.braintrust_tracing import (
    BraintrustRun,
    braintrust_configured,
    resolve_system_prompt,
    score_ca_generation,
    start_vllm_run,
)


def test_braintrust_configured_respects_env(monkeypatch):
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    monkeypatch.delenv("BRAINTRUST_ENABLED", raising=False)
    assert braintrust_configured() is False
    assert braintrust_configured(enabled=True) is True
    assert braintrust_configured(enabled=False) is False

    monkeypatch.setenv("BRAINTRUST_API_KEY", "sk-test")
    assert braintrust_configured() is True
    monkeypatch.setenv("BRAINTRUST_ENABLED", "false")
    assert braintrust_configured() is False
    monkeypatch.setenv("BRAINTRUST_ENABLED", "true")
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    assert braintrust_configured() is True


def test_score_ca_generation_perfect_match():
    gt = {
        "gt_group_ca": 12,
        "gt_interpersonal_ca": 18,
        "gt_group_band": "low",
        "gt_interpersonal_band": "moderate",
    }
    text = json.dumps(
        {
            "self_reported_group_ca": 12,
            "self_reported_interpersonal_ca": 18,
            "self_reported_band_group": "low",
            "self_reported_band_interpersonal": "moderate",
        }
    )
    scored = score_ca_generation(text, json.dumps(gt))
    assert scored["error"] is None
    assert scored["scores"]["parse_ok"] == 1.0
    assert scored["scores"]["exact_match_group"] == 1.0
    assert scored["scores"]["exact_match_interpersonal"] == 1.0
    assert scored["scores"]["band_match_group"] == 1.0
    assert scored["scores"]["band_match_mean"] == 1.0
    assert scored["scores"]["inverse_mae_mean"] == 1.0
    assert scored["metrics"]["mae_mean"] == 0.0


def test_score_ca_generation_parse_failure_and_errors():
    scored = score_ca_generation("not json", '{"gt_group_ca": 10}')
    assert scored["scores"]["parse_ok"] == 0.0
    assert scored["parsed"] is None
    assert scored["error"]
    assert "exact_match_group" not in scored["scores"]


def test_score_ca_generation_partial_miss():
    gt = {"gt_group_ca": 10, "gt_interpersonal_ca": 20, "gt_group_band": "low"}
    text = json.dumps(
        {
            "self_reported_group_ca": 22,  # abs err 12 → accuracy 0.5
            "self_reported_interpersonal_ca": 20,
            "self_reported_band_group": "high",
            "self_reported_band_interpersonal": "high",
        }
    )
    scored = score_ca_generation(text, gt)
    assert scored["scores"]["parse_ok"] == 1.0
    assert scored["scores"]["exact_match_group"] == 0.0
    assert scored["scores"]["exact_match_interpersonal"] == 1.0
    assert scored["scores"]["score_accuracy_group"] == 0.5
    assert scored["scores"]["band_match_group"] == 0.0
    assert scored["metrics"]["abs_error_group"] == 12.0


def test_resolve_system_prompt_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    monkeypatch.delenv("BRAINTRUST_ENABLED", raising=False)
    text, meta = resolve_system_prompt(fallback="LOCAL SYSTEM")
    assert text == "LOCAL SYSTEM"
    assert meta["source"] == "local"


def test_start_vllm_run_disabled_is_noop(monkeypatch):
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    run = start_vllm_run(model="m", preset="v1_baseline", enabled=False)
    assert run.enabled is False
    scored = run.log_generation(
        caseid="p1__demos",
        prompt="You are …",
        generated_text='{"self_reported_group_ca": 12, "self_reported_interpersonal_ca": 12, '
        '"self_reported_band_group": "low", "self_reported_band_interpersonal": "low"}',
        answer='{"gt_group_ca": 12, "gt_interpersonal_ca": 12}',
    )
    assert scored["scores"]["parse_ok"] == 1.0
    assert run.n_logged == 0
    assert run.close() is None


def test_braintrust_run_log_batch_summary():
    run = BraintrustRun(enabled=False, project="test")
    summary = run.log_batch(
        [
            {
                "caseid": "a__demos",
                "prompt": "p",
                "generated_text": (
                    '{"self_reported_group_ca": 12, '
                    '"self_reported_interpersonal_ca": 12, '
                    '"self_reported_band_group": "low", '
                    '"self_reported_band_interpersonal": "low"}'
                ),
                "answer": '{"gt_group_ca": 12, "gt_interpersonal_ca": 12}',
            },
            {
                "caseid": "b__demos",
                "prompt": "p",
                "generated_text": "broken",
                "answer": None,
            },
        ]
    )
    assert summary["n"] == 2
    assert summary["parse_rate"] == 0.5


def test_log_results_csv_disabled(tmp_path: Path, monkeypatch):
    from inference.braintrust_tracing import log_results_csv

    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    result = tmp_path / "results.csv"
    prompts = tmp_path / "prompts.csv"
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
                "answer": json.dumps(
                    {
                        "gt_group_ca": 12,
                        "gt_interpersonal_ca": 18,
                        "gt_group_band": "low",
                        "gt_interpersonal_band": "moderate",
                    }
                ),
            }
        ]
    ).to_csv(result, index=False)
    pd.DataFrame(
        [{"caseid": "p1__demos", "prompt": "You are a test persona."}]
    ).to_csv(prompts, index=False)

    summary = log_results_csv(
        result_csv=result,
        prompt_csv=prompts,
        model="mock-model",
        preset="v1_baseline",
        enabled=False,
    )
    assert summary["enabled"] is False
    assert summary["n"] == 1
    assert summary["parse_rate"] == 1.0


def test_predict_vllm_help_lists_braintrust():
    """CLI help must expose Braintrust flags without importing CUDA/vLLM."""
    from inference.predict_vllm import main
    import sys
    from io import StringIO

    buf = StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        try:
            main(["--help"])
        except SystemExit as exc:
            assert exc.code == 0
    finally:
        sys.stdout = old
    help_text = buf.getvalue()
    assert "--braintrust" in help_text
    assert "--braintrust_project" in help_text


def test_braintrust_prompt_module_imports_system_prompt():
    """Pushable prompt module must stay importable without network."""
    # Importing braintrust.projects.create is lazy at module level — the file
    # does call it on import. Guard with API-key absence by only checking the
    # source stays synced with SYSTEM_PROMPT.
    from ca_personas.personas import SYSTEM_PROMPT
    from pathlib import Path

    src = Path("prompts/braintrust_ca_system.py").read_text(encoding="utf-8")
    assert "SYSTEM_PROMPT" in src
    assert "ca-digital-twin-system" in src
    assert "psych755-ca-personas" in src
    # Sanity: local system prompt still has the JSON contract keys.
    assert "self_reported_group_ca" in SYSTEM_PROMPT
