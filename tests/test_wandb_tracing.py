"""Unit tests for Weights & Biases opt-in wiring (no network)."""

from __future__ import annotations

from inference.wandb_tracing import WandbRun, start_wandb_run, wandb_configured


def test_wandb_configured_respects_env(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENABLED", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    assert wandb_configured() is False
    assert wandb_configured(enabled=True) is True
    assert wandb_configured(enabled=False) is False

    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    assert wandb_configured() is True
    monkeypatch.setenv("WANDB_ENABLED", "false")
    assert wandb_configured() is False
    monkeypatch.setenv("WANDB_ENABLED", "true")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert wandb_configured() is True
    monkeypatch.delenv("WANDB_ENABLED", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert wandb_configured() is True


def test_start_wandb_run_disabled_is_noop(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    run = start_wandb_run(model="m", preset="v1_baseline", enabled=False)
    assert run.enabled is False
    summary = run.log_chunk(
        chunk_idx=0,
        n_chunks=1,
        rows=[
            {
                "caseid": "p1__demos",
                "generated_text": (
                    '{"self_reported_group_ca": 12, '
                    '"self_reported_interpersonal_ca": 12, '
                    '"self_reported_band_group": "low", '
                    '"self_reported_band_interpersonal": "low"}'
                ),
                "answer": '{"gt_group_ca": 12, "gt_interpersonal_ca": 12}',
            }
        ],
    )
    assert summary["n"] == 1
    assert summary["parse_rate"] == 1.0
    assert run.n_logged == 0
    assert run.close() is None


def test_wandb_run_log_chunk_summary_without_backend():
    run = WandbRun(enabled=False, project="test")
    summary = run.log_chunk(
        chunk_idx=0,
        n_chunks=1,
        rows=[
            {
                "caseid": "a__demos",
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
                "generated_text": "broken",
                "answer": None,
            },
        ],
    )
    assert summary["n"] == 2
    assert summary["parse_rate"] == 0.5
