"""Braintrust Eval() entrypoint for CA digital-twin prompt iteration.

Runs offline scorers over a results CSV (task is a no-op that returns the
already-generated text). Use from the Braintrust playground / ``bt eval``
to compare prompt versions without re-running vLLM.

Example::

    BRAINTRUST_API_KEY=... \\
      VLLM_RESULT_CSV=outputs/vllm_results/results.csv \\
      VLLM_PROMPT_CSV=outputs/vllm_prompts/prompts.csv \\
      python -m inference.braintrust_eval

Or::

    bt eval src/inference/braintrust_eval.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from inference.braintrust_tracing import (
    DEFAULT_PROJECT,
    braintrust_configured,
    project_name,
    score_ca_generation,
)
from inference.utils import normalize_caseid

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_result_csv() -> Path:
    return Path(os.getenv("VLLM_RESULT_CSV", _REPO_ROOT / "outputs/vllm_results/results.csv"))


def _default_prompt_csv() -> Path | None:
    raw = os.getenv("VLLM_PROMPT_CSV", str(_REPO_ROOT / "outputs/vllm_prompts/prompts.csv"))
    path = Path(raw)
    return path if path.is_file() else None


def load_eval_cases(
    result_csv: Path | None = None,
    prompt_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Build Braintrust Eval cases from a vLLM results (+ optional prompts) CSV."""
    result_path = result_csv or _default_result_csv()
    if not result_path.is_file():
        raise FileNotFoundError(
            f"Result CSV not found: {result_path}. "
            "Set VLLM_RESULT_CSV or generate with scripts/run_vllm.sh"
        )
    df = pd.read_csv(result_path)
    df["caseid"] = df["caseid"].map(normalize_caseid)

    p_csv = prompt_csv if prompt_csv is not None else _default_prompt_csv()
    if p_csv is not None and Path(p_csv).is_file():
        prompts = pd.read_csv(p_csv)
        prompts["caseid"] = prompts["caseid"].map(normalize_caseid)
        cols = ["caseid"]
        for c in ("prompt", "answer", "tier", "participant_id"):
            if c in prompts.columns:
                cols.append(c)
        df = df.merge(prompts[cols].drop_duplicates("caseid"), on="caseid", how="left",
                      suffixes=("", "_prompt"))
        if "answer" not in df.columns and "answer_prompt" in df.columns:
            df["answer"] = df["answer_prompt"]
        if "prompt" not in df.columns and "prompt_prompt" in df.columns:
            df["prompt"] = df["prompt_prompt"]

    cases: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        expected = None
        if "answer" in row and not (isinstance(row["answer"], float) and pd.isna(row["answer"])):
            try:
                expected = json.loads(str(row["answer"]))
            except json.JSONDecodeError:
                expected = {"raw_answer": str(row["answer"])}
        cases.append(
            {
                "input": {
                    "caseid": row["caseid"],
                    "prompt": None if "prompt" not in row or pd.isna(row.get("prompt")) else str(row["prompt"]),
                    "generated_text": (
                        "" if pd.isna(row.get("generated_text")) else str(row["generated_text"])
                    ),
                },
                "expected": expected,
                "metadata": {
                    "caseid": row["caseid"],
                    "tier": (
                        None
                        if "tier" not in row or pd.isna(row.get("tier"))
                        else str(row["tier"])
                    ),
                    "provider": "vllm",
                },
                "tags": ["vllm", "ca-digital-twin"],
            }
        )
    return cases


def _task(input_row: dict[str, Any]) -> dict[str, Any]:
    """Replay stored generation (no live LLM) for offline scoring."""
    return {
        "generated_text": input_row.get("generated_text", ""),
        "caseid": input_row.get("caseid"),
    }


def parse_ok_scorer(input, output, expected) -> dict[str, Any]:  # noqa: A002
    scored = score_ca_generation(
        (output or {}).get("generated_text") if isinstance(output, dict) else str(output or ""),
        expected,
    )
    return {"name": "parse_ok", "score": scored["scores"].get("parse_ok", 0.0)}


def exact_match_mean_scorer(input, output, expected) -> dict[str, Any]:  # noqa: A002
    scored = score_ca_generation(
        (output or {}).get("generated_text") if isinstance(output, dict) else str(output or ""),
        expected,
    )
    val = scored["scores"].get("exact_match_mean")
    return {"name": "exact_match_mean", "score": val if val is not None else 0.0}


def band_match_mean_scorer(input, output, expected) -> dict[str, Any]:  # noqa: A002
    scored = score_ca_generation(
        (output or {}).get("generated_text") if isinstance(output, dict) else str(output or ""),
        expected,
    )
    val = scored["scores"].get("band_match_mean")
    return {"name": "band_match_mean", "score": val if val is not None else 0.0}


def inverse_mae_mean_scorer(input, output, expected) -> dict[str, Any]:  # noqa: A002
    scored = score_ca_generation(
        (output or {}).get("generated_text") if isinstance(output, dict) else str(output or ""),
        expected,
    )
    val = scored["scores"].get("inverse_mae_mean")
    return {"name": "inverse_mae_mean", "score": val if val is not None else 0.0}


def score_accuracy_mean_scorer(input, output, expected) -> dict[str, Any]:  # noqa: A002
    scored = score_ca_generation(
        (output or {}).get("generated_text") if isinstance(output, dict) else str(output or ""),
        expected,
    )
    val = scored["scores"].get("score_accuracy_mean")
    return {"name": "score_accuracy_mean", "score": val if val is not None else 0.0}


def run_eval(*, no_send_logs: bool | None = None) -> Any:
    """Execute Braintrust Eval over the configured results CSV."""
    from braintrust import Eval

    send = no_send_logs
    if send is None:
        send = not braintrust_configured()

    return Eval(
        project_name() or DEFAULT_PROJECT,
        data=load_eval_cases,
        task=_task,
        scores=[
            parse_ok_scorer,
            exact_match_mean_scorer,
            band_match_mean_scorer,
            score_accuracy_mean_scorer,
            inverse_mae_mean_scorer,
        ],
        metadata={
            "provider": "vllm",
            "task": "ca-digital-twin",
            "result_csv": str(_default_result_csv()),
        },
        experiment_name=os.getenv("BRAINTRUST_EXPERIMENT") or None,
        no_send_logs=send,
    )


if __name__ == "__main__":
    result = run_eval()
    # Eval returns an EvalResultWithSummary; print a compact view when possible.
    summary = getattr(result, "summary", result)
    print(summary)
