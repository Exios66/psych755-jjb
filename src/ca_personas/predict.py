"""Run LLM predictions for persona prompts."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from ca_personas.llm.base import LLMClient, extract_json_object, validate_prediction
from ca_personas.personas import PersonaPrompt


def predict_one(client: LLMClient, prompt: PersonaPrompt) -> dict[str, Any]:
    """Call the LLM once and normalize the prediction row."""
    started = time.time()
    error: str | None = None
    raw_text = ""
    parsed: dict[str, Any] = {
        "pred_group_ca": None,
        "pred_interpersonal_ca": None,
        "pred_group_band": None,
        "pred_interpersonal_band": None,
    }
    try:
        response = client.complete(prompt.system_prompt, prompt.user_prompt)
        raw_text = response.content
        payload = extract_json_object(raw_text)
        parsed = validate_prediction(payload)
        provider = response.provider
        model = response.model
    except Exception as exc:  # noqa: BLE001 - capture per-row failures for batch runs
        error = f"{type(exc).__name__}: {exc}"
        provider = getattr(client, "provider", "unknown")
        model = getattr(client, "model", "unknown")

    return {
        "participant_id": prompt.participant_id,
        "tier": prompt.tier,
        "provider": provider,
        "model": model,
        "latency_seconds": round(time.time() - started, 3),
        "raw_response": raw_text,
        "error": error,
        **parsed,
    }


def run_predictions(
    client: LLMClient,
    prompts: list[PersonaPrompt],
    *,
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    """Predict CA scores for each persona prompt."""
    rows: list[dict[str, Any]] = []
    for i, prompt in enumerate(prompts):
        rows.append(predict_one(client, prompt))
        if sleep_seconds > 0 and i < len(prompts) - 1:
            time.sleep(sleep_seconds)
    return pd.DataFrame(rows)
