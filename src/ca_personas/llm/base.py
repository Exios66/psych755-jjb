"""Shared LLM client interface and factory."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    raw: dict[str, Any] | None = None


class LLMClient(ABC):
    provider: str
    model: str

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a chat completion as plain text content."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating markdown fences."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def validate_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize predicted CA scores."""
    required = (
        "self_reported_group_ca",
        "self_reported_interpersonal_ca",
    )
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Prediction JSON missing keys: {missing}")

    group = int(payload["self_reported_group_ca"])
    interpersonal = int(payload["self_reported_interpersonal_ca"])
    if not 6 <= group <= 30:
        raise ValueError(f"group CA out of range: {group}")
    if not 6 <= interpersonal <= 30:
        raise ValueError(f"interpersonal CA out of range: {interpersonal}")

    return {
        "pred_group_ca": group,
        "pred_interpersonal_ca": interpersonal,
        "pred_group_band": payload.get("self_reported_band_group"),
        "pred_interpersonal_band": payload.get("self_reported_band_interpersonal"),
    }


def get_client(
    provider: str | None = None,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 256,
    timeout_seconds: int = 120,
) -> LLMClient:
    """Construct an LLM client from explicit args or environment variables."""
    chosen = (provider or os.getenv("CA_LLM_PROVIDER", "ollama")).strip().lower()
    if chosen == "mock":
        from ca_personas.llm.mock import MockLLMClient

        return MockLLMClient(model=model or "mock-persona")
    if chosen == "ollama":
        from ca_personas.llm.ollama import OllamaClient

        return OllamaClient(
            model=model or os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
    if chosen == "openrouter":
        from ca_personas.llm.openrouter import OpenRouterClient

        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for provider=openrouter")
        return OpenRouterClient(
            api_key=api_key,
            model=model or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unsupported LLM provider: {chosen!r} (use ollama, openrouter, or mock)")
