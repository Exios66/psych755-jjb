"""Deterministic mock LLM for offline tests and dry runs."""

from __future__ import annotations

import hashlib
import json

from ca_personas.llm.base import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    provider = "mock"

    def __init__(self, model: str = "mock-persona") -> None:
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        digest = hashlib.sha256((system_prompt + "\n" + user_prompt).encode("utf-8")).hexdigest()
        # Map hash bytes into the legal 6–30 range.
        group = 6 + (int(digest[:2], 16) % 25)
        interpersonal = 6 + (int(digest[2:4], 16) % 25)

        def band(score: int) -> str:
            if score <= 13:
                return "low"
            if score >= 20:
                return "high"
            return "moderate"

        payload = {
            "self_reported_group_ca": group,
            "self_reported_interpersonal_ca": interpersonal,
            "self_reported_band_group": band(group),
            "self_reported_band_interpersonal": band(interpersonal),
        }
        return LLMResponse(
            content=json.dumps(payload),
            provider=self.provider,
            model=self.model,
            raw={"mock": True, "digest": digest},
        )
