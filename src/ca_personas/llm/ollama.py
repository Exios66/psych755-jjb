"""Ollama chat-completions client."""

from __future__ import annotations

from typing import Any

import requests

from ca_personas.llm.base import LLMClient, LLMResponse


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 256,
        timeout_seconds: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = requests.post(url, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            raise RuntimeError(f"Empty Ollama response: {data}")
        return LLMResponse(content=content, provider=self.provider, model=self.model, raw=data)
