"""OpenRouter OpenAI-compatible chat client."""

from __future__ import annotations

import os
from typing import Any

import requests

from ca_personas.llm.base import LLMClient, LLMResponse


class OpenRouterClient(LLMClient):
    provider = "openrouter"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.2,
        max_tokens: int = 256,
        timeout_seconds: int = 120,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv(
                "OPENROUTER_HTTP_REFERER",
                "https://github.com/Exios66/psych755-jjb",
            ),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "psych755-ca-personas"),
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenRouter response: {data}") from exc
        if not content:
            raise RuntimeError(f"Empty OpenRouter response: {data}")
        return LLMResponse(content=content, provider=self.provider, model=self.model, raw=data)
