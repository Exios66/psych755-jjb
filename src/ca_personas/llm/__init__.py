"""LLM provider adapters (Ollama, OpenRouter)."""

from ca_personas.llm.base import LLMClient, LLMResponse, get_client

__all__ = ["LLMClient", "LLMResponse", "get_client"]
