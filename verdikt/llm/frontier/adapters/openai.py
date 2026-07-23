"""OpenAI Chat Completions wire protocol.

Also used by any OpenAI-compatible endpoint: Kimi (Moonshot), Mistral,
OpenRouter, xAI, DeepSeek, vLLM, Ollama, and most self-hosted servers.
"""
from __future__ import annotations

from typing import Any

import httpx

from .base import ProtocolAdapter, post_json

PROTOCOL = "openai"

DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "kimi": "https://api.moonshot.ai/v1",
    "moonshot": "https://api.moonshot.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
}


class OpenAIAdapter(ProtocolAdapter):
    async def complete(
        self,
        http: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        params: dict[str, Any],
    ) -> tuple[str, int, int]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **params,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = await post_json(
            http,
            f"{base_url}/chat/completions",
            {"Authorization": f"Bearer {api_key}"},
            payload,
        )
        usage = data.get("usage", {})
        return (
            data["choices"][0]["message"].get("content") or "",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
