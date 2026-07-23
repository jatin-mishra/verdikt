"""Anthropic Messages API (Claude) wire protocol."""
from __future__ import annotations

from typing import Any

import httpx

from .base import ProtocolAdapter, post_json

PROTOCOL = "anthropic"
ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_BASE_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
}


class AnthropicAdapter(ProtocolAdapter):
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
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        chat = [m for m in messages if m["role"] != "system"]
        payload: dict[str, Any] = {
            "model": model,
            "messages": chat,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **params,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        # Anthropic has no JSON response_format; the prompt templates already
        # demand a JSON-only reply and extract_json() handles any wrapping.
        data = await post_json(
            http,
            f"{base_url}/v1/messages",
            {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
            payload,
        )
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
