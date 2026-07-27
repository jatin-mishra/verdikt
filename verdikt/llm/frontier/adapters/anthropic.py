"""Anthropic (Claude) — backed by the official ``anthropic`` SDK.

Only this module knows about ``anthropic.AsyncAnthropic`` or its request/
response types; ``FrontierClient`` only ever sees the generic
``(text, input_tokens, output_tokens)`` tuple from ``complete()``.
"""
from __future__ import annotations

from typing import Any, Optional

import anthropic
import httpx

from .base import ProtocolAdapter

PROTOCOL = "anthropic"

DEFAULT_BASE_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
}


class AnthropicAdapter(ProtocolAdapter):
    def _client(
        self, base_url: str, api_key: str, timeout: float, transport: Optional[httpx.AsyncBaseTransport]
    ) -> anthropic.AsyncAnthropic:
        def factory() -> anthropic.AsyncAnthropic:
            http_client = httpx.AsyncClient(transport=transport) if transport is not None else None
            return anthropic.AsyncAnthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=0,  # retries/backoff are centralized in RetryingClient
                http_client=http_client,
            )

        return self._client_for((base_url, api_key), factory)

    async def complete(
        self,
        base_url: str,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        params: dict[str, Any],
        *,
        timeout: float,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> tuple[str, int, int]:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        chat = [m for m in messages if m["role"] != "system"]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **params,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        # Anthropic has no JSON response_format; the prompt templates already
        # demand a JSON-only reply and extract_json() handles any wrapping.
        client = self._client(base_url, api_key, timeout, transport)
        try:
            resp = await client.messages.create(**kwargs)
        except anthropic.AnthropicError as exc:
            raise RuntimeError(str(exc)) from exc
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return text, resp.usage.input_tokens or 0, resp.usage.output_tokens or 0
