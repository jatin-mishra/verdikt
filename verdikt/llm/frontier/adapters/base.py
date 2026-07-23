"""Shared adapter interface and HTTP helper for frontier wire protocols."""
from __future__ import annotations

import abc
from typing import Any

import httpx


class ProtocolAdapter(abc.ABC):
    """One wire protocol (OpenAI/Anthropic/Gemini), shared by every provider
    that speaks it."""

    @abc.abstractmethod
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
        """-> (text, input_tokens, output_tokens)"""


async def post_json(
    http: httpx.AsyncClient, url: str, headers: dict[str, str], payload: dict[str, Any]
) -> dict:
    resp = await http.post(url, headers=headers, json=payload)
    if resp.status_code >= 400:
        # include status so RetryingClient can spot 429/503 as transient
        raise RuntimeError(f"HTTP {resp.status_code} from {url}: {resp.text[:500]}")
    return resp.json()
