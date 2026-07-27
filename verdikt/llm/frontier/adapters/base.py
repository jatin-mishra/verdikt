"""Shared adapter interface: each protocol adapter owns one provider's
official SDK entirely — its request/response shapes never leak past the
``complete()`` boundary below."""
from __future__ import annotations

import abc
import inspect
from typing import Any

import httpx


class ProtocolAdapter(abc.ABC):
    """One wire protocol (Anthropic, Gemini), backed by that provider's
    official SDK. Caches SDK clients per (base_url, api_key) so repeated
    calls reuse connections instead of reconnecting every time."""

    def __init__(self) -> None:
        self._clients: dict[tuple[Any, ...], Any] = {}

    def _client_for(self, key: tuple[Any, ...], factory) -> Any:
        if key not in self._clients:
            self._clients[key] = factory()
        return self._clients[key]

    async def aclose(self) -> None:
        for client in self._clients.values():
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close is None:
                continue
            result = close()
            if inspect.isawaitable(result):
                await result
        self._clients.clear()

    @abc.abstractmethod
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
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> tuple[str, int, int]:
        """-> (text, input_tokens, output_tokens)"""
