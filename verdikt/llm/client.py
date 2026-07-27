"""LLM client abstraction. Any backend works if it implements ``LLMClient``."""
from __future__ import annotations

import abc
import asyncio
import random
from typing import Any

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    extra: dict[str, Any] = Field(default_factory=dict)


class LLMClient(abc.ABC):
    """Minimal interface every backend must implement."""

    @abc.abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = True,
        **kwargs: Any,
    ) -> LLMResponse:
        ...


TRANSIENT_MARKERS = (
    "rate limit",
    "429",
    "overloaded",
    "timeout",
    "temporarily",
    "503",
    "quota",
)


def is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in TRANSIENT_MARKERS)


class RetryingClient(LLMClient):
    """Wraps another client with exponential-backoff retries and model fallbacks.

    Transient errors (rate limits, timeouts, overload) are retried with
    backoff; anything else moves on to the next fallback model. If every
    candidate fails, the last error propagates and the Executor turns it
    into an error Verdict.
    """

    def __init__(
        self,
        inner: LLMClient,
        max_retries: int = 3,
        base_delay: float = 1.0,
        fallbacks: dict[str, list[str]] | None = None,  # model -> fallback models
    ):
        self.inner = inner
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.fallbacks = fallbacks or {}

    async def complete(self, model: str, messages: list[dict[str, str]], **kw: Any) -> LLMResponse:
        candidates = [model, *self.fallbacks.get(model, [])]
        last_exc: Exception = RuntimeError("no models tried")
        for candidate in candidates:
            for attempt in range(self.max_retries + 1):
                try:
                    return await self.inner.complete(candidate, messages, **kw)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if not is_transient(exc) or attempt == self.max_retries:
                        break  # try next fallback model
                    delay = self.base_delay * (2**attempt) * (1 + random.random() * 0.2)
                    await asyncio.sleep(delay)
        raise last_exc
