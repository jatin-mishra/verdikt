"""FrontierClient — dispatches to per-provider SDK-backed adapters."""
from __future__ import annotations

import time
from typing import Any, Optional

import httpx

from .. import logging as llm_logging
from ..client import LLMClient, LLMResponse
from ..cost import estimate_cost
from ..providers import ProviderRegistry
from .adapters import PROTOCOL_ADAPTER_CLASSES, PROVIDER_BASE_URLS, PROVIDER_PROTOCOLS


class FrontierClient(LLMClient):
    """LLM client for frontier providers (Anthropic/Claude, Google Gemini),
    backed by each provider's official Python SDK."""

    def __init__(
        self,
        registry: Optional[ProviderRegistry] = None,
        timeout: float = 120.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,  # injectable for tests
        log_calls: Optional[bool] = None,  # None -> defer to the global flag
    ):
        self.registry = registry or ProviderRegistry()
        self.timeout = timeout
        self.transport = transport
        self.log_calls = log_calls
        self._adapters = {proto: cls() for proto, cls in PROTOCOL_ADAPTER_CLASSES.items()}

    async def aclose(self) -> None:
        for adapter in self._adapters.values():
            await adapter.aclose()

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
        cfg, provider, bare = self.registry.resolve(model)
        protocol = cfg.protocol or PROVIDER_PROTOCOLS.get(provider)
        if protocol is None:
            raise ValueError(
                f"unknown provider '{provider}' for model '{model}': set 'protocol' "
                f"('anthropic' or 'gemini') in its ProviderConfig"
            )
        base_url = (cfg.base_url or PROVIDER_BASE_URLS.get(provider, "")).rstrip("/")
        if not base_url:
            raise ValueError(
                f"unknown provider '{provider}' for model '{model}': set base_url "
                f"in its ProviderConfig"
            )
        if not cfg.api_key:
            raise ValueError(
                f"no API key for provider '{provider}': set it in ProviderConfig "
                f"or the provider's environment variable"
            )
        adapter = self._adapters.get(protocol)
        if adapter is None:
            raise ValueError(f"unknown protocol '{protocol}' for model '{model}'")
        params = {**cfg.default_params, **kwargs}

        call_id = llm_logging.log_request(
            provider, model, messages, temperature, max_tokens, json_mode, params,
            enabled=self.log_calls,
        )
        start = time.perf_counter()
        try:
            async with self.registry.semaphore(provider):
                text, in_tok, out_tok = await adapter.complete(
                    base_url, cfg.api_key, bare, messages, temperature, max_tokens,
                    json_mode, params, timeout=self.timeout, transport=self.transport,
                )
        except Exception as exc:  # noqa: BLE001
            llm_logging.log_error(call_id, model, exc, enabled=self.log_calls)
            raise
        resp = LLMResponse(
            text=text,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=estimate_cost(model, in_tok, out_tok),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        llm_logging.log_response(call_id, model, resp, enabled=self.log_calls)
        return resp
