"""Native client for frontier LLM providers — no litellm, no provider SDKs.

Talks directly to each provider's HTTP API over three wire protocols:

- ``openai``    — OpenAI Chat Completions, also used by Kimi (Moonshot),
                  Mistral, OpenRouter, vLLM, Ollama, and most self-hosted
                  OpenAI-compatible endpoints
- ``anthropic`` — Anthropic Messages API (Claude)
- ``gemini``    — Google Gemini generateContent API

The protocol is inferred from the provider prefix of the model string
("anthropic/claude-sonnet-4-5" -> anthropic protocol). Custom providers can
set ``protocol`` and ``base_url`` in their ProviderConfig.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import httpx

from .client import LLMClient, LLMResponse
from .cost import estimate_cost
from .providers import ProviderRegistry

_PROTOCOLS: dict[str, str] = {
    "openai": "openai",
    "kimi": "openai",
    "moonshot": "openai",
    "mistral": "openai",
    "openrouter": "openai",
    "xai": "openai",
    "deepseek": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
}

_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "kimi": "https://api.moonshot.ai/v1",
    "moonshot": "https://api.moonshot.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "google": "https://generativelanguage.googleapis.com/v1beta",
}

ANTHROPIC_VERSION = "2023-06-01"


class FrontierClient(LLMClient):
    """Direct HTTP client for frontier providers (OpenAI, Anthropic/Claude,
    Google Gemini, Kimi K-series, and any OpenAI-compatible endpoint)."""

    def __init__(
        self,
        registry: Optional[ProviderRegistry] = None,
        timeout: float = 120.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,  # injectable for tests
    ):
        self.registry = registry or ProviderRegistry()
        self._http = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        await self._http.aclose()

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
        protocol = cfg.protocol or _PROTOCOLS.get(provider, "openai")
        base_url = (cfg.base_url or _BASE_URLS.get(provider, "")).rstrip("/")
        if not base_url:
            raise ValueError(
                f"unknown provider '{provider}' for model '{model}': set base_url "
                f"(and protocol) in its ProviderConfig"
            )
        if not cfg.api_key:
            raise ValueError(
                f"no API key for provider '{provider}': set it in ProviderConfig "
                f"or the provider's environment variable"
            )
        params = {**cfg.default_params, **kwargs}
        start = time.perf_counter()
        async with self.registry.semaphore(provider):
            if protocol == "anthropic":
                text, in_tok, out_tok = await self._anthropic(
                    base_url, cfg.api_key, bare, messages, temperature, max_tokens, params
                )
            elif protocol == "gemini":
                text, in_tok, out_tok = await self._gemini(
                    base_url, cfg.api_key, bare, messages, temperature, max_tokens,
                    json_mode, params,
                )
            else:
                text, in_tok, out_tok = await self._openai(
                    base_url, cfg.api_key, bare, messages, temperature, max_tokens,
                    json_mode, params,
                )
        return LLMResponse(
            text=text,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=estimate_cost(model, in_tok, out_tok),
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    # -- protocol adapters ---------------------------------------------------

    async def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict:
        resp = await self._http.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            # include status so RetryingClient can spot 429/503 as transient
            raise RuntimeError(f"HTTP {resp.status_code} from {url}: {resp.text[:500]}")
        return resp.json()

    async def _openai(
        self, base_url: str, api_key: str, model: str,
        messages: list[dict[str, str]], temperature: float, max_tokens: int,
        json_mode: bool, params: dict[str, Any],
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
        data = await self._post(
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

    async def _anthropic(
        self, base_url: str, api_key: str, model: str,
        messages: list[dict[str, str]], temperature: float, max_tokens: int,
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
        data = await self._post(
            f"{base_url}/v1/messages",
            {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
            payload,
        )
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    async def _gemini(
        self, base_url: str, api_key: str, model: str,
        messages: list[dict[str, str]], temperature: float, max_tokens: int,
        json_mode: bool, params: dict[str, Any],
    ) -> tuple[str, int, int]:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m["content"]}]}
            for m in messages
            if m["role"] != "system"
        ]
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            **params,
        }
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation_config}
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        data = await self._post(
            f"{base_url}/models/{model}:generateContent",
            {"x-goog-api-key": api_key},
            payload,
        )
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            text = "".join(
                p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])
            )
        usage = data.get("usageMetadata", {})
        return text, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)
