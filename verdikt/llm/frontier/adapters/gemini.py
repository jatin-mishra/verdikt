"""Google Gemini — backed by the official ``google-genai`` SDK.

Only this module knows about ``genai.Client`` or its request/response types;
``FrontierClient`` only ever sees the generic ``(text, input_tokens,
output_tokens)`` tuple from ``complete()``.
"""
from __future__ import annotations

from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .base import ProtocolAdapter

PROTOCOL = "gemini"

DEFAULT_BASE_URLS: dict[str, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "google": "https://generativelanguage.googleapis.com/v1beta",
}


class GeminiAdapter(ProtocolAdapter):
    def _client(
        self, base_url: str, api_key: str, timeout: float, transport: httpx.AsyncBaseTransport | None
    ) -> genai.Client:
        def factory() -> genai.Client:
            # base_url already carries the API version (".../v1beta"); clear
            # api_version so the SDK doesn't append its own default on top.
            http_options = types.HttpOptions(
                base_url=base_url,
                api_version="",
                timeout=int(timeout * 1000),
                async_client_args={"transport": transport} if transport is not None else None,
            )
            return genai.Client(api_key=api_key, http_options=http_options)

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
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> tuple[str, int, int]:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
            if m["role"] != "system"
        ]
        params = dict(params)
        # verdikt-level convenience flag Anthropic understands (see
        # AnthropicAdapter.complete) but Gemini has no equivalent inline
        # mechanism for (its context caching is a separate, heavier API) --
        # drop it rather than let it crash a judge broadcasting to both
        # providers with the same llm_params.
        params.pop("cache_system_prompt", None)
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction="\n\n".join(system_parts) if system_parts else None,
            response_mime_type="application/json" if json_mode else None,
            **params,
        )
        client = self._client(base_url, api_key, timeout, transport)
        try:
            resp = await client.aio.models.generate_content(model=model, contents=contents, config=config)
        except genai_errors.APIError as exc:
            raise RuntimeError(str(exc)) from exc
        usage = resp.usage_metadata
        input_tokens = (usage.prompt_token_count if usage else None) or 0
        output_tokens = (usage.candidates_token_count if usage else None) or 0
        return resp.text or "", input_tokens, output_tokens
