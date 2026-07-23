"""Google Gemini generateContent API wire protocol."""
from __future__ import annotations

from typing import Any

import httpx

from .base import ProtocolAdapter, post_json

PROTOCOL = "gemini"

DEFAULT_BASE_URLS: dict[str, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "google": "https://generativelanguage.googleapis.com/v1beta",
}


class GeminiAdapter(ProtocolAdapter):
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
        data = await post_json(
            http,
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
