"""FrontierClient protocol adapters, tested against a mock HTTP transport."""
from __future__ import annotations

import json

import httpx
import pytest

from verdikt import ProviderConfig
from verdikt.llm.frontier import FrontierClient
from verdikt.llm.providers import ProviderRegistry

MESSAGES = [
    {"role": "system", "content": "be strict"},
    {"role": "user", "content": "judge this"},
]


def make_client(handler) -> tuple[FrontierClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    registry = ProviderRegistry(
        {
            "openai": ProviderConfig(api_key="sk-oai"),
            "anthropic": ProviderConfig(api_key="sk-ant"),
            "gemini": ProviderConfig(api_key="sk-gem"),
            "kimi": ProviderConfig(api_key="sk-kimi"),
            "myhost": ProviderConfig(
                api_key="sk-local", base_url="http://localhost:8000/v1", protocol="openai"
            ),
        }
    )
    client = FrontierClient(registry, transport=httpx.MockTransport(_handler))
    return client, seen


async def test_openai_protocol():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"score": 4}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
            },
        )

    client, seen = make_client(handler)
    resp = await client.complete("openai/gpt-4.1", MESSAGES)
    req = seen[0]
    assert str(req.url) == "https://api.openai.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-oai"
    body = json.loads(req.content)
    assert body["model"] == "gpt-4.1"
    assert body["response_format"] == {"type": "json_object"}
    assert resp.text == '{"score": 4}'
    assert resp.input_tokens == 12 and resp.output_tokens == 7


async def test_kimi_uses_moonshot_endpoint_with_openai_protocol():
    def handler(request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}}
        )

    client, seen = make_client(handler)
    await client.complete("kimi/kimi-k3", MESSAGES)
    assert str(seen[0].url) == "https://api.moonshot.ai/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer sk-kimi"


async def test_anthropic_protocol():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": '{"score": 5}'}],
                "usage": {"input_tokens": 20, "output_tokens": 9},
            },
        )

    client, seen = make_client(handler)
    resp = await client.complete("anthropic/claude-sonnet-4-5", MESSAGES)
    req = seen[0]
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "sk-ant"
    assert "anthropic-version" in req.headers
    body = json.loads(req.content)
    assert body["system"] == "be strict"  # system message lifted out
    assert all(m["role"] != "system" for m in body["messages"])
    assert resp.text == '{"score": 5}'
    assert resp.input_tokens == 20


async def test_gemini_protocol():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"score": 3}'}]}}],
                "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 5},
            },
        )

    client, seen = make_client(handler)
    resp = await client.complete("gemini/gemini-2.5-pro", MESSAGES)
    req = seen[0]
    assert "generativelanguage.googleapis.com" in str(req.url)
    assert str(req.url).endswith("models/gemini-2.5-pro:generateContent")
    assert req.headers["x-goog-api-key"] == "sk-gem"
    body = json.loads(req.content)
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["systemInstruction"]["parts"][0]["text"] == "be strict"
    assert resp.text == '{"score": 3}'
    assert resp.output_tokens == 5


async def test_custom_openai_compatible_endpoint():
    def handler(request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}}
        )

    client, seen = make_client(handler)
    await client.complete("myhost/llama-3.3-70b", MESSAGES)
    assert str(seen[0].url) == "http://localhost:8000/v1/chat/completions"


async def test_http_error_includes_status_for_retry_classification():
    def handler(request):
        return httpx.Response(429, text="rate limited")

    client, _ = make_client(handler)
    with pytest.raises(RuntimeError, match="429"):
        await client.complete("openai/gpt-4.1", MESSAGES)


async def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    client = FrontierClient(ProviderRegistry(), transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(ValueError, match="no API key"):
        await client.complete("mistral/mistral-large", MESSAGES)


async def test_unknown_provider_without_base_url_raises():
    registry = ProviderRegistry({"weird": ProviderConfig(api_key="k")})
    client = FrontierClient(registry, transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(ValueError, match="base_url"):
        await client.complete("weird/some-model", MESSAGES)
