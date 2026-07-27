"""FrontierClient protocol adapters, tested against a mock HTTP transport
injected straight into each provider's official SDK."""
from __future__ import annotations

import json

import httpx
import pytest

from verdikt import ProviderConfig
from verdikt.llm import logging as llm_logging
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
            "anthropic": ProviderConfig(api_key="sk-ant"),
            "gemini": ProviderConfig(api_key="sk-gem"),
            "myhost": ProviderConfig(
                api_key="sk-local",
                base_url="http://localhost:8000",
                protocol="anthropic",
            ),
        }
    )
    client = FrontierClient(registry, transport=httpx.MockTransport(_handler))
    return client, seen


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
    assert body["model"] == "claude-sonnet-4-5"
    assert body["system"] == "be strict"  # system message lifted out
    assert all(m["role"] != "system" for m in body["messages"])
    assert resp.text == '{"score": 5}'
    assert resp.input_tokens == 20 and resp.output_tokens == 9
    await client.aclose()


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
    await client.aclose()


async def test_custom_base_url_override():
    def handler(request):
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "ok"}], "usage": {}},
        )

    client, seen = make_client(handler)
    await client.complete("myhost/some-claude-compatible-model", MESSAGES)
    assert str(seen[0].url) == "http://localhost:8000/v1/messages"
    assert seen[0].headers["x-api-key"] == "sk-local"
    await client.aclose()


async def test_anthropic_error_includes_status_for_retry_classification():
    def handler(request):
        return httpx.Response(
            429,
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
        )

    client, _ = make_client(handler)
    with pytest.raises(RuntimeError, match="429"):
        await client.complete("anthropic/claude-sonnet-4-5", MESSAGES)
    await client.aclose()


async def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = FrontierClient(ProviderRegistry(), transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(ValueError, match="no API key"):
        await client.complete("gemini/gemini-2.5-flash", MESSAGES)
    await client.aclose()


async def test_unknown_provider_raises():
    registry = ProviderRegistry({"weird": ProviderConfig(api_key="k")})
    client = FrontierClient(registry, transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(ValueError, match="protocol"):
        await client.complete("weird/some-model", MESSAGES)
    await client.aclose()


async def test_llm_call_logging_flag(capsys):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "logged output"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    try:
        client, _ = make_client(handler)
        llm_logging.set_llm_logging(True)
        await client.complete("anthropic/claude-sonnet-4-5", MESSAGES)
        out = capsys.readouterr().out
        assert "REQUEST" in out and "RESPONSE" in out
        assert "be strict" in out  # exact system prompt printed
        assert "logged output" in out  # exact response text printed

        llm_logging.set_llm_logging(False)
        client2, _ = make_client(handler)
        await client2.complete("anthropic/claude-sonnet-4-5", MESSAGES)
        assert capsys.readouterr().out == ""
        await client.aclose()
        await client2.aclose()
    finally:
        llm_logging.set_llm_logging(None)
