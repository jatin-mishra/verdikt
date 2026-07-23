from __future__ import annotations

import json
from typing import Any, Callable, Optional, Union

import pytest

from verdikt.llm.client import LLMClient, LLMResponse

Handler = Callable[[str, list[dict[str, str]]], Union[str, Exception]]


class FakeLLMClient(LLMClient):
    """Test double. Provide either a handler(model, messages) -> str|Exception,
    or a list of canned responses consumed in order."""

    def __init__(self, handler: Optional[Handler] = None, responses: Optional[list] = None):
        self.handler = handler
        self.responses = list(responses or [])
        self.calls: list[tuple[str, str]] = []  # (model, prompt)

    async def complete(self, model: str, messages: list[dict[str, str]], **kw: Any) -> LLMResponse:
        prompt = messages[-1]["content"]
        self.calls.append((model, prompt))
        if self.handler is not None:
            out = self.handler(model, messages)
        elif self.responses:
            out = self.responses.pop(0)
        else:
            out = json.dumps({"reasoning": "looks fine", "score": 4, "confidence": 0.9})
        if isinstance(out, Exception):
            raise out
        return LLMResponse(text=out, model=model, input_tokens=10, output_tokens=5, cost_usd=0.001)


@pytest.fixture
def fake_client() -> FakeLLMClient:
    return FakeLLMClient()


def score_response(score: float, reasoning: str = "r", confidence: float = 0.9) -> str:
    return json.dumps({"reasoning": reasoning, "score": score, "confidence": confidence})


def label_response(label: str, reasoning: str = "r") -> str:
    return json.dumps({"reasoning": reasoning, "label": label, "confidence": 0.9})


def winner_response(winner: str, reasoning: str = "r") -> str:
    return json.dumps({"reasoning": reasoning, "winner": winner, "confidence": 0.9})
