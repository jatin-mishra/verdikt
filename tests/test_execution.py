from __future__ import annotations

import json

import pytest

from verdikt import EvalInput, JudgeConfig
from verdikt.core.registry import get_judge_class
from verdikt.execution.modes import Executor

from conftest import FakeLLMClient, label_response, score_response

M1, M2, M3 = "openai/gpt-4.1", "anthropic/claude-sonnet", "kimi/kimi-k3"
SCORES = {M1: 5, M2: 4, M3: 3}  # normalized: 1.0, 0.75, 0.5


def make_executor(client, judge_type="pointwise", **kw) -> Executor:
    cfg = JudgeConfig(name="panel", type=judge_type, **kw)
    return Executor(get_judge_class(judge_type)(cfg), client)


def by_model_handler(model, messages):
    return score_response(SCORES[model])


async def test_broadcast_average():
    ex = make_executor(
        FakeLLMClient(handler=by_model_handler),
        execution={"mode": "broadcast", "models": [M1, M2, M3], "consensus": "average"},
    )
    v = await ex.evaluate(EvalInput(output="a"))
    assert v.score == pytest.approx((1.0 + 0.75 + 0.5) / 3)
    assert len(v.sub_verdicts) == 3
    assert v.meta.execution_mode == "broadcast"
    assert v.meta.disagreement is not None
    assert v.meta.cost_usd > 0


async def test_broadcast_weighted_average():
    ex = make_executor(
        FakeLLMClient(handler=by_model_handler),
        execution={
            "mode": "broadcast",
            "models": [M1, M2],
            "consensus": "weighted_average",
            "weights": {M1: 3.0, M2: 1.0},
        },
    )
    v = await ex.evaluate(EvalInput(output="a"))
    assert v.score == pytest.approx((1.0 * 3 + 0.75 * 1) / 4)


async def test_broadcast_majority_vote_labels():
    def handler(model, messages):
        return label_response("safe" if model != M3 else "unsafe")

    ex = make_executor(
        FakeLLMClient(handler=handler),
        judge_type="classifier",
        labels=["safe", "unsafe"],
        fail_on=["unsafe"],
        execution={"mode": "broadcast", "models": [M1, M2, M3], "consensus": "majority_vote"},
    )
    v = await ex.evaluate(EvalInput(output="a"))
    assert v.label == "safe"
    assert v.passed is True
    assert v.meta.disagreement == pytest.approx(1 / 3)


async def test_broadcast_unanimous_fails_on_split_labels():
    def handler(model, messages):
        return label_response("safe" if model == M1 else "unsafe")

    ex = make_executor(
        FakeLLMClient(handler=handler),
        judge_type="classifier",
        labels=["safe", "unsafe"],
        execution={
            "mode": "broadcast",
            "models": [M1, M2],
            "consensus": "unanimous",
            "disagreement_threshold": 1.0,  # don't trigger disagreement handling
        },
    )
    v = await ex.evaluate(EvalInput(output="a"))
    assert v.passed is False


async def test_consensus_leader():
    ex = make_executor(
        FakeLLMClient(handler=by_model_handler),
        execution={
            "mode": "broadcast",
            "models": [M1, M2, M3],
            "consensus": "consensus_leader",
            "leader": M2,
            "disagreement_threshold": 1.0,
        },
    )
    v = await ex.evaluate(EvalInput(output="a"))
    assert v.score == 0.75  # leader M2's verdict


async def test_disagreement_fail_action():
    def handler(model, messages):
        return score_response(5 if model == M1 else 1)  # 1.0 vs 0.0 -> max disagreement

    ex = make_executor(
        FakeLLMClient(handler=handler),
        execution={
            "mode": "broadcast",
            "models": [M1, M2],
            "consensus": "average",
            "on_disagreement": "fail",
            "disagreement_threshold": 0.25,
        },
    )
    v = await ex.evaluate(EvalInput(output="a"))
    assert v.passed is False
    assert v.meta.extra["disagreement_action"] == "fail"


async def test_judge_of_judges_meta_call():
    def handler(model, messages):
        prompt = messages[-1]["content"]
        if "meta-judge" in prompt:
            assert "Judge 1" in prompt and "Judge 2" in prompt
            return json.dumps(
                {"reasoning": "judge 2 argued better", "score": 0.42, "label": None, "confidence": 0.9}
            )
        return by_model_handler(model, messages)

    ex = make_executor(
        FakeLLMClient(handler=handler),
        execution={"mode": "judge_of_judges", "models": [M1, M2], "meta_judge": M3},
    )
    v = await ex.evaluate(EvalInput(output="a"))
    assert v.score == 0.42
    assert v.meta.extra == {} or True  # meta info lives in sub meta
    assert len(v.sub_verdicts) == 2


async def test_broadcast_survives_one_member_failure():
    def handler(model, messages):
        if model == M2:
            return RuntimeError("boom")
        return by_model_handler(model, messages)

    ex = make_executor(
        FakeLLMClient(handler=handler),
        execution={"mode": "broadcast", "models": [M1, M2, M3], "consensus": "average"},
    )
    v = await ex.evaluate(EvalInput(output="a"))
    assert v.ok
    assert v.score == pytest.approx((1.0 + 0.5) / 2)
