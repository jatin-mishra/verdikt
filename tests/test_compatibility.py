"""Judge-type x execution-mode compatibility matrix."""
from __future__ import annotations

import json

import pytest
from conftest import FakeLLMClient, winner_response

from verdikt import EvalInput, JudgeConfig
from verdikt.core.registry import get_judge_class
from verdikt.execution.modes import Executor

M1, M2, M3 = "anthropic/claude-sonnet-4-5", "anthropic/claude-haiku", "gemini/gemini-2.5-pro"


def make_executor(judge_type: str, client=None, **kw) -> Executor:
    cfg = JudgeConfig(name="j", type=judge_type, **kw)
    return Executor(get_judge_class(judge_type)(cfg), client or FakeLLMClient())


def test_classifier_rejects_average_consensus():
    with pytest.raises(ValueError, match="cannot use consensus 'average'"):
        make_executor(
            "classifier",
            labels=["safe", "unsafe"],
            execution={"mode": "broadcast", "models": [M1, M2], "consensus": "average"},
        )


def test_pairwise_rejects_weighted_average_consensus():
    with pytest.raises(ValueError, match="cannot use consensus 'weighted_average'"):
        make_executor(
            "pairwise",
            execution={"mode": "broadcast", "models": [M1, M2], "consensus": "weighted_average"},
        )


def test_classifier_broadcast_defaults_to_majority_vote():
    ex = make_executor(
        "classifier",
        labels=["safe", "unsafe"],
        execution={"mode": "broadcast", "models": [M1, M2, M3]},  # consensus not set
    )
    assert ex.cfg.consensus == "majority_vote"


def test_pairwise_broadcast_defaults_to_majority_vote():
    ex = make_executor("pairwise", execution={"mode": "broadcast", "models": [M1, M2, M3]})
    assert ex.cfg.consensus == "majority_vote"


def test_score_judge_broadcast_defaults_to_average():
    ex = make_executor("pointwise", execution={"mode": "broadcast", "models": [M1, M2]})
    assert ex.cfg.consensus == "average"


def test_explicit_valid_consensus_is_kept():
    ex = make_executor(
        "classifier",
        labels=["safe", "unsafe"],
        execution={"mode": "broadcast", "models": [M1, M2], "consensus": "unanimous"},
    )
    assert ex.cfg.consensus == "unanimous"


def test_consensus_leader_defaults_to_first_model():
    ex = make_executor(
        "pointwise",
        execution={"mode": "broadcast", "models": [M2, M1], "consensus": "consensus_leader"},
    )
    assert ex.cfg.leader == M2


async def test_pairwise_broadcast_majority_across_models():
    def handler(model, messages):
        prompt = messages[-1]["content"]
        a = prompt.split("<response_A>")[1].split("</response_A>")[0]
        prefers_good = model != M3  # M3 is a contrarian: prefers the bad answer
        wants_a = ("good" in a) == prefers_good
        return winner_response("A" if wants_a else "B")

    ex = make_executor(
        "pairwise",
        client=FakeLLMClient(handler=handler),
        execution={"mode": "broadcast", "models": [M1, M2, M3], "disagreement_threshold": 1.0},
    )
    v = await ex.evaluate(EvalInput(output="-", candidates=["good answer", "bad answer"]))
    assert v.label == "A"  # 2 of 3 models prefer A even with position swap


async def test_rubric_broadcast_merges_criteria_breakdown():
    def handler(model, messages):
        hi = model == M1
        return json.dumps(
            {
                "criteria": [
                    {"criterion": "correct", "reasoning": "r", "score": 5 if hi else 3},
                    {"criterion": "concise", "reasoning": "r", "score": 3},
                ],
                "reasoning": "overall",
                "confidence": 0.9,
            }
        )

    ex = make_executor(
        "rubric",
        client=FakeLLMClient(handler=handler),
        criteria=["correct", "concise"],
        execution={"mode": "broadcast", "models": [M1, M2], "disagreement_threshold": 1.0},
    )
    v = await ex.evaluate(EvalInput(output="a"))
    assert v.criteria_breakdown is not None
    by_name = {c.criterion: c.score for c in v.criteria_breakdown}
    assert by_name["correct"] == pytest.approx((1.0 + 0.5) / 2)
    assert by_name["concise"] == pytest.approx(0.5)
