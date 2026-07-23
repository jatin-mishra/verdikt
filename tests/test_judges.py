from __future__ import annotations

import json

import pytest

from verdikt import EvalInput, JudgeConfig, Step
from verdikt.core.registry import available_types, get_judge_class
from verdikt.execution.modes import Executor

from conftest import FakeLLMClient, label_response, score_response, winner_response

MODEL = "openai/gpt-4.1-mini"


def make(judge_type: str, **kw) -> "tuple":
    cfg = JudgeConfig(name=f"{judge_type}_j", type=judge_type, model=MODEL, **kw)
    return get_judge_class(judge_type)(cfg)


def test_builtins_registered():
    types = available_types()
    for t in [
        "pointwise",
        "reference",
        "pairwise",
        "rubric",
        "classifier",
        "rag_faithfulness",
        "rag_context_relevance",
        "rag_answer_relevance",
        "trajectory",
    ]:
        assert t in types


async def test_pointwise_score_normalized_and_threshold():
    judge = make("pointwise", threshold=0.7)
    client = FakeLLMClient(responses=[score_response(4)])
    v = await judge.evaluate_with_model(EvalInput(input="q", output="a"), client, MODEL)
    assert v.score == 0.75  # (4-1)/(5-1)
    assert v.passed is True
    assert v.meta.model == MODEL
    assert v.meta.cost_usd > 0


async def test_pointwise_fails_threshold():
    judge = make("pointwise", threshold=0.9)
    client = FakeLLMClient(responses=[score_response(3)])
    v = await judge.evaluate_with_model(EvalInput(output="a"), client, MODEL)
    assert v.passed is False


async def test_multi_sample_median():
    judge = make("pointwise", samples=3)
    client = FakeLLMClient(responses=[score_response(1), score_response(5), score_response(4)])
    v = await judge.evaluate_with_model(EvalInput(output="a"), client, MODEL)
    assert v.score == 0.75  # median of 0, 1, 0.75
    assert v.meta.samples == 3
    assert len(client.calls) == 3


async def test_reference_requires_expected_output():
    judge = make("reference")
    with pytest.raises(ValueError, match="expected_output"):
        await judge.evaluate_with_model(EvalInput(output="a"), FakeLLMClient(), MODEL)


async def test_rubric_breakdown():
    judge = make("rubric", criteria=["correct", "concise"])
    resp = json.dumps(
        {
            "criteria": [
                {"criterion": "correct", "reasoning": "ok", "score": 5},
                {"criterion": "concise", "reasoning": "meh", "score": 3},
            ],
            "reasoning": "overall",
            "confidence": 0.8,
        }
    )
    client = FakeLLMClient(responses=[resp])
    v = await judge.evaluate_with_model(EvalInput(output="a"), client, MODEL)
    assert v.criteria_breakdown is not None and len(v.criteria_breakdown) == 2
    assert v.score == pytest.approx((1.0 + 0.5) / 2)


async def test_classifier_label_and_fail_on():
    judge = make("classifier", labels=["safe", "unsafe"], fail_on=["unsafe"])
    client = FakeLLMClient(responses=[label_response("unsafe")])
    v = await judge.evaluate_with_model(EvalInput(output="bad stuff"), client, MODEL)
    assert v.label == "unsafe"
    assert v.passed is False


async def test_classifier_rejects_unknown_label():
    judge = make("classifier", labels=["safe", "unsafe"])
    client = FakeLLMClient(responses=[label_response("meh")])
    v = await judge.evaluate_with_model(EvalInput(output="x"), client, MODEL)
    assert v.error is not None  # bad sample -> error verdict, no exception


async def test_pairwise_consistent_winner_with_swap():
    def handler(model, messages):
        prompt = messages[-1]["content"]
        # prefer the candidate containing "good" whichever slot it is in
        a_section = prompt.split("<response_A>")[1].split("</response_A>")[0]
        return winner_response("A" if "good" in a_section else "B")

    judge = make("pairwise")
    client = FakeLLMClient(handler=handler)
    v = await judge.evaluate_with_model(
        EvalInput(output="-", candidates=["good answer", "bad answer"]), client, MODEL
    )
    assert v.label == "A"
    assert v.score == 1.0
    assert len(client.calls) == 2  # position swap ran both orders


async def test_pairwise_position_bias_becomes_tie():
    # always prefers slot A -> inconsistent across swap -> tie
    judge = make("pairwise")
    client = FakeLLMClient(handler=lambda m, msgs: winner_response("A"))
    v = await judge.evaluate_with_model(
        EvalInput(output="-", candidates=["x", "y"]), client, MODEL
    )
    assert v.label == "tie"
    assert "disagreement" in v.reasoning


async def test_rag_faithfulness_requires_context():
    judge = make("rag_faithfulness")
    with pytest.raises(ValueError, match="context"):
        await judge.evaluate_with_model(EvalInput(output="a"), FakeLLMClient(), MODEL)


async def test_trajectory_judge():
    judge = make("trajectory")
    client = FakeLLMClient(responses=[score_response(5)])
    inp = EvalInput(
        input="find the weather",
        output="It is sunny.",
        trajectory=[Step(thought="check API", tool="weather", tool_input={"city": "X"}, observation="sunny")],
    )
    v = await judge.evaluate_with_model(inp, client, MODEL)
    assert v.score == 1.0
    assert "weather" in client.calls[0][1]


async def test_prior_verdicts_rendered_into_prompt():
    from verdikt import Verdict

    judge = make("pointwise")
    client = FakeLLMClient(responses=[score_response(4)])
    prior = Verdict(judge_name="safety", label="safe", passed=True, reasoning="no issues")
    await judge.evaluate_with_model(
        EvalInput(output="a", prior_verdicts=[prior]), client, MODEL
    )
    assert "safety" in client.calls[0][1]
    assert "no issues" in client.calls[0][1]


async def test_executor_wraps_errors_as_verdict():
    judge = make("pointwise")
    client = FakeLLMClient(handler=lambda m, msgs: RuntimeError("boom"))
    ex = Executor(judge, client)
    v = await ex.evaluate(EvalInput(output="a"))
    assert v.error is not None and "boom" in v.error
