from __future__ import annotations

import pytest

from verdikt import EvalInput, JudgeConfig, PipelineConfig, Verdict
from verdikt.core.registry import get_judge_class
from verdikt.execution.modes import Executor
from verdikt.pipeline.runner import PipelineRunner, eval_condition

from conftest import FakeLLMClient, label_response, score_response

MODEL = "openai/gpt-4.1-mini"


def executors(client) -> dict:
    out = {}
    for cfg in [
        JudgeConfig(name="safety", type="classifier", model=MODEL, labels=["safe", "unsafe"], fail_on=["unsafe"]),
        JudgeConfig(name="quality", type="pointwise", model=MODEL, threshold=0.6),
        JudgeConfig(name="recheck", type="pointwise", model=MODEL, threshold=0.6),
    ]:
        out[cfg.name] = Executor(get_judge_class(cfg.type)(cfg), client)
    return out


def route(model, messages):
    prompt = messages[-1]["content"]
    if "Allowed labels" in prompt:
        return label_response("safe")
    return score_response(4)


async def test_sequential_pipeline_with_prior_verdicts():
    client = FakeLLMClient(handler=route)
    cfg = PipelineConfig(
        name="p",
        steps=[{"judge": "safety"}, {"judge": "quality", "pass_prior_verdicts": True}],
        aggregation="all_pass",
    )
    runner = PipelineRunner(cfg, executors(client))
    pv = await runner.run(EvalInput(output="hello"))
    assert pv.passed is True
    assert [v.judge_name for v in pv.verdicts] == ["safety", "quality"]
    # quality judge saw safety's verdict in its prompt
    assert "safety" in client.calls[-1][1]


async def test_gate_stops_pipeline():
    def handler(model, messages):
        prompt = messages[-1]["content"]
        if "Allowed labels" in prompt:
            return label_response("unsafe")
        return score_response(5)

    client = FakeLLMClient(handler=handler)
    cfg = PipelineConfig(
        name="p",
        steps=[{"judge": "safety", "on_fail": "stop"}, {"judge": "quality"}],
    )
    runner = PipelineRunner(cfg, executors(client))
    pv = await runner.run(EvalInput(output="bad"))
    assert pv.passed is False
    assert pv.stopped_early is True
    assert pv.skipped_steps == ["quality"]
    assert len(client.calls) == 1  # quality never ran


async def test_run_if_condition():
    client = FakeLLMClient(handler=route)  # quality scores 0.75
    cfg = PipelineConfig(
        name="p",
        steps=[
            {"judge": "quality"},
            {"judge": "recheck", "run_if": "quality.score < 0.6"},
        ],
    )
    runner = PipelineRunner(cfg, executors(client))
    pv = await runner.run(EvalInput(output="x"))
    assert pv.skipped_steps == ["recheck"]
    assert len(pv.verdicts) == 1


async def test_parallel_step():
    client = FakeLLMClient(handler=route)
    cfg = PipelineConfig(name="p", steps=[{"parallel": ["safety", "quality"]}])
    runner = PipelineRunner(cfg, executors(client))
    pv = await runner.run(EvalInput(output="x"))
    assert {v.judge_name for v in pv.verdicts} == {"safety", "quality"}


async def test_weighted_average_aggregation():
    client = FakeLLMClient(handler=route)
    cfg = PipelineConfig(
        name="p",
        steps=[{"judge": "quality"}, {"judge": "recheck"}],
        aggregation="weighted_average",
        weights={"quality": 3.0, "recheck": 1.0},
        threshold=0.5,
    )
    runner = PipelineRunner(cfg, executors(client))
    pv = await runner.run(EvalInput(output="x"))
    assert pv.score == pytest.approx(0.75)
    assert pv.passed is True


def test_eval_condition():
    vs = {"q": Verdict(judge_name="q", score=0.4, passed=False, label="bad")}
    assert eval_condition("q.score < 0.6", vs) is True
    assert eval_condition("q.passed == false", vs) is True
    assert eval_condition("q.label == 'bad'", vs) is True
    assert eval_condition("q.score >= 0.6", vs) is False
    assert eval_condition("missing.score < 1", vs) is False
    with pytest.raises(ValueError):
        eval_condition("garbage!!", vs)


async def test_batch_runner():
    from verdikt.pipeline.batch import BatchRunner

    async def evaluate(inp: EvalInput):
        bad = "bad" in inp.output
        return Verdict(judge_name="j", score=0.1 if bad else 0.9, passed=not bad)

    runner = BatchRunner(evaluate, concurrency=2)
    items = [EvalInput(output=x) for x in ["ok 1", "bad 2", "ok 3"]]
    result = await runner.run(items)
    assert result.total == 3
    assert len(result.results) == 3
    assert result.failed_indices == [1]
