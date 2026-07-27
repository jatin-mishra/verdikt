from __future__ import annotations

import pytest
from conftest import FakeLLMClient, contains, label_response, score_response

from verdikt import EvalInput, Verdikt, parse_config
from verdikt.config.loader import load_config

YAML = """
providers:
  anthropic:
    api_key: ${TEST_ANTHROPIC_KEY}
  gemini:
    api_key: ${TEST_GEMINI_KEY}
    base_url: "https://gemini-proxy.example.com"

judges:
  - name: helpfulness
    type: pointwise
    model: anthropic/claude-haiku
    threshold: 0.6
  - name: safety
    type: classifier
    model: anthropic/claude-haiku
    labels: [safe, unsafe]
    fail_on: [unsafe]
  - name: panel
    type: pointwise
    execution:
      mode: broadcast
      models: [anthropic/claude-sonnet-4-5, gemini/gemini-2.5-pro]
      consensus: average

pipelines:
  - name: main
    steps:
      - judge: safety
        on_fail: stop
      - judge: helpfulness
    aggregation: all_pass
"""


@pytest.fixture
def yaml_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-test-123")
    monkeypatch.setenv("TEST_GEMINI_KEY", "gk-test-456")
    p = tmp_path / "verdikt.yaml"
    p.write_text(YAML)
    return str(p)


def test_yaml_env_interpolation(yaml_path):
    cfg = load_config(yaml_path)
    assert cfg.providers["anthropic"].api_key == "sk-test-123"
    assert cfg.providers["gemini"].base_url == "https://gemini-proxy.example.com"
    assert len(cfg.judges) == 3
    assert cfg.pipelines[0].steps[0].on_fail == "stop"


def test_unknown_judge_in_pipeline_rejected():
    cfg = parse_config(
        {
            "judges": [{"name": "a", "type": "pointwise", "model": "anthropic/x"}],
            "pipelines": [{"name": "p", "steps": [{"judge": "nope"}]}],
        }
    )
    with pytest.raises(ValueError, match="unknown judges"):
        Verdikt.from_config(cfg, client=FakeLLMClient())


async def test_facade_judge_and_pipeline(yaml_path):
    def handler(model, messages):
        if contains(messages, "Allowed labels"):
            return label_response("safe")
        return score_response(4)

    cfg = load_config(yaml_path)
    v = Verdikt.from_config(cfg, client=FakeLLMClient(handler=handler))

    verdict = await v.evaluate("helpfulness", EvalInput(input="q", output="a"))
    assert verdict.passed is True

    pv = await v.evaluate("main", EvalInput(output="a"))
    assert pv.passed is True
    assert len(pv.verdicts) == 2

    with pytest.raises(KeyError, match="unknown judge/pipeline"):
        await v.evaluate("nope", EvalInput(output="a"))


async def test_facade_batch(yaml_path):
    cfg = load_config(yaml_path)
    v = Verdikt.from_config(cfg, client=FakeLLMClient())
    items = [EvalInput(output=f"answer {i}") for i in range(3)]
    result = await v.evaluate_batch("helpfulness", items)
    assert len(result.results) == 3
    assert result.failed_indices == []


def test_custom_judge_registration():
    from verdikt import BaseJudge, JudgeConfig, register
    from verdikt.core.base import ScoreParsingMixin
    from verdikt.core.registry import available_types

    @register("my_custom")
    class MyJudge(ScoreParsingMixin, BaseJudge):
        template = "pointwise.j2"

    assert "my_custom" in available_types()
    v = Verdikt(
        judges=[JudgeConfig(name="c", type="my_custom", model="anthropic/x")],
        client=FakeLLMClient(),
    )
    assert "c" in v.executors
