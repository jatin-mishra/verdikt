from __future__ import annotations

import pytest

from verdikt import EvalInput, Verdikt, parse_config
from verdikt.config.loader import load_config

from conftest import FakeLLMClient, label_response, score_response

YAML = """
providers:
  openai:
    api_key: ${TEST_OPENAI_KEY}
  kimi:
    api_key: ${TEST_MOONSHOT_KEY}
    base_url: "https://api.moonshot.ai/v1"

judges:
  - name: helpfulness
    type: pointwise
    model: openai/gpt-4.1-mini
    threshold: 0.6
  - name: safety
    type: classifier
    model: openai/gpt-4.1-mini
    labels: [safe, unsafe]
    fail_on: [unsafe]
  - name: panel
    type: pointwise
    execution:
      mode: broadcast
      models: [openai/gpt-4.1, kimi/kimi-k3]
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
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-123")
    monkeypatch.setenv("TEST_MOONSHOT_KEY", "mk-test-456")
    p = tmp_path / "verdikt.yaml"
    p.write_text(YAML)
    return str(p)


def test_yaml_env_interpolation(yaml_path):
    cfg = load_config(yaml_path)
    assert cfg.providers["openai"].api_key == "sk-test-123"
    assert cfg.providers["kimi"].base_url == "https://api.moonshot.ai/v1"
    assert len(cfg.judges) == 3
    assert cfg.pipelines[0].steps[0].on_fail == "stop"


def test_unknown_judge_in_pipeline_rejected():
    cfg = parse_config(
        {
            "judges": [{"name": "a", "type": "pointwise", "model": "openai/x"}],
            "pipelines": [{"name": "p", "steps": [{"judge": "nope"}]}],
        }
    )
    with pytest.raises(ValueError, match="unknown judges"):
        Verdikt.from_config(cfg, client=FakeLLMClient())


async def test_facade_judge_and_pipeline(yaml_path):
    def handler(model, messages):
        if "Allowed labels" in messages[-1]["content"]:
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
        judges=[JudgeConfig(name="c", type="my_custom", model="openai/x")],
        client=FakeLLMClient(),
    )
    assert "c" in v.executors
