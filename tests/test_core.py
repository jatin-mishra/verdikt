from __future__ import annotations

import pytest

from verdikt import EvalInput, JudgeConfig, Scale
from verdikt.core.parsing import ParseError, extract_json


def test_scale_normalize():
    s = Scale(min=1, max=5)
    assert s.normalize(1) == 0.0
    assert s.normalize(5) == 1.0
    assert s.normalize(3) == 0.5
    assert s.normalize(9) == 1.0  # clamped
    assert s.normalize(-2) == 0.0


def test_judge_config_requires_model_for_single():
    with pytest.raises(ValueError, match="model"):
        JudgeConfig(name="j", type="pointwise")
    JudgeConfig(name="j", type="pointwise", model="openai/gpt-4.1-mini")


def test_judge_config_broadcast_needs_models():
    with pytest.raises(ValueError, match="models"):
        JudgeConfig(name="j", type="pointwise", execution={"mode": "broadcast"})
    JudgeConfig(
        name="j",
        type="pointwise",
        execution={"mode": "broadcast", "models": ["openai/gpt-4.1", "kimi/kimi-k3"]},
    )


def test_extract_json_plain_and_fenced_and_embedded():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here is it: {"a": {"b": 2}} hope it helps') == {"a": {"b": 2}}
    assert extract_json('{"s": "brace } in string"}') == {"s": "brace } in string"}
    with pytest.raises(ParseError):
        extract_json("no json here")


def test_eval_input_defaults():
    inp = EvalInput(output="hello")
    assert inp.prior_verdicts == []
    assert inp.metadata == {}
