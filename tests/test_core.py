from __future__ import annotations

import pytest

from verdikt import EvalInput, JudgeConfig, Scale, add_template_dir
from verdikt.core.parsing import ParseError, extract_json
from verdikt.prompts import _env, render


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
    JudgeConfig(name="j", type="pointwise", model="anthropic/claude-haiku")


def test_judge_config_broadcast_needs_models():
    with pytest.raises(ValueError, match="models"):
        JudgeConfig(name="j", type="pointwise", execution={"mode": "broadcast"})
    JudgeConfig(
        name="j",
        type="pointwise",
        execution={"mode": "broadcast", "models": ["anthropic/claude-sonnet-4-5", "gemini/gemini-2.5-pro"]},
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


def test_add_template_dir_registers_custom_templates(tmp_path):
    (tmp_path / "greeting.j2").write_text("hello {{ name }}")
    add_template_dir(tmp_path)
    try:
        assert render("greeting.j2", name="world") == "hello world"
    finally:
        _env.loader.searchpath.remove(str(tmp_path))
