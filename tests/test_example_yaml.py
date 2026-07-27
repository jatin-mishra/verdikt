"""Keep examples/verdikt.example.yaml valid: it must load, build, and run."""
from __future__ import annotations

from pathlib import Path

from conftest import FakeLLMClient, contains, label_response, score_response

from verdikt import EvalInput, Verdikt
from verdikt.config.loader import load_config

EXAMPLE = str(Path(__file__).parent.parent / "examples" / "verdikt.example.yaml")


def test_example_yaml_loads_and_builds(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # cache_path is relative; keep repo clean
    cfg = load_config(EXAMPLE)
    assert {j.name for j in cfg.judges} >= {
        "helpfulness", "correctness", "ab_test", "support_quality", "safety",
        "grounded", "ctx_relevance", "ans_relevance", "agent_run",
        "quality_panel", "weighted_panel", "safety_jury", "strict_safety",
        "led_panel", "final_review",
    }
    vd = Verdikt.from_config(cfg, client=FakeLLMClient())
    assert set(vd.pipelines) == {"support_bot_eval", "rag_eval"}
    # every judge type + execution mode in the example passed Executor validation
    assert len(vd.executors) == len(cfg.judges)


async def test_example_pipeline_runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def handler(model, messages):
        if contains(messages, "Allowed labels"):
            return label_response("safe")
        return score_response(5)

    cfg = load_config(EXAMPLE)
    vd = Verdikt.from_config(cfg, client=FakeLLMClient(handler=handler))
    pv = await vd.evaluate("support_bot_eval", EvalInput(input="q", output="a"))
    assert pv.passed is True
    # final_review skipped: quality_panel scored 1.0, run_if wanted < 0.9
    assert pv.skipped_steps == ["final_review"]
