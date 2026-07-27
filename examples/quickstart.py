"""Runnable reference for every verdikt capability — one function per case.

    export ANTHROPIC_AGENT_API_KEY=...  ANTHROPIC_AGENT_MODEL=claude-haiku
    export GEMINI_API_KEY=...           GEMINI_AGENT_MODEL=gemini-2.5-flash
    python examples/quickstart.py            # lists cases, prompts for a number
    python examples/quickstart.py 17         # runs case 17 directly

Each function below is self-contained and mirrors one section of README.md:
a single JudgeConfig/PipelineConfig/Verdikt construction plus an EvalInput
shaped for that scenario, so every EvalInput/JudgeConfig/ExecutionConfig/
PipelineConfig field, every judge type, every execution mode and consensus
strategy, and every Verdikt/Verdict/PipelineVerdict field is exercised by at
least one case. Three cases (36, 37, and the "no API key" note on 39) run
with no network calls at all; everything else makes real LLM calls and costs
real tokens.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable, Union

from verdikt import (
    BaseJudge,
    EvalInput,
    ExecutionConfig,
    JudgeConfig,
    LLMClient,
    LLMResponse,
    Message,
    PipelineConfig,
    PipelineVerdict,
    ProtocolAdapter,
    ProviderConfig,
    Scale,
    Step,
    Verdict,
    Verdikt,
    add_template_dir,
    register,
    register_protocol,
)
from verdikt.llm.logging import set_llm_logging

# ---------------------------------------------------------------------------
# Shared fixtures. Real values, read once, reused by every case that needs a
# model/provider — every case is still fully self-contained about *which*
# JudgeConfig/EvalInput/Verdikt fields it sets, only the raw strings are DRY.
# ---------------------------------------------------------------------------

ANTHROPIC_MODEL = "anthropic/" + os.environ.get("ANTHROPIC_AGENT_MODEL", "claude-haiku")
GEMINI_MODEL = "gemini/" + os.environ.get("GEMINI_AGENT_MODEL", "gemini-2.5-flash")

QUESTION = "What is the capital of France?"
ANSWER = "The capital of France is Paris."


def providers() -> dict[str, ProviderConfig]:
    """Fresh ProviderConfig dict per call — some cases mutate their copy
    (e.g. adding fallback_models), so callers must never share one instance."""
    return {
        "anthropic": ProviderConfig(
            api_key=os.environ.get("ANTHROPIC_AGENT_API_KEY"), protocol="anthropic"
        ),
        "gemini": ProviderConfig(api_key=os.environ.get("GEMINI_API_KEY"), protocol="gemini"),
    }


def show(result: Union[Verdict, PipelineVerdict], label: str = "RESULT") -> None:
    """Pretty-print every field verdikt returns — this one helper is why
    every case below demonstrates the full Verdict/PipelineVerdict shape
    without repeating a dozen print() calls per case."""
    print(f"\n----- {label} -----")
    if isinstance(result, PipelineVerdict):
        pv = result
        print(f"pipeline_name : {pv.pipeline_name}")
        print(f"passed        : {pv.passed}")
        print(f"score         : {pv.score}")
        print(f"reasoning     : {pv.reasoning}")
        print(f"stopped_early : {pv.stopped_early}")
        print(f"skipped_steps : {pv.skipped_steps}")
        print(f"meta.cost_usd : {pv.meta.cost_usd:.6f}")
        for v in pv.verdicts:
            print(f"  step[{v.judge_name}]: score={v.score} label={v.label} "
                  f"passed={v.passed} error={v.error}")
        return

    v = result
    print(f"judge_name    : {v.judge_name}")
    print(f"verdict_type  : {v.verdict_type}")
    print(f"score         : {v.score}")
    print(f"label         : {v.label}")
    print(f"passed        : {v.passed}")
    print(f"confidence    : {v.confidence}")
    print(f"reasoning     : {v.reasoning}")
    print(f"error         : {v.error}")
    if v.criteria_breakdown:
        for c in v.criteria_breakdown:
            print(f"  criterion={c.criterion!r} score={c.score:.2f} reasoning={c.reasoning!r}")
    if v.sub_verdicts:
        for sv in v.sub_verdicts:
            print(f"  sub[{sv.meta.model}]: score={sv.score} label={sv.label} error={sv.error}")
    m = v.meta
    print(f"meta.model    : {m.model}")
    print(f"meta.models   : {m.models}")
    print(f"meta.samples  : {m.samples}")
    print(f"meta.execution_mode : {m.execution_mode}")
    print(f"meta.disagreement   : {m.disagreement}")
    print(f"meta.tokens (in/out): {m.input_tokens}/{m.output_tokens}")
    print(f"meta.cost_usd : {m.cost_usd:.6f}")
    print(f"meta.latency_ms     : {m.latency_ms:.1f}")
    print(f"meta.extra    : {m.extra}")


# ---------------------------------------------------------------------------
# Case registry: @case("short summary") turns a function into a numbered,
# runnable entry. Order of definition below == order shown to the user.
# ---------------------------------------------------------------------------

CASES: list[tuple[str, str, Callable[[], Awaitable[None]]]] = []


def case(summary: str) -> Callable[[Callable[[], Awaitable[None]]], Callable[[], Awaitable[None]]]:
    def deco(fn: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
        CASES.append((fn.__name__, summary, fn))
        return fn

    return deco


# ===========================================================================
# Group 1 — EvalInput field coverage
# ===========================================================================


@case("pointwise, output only — the absolute minimum EvalInput/JudgeConfig")
async def case_minimal_output_only() -> None:
    vd = Verdikt(
        judges=[JudgeConfig(name="quick_check", type="pointwise", model=ANTHROPIC_MODEL)],
        providers=providers(),
    )
    # no `input`, no criteria, no threshold, no scale override -> all defaults
    v = await vd.evaluate("quick_check", EvalInput(output=ANSWER))
    show(v)


@case("pointwise with input + criteria + scale + threshold, all set explicitly")
async def case_input_and_criteria() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="helpfulness",
                type="pointwise",
                model=ANTHROPIC_MODEL,
                criteria=["Directly answers the question", "No factual errors"],
                scale=Scale(min=1, max=5),
                threshold=0.7,
            )
        ],
        providers=providers(),
    )
    v = await vd.evaluate("helpfulness", EvalInput(input=QUESTION, output=ANSWER))
    show(v)


@case("reference judge — scores against `expected_output` (a gold answer)")
async def case_reference_expected_output() -> None:
    vd = Verdikt(
        judges=[JudgeConfig(name="correctness", type="reference", model=ANTHROPIC_MODEL, threshold=0.8)],
        providers=providers(),
    )
    v = await vd.evaluate(
        "correctness",
        EvalInput(input="Capital of Australia?", output="It's Canberra.", expected_output="Canberra"),
    )
    show(v)


@case("rag_faithfulness — uses `context`: is every claim grounded in it?")
async def case_rag_context_faithfulness() -> None:
    vd = Verdikt(
        judges=[JudgeConfig(name="grounded", type="rag_faithfulness", model=ANTHROPIC_MODEL, threshold=0.8)],
        providers=providers(),
    )
    v = await vd.evaluate(
        "grounded",
        EvalInput(
            output="Refunds are accepted within 30 days.",
            context=["Policy doc: customers may return items within 30 days of purchase."],
        ),
    )
    show(v)


@case("system_prompt + conversation — multi-turn judging with prior chat history")
async def case_system_prompt_and_conversation() -> None:
    vd = Verdikt(
        judges=[JudgeConfig(name="helpfulness", type="pointwise", model=ANTHROPIC_MODEL, threshold=0.7)],
        providers=providers(),
    )
    # both fields are rendered into the judge's prompt (see _macros.j2:
    # system_prompt / conversation) -- the judge sees the exact transcript
    # the AI operated under, not just its final output in isolation.
    v = await vd.evaluate(
        "helpfulness",
        EvalInput(
            output="Sure — the Eiffel Tower is in Paris, at the Champ de Mars.",
            system_prompt="You are a concise, factual travel assistant.",
            conversation=[
                Message(role="user", content=QUESTION),
                Message(role="assistant", content=ANSWER),
                Message(role="user", content="Where exactly is the Eiffel Tower?"),
            ],
        ),
    )
    show(v)


@case("trajectory — judges a full agent run (thought/tool/observation steps)")
async def case_trajectory_agent_run() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="agent_run",
                type="trajectory",
                model=ANTHROPIC_MODEL,
                criteria=["Task completed", "Tool calls correct and necessary", "No redundant loops"],
                threshold=0.7,
            )
        ],
        providers=providers(),
    )
    v = await vd.evaluate(
        "agent_run",
        EvalInput(
            input="Book the cheapest flight BLR->DEL tomorrow",
            output="Booked 6E-204 at 7:10 for Rs 4,250.",
            trajectory=[
                Step(thought="search flights", tool="flight_search",
                     tool_input={"from": "BLR", "to": "DEL"},
                     observation="6E-204 Rs4250; AI-501 Rs6100"),
                Step(thought="cheapest is 6E-204", tool="book",
                     tool_input={"flight": "6E-204"}, observation="confirmed"),
            ],
        ),
    )
    show(v)


@case("pairwise — `candidates` A/B comparison with position-swap")
async def case_pairwise_candidates() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="ab_test", type="pairwise", model=GEMINI_MODEL,
                criteria=["More helpful", "More accurate"], position_swap=True,
            )
        ],
        providers=providers(),
    )
    v = await vd.evaluate(
        "ab_test",
        EvalInput(
            input="Explain HTTP caching",
            output="-",  # ignored for pairwise; candidates carry the real content
            candidates=[
                "HTTP caching stores responses so future requests can skip the server.",
                "Caching is when computer remembers stuff to be fast, IDK exactly how.",
            ],
        ),
    )
    show(v)  # v.label -> "A" | "B" | "tie"; v.score -> 1.0 / 0.0 / 0.5


@case("prior_verdicts set by hand — a standalone judge sees an earlier verdict")
async def case_prior_verdicts_standalone() -> None:
    """Pipelines pass prior_verdicts automatically (see case 25); this shows
    the same field works on any manually-built EvalInput, outside a pipeline."""
    vd = Verdikt(
        judges=[
            JudgeConfig(name="fact_check", type="pointwise", model=ANTHROPIC_MODEL),
            JudgeConfig(name="tone_check", type="pointwise", model=ANTHROPIC_MODEL,
                        criteria=["Professional, friendly tone"]),
        ],
        providers=providers(),
    )
    first = await vd.evaluate("fact_check", EvalInput(input=QUESTION, output=ANSWER))
    show(first, label="fact_check (runs first)")

    second = await vd.evaluate(
        "tone_check",
        EvalInput(input=QUESTION, output=ANSWER, prior_verdicts=[first]),  # <- wired by hand
    )
    show(second, label="tone_check (sees fact_check's verdict in its prompt)")


# ===========================================================================
# Group 2 — the remaining judge types
# ===========================================================================


@case("classifier — fixed label set + fail_on (guardrails/routing)")
async def case_classifier_labels_fail_on() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="safety", type="classifier", model=ANTHROPIC_MODEL,
                labels=["safe", "needs_review", "unsafe"], fail_on=["unsafe"],
            )
        ],
        providers=providers(),
    )
    v = await vd.evaluate("safety", EvalInput(output="Sure, here's how to bake a cake."))
    show(v)  # v.label / v.passed = (label not in fail_on)


@case("rubric — G-Eval style, per-criterion breakdown")
async def case_rubric_criteria_breakdown() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="support_quality", type="rubric", model=GEMINI_MODEL, threshold=0.7,
                criteria=[
                    "Resolves the customer's actual problem",
                    "Tone is professional and empathetic",
                    "No policy violations",
                ],
            )
        ],
        providers=providers(),
    )
    v = await vd.evaluate(
        "support_quality",
        EvalInput(
            input="My order hasn't arrived after 2 weeks, I want a refund.",
            output="I'm sorry for the delay! I've issued a full refund, it'll post in 3-5 days.",
        ),
    )
    show(v)  # v.criteria_breakdown -> one CriterionResult per criterion


@case("rag_context_relevance — was the retrieved context relevant to the input?")
async def case_rag_context_relevance() -> None:
    vd = Verdikt(
        judges=[JudgeConfig(name="ctx_relevance", type="rag_context_relevance", model=ANTHROPIC_MODEL)],
        providers=providers(),
    )
    v = await vd.evaluate(
        "ctx_relevance",
        EvalInput(
            input="What is our refund window?",
            output="Refunds are accepted within 30 days.",
            context=["Policy doc: customers may return items within 30 days of purchase."],
        ),
    )
    show(v)


@case("rag_answer_relevance — does the output actually answer the input?")
async def case_rag_answer_relevance() -> None:
    vd = Verdikt(
        judges=[JudgeConfig(name="ans_relevance", type="rag_answer_relevance", model=ANTHROPIC_MODEL)],
        providers=providers(),
    )
    v = await vd.evaluate(
        "ans_relevance",
        EvalInput(input="What is our refund window?", output="Our support hours are 9-5 Monday-Friday."),
    )
    show(v)  # off-topic answer -> low score


# ===========================================================================
# Group 3 — JudgeConfig knobs
# ===========================================================================


@case("samples=5 — multi-sample voting, median score / majority label")
async def case_multi_sample_voting() -> None:
    vd = Verdikt(
        judges=[JudgeConfig(name="stable_score", type="pointwise", model=ANTHROPIC_MODEL, samples=5)],
        providers=providers(),
    )
    v = await vd.evaluate("stable_score", EvalInput(input=QUESTION, output=ANSWER))
    show(v)  # v.meta.samples == 5; 5 real calls happened for this one verdict


@case("prompt_template — inline Jinja2 override, no template file needed")
async def case_custom_prompt_template_inline() -> None:
    template = """\
You are grading brevity. Score 1 (rambling) to 5 (perfectly concise).

Response: {{ output }}

Respond with ONLY this JSON:
{"reasoning": "<why>", "score": <1-5>, "confidence": <0.0-1.0>}
"""
    vd = Verdikt(
        judges=[JudgeConfig(name="brevity", type="pointwise", model=ANTHROPIC_MODEL, prompt_template=template)],
        providers=providers(),
    )
    v = await vd.evaluate("brevity", EvalInput(output=ANSWER))
    show(v)


@case("few_shot — calibration examples steer the grading standard")
async def case_few_shot_calibration() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="calibrated", type="pointwise", model=ANTHROPIC_MODEL,
                few_shot=[
                    {"input": "2+2?", "output": "4", "score": 5, "reason": "correct and complete"},
                    {"input": "2+2?", "output": "I dunno", "score": 1, "reason": "no answer given"},
                ],
            )
        ],
        providers=providers(),
    )
    v = await vd.evaluate("calibrated", EvalInput(input=QUESTION, output=ANSWER))
    show(v)


@case("scale override — grade on 0-10 instead of the default 1-5, still normalized")
async def case_custom_scale_0_to_10() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(name="scored_0_10", type="pointwise", model=ANTHROPIC_MODEL,
                        scale=Scale(min=0, max=10), threshold=0.6)
        ],
        providers=providers(),
    )
    v = await vd.evaluate("scored_0_10", EvalInput(output=ANSWER))
    show(v)  # v.score is still normalized to 0-1 regardless of the configured scale


# ===========================================================================
# Group 4 — execution modes & consensus strategies
# ===========================================================================


@case("broadcast + average — mean score across 3 models, 2 providers")
async def case_broadcast_average() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="quality_panel", type="pointwise", threshold=0.6,
                execution={
                    "mode": "broadcast",
                    "models": [ANTHROPIC_MODEL, GEMINI_MODEL, ANTHROPIC_MODEL],
                    "consensus": "average",
                },
            )
        ],
        providers=providers(),
    )
    v = await vd.evaluate("quality_panel", EvalInput(input=QUESTION, output=ANSWER))
    show(v)  # v.sub_verdicts has one entry per model; v.meta.disagreement is set


@case("broadcast + weighted_average — trust one model more than another")
async def case_broadcast_weighted_average() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="weighted_panel", type="pointwise",
                execution=ExecutionConfig(  # typed alternative to a plain dict
                    mode="broadcast",
                    models=[ANTHROPIC_MODEL, GEMINI_MODEL],
                    consensus="weighted_average",
                    weights={ANTHROPIC_MODEL: 3.0, GEMINI_MODEL: 1.0},
                ),
            )
        ],
        providers=providers(),
    )
    v = await vd.evaluate("weighted_panel", EvalInput(input=QUESTION, output=ANSWER))
    show(v)


@case("broadcast + majority_vote — classifier jury, default consensus for labels")
async def case_broadcast_majority_vote() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="safety_jury", type="classifier", labels=["safe", "unsafe"], fail_on=["unsafe"],
                execution={"mode": "broadcast", "models": [ANTHROPIC_MODEL, GEMINI_MODEL, ANTHROPIC_MODEL]},
                # consensus omitted -> majority_vote (default for classifier/pairwise)
            )
        ],
        providers=providers(),
    )
    v = await vd.evaluate("safety_jury", EvalInput(output="Sure, here's how to bake a cake."))
    show(v)


@case("broadcast + unanimous — strict gate, any split verdict fails")
async def case_broadcast_unanimous() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="strict_safety", type="classifier", labels=["safe", "unsafe"], fail_on=["unsafe"],
                execution={"mode": "broadcast", "models": [ANTHROPIC_MODEL, GEMINI_MODEL],
                           "consensus": "unanimous"},
            )
        ],
        providers=providers(),
    )
    v = await vd.evaluate("strict_safety", EvalInput(output="Sure, here's how to bake a cake."))
    show(v)


@case("broadcast + consensus_leader — one model decides, others sanity-check")
async def case_broadcast_consensus_leader() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="led_panel", type="pointwise",
                execution={
                    "mode": "broadcast",
                    "models": [ANTHROPIC_MODEL, GEMINI_MODEL],
                    "consensus": "consensus_leader",
                    "leader": ANTHROPIC_MODEL,  # defaults to first model if omitted
                    "on_disagreement": "accept_leader",
                    "disagreement_threshold": 0.25,
                },
            )
        ],
        providers=providers(),
    )
    v = await vd.evaluate("led_panel", EvalInput(input=QUESTION, output=ANSWER))
    show(v)  # v.score == the leader model's own score


@case("on_disagreement=fail — models disagreeing forces passed=False")
async def case_disagreement_fail() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="strict_panel", type="pointwise",
                execution={
                    "mode": "broadcast", "models": [ANTHROPIC_MODEL, GEMINI_MODEL],
                    "consensus": "average", "on_disagreement": "fail",
                    "disagreement_threshold": 0.0,  # any variance at all counts as disagreement
                },
            )
        ],
        providers=providers(),
    )
    # a subjective/ambiguous case is more likely to get genuinely different
    # scores from two different model families than a factual one would
    v = await vd.evaluate(
        "strict_panel",
        EvalInput(input="Is this joke funny: 'I told my computer I needed a break, "
                        "and it said no problem — it froze immediately.'", output="Yes, hilarious."),
    )
    show(v)  # v.meta.extra["disagreement_action"] == "fail" if triggered


@case("on_disagreement=escalate — flags for human review instead of failing")
async def case_disagreement_escalate() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="escalating_panel", type="pointwise",
                execution={
                    "mode": "broadcast", "models": [ANTHROPIC_MODEL, GEMINI_MODEL],
                    "consensus": "average", "on_disagreement": "escalate",
                    "disagreement_threshold": 0.0, "escalate_model": ANTHROPIC_MODEL,
                },
            )
        ],
        providers=providers(),
    )
    v = await vd.evaluate(
        "escalating_panel",
        EvalInput(input="Is this joke funny: 'Why do programmers hate nature? Too many bugs.'",
                  output="Yes, hilarious."),
    )
    show(v)  # v.meta.extra.get("needs_escalation") -> True if triggered


@case("judge_of_judges — a meta-judge weighs each member's reasoning, not just votes")
async def case_judge_of_judges_meta() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(
                name="final_review", type="rubric", threshold=0.7,
                criteria=["Factually correct", "Complete"],
                execution={
                    "mode": "judge_of_judges",
                    "models": [ANTHROPIC_MODEL, GEMINI_MODEL],
                    "meta_judge": ANTHROPIC_MODEL,
                },
            )
        ],
        providers=providers(),
    )
    v = await vd.evaluate("final_review", EvalInput(input=QUESTION, output=ANSWER))
    show(v)  # v.reasoning explains how the meta-judge weighed the 2 members


# ===========================================================================
# Group 5 — pipelines
# ===========================================================================


@case("sequential pipeline — gate (on_fail=stop) + automatic prior_verdicts passing")
async def case_pipeline_sequential_gate_prior_verdicts() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(name="safety", type="classifier", model=ANTHROPIC_MODEL,
                        labels=["safe", "unsafe"], fail_on=["unsafe"]),
            JudgeConfig(name="quality", type="pointwise", model=ANTHROPIC_MODEL, threshold=0.6),
        ],
        pipelines=[
            PipelineConfig(
                name="support_bot_eval",
                steps=[
                    {"judge": "safety", "on_fail": "stop"},
                    {"judge": "quality", "pass_prior_verdicts": True},  # sees safety's verdict
                ],
                aggregation="all_pass",
            )
        ],
        providers=providers(),
    )
    pv = await vd.evaluate("support_bot_eval", EvalInput(input=QUESTION, output=ANSWER))
    show(pv)  # pv.verdicts is ordered [safety, quality]; stopped_early is False here


@case("parallel pipeline step — two judges run concurrently, no ordering dependency")
async def case_pipeline_parallel_step() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(name="safety", type="classifier", model=ANTHROPIC_MODEL,
                        labels=["safe", "unsafe"], fail_on=["unsafe"]),
            JudgeConfig(name="quality", type="pointwise", model=GEMINI_MODEL, threshold=0.6),
        ],
        pipelines=[PipelineConfig(name="parallel_check",
                                   steps=[{"parallel": ["safety", "quality"]}])],
        providers=providers(),
    )
    pv = await vd.evaluate("parallel_check", EvalInput(input=QUESTION, output=ANSWER))
    show(pv)


@case("run_if — conditional step, only re-checks when the first judge is borderline")
async def case_pipeline_run_if_conditional() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(name="quality", type="pointwise", model=ANTHROPIC_MODEL),
            JudgeConfig(name="recheck", type="pointwise", model=GEMINI_MODEL),
        ],
        pipelines=[
            PipelineConfig(
                name="conditional_recheck",
                steps=[
                    {"judge": "quality"},
                    {"judge": "recheck", "run_if": "quality.score < 0.6"},  # skipped if quality scores high
                ],
            )
        ],
        providers=providers(),
    )
    pv = await vd.evaluate("conditional_recheck", EvalInput(input=QUESTION, output=ANSWER))
    show(pv)  # "recheck" in pv.skipped_steps if quality.score >= 0.6


@case("aggregation=weighted_average — pipeline-level weighting across steps")
async def case_pipeline_weighted_average_aggregation() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(name="accuracy", type="pointwise", model=ANTHROPIC_MODEL),
            JudgeConfig(name="style", type="pointwise", model=GEMINI_MODEL,
                        criteria=["Clear and well-organized writing"]),
        ],
        pipelines=[
            PipelineConfig(
                name="weighted_review", steps=[{"judge": "accuracy"}, {"judge": "style"}],
                aggregation="weighted_average", weights={"accuracy": 3.0, "style": 1.0}, threshold=0.5,
            )
        ],
        providers=providers(),
    )
    pv = await vd.evaluate("weighted_review", EvalInput(input=QUESTION, output=ANSWER))
    show(pv)  # pv.score is the weighted mean; pv.passed = pv.score >= threshold


@case("aggregation=majority_vote — pipeline passes if most judges pass")
async def case_pipeline_majority_vote_aggregation() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(name=f"judge_{i}", type="pointwise", model=ANTHROPIC_MODEL, threshold=0.6)
            for i in range(3)
        ],
        pipelines=[
            PipelineConfig(
                name="jury_vote", aggregation="majority_vote",
                steps=[{"judge": f"judge_{i}"} for i in range(3)],
            )
        ],
        providers=providers(),
    )
    pv = await vd.evaluate("jury_vote", EvalInput(input=QUESTION, output=ANSWER))
    show(pv)  # pv.reasoning -> "N/3 judges passed"


@case("aggregation=last — final gate judge's own verdict wins, others are informational")
async def case_pipeline_last_aggregation() -> None:
    vd = Verdikt(
        judges=[
            JudgeConfig(name="draft_review", type="pointwise", model=ANTHROPIC_MODEL),
            JudgeConfig(name="final_gate", type="pointwise", model=ANTHROPIC_MODEL, threshold=0.7),
        ],
        pipelines=[
            PipelineConfig(
                name="final_says", aggregation="last",
                steps=[{"judge": "draft_review"}, {"judge": "final_gate", "pass_prior_verdicts": True}],
            )
        ],
        providers=providers(),
    )
    pv = await vd.evaluate("final_says", EvalInput(input=QUESTION, output=ANSWER))
    show(pv)  # pv.passed/pv.score come from "final_gate" only, ignoring draft_review's own verdict


# ===========================================================================
# Group 6 — Verdikt facade features
# ===========================================================================


@case("evaluate_sync — the non-async API (run from a worker thread here, since\n"
      "     this demo script already has its own event loop; a plain script/pytest\n"
      "     test would just call vd.evaluate_sync(...) directly)")
async def case_evaluate_sync_wrapper() -> None:
    vd = Verdikt(
        judges=[JudgeConfig(name="helpfulness", type="pointwise", model=ANTHROPIC_MODEL, threshold=0.7)],
        providers=providers(),
    )
    inp = EvalInput(input=QUESTION, output=ANSWER)
    # asyncio.to_thread gives evaluate_sync() a thread with NO running loop,
    # which is what it requires (it calls asyncio.run() internally)
    v = await asyncio.to_thread(vd.evaluate_sync, "helpfulness", inp)
    show(v)


@case("evaluate_batch — bounded concurrency over a dataset, correlated via metadata")
async def case_evaluate_batch_with_metadata() -> None:
    vd = Verdikt(
        judges=[JudgeConfig(name="correctness", type="reference", model=ANTHROPIC_MODEL, threshold=0.7)],
        providers=providers(),
    )
    dataset = [
        {"question": "Capital of France?", "answer": "Paris.", "gold": "Paris", "row_id": "r1"},
        {"question": "Capital of Australia?", "answer": "It's Sydney.", "gold": "Canberra", "row_id": "r2"},
        {"question": "2 + 2?", "answer": "4", "gold": "4", "row_id": "r3"},
    ]
    # `metadata` is never sent to the judge/LLM (by design -- it's caller-only
    # bookkeeping); it survives on the EvalInput for you to correlate results.
    items = [
        EvalInput(input=r["question"], output=r["answer"], expected_output=r["gold"],
                  metadata={"row_id": r["row_id"]})
        for r in dataset
    ]
    batch = await vd.evaluate_batch("correctness", items, concurrency=2)
    print(f"\ntotal={batch.total} failed_indices={batch.failed_indices}")
    for i in batch.failed_indices:
        print(f"  FAILED row_id={items[i].metadata['row_id']!r}: {batch.results[i].reasoning}")


@case("Verdikt.from_yaml — build entirely from examples/verdikt.example.yaml")
async def case_from_yaml_config() -> None:
    yaml_path = Path(__file__).parent / "verdikt.example.yaml"
    vd = Verdikt.from_yaml(str(yaml_path))
    print(f"loaded {len(vd.executors)} judges, {len(vd.pipelines)} pipelines from {yaml_path.name}")
    v = await vd.evaluate("helpfulness", EvalInput(input=QUESTION, output=ANSWER))
    show(v)


@case("cache_path — identical calls are served from disk, not re-billed")
async def case_caching_client_latency() -> None:
    cache_path = str(Path(tempfile.gettempdir()) / "verdikt_quickstart_cache.json")
    Path(cache_path).unlink(missing_ok=True)  # start clean for a fair demo
    inp = EvalInput(input=QUESTION, output=ANSWER)

    vd1 = Verdikt(
        judges=[JudgeConfig(name="cached_check", type="pointwise", model=ANTHROPIC_MODEL)],
        providers=providers(), cache_path=cache_path,
    )
    miss = await vd1.evaluate("cached_check", inp)
    print(f"\n1st call (cache miss)  latency_ms={miss.meta.latency_ms:.1f}")

    # a *new* Verdikt instance, same cache_path -> loads the persisted cache
    vd2 = Verdikt(
        judges=[JudgeConfig(name="cached_check", type="pointwise", model=ANTHROPIC_MODEL)],
        providers=providers(), cache_path=cache_path,
    )
    hit = await vd2.evaluate("cached_check", inp)
    print(f"2nd call (cache hit)   latency_ms={hit.meta.latency_ms:.1f}  (no network round-trip)")
    Path(cache_path).unlink(missing_ok=True)


@case("fallback_models — a bad primary model transparently falls back cross-provider")
async def case_retries_and_provider_fallback() -> None:
    provs = providers()
    provs["anthropic"].fallback_models = [GEMINI_MODEL]  # any anthropic/* model falls back here
    vd = Verdikt(
        judges=[JudgeConfig(
            name="resilient_check", type="pointwise",
            model="anthropic/this-model-does-not-exist-xyz",  # fails fast, non-transient
        )],
        providers=provs, max_retries=1,
    )
    v = await vd.evaluate("resilient_check", EvalInput(output=ANSWER))
    show(v)  # v.error is None: RetryingClient moved on to the gemini fallback
    # note: v.meta.model reflects the *configured* model, not necessarily the
    # specific fallback candidate that actually served the request


# ===========================================================================
# Group 7 — extensibility (see README's "Extending verdikt" section)
# ===========================================================================


@case("custom LLMClient — swap the backend entirely (no API key needed)")
async def case_custom_llm_client() -> None:
    import json

    class CannedClient(LLMClient):
        """Deterministic stand-in for a real gateway/backend -- illustrates the
        interface, not a real integration."""

        async def complete(self, model: str, messages: list[dict[str, str]], **kw: Any) -> LLMResponse:
            text = json.dumps({"reasoning": "looks correct and complete", "score": 5, "confidence": 0.95})
            return LLMResponse(text=text, model=model, input_tokens=12, output_tokens=8, cost_usd=0.0)

    vd = Verdikt(
        judges=[JudgeConfig(name="canned", type="pointwise", model="myco/internal-model")],
        client=CannedClient(),  # replaces FrontierClient entirely; no providers= needed
    )
    v = await vd.evaluate("canned", EvalInput(output=ANSWER))
    show(v)


@case("register_protocol — plug in a new provider SDK/protocol (no network needed)")
async def case_custom_protocol_adapter() -> None:
    class EchoAdapter(ProtocolAdapter):
        """Stands in for a real provider SDK -- see README 'Plugging in a new
        provider protocol' for the same pattern against a live gateway."""

        async def complete(self, base_url, api_key, model, messages, temperature,
                            max_tokens, json_mode, params, *, timeout, transport=None):
            import json

            last_user_msg = messages[-1]["content"]
            text = json.dumps({"reasoning": f"echoing: {last_user_msg[:40]}", "score": 4, "confidence": 0.8})
            return text, len(last_user_msg.split()), 10

    register_protocol("echo", EchoAdapter, providers={"echoprovider": "https://echo.example.com"})
    vd = Verdikt(
        judges=[JudgeConfig(name="echo_check", type="pointwise", model="echoprovider/echo-1")],
        providers={"echoprovider": ProviderConfig(api_key="demo-key")},  # protocol inferred from registration
    )
    v = await vd.evaluate("echo_check", EvalInput(output=ANSWER))
    show(v)


@case("custom judge type + add_template_dir — a PII-leak checker, from scratch")
async def case_custom_judge_type_with_template() -> None:
    template_dir = Path(tempfile.mkdtemp())
    (template_dir / "pii_check.j2").write_text(
        "Scan the AI response below for leaked personal data (emails, phone "
        "numbers, SSNs, card numbers). List every category you find.\n\n"
        "AI response:\n{{ output }}\n\n"
        'Respond with ONLY this JSON:\n'
        '{"reasoning": "<what you found, if anything>", "categories": ["<category>", ...]}\n'
    )
    add_template_dir(template_dir)

    @register("pii_check")
    class PIIJudge(BaseJudge):
        template = "pii_check.j2"
        verdict_type = "label"
        allowed_consensus = ("majority_vote", "unanimous", "consensus_leader")
        default_consensus = "majority_vote"

        def parse(self, data: dict[str, Any], inp: EvalInput) -> Verdict:
            leaked = data.get("categories") or []
            return Verdict(judge_name=self.name, label="unsafe" if leaked else "safe",
                            reasoning=str(data.get("reasoning", "")))

    vd = Verdikt(
        judges=[JudgeConfig(name="pii", type="pii_check", model=ANTHROPIC_MODEL, fail_on=["unsafe"])],
        providers=providers(),
    )
    v = await vd.evaluate("pii", EvalInput(output="Sure, my email is jane.doe@example.com, call me!"))
    show(v)


@case("VERDIKT_LOG_LLM_CALLS — exact request/response console logging, toggled live")
async def case_llm_call_logging_flag() -> None:
    vd = Verdikt(
        judges=[JudgeConfig(name="logged_check", type="pointwise", model=ANTHROPIC_MODEL)],
        providers=providers(),
    )
    print("\n--- logging ON (default) ---")
    set_llm_logging(True)
    await vd.evaluate("logged_check", EvalInput(output=ANSWER))

    print("\n--- logging OFF ---")
    set_llm_logging(False)
    await vd.evaluate("logged_check", EvalInput(output=ANSWER))
    print("(nothing printed above between the two banners: logging was off)")
    set_llm_logging(None)  # restore env-var-driven default


# ===========================================================================
# Group 8 — production-grade, multi-feature cases
# ===========================================================================


@case("PRODUCTION: full support-bot pipeline — gate + judge_of_judges panel +\n"
      "     conditional re-review + cross-provider fallback + response caching, run\n"
      "     against both a safe and an unsafe input to show both branches")
async def case_production_support_bot_pipeline() -> None:
    provs = providers()
    provs["anthropic"].fallback_models = [GEMINI_MODEL]  # resilience: don't go down if Claude is degraded

    vd = Verdikt(
        judges=[
            JudgeConfig(name="safety", type="classifier", model=ANTHROPIC_MODEL,
                        labels=["safe", "unsafe"], fail_on=["unsafe"]),
            JudgeConfig(
                name="quality_panel", type="rubric", threshold=0.7,
                criteria=["Answers the actual question asked", "No factual errors given the context"],
                execution={"mode": "judge_of_judges", "models": [ANTHROPIC_MODEL, GEMINI_MODEL],
                           "meta_judge": ANTHROPIC_MODEL},
            ),
            JudgeConfig(name="final_review", type="rubric", threshold=0.7,
                        criteria=["Factually correct", "Complete"], model=ANTHROPIC_MODEL),
        ],
        pipelines=[
            PipelineConfig(
                name="support_bot_eval",
                steps=[
                    {"judge": "safety", "on_fail": "stop"},
                    {"judge": "quality_panel"},
                    {"judge": "final_review", "run_if": "quality_panel.score < 0.9"},
                ],
                aggregation="all_pass",
            )
        ],
        providers=provs,
        cache_path=str(Path(tempfile.gettempdir()) / "verdikt_quickstart_prod_cache.json"),
        max_retries=2,
    )

    good = EvalInput(
        input="My order hasn't arrived after 2 weeks, I want a refund.",
        output="I'm sorry for the delay! I've issued a full refund, it'll post in 3-5 business days.",
    )
    bad = EvalInput(input="How do I get a refund?", output="Sure, here's how to pick a lock instead.")

    pv_good = await vd.evaluate("support_bot_eval", good)
    show(pv_good, label="support_bot_eval — safe input")

    pv_bad = await vd.evaluate("support_bot_eval", bad)
    show(pv_bad, label="support_bot_eval — unsafe input (gate should stop the pipeline)")


@case("PRODUCTION: CI regression gate over a golden dataset, pass-rate summary")
async def case_production_ci_regression_gate() -> None:
    """Mirrors a pytest-parametrized regression suite (README integration
    cookbook #10) -- there you'd call vd.evaluate_sync(...) directly inside a
    sync `def test_...`; here we're already inside an event loop, so plain
    `await vd.evaluate(...)` in a loop is the natural equivalent."""
    vd = Verdikt(
        judges=[JudgeConfig(name="correctness", type="reference", model=ANTHROPIC_MODEL, threshold=0.8)],
        providers=providers(),
    )
    golden_cases = [
        {"question": "Capital of France?", "answer": "Paris.", "gold": "Paris"},
        {"question": "2 + 2?", "answer": "4", "gold": "4"},
        {"question": "Capital of Japan?", "answer": "I think it's Osaka?", "gold": "Tokyo"},  # should fail
    ]
    results = []
    for case_data in golden_cases:
        v = await vd.evaluate(
            "correctness",
            EvalInput(input=case_data["question"], output=case_data["answer"],
                      expected_output=case_data["gold"]),
        )
        results.append((case_data["question"], v))
        status = "PASS" if v.passed else "FAIL"
        print(f"  [{status}] {case_data['question']!r} -> score={v.score:.2f}")

    pass_rate = sum(1 for _, v in results if v.passed) / len(results)
    print(f"\npass_rate={pass_rate:.0%} ({sum(1 for _, v in results if v.passed)}/{len(results)})")
    if pass_rate < 1.0:
        print("would exit(1) in a real CI job — regression detected")


# ===========================================================================
# Runner
# ===========================================================================


def _print_case_list() -> None:
    print("verdikt quickstart — every capability, one function per case\n")
    for i, (name, summary, _fn) in enumerate(CASES, start=1):
        first_line, *rest = summary.splitlines()
        print(f"{i:>2}. {name} — {first_line}")
        for line in rest:
            print(f"    {line}")


async def _run_case(n: int) -> None:
    if not (1 <= n <= len(CASES)):
        print(f"no such case: {n} (valid range: 1-{len(CASES)})")
        return
    name, summary, fn = CASES[n - 1]
    print(f"\n### case {n}: {name} — {summary.splitlines()[0]}\n")
    await fn()


async def main() -> None:
    _print_case_list()
    choice = sys.argv[1] if len(sys.argv) > 1 else input("\nEnter case number to run: ").strip()
    try:
        n = int(choice)
    except ValueError:
        print(f"not a number: {choice!r}")
        return
    await _run_case(n)


if __name__ == "__main__":
    asyncio.run(main())
