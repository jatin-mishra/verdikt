"""BaseJudge: validate input -> render prompt -> call LLM -> parse -> Verdict.

Handles multi-sample voting. Execution modes (broadcast, judge-of-judges)
live in ``verdikt.execution`` and call ``evaluate_with_model`` per model.
"""
from __future__ import annotations

import abc
import asyncio
import statistics
import time
from collections import Counter
from typing import Any, Optional

from ..llm.client import LLMClient, LLMResponse
from ..prompts import render, render_inline
from .parsing import extract_json
from .schemas import EvalInput, JudgeConfig, JudgeMeta, Verdict


class BaseJudge(abc.ABC):
    template: str = "pointwise.j2"
    verdict_type: str = "score"

    # -- execution-mode compatibility (see verdikt.execution.modes) ----------
    # Score-based judges can use any consensus strategy. Label/comparison
    # judges override these: averaging labels or A/B/tie outcomes is invalid.
    supported_modes: tuple[str, ...] = ("single", "broadcast", "judge_of_judges")
    allowed_consensus: tuple[str, ...] = (
        "average",
        "weighted_average",
        "majority_vote",
        "unanimous",
        "consensus_leader",
    )
    default_consensus: str = "average"

    def __init__(self, config: JudgeConfig):
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    # -- hooks ---------------------------------------------------------------

    def required_fields(self) -> list[str]:
        return ["output"]

    def template_context(self, inp: EvalInput) -> dict[str, Any]:
        return {
            "input": inp.input,
            "output": inp.output,
            "expected_output": inp.expected_output,
            "context": inp.context,
            "criteria": self.config.criteria,
            "rubric": self.config.rubric,
            "labels": self.config.labels,
            "scale": self.config.scale,
            "few_shot": self.config.few_shot,
            "priors": inp.prior_verdicts,
            "trajectory": inp.trajectory,
            "metric": self.config.extra.get("metric"),
        }

    @abc.abstractmethod
    def parse(self, data: dict[str, Any], inp: EvalInput) -> Verdict:
        """Turn the judge model's JSON into a Verdict (without meta/threshold)."""

    # -- orchestration -------------------------------------------------------

    def validate_input(self, inp: EvalInput) -> None:
        missing = [f for f in self.required_fields() if getattr(inp, f, None) in (None, [], "")]
        if missing:
            raise ValueError(f"judge '{self.name}' requires input fields: {missing}")

    def build_prompt(self, inp: EvalInput, **overrides: Any) -> str:
        ctx = {**self.template_context(inp), **overrides}
        if self.config.prompt_template:
            return render_inline(self.config.prompt_template, **ctx)
        return render(self.template, **ctx)

    def apply_threshold(self, v: Verdict) -> Verdict:
        if v.error:
            return v
        if self.config.threshold is not None and v.score is not None:
            v.passed = v.score >= self.config.threshold
        if self.config.fail_on and v.label is not None:
            v.passed = v.label not in self.config.fail_on
        return v

    async def judge_once(
        self, inp: EvalInput, client: LLMClient, model: str, **prompt_overrides: Any
    ) -> tuple[Verdict, LLMResponse]:
        prompt = self.build_prompt(inp, **prompt_overrides)
        resp = await client.complete(
            model,
            [{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        data = extract_json(resp.text)
        verdict = self.parse(data, inp)
        verdict.judge_name = self.name
        verdict.verdict_type = self.verdict_type  # type: ignore[assignment]
        return verdict, resp

    async def evaluate_with_model(
        self, inp: EvalInput, client: LLMClient, model: str
    ) -> Verdict:
        """Run this judge on one model, with multi-sample voting if configured."""
        self.validate_input(inp)
        start = time.perf_counter()
        n = max(1, self.config.samples)
        results = await asyncio.gather(
            *(self.judge_once(inp, client, model) for _ in range(n)),
            return_exceptions=True,
        )
        verdicts: list[Verdict] = []
        responses: list[LLMResponse] = []
        errors: list[str] = []
        for r in results:
            if isinstance(r, BaseException):
                errors.append(str(r))
            else:
                verdicts.append(r[0])
                responses.append(r[1])

        meta = JudgeMeta(
            model=model,
            samples=n,
            latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=sum(r.input_tokens for r in responses),
            output_tokens=sum(r.output_tokens for r in responses),
            cost_usd=sum(r.cost_usd for r in responses),
        )
        if not verdicts:
            return Verdict(
                judge_name=self.name,
                verdict_type=self.verdict_type,  # type: ignore[arg-type]
                error="; ".join(errors) or "all samples failed",
                meta=meta,
            )
        final = self.aggregate_samples(verdicts) if len(verdicts) > 1 else verdicts[0]
        final.meta = meta
        return self.apply_threshold(final)

    def aggregate_samples(self, verdicts: list[Verdict]) -> Verdict:
        """Combine k samples of the same judge+model: median score, majority label."""
        base = verdicts[0].model_copy(deep=True)
        scores = [v.score for v in verdicts if v.score is not None]
        if scores:
            base.score = statistics.median(scores)
        labels = [v.label for v in verdicts if v.label is not None]
        if labels:
            base.label = Counter(labels).most_common(1)[0][0]
        confs = [v.confidence for v in verdicts if v.confidence is not None]
        if confs:
            base.confidence = sum(confs) / len(confs)
        base.reasoning = max((v.reasoning for v in verdicts), key=len)
        return base


class ScoreParsingMixin:
    """Shared parse() for judges whose JSON is {reasoning, score, confidence}."""

    def parse(self, data: dict[str, Any], inp: EvalInput) -> Verdict:  # type: ignore[override]
        config: JudgeConfig = self.config  # type: ignore[attr-defined]
        raw = data.get("score")
        if raw is None:
            raise ValueError(f"judge response missing 'score': {data}")
        score = config.scale.normalize(float(raw))
        return Verdict(
            judge_name=config.name,
            score=score,
            reasoning=str(data.get("reasoning", "")),
            confidence=_opt_float(data.get("confidence")),
        )


def _opt_float(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None
