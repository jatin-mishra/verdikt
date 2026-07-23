"""Executor: runs a judge in single, broadcast, or judge_of_judges mode."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from ..core.base import BaseJudge, _opt_float
from ..core.parsing import extract_json
from ..core.schemas import CriterionResult, EvalInput, JudgeMeta, Verdict
from ..llm.client import LLMClient
from ..prompts import render
from .consensus import combine, disagreement_score


def _merge_breakdowns(members: list[Verdict]) -> list[CriterionResult] | None:
    """Average per-criterion scores across broadcast members (rubric judges)."""
    per: dict[str, list[float]] = {}
    for v in members:
        if v.ok and v.criteria_breakdown:
            for c in v.criteria_breakdown:
                per.setdefault(c.criterion, []).append(c.score)
    if not per:
        return None
    return [
        CriterionResult(
            criterion=name,
            score=sum(scores) / len(scores),
            reasoning=f"mean across {len(scores)} model(s)",
        )
        for name, scores in per.items()
    ]


class Executor:
    """Runs a judge in its configured execution mode.

    Validates judge-type / execution-mode compatibility at construction:
    - mode must be in ``judge.supported_modes``
    - an explicitly configured consensus must be in ``judge.allowed_consensus``
      (e.g. 'average' is rejected for classifier/pairwise judges)
    - if consensus was NOT explicitly configured, the judge type's
      ``default_consensus`` is used (majority_vote for label/comparison
      judges, average for score judges) instead of the schema default.
    """

    def __init__(self, judge: BaseJudge, client: LLMClient):
        self.judge = judge
        self.client = client
        self.cfg = judge.config.execution.model_copy(deep=True)
        self._validate_compatibility()

    def _validate_compatibility(self) -> None:
        cfg_in = self.judge.config.execution
        if self.cfg.mode not in self.judge.supported_modes:
            raise ValueError(
                f"judge '{self.judge.name}' (type={self.judge.config.type}) does not "
                f"support execution mode '{self.cfg.mode}'; supported: "
                f"{list(self.judge.supported_modes)}"
            )
        if self.cfg.mode == "single":
            return
        explicit = "consensus" in cfg_in.model_fields_set
        if explicit and self.cfg.consensus not in self.judge.allowed_consensus:
            raise ValueError(
                f"judge '{self.judge.name}' (type={self.judge.config.type}, "
                f"verdict_type={self.judge.verdict_type}) cannot use consensus "
                f"'{self.cfg.consensus}': labels/comparisons cannot be averaged. "
                f"Valid options: {list(self.judge.allowed_consensus)}"
            )
        if not explicit:
            self.cfg.consensus = self.judge.default_consensus  # type: ignore[assignment]
        if self.cfg.consensus == "consensus_leader" and not self.cfg.leader:
            self.cfg.leader = self._models()[0] if self._models() else None

    async def evaluate(self, inp: EvalInput) -> Verdict:
        try:
            if self.cfg.mode == "single":
                return await self.judge.evaluate_with_model(
                    inp, self.client, self.judge.config.model  # type: ignore[arg-type]
                )
            return await self._broadcast(inp, meta_judge=self.cfg.mode == "judge_of_judges")
        except Exception as exc:  # noqa: BLE001 - never crash the caller's agent
            return Verdict(judge_name=self.judge.name, error=str(exc))

    # ------------------------------------------------------------------

    def _models(self) -> list[str]:
        models = list(self.cfg.models)
        if not models and self.judge.config.model:
            models = [self.judge.config.model]
        return models

    async def _broadcast(self, inp: EvalInput, meta_judge: bool) -> Verdict:
        start = time.perf_counter()
        models = self._models()
        results = await asyncio.gather(
            *(self.judge.evaluate_with_model(inp, self.client, m) for m in models),
            return_exceptions=True,
        )
        members: list[Verdict] = []
        for model, r in zip(models, results):
            if isinstance(r, BaseException):
                members.append(
                    Verdict(judge_name=self.judge.name, error=str(r), meta=JudgeMeta(model=model))
                )
            else:
                members.append(r)

        disagreement = disagreement_score(members)

        if meta_judge:
            final = await self._meta_judge(inp, members)
        else:
            final = combine(members, self.cfg, self.judge.name)
            final = self._handle_disagreement(final, members, disagreement)

        final.sub_verdicts = members
        final.verdict_type = members[0].verdict_type if members else final.verdict_type
        if final.criteria_breakdown is None:  # e.g. rubric judges: keep per-criterion detail
            final.criteria_breakdown = _merge_breakdowns(members)
        final.meta = JudgeMeta(
            models=models,
            execution_mode=self.cfg.mode,
            disagreement=disagreement,
            latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=sum(v.meta.input_tokens for v in members) + final.meta.input_tokens,
            output_tokens=sum(v.meta.output_tokens for v in members) + final.meta.output_tokens,
            cost_usd=sum(v.meta.cost_usd for v in members) + final.meta.cost_usd,
            extra=final.meta.extra,
        )
        return self.judge.apply_threshold(final)

    def _handle_disagreement(
        self, final: Verdict, members: list[Verdict], disagreement: float | None
    ) -> Verdict:
        if disagreement is None or disagreement <= self.cfg.disagreement_threshold:
            return final
        action = self.cfg.on_disagreement
        final.meta.extra["disagreement_action"] = action
        if action == "fail":
            final.passed = False
            final.reasoning = (
                f"models disagree (disagreement={disagreement:.2f} > "
                f"{self.cfg.disagreement_threshold}) -> failed. " + final.reasoning
            )
        elif action == "accept_leader":
            leader_model = self.cfg.leader or (self.cfg.models[0] if self.cfg.models else "")
            leader = next((v for v in members if v.meta.model == leader_model and v.ok), None)
            if leader:
                final.score, final.label = leader.score, leader.label
                final.reasoning = f"[disagreement -> leader {leader_model}] {leader.reasoning}"
        elif action == "escalate":
            final.meta.extra["needs_escalation"] = True
            final.meta.extra["escalate_model"] = self.cfg.escalate_model
        return final

    async def _meta_judge(self, inp: EvalInput, members: list[Verdict]) -> Verdict:
        model = self.cfg.meta_judge or self._models()[0]
        ok_members = [v for v in members if v.ok]
        if not ok_members:
            return combine(members, self.cfg, self.judge.name)
        task = (
            f"type={self.judge.config.type}; "
            f"criteria={self.judge.config.criteria or 'overall quality'}"
        )
        prompt = render(
            "meta_judge.j2",
            task_description=task,
            input=inp.input,
            output=inp.output,
            sub_verdicts=ok_members,
        )
        resp = await self.client.complete(
            model,
            [{"role": "user", "content": prompt}],
            temperature=self.judge.config.temperature,
            max_tokens=self.judge.config.max_tokens,
        )
        data: dict[str, Any] = extract_json(resp.text)
        verdict = Verdict(
            judge_name=self.judge.name,
            score=_opt_float(data.get("score")),
            label=(str(data["label"]) if data.get("label") not in (None, "null") else None),
            reasoning=str(data.get("reasoning", "")),
            confidence=_opt_float(data.get("confidence")),
        )
        verdict.meta.model = model
        verdict.meta.input_tokens = resp.input_tokens
        verdict.meta.output_tokens = resp.output_tokens
        verdict.meta.cost_usd = resp.cost_usd
        verdict.meta.extra["meta_judge"] = model
        return verdict
