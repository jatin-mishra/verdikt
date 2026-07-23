"""Pipeline runner: sequential/parallel steps, gating, conditions,
and prior-verdict passing."""
from __future__ import annotations

import asyncio
import re

from ..core.schemas import EvalInput, PipelineConfig, PipelineVerdict, Verdict
from ..execution.modes import Executor
from .aggregation import aggregate

_COND = re.compile(
    r"^\s*(?P<judge>[\w-]+)\.(?P<field>score|passed|label|error)\s*"
    r"(?P<op>==|!=|<=|>=|<|>)\s*(?P<value>.+?)\s*$"
)


def eval_condition(expr: str, verdicts: dict[str, Verdict]) -> bool:
    """Evaluate e.g. "helpfulness.score < 0.6" against verdicts so far.

    Unknown judge -> False (step is skipped).
    """
    m = _COND.match(expr)
    if not m:
        raise ValueError(f"bad run_if expression: {expr!r}")
    v = verdicts.get(m["judge"])
    if v is None:
        return False
    actual = getattr(v, m["field"])
    raw = m["value"].strip().strip("'\"")
    expected: object
    if raw.lower() in ("true", "false"):
        expected = raw.lower() == "true"
    elif raw.lower() in ("none", "null"):
        expected = None
    else:
        try:
            expected = float(raw)
        except ValueError:
            expected = raw
    op = m["op"]
    try:
        if op == "==":
            return actual == expected
        if op == "!=":
            return actual != expected
        if actual is None:
            return False
        return {
            "<": actual < expected,
            "<=": actual <= expected,
            ">": actual > expected,
            ">=": actual >= expected,
        }[op]
    except TypeError:
        return False


class PipelineRunner:
    def __init__(self, config: PipelineConfig, executors: dict[str, Executor]):
        self.config = config
        self.executors = executors

    async def run(self, inp: EvalInput) -> PipelineVerdict:
        done: dict[str, Verdict] = {}
        ordered: list[Verdict] = []
        skipped: list[str] = []
        stopped_early = False

        for step in self.config.steps:
            names = [n for n in ([step.judge] if step.judge else step.parallel) if n]

            if step.run_if and not eval_condition(step.run_if, done):
                skipped.extend(names)
                continue

            step_inp = inp
            if step.pass_prior_verdicts and ordered:
                step_inp = inp.model_copy(update={"prior_verdicts": list(ordered)})

            results = await asyncio.gather(
                *(self.executors[n].evaluate(step_inp) for n in names)
            )
            for v in results:
                done[v.judge_name] = v
                ordered.append(v)

            if step.on_fail == "stop" and any(v.passed is False or v.error for v in results):
                stopped_early = True
                remaining = [
                    n
                    for s in self.config.steps[self.config.steps.index(step) + 1 :]
                    for n in ([s.judge] if s.judge else s.parallel)
                    if n and n not in done
                ]
                skipped.extend(remaining)
                break

        return aggregate(
            self.config,
            ordered,
            skipped_steps=skipped,
            stopped_early=stopped_early,
        )
