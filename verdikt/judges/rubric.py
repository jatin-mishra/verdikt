from __future__ import annotations

from typing import Any

from ..core.base import BaseJudge, _opt_float
from ..core.registry import register
from ..core.schemas import CriterionResult, EvalInput, Verdict


@register("rubric")
class RubricJudge(BaseJudge):
    """G-Eval style: grade each criterion with chain-of-thought, then average."""

    template = "rubric.j2"
    verdict_type = "rubric"

    def validate_input(self, inp: EvalInput) -> None:
        super().validate_input(inp)
        if not self.config.criteria:
            raise ValueError(f"rubric judge '{self.name}' needs at least one criterion")

    def parse(self, data: dict[str, Any], inp: EvalInput) -> Verdict:
        raw_criteria = data.get("criteria")
        if not isinstance(raw_criteria, list) or not raw_criteria:
            raise ValueError(f"rubric response missing 'criteria' list: {data}")
        breakdown = [
            CriterionResult(
                criterion=str(c.get("criterion", f"criterion_{i + 1}")),
                score=self.config.scale.normalize(float(c["score"])),
                reasoning=str(c.get("reasoning", "")),
            )
            for i, c in enumerate(raw_criteria)
        ]
        overall = sum(c.score for c in breakdown) / len(breakdown)
        return Verdict(
            judge_name=self.name,
            score=overall,
            reasoning=str(data.get("reasoning", "")),
            criteria_breakdown=breakdown,
            confidence=_opt_float(data.get("confidence")),
        )
