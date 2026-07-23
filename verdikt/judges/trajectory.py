from __future__ import annotations

from ..core.base import BaseJudge, ScoreParsingMixin
from ..core.registry import register


@register("trajectory")
class TrajectoryJudge(ScoreParsingMixin, BaseJudge):
    """Evaluates a full agent run: task completion, tool-call correctness,
    plan efficiency, and answer/observation consistency."""

    template = "trajectory.j2"
    verdict_type = "score"

    def required_fields(self) -> list[str]:
        return ["output", "trajectory"]
