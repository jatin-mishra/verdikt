from __future__ import annotations

from ..core.base import BaseJudge, ScoreParsingMixin
from ..core.registry import register


@register("pointwise")
class PointwiseJudge(ScoreParsingMixin, BaseJudge):
    """Scores one output on the configured scale (reference-free by default;
    uses ``expected_output`` and ``context`` automatically when present)."""

    # uses BaseJudge's default system_template/user_template (pointwise_*.j2)
    verdict_type = "score"


@register("reference")
class ReferenceJudge(PointwiseJudge):
    """Pointwise scoring against a required gold answer."""

    def required_fields(self) -> list[str]:
        return ["output", "expected_output"]
