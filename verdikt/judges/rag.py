from __future__ import annotations

from typing import Any

from ..core.base import BaseJudge, ScoreParsingMixin
from ..core.registry import register
from ..core.schemas import EvalInput


class _RagJudge(ScoreParsingMixin, BaseJudge):
    template = "rag.j2"
    verdict_type = "score"
    metric = "faithfulness"

    def template_context(self, inp: EvalInput) -> dict[str, Any]:
        ctx = super().template_context(inp)
        ctx["metric"] = self.metric
        return ctx


@register("rag_faithfulness")
class FaithfulnessJudge(_RagJudge):
    """Is every claim in the output grounded in the retrieved context?"""

    metric = "faithfulness"

    def required_fields(self) -> list[str]:
        return ["output", "context"]


@register("rag_context_relevance")
class ContextRelevanceJudge(_RagJudge):
    """Is the retrieved context relevant to the user input?"""

    metric = "context_relevance"

    def required_fields(self) -> list[str]:
        return ["input", "output", "context"]


@register("rag_answer_relevance")
class AnswerRelevanceJudge(_RagJudge):
    """Does the output actually answer the user input?"""

    metric = "answer_relevance"

    def required_fields(self) -> list[str]:
        return ["input", "output"]
