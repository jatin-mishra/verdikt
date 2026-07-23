from __future__ import annotations

from typing import Any

from ..core.base import BaseJudge, _opt_float
from ..core.registry import register
from ..core.schemas import EvalInput, Verdict
from ..llm.client import LLMClient, LLMResponse

_SCORE = {"A": 1.0, "B": 0.0, "tie": 0.5}


@register("pairwise")
class PairwiseJudge(BaseJudge):
    """Compares candidates[0] (A) vs candidates[1] (B).

    With ``position_swap`` (default), the comparison runs twice with the order
    flipped; a winner is only declared if both orders agree, otherwise "tie".
    label: "A" | "B" | "tie"; score: 1.0 (A) / 0.0 (B) / 0.5 (tie).
    """

    template = "pairwise.j2"
    verdict_type = "comparison"

    # averaging A/B/tie outcomes produces meaningless mid-values (e.g. 0.67):
    # only vote-style consensus makes sense for comparisons
    allowed_consensus = ("majority_vote", "unanimous", "consensus_leader")
    default_consensus = "majority_vote"

    def required_fields(self) -> list[str]:
        return ["candidates"]

    def validate_input(self, inp: EvalInput) -> None:
        super().validate_input(inp)
        if not inp.candidates or len(inp.candidates) != 2:
            raise ValueError(f"pairwise judge '{self.name}' needs exactly 2 candidates")

    def parse(self, data: dict[str, Any], inp: EvalInput) -> Verdict:
        winner = str(data.get("winner", "")).strip()
        if winner not in ("A", "B", "tie"):
            raise ValueError(f"pairwise winner must be A|B|tie, got {winner!r}")
        return Verdict(
            judge_name=self.name,
            label=winner,
            score=_SCORE[winner],
            reasoning=str(data.get("reasoning", "")),
            confidence=_opt_float(data.get("confidence")),
        )

    async def judge_once(
        self, inp: EvalInput, client: LLMClient, model: str, **_: Any
    ) -> tuple[Verdict, LLMResponse]:
        a, b = inp.candidates  # type: ignore[misc]
        v1, r1 = await super().judge_once(inp, client, model, candidate_a=a, candidate_b=b)
        if not self.config.position_swap:
            return v1, r1
        v2, r2 = await super().judge_once(inp, client, model, candidate_a=b, candidate_b=a)
        # unswap the second verdict: its "A" is really B
        unswapped = {"A": "B", "B": "A", "tie": "tie"}[v2.label or "tie"]
        winner = v1.label if v1.label == unswapped else "tie"
        merged = Verdict(
            judge_name=self.name,
            label=winner,
            score=_SCORE[winner or "tie"],
            reasoning=(
                f"[order A,B] {v1.reasoning}\n[order B,A] {v2.reasoning}"
                + ("" if v1.label == unswapped else "\n[position-swap disagreement -> tie]")
            ),
            confidence=min(v1.confidence or 1.0, v2.confidence or 1.0),
        )
        combined = LLMResponse(
            text="",
            model=model,
            input_tokens=r1.input_tokens + r2.input_tokens,
            output_tokens=r1.output_tokens + r2.output_tokens,
            cost_usd=r1.cost_usd + r2.cost_usd,
            latency_ms=r1.latency_ms + r2.latency_ms,
        )
        return merged, combined
