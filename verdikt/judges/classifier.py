from __future__ import annotations

from typing import Any

from ..core.base import BaseJudge, _opt_float
from ..core.registry import register
from ..core.schemas import EvalInput, Verdict


@register("classifier")
class ClassifierJudge(BaseJudge):
    """Returns one label from a fixed set (e.g. safe/unsafe)."""

    template = "classifier.j2"
    verdict_type = "label"

    # labels cannot be averaged: only vote-style consensus makes sense
    allowed_consensus = ("majority_vote", "unanimous", "consensus_leader")
    default_consensus = "majority_vote"

    def validate_input(self, inp: EvalInput) -> None:
        super().validate_input(inp)
        if not self.config.labels:
            raise ValueError(f"classifier judge '{self.name}' needs 'labels'")

    def parse(self, data: dict[str, Any], inp: EvalInput) -> Verdict:
        label = str(data.get("label", "")).strip()
        if label not in self.config.labels:
            # tolerate case mismatch before failing
            lowered = {l.lower(): l for l in self.config.labels}
            if label.lower() in lowered:
                label = lowered[label.lower()]
            else:
                raise ValueError(
                    f"label {label!r} not in allowed labels {self.config.labels}"
                )
        return Verdict(
            judge_name=self.name,
            label=label,
            reasoning=str(data.get("reasoning", "")),
            confidence=_opt_float(data.get("confidence")),
        )
