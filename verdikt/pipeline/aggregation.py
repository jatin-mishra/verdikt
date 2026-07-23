"""Combine step verdicts into one PipelineVerdict."""
from __future__ import annotations

from ..core.schemas import PipelineConfig, PipelineVerdict, Verdict


def _failed(v: Verdict) -> bool:
    return v.error is not None or v.passed is False


def aggregate(cfg: PipelineConfig, verdicts: list[Verdict], **kw: object) -> PipelineVerdict:
    pv = PipelineVerdict(pipeline_name=cfg.name, passed=True, verdicts=verdicts, **kw)  # type: ignore[arg-type]
    if not verdicts:
        pv.passed = False
        pv.reasoning = "no verdicts produced"
        return pv

    scores = [v.score for v in verdicts if v.score is not None and v.ok]

    if cfg.aggregation == "all_pass":
        failed = [v.judge_name for v in verdicts if _failed(v)]
        pv.passed = not failed
        pv.score = sum(scores) / len(scores) if scores else None
        pv.reasoning = (
            f"failed judges: {failed}" if failed else "all judges passed"
        )
    elif cfg.aggregation == "weighted_average":
        weighted = [
            (v.score, cfg.weights.get(v.judge_name, 1.0))
            for v in verdicts
            if v.score is not None and v.ok
        ]
        if weighted:
            total_w = sum(w for _, w in weighted)
            pv.score = sum(s * w for s, w in weighted) / total_w
            pv.passed = pv.score >= cfg.threshold
            pv.reasoning = f"weighted score {pv.score:.3f} vs threshold {cfg.threshold}"
        else:
            pv.passed = False
            pv.reasoning = "no scored verdicts to aggregate"
    elif cfg.aggregation == "majority_vote":
        votes = [not _failed(v) for v in verdicts if v.passed is not None or v.error]
        pv.passed = votes.count(True) > len(votes) / 2 if votes else False
        pv.score = sum(scores) / len(scores) if scores else None
        pv.reasoning = f"{votes.count(True)}/{len(votes)} judges passed"
    else:  # last
        last = verdicts[-1]
        pv.passed = not _failed(last)
        pv.score = last.score
        pv.reasoning = f"decided by last judge '{last.judge_name}': {last.reasoning}"

    # roll up costs
    pv.meta.cost_usd = sum(v.meta.cost_usd for v in verdicts)
    pv.meta.input_tokens = sum(v.meta.input_tokens for v in verdicts)
    pv.meta.output_tokens = sum(v.meta.output_tokens for v in verdicts)
    return pv
