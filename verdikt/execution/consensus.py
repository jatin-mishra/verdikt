"""Consensus strategies for combining verdicts from multiple models."""
from __future__ import annotations

import statistics
from collections import Counter

from ..core.schemas import ExecutionConfig, Verdict


def disagreement_score(verdicts: list[Verdict]) -> float | None:
    """0 = all models agree, 1 = maximal disagreement."""
    ok = [v for v in verdicts if v.ok]
    if len(ok) < 2:
        return None
    scores = [v.score for v in ok if v.score is not None]
    if len(scores) >= 2:
        # scores live in 0-1; max possible pstdev is 0.5
        return min(1.0, 2 * statistics.pstdev(scores))
    labels = [v.label for v in ok if v.label is not None]
    if len(labels) >= 2:
        top = Counter(labels).most_common(1)[0][1]
        return 1.0 - top / len(labels)
    return None


def _weight(v: Verdict, weights: dict[str, float]) -> float:
    return weights.get(v.meta.model or "", 1.0)


def combine(verdicts: list[Verdict], cfg: ExecutionConfig, judge_name: str) -> Verdict:
    """Combine broadcast member verdicts per the configured consensus strategy."""
    ok = [v for v in verdicts if v.ok]
    if not ok:
        return Verdict(
            judge_name=judge_name,
            error="all broadcast members failed: "
            + "; ".join(v.error or "?" for v in verdicts),
        )

    strategy = cfg.consensus
    if strategy == "consensus_leader":
        leader_model = cfg.leader or (ok[0].meta.model or "")
        leader = next((v for v in ok if v.meta.model == leader_model), ok[0])
        out = leader.model_copy(deep=True)
    elif strategy == "majority_vote":
        out = _majority(ok, judge_name)
    elif strategy == "unanimous":
        out = _unanimous(ok, judge_name)
    else:  # average / weighted_average
        weights = cfg.weights if strategy == "weighted_average" else {}
        out = _average(ok, weights, judge_name)

    out.judge_name = judge_name
    return out


def _majority(ok: list[Verdict], judge_name: str) -> Verdict:
    labels = [v.label for v in ok if v.label is not None]
    if labels:
        winner, count = Counter(labels).most_common(1)[0]
        members = [v for v in ok if v.label == winner]
        scores = [v.score for v in members if v.score is not None]
        return Verdict(
            judge_name=judge_name,
            label=winner,
            score=statistics.median(scores) if scores else None,
            reasoning=f"majority vote: {count}/{len(labels)} models chose '{winner}'. "
            + " | ".join(f"[{v.meta.model}] {v.reasoning}" for v in ok),
        )
    # no labels -> vote on passed, fall back to median score
    passed_votes = [v.passed for v in ok if v.passed is not None]
    scores = [v.score for v in ok if v.score is not None]
    verdict = Verdict(
        judge_name=judge_name,
        score=statistics.median(scores) if scores else None,
        reasoning=" | ".join(f"[{v.meta.model}] {v.reasoning}" for v in ok),
    )
    if passed_votes:
        verdict.passed = passed_votes.count(True) > len(passed_votes) / 2
    return verdict


def _unanimous(ok: list[Verdict], judge_name: str) -> Verdict:
    labels = {v.label for v in ok if v.label is not None}
    scores = [v.score for v in ok if v.score is not None]
    passed_flags = [v.passed for v in ok if v.passed is not None]
    verdict = Verdict(
        judge_name=judge_name,
        score=min(scores) if scores else None,  # strictest member
        label=labels.pop() if len(labels) == 1 else None,
        reasoning=" | ".join(f"[{v.meta.model}] {v.reasoning}" for v in ok),
    )
    if passed_flags:
        verdict.passed = all(passed_flags)
    elif len({v.label for v in ok if v.label is not None}) > 1:
        verdict.passed = False
    return verdict


def _average(ok: list[Verdict], weights: dict[str, float], judge_name: str) -> Verdict:
    scored = [v for v in ok if v.score is not None]
    score = None
    if scored:
        total_w = sum(_weight(v, weights) for v in scored)
        score = sum(v.score * _weight(v, weights) for v in scored) / total_w  # type: ignore[operator]
    labels = [v.label for v in ok if v.label is not None]
    label = Counter(labels).most_common(1)[0][0] if labels else None
    confs = [v.confidence for v in ok if v.confidence is not None]
    return Verdict(
        judge_name=judge_name,
        score=score,
        label=label,
        confidence=sum(confs) / len(confs) if confs else None,
        reasoning=" | ".join(f"[{v.meta.model}] {v.reasoning}" for v in ok),
    )
