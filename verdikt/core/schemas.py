"""Core data contracts: every judge consumes EvalInput and produces Verdict."""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class Step(BaseModel):
    """One step of an agent trajectory."""

    thought: str | None = None
    tool: str | None = None
    tool_input: dict[str, Any] | None = None
    observation: str | None = None


class EvalInput(BaseModel):
    """Generic input contract accepted by every judge.

    Only ``output`` is always required; each judge declares which other
    fields it needs and validates at runtime.
    """

    input: str | None = None
    output: str
    expected_output: str | None = None
    context: list[str] | None = None
    system_prompt: str | None = None
    conversation: list[Message] | None = None
    trajectory: list[Step] | None = None
    candidates: list[str] | None = None  # pairwise: [A, B] (output ignored)
    prior_verdicts: list[Verdict] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


class CriterionResult(BaseModel):
    criterion: str
    score: float  # normalized 0-1
    reasoning: str = ""


class JudgeMeta(BaseModel):
    model: str | None = None
    models: list[str] = Field(default_factory=list)  # broadcast mode
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    samples: int = 1
    execution_mode: str = "single"
    disagreement: float | None = None  # 0-1 variance across broadcast models
    timestamp: float = Field(default_factory=time.time)
    extra: dict[str, Any] = Field(default_factory=dict)


class Verdict(BaseModel):
    judge_name: str
    verdict_type: Literal["score", "label", "comparison", "rubric"] = "score"
    score: float | None = None  # normalized 0-1
    label: str | None = None  # e.g. "safe", "A", "B", "tie"
    passed: bool | None = None
    reasoning: str = ""
    criteria_breakdown: list[CriterionResult] | None = None
    confidence: float | None = None
    error: str | None = None
    sub_verdicts: list[Verdict] = Field(default_factory=list)  # broadcast members
    meta: JudgeMeta = Field(default_factory=JudgeMeta)

    @property
    def ok(self) -> bool:
        return self.error is None


class PipelineVerdict(BaseModel):
    pipeline_name: str
    passed: bool
    score: float | None = None
    verdicts: list[Verdict] = Field(default_factory=list)
    skipped_steps: list[str] = Field(default_factory=list)
    stopped_early: bool = False
    reasoning: str = ""
    meta: JudgeMeta = Field(default_factory=JudgeMeta)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class Scale(BaseModel):
    min: float = 1
    max: float = 5

    def normalize(self, raw: float) -> float:
        if self.max == self.min:
            return 0.0
        v = (raw - self.min) / (self.max - self.min)
        return max(0.0, min(1.0, v))


ConsensusStrategy = Literal[
    "majority_vote", "average", "weighted_average", "unanimous", "consensus_leader"
]


class ExecutionConfig(BaseModel):
    mode: Literal["single", "broadcast", "judge_of_judges"] = "single"
    models: list[str] = Field(default_factory=list)  # broadcast / judge_of_judges
    consensus: ConsensusStrategy = "average"
    weights: dict[str, float] = Field(default_factory=dict)
    leader: str | None = None  # for consensus_leader
    meta_judge: str | None = None  # for judge_of_judges
    on_disagreement: Literal["accept_consensus", "accept_leader", "fail", "escalate"] = (
        "accept_consensus"
    )
    disagreement_threshold: float = 0.25
    escalate_model: str | None = None  # used when on_disagreement == "escalate"


class JudgeConfig(BaseModel):
    name: str
    type: str  # registry key: pointwise, pairwise, rubric, classifier, ...
    model: str | None = None  # "provider/model"; required for mode=single
    criteria: list[str] = Field(default_factory=list)
    rubric: str | None = None  # free-text rubric override
    labels: list[str] = Field(default_factory=list)  # classifier
    fail_on: list[str] = Field(default_factory=list)  # classifier labels that fail
    scale: Scale = Field(default_factory=Scale)
    threshold: float | None = None  # normalized 0-1; verdict.passed = score >= threshold
    samples: int = 1  # multi-sample voting
    position_swap: bool = True  # pairwise only
    temperature: float = 0.0
    max_tokens: int = 1024
    prompt_template: str | None = None  # inline Jinja2 override
    few_shot: list[dict[str, Any]] = Field(default_factory=list)  # calibration examples
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    # provider-native passthrough params for this judge's own LLM calls (top_p,
    # top_k, stop_sequences, thinking budgets, cache_system_prompt, ...) -- each
    # ProtocolAdapter decides what it understands; unknown keys reach the SDK
    # call as-is, so new provider features never need a verdikt code change.
    llm_params: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)  # judge-type-specific data (not LLM params)

    @model_validator(mode="after")
    def _check_models(self) -> JudgeConfig:
        if self.execution.mode == "single" and not self.model:
            raise ValueError(f"judge '{self.name}': 'model' is required for single mode")
        if self.execution.mode != "single" and not (self.execution.models or self.model):
            raise ValueError(
                f"judge '{self.name}': 'execution.models' required for {self.execution.mode}"
            )
        return self


class PipelineStep(BaseModel):
    judge: str | None = None  # single judge name
    parallel: list[str] = Field(default_factory=list)  # or a concurrent group
    on_fail: Literal["continue", "stop"] = "continue"
    run_if: str | None = None  # e.g. "helpfulness.score < 0.6"
    pass_prior_verdicts: bool = True

    @model_validator(mode="after")
    def _check_target(self) -> PipelineStep:
        if bool(self.judge) == bool(self.parallel):
            raise ValueError("step needs exactly one of 'judge' or 'parallel'")
        return self


class PipelineConfig(BaseModel):
    name: str
    steps: list[PipelineStep]
    aggregation: Literal["all_pass", "weighted_average", "majority_vote", "last"] = "all_pass"
    weights: dict[str, float] = Field(default_factory=dict)
    threshold: float = 0.5  # for weighted_average aggregation


class ProviderConfig(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    # wire protocol for this provider's API; inferred from the provider name
    # for known providers (anthropic, gemini). Set it explicitly for
    # custom/self-hosted endpoints speaking one of these protocols.
    protocol: Literal["anthropic", "gemini"] | None = None
    default_params: dict[str, Any] = Field(default_factory=dict)
    max_concurrency: int = 8
    fallback_models: list[str] = Field(default_factory=list)
