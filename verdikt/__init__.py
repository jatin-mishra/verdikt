"""verdikt — configurable LLM-as-judge library.

Plug into any agent: configure judges (in code or YAML), feed an EvalInput,
get a structured Verdict back. Supports single-model judging, multi-model
broadcast with consensus, judge-of-judges, and multi-step pipelines with
prior-verdict passing.
"""
from .config.loader import VerdiktConfig, load_config, parse_config
from .core.base import BaseJudge
from .core.registry import available_types, register
from .core.schemas import (
    CriterionResult,
    EvalInput,
    ExecutionConfig,
    JudgeConfig,
    JudgeMeta,
    Message,
    PipelineConfig,
    PipelineStep,
    PipelineVerdict,
    ProviderConfig,
    Scale,
    Step,
    Verdict,
)
from .facade import Verdikt
from .llm.client import LLMClient, LLMResponse
from .llm.frontier import FrontierClient
from .pipeline.batch import BatchResult

__version__ = "0.1.0"

__all__ = [
    "Verdikt",
    "EvalInput",
    "Verdict",
    "PipelineVerdict",
    "CriterionResult",
    "JudgeMeta",
    "Message",
    "Step",
    "Scale",
    "JudgeConfig",
    "ExecutionConfig",
    "PipelineConfig",
    "PipelineStep",
    "ProviderConfig",
    "VerdiktConfig",
    "BaseJudge",
    "register",
    "available_types",
    "load_config",
    "parse_config",
    "LLMClient",
    "LLMResponse",
    "FrontierClient",
    "BatchResult",
]
