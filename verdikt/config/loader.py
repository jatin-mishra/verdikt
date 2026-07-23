"""YAML config loading with ${ENV_VAR} interpolation."""
from __future__ import annotations

import os
import re
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..core.schemas import JudgeConfig, PipelineConfig, ProviderConfig

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def sub(m: "re.Match[str]") -> str:
            return os.environ.get(m.group(1), "")

        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


class VerdiktConfig(BaseModel):
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    judges: list[JudgeConfig] = Field(default_factory=list)
    pipelines: list[PipelineConfig] = Field(default_factory=list)
    cache_path: str | None = None
    max_retries: int = 3


def load_config(path: str) -> VerdiktConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> VerdiktConfig:
    raw = _interpolate(raw)
    return VerdiktConfig(**raw)
