"""Top-level entry point: the ``Verdikt`` class."""
from __future__ import annotations

import asyncio
from typing import Optional, Sequence, Union

from .config.loader import VerdiktConfig, load_config
from .core.registry import get_judge_class
from .core.schemas import (
    EvalInput,
    JudgeConfig,
    PipelineConfig,
    PipelineVerdict,
    ProviderConfig,
    Verdict,
)
from .execution.modes import Executor
from .llm.cache import CachingClient
from .llm.client import LLMClient, RetryingClient
from .llm.providers import ProviderRegistry
from .pipeline.batch import BatchResult, BatchRunner
from .pipeline.runner import PipelineRunner


class Verdikt:
    """Facade over judges, pipelines, and providers.

    >>> v = Verdikt.from_yaml("verdikt.yaml")
    >>> verdict = await v.evaluate("helpfulness", EvalInput(input=q, output=a))
    """

    def __init__(
        self,
        judges: Sequence[JudgeConfig] = (),
        pipelines: Sequence[PipelineConfig] = (),
        providers: Optional[dict[str, ProviderConfig]] = None,
        client: Optional[LLMClient] = None,
        cache_path: Optional[str] = None,
        max_retries: int = 3,
    ):
        self.registry = ProviderRegistry(dict(providers or {}))
        base = client or self._default_client()
        fallbacks: dict[str, list[str]] = {}
        for jc in judges:
            for model in [jc.model, *jc.execution.models]:
                if not model:
                    continue
                provider, _ = ProviderRegistry.split(model)
                cfg = self.registry.get(provider)
                if cfg.fallback_models:
                    fallbacks[model] = [m for m in cfg.fallback_models if m != model]
        wrapped: LLMClient = RetryingClient(base, max_retries=max_retries, fallbacks=fallbacks)
        self.client: LLMClient = CachingClient(wrapped, path=cache_path) if cache_path else wrapped

        self.executors: dict[str, Executor] = {}
        for jc in judges:
            judge = get_judge_class(jc.type)(jc)
            self.executors[jc.name] = Executor(judge, self.client)

        self.pipelines: dict[str, PipelineRunner] = {}
        for pc in pipelines:
            missing = [
                n
                for step in pc.steps
                for n in ([step.judge] if step.judge else step.parallel)
                if n and n not in self.executors
            ]
            if missing:
                raise ValueError(f"pipeline '{pc.name}' references unknown judges: {missing}")
            self.pipelines[pc.name] = PipelineRunner(pc, self.executors)

    # ------------------------------------------------------------------

    def _default_client(self) -> LLMClient:
        from .llm.frontier import FrontierClient

        return FrontierClient(self.registry)

    @classmethod
    def from_yaml(cls, path: str, client: Optional[LLMClient] = None) -> "Verdikt":
        return cls.from_config(load_config(path), client=client)

    @classmethod
    def from_config(cls, cfg: VerdiktConfig, client: Optional[LLMClient] = None) -> "Verdikt":
        return cls(
            judges=cfg.judges,
            pipelines=cfg.pipelines,
            providers=cfg.providers,
            client=client,
            cache_path=cfg.cache_path,
            max_retries=cfg.max_retries,
        )

    # ------------------------------------------------------------------

    async def evaluate(self, name: str, inp: EvalInput) -> Union[Verdict, PipelineVerdict]:
        """Run a judge or a pipeline by name."""
        if name in self.pipelines:
            return await self.pipelines[name].run(inp)
        if name in self.executors:
            return await self.executors[name].evaluate(inp)
        raise KeyError(
            f"unknown judge/pipeline '{name}'. "
            f"judges={sorted(self.executors)}, pipelines={sorted(self.pipelines)}"
        )

    def evaluate_sync(self, name: str, inp: EvalInput) -> Union[Verdict, PipelineVerdict]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.evaluate(name, inp))
        raise RuntimeError(
            "evaluate_sync() cannot be called from a running event loop; use await evaluate()"
        )

    async def evaluate_batch(
        self, name: str, items: list[EvalInput], concurrency: int = 8
    ) -> BatchResult:
        """Evaluate a dataset with bounded concurrency; failures become error
        verdicts in the results, never exceptions."""

        async def _fn(inp: EvalInput) -> Union[Verdict, PipelineVerdict]:
            return await self.evaluate(name, inp)

        return await BatchRunner(_fn, concurrency=concurrency).run(items)
