"""Batch evaluation over a dataset with bounded concurrency.

Individual failures become error verdicts (the Executor never raises),
so one bad item never sinks the batch.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from ..core.schemas import EvalInput, PipelineVerdict, Verdict

AnyVerdict = Verdict | PipelineVerdict
EvalFn = Callable[[EvalInput], Awaitable[AnyVerdict]]


class BatchResult(BaseModel):
    total: int
    results: list[AnyVerdict] = Field(default_factory=list)  # same order as items

    @property
    def failed_indices(self) -> list[int]:
        out = []
        for i, v in enumerate(self.results):
            error = getattr(v, "error", None)
            if error is not None or v.passed is False:
                out.append(i)
        return out


class BatchRunner:
    def __init__(self, evaluate: EvalFn, concurrency: int = 8):
        self.evaluate = evaluate
        self.concurrency = concurrency

    async def run(self, items: list[EvalInput]) -> BatchResult:
        sem = asyncio.Semaphore(self.concurrency)

        async def _one(inp: EvalInput) -> AnyVerdict:
            async with sem:
                return await self.evaluate(inp)

        results = await asyncio.gather(*(_one(i) for i in items))
        return BatchResult(total=len(items), results=list(results))
