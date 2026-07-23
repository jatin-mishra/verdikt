"""Response cache: identical (model, messages, params) -> cached LLMResponse.

In-memory by default; pass ``path`` for a JSON file that survives restarts.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

from .client import LLMClient, LLMResponse


def _key(model: str, messages: list[dict[str, str]], kw: dict[str, Any]) -> str:
    payload = json.dumps({"m": model, "msgs": messages, "kw": kw}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class CachingClient(LLMClient):
    def __init__(self, inner: LLMClient, path: Optional[str] = None):
        self.inner = inner
        self.path = path
        self._mem: dict[str, dict[str, Any]] = {}
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    self._mem = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._mem = {}

    def _persist(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._mem, f)
        os.replace(tmp, self.path)

    async def complete(self, model: str, messages: list[dict[str, str]], **kw: Any) -> LLMResponse:
        key = _key(model, messages, kw)
        if key in self._mem:
            resp = LLMResponse(**self._mem[key])
            resp.extra["cached"] = True
            return resp
        resp = await self.inner.complete(model, messages, **kw)
        self._mem[key] = resp.model_dump()
        self._persist()
        return resp
