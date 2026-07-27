"""Provider registry: API keys, base URLs, concurrency limits, fallbacks.

Model names are "provider/model" strings, e.g. "anthropic/claude-sonnet-4-5",
"gemini/gemini-2.5-pro".
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from ..core.schemas import ProviderConfig

# env var conventions per provider prefix
_ENV_KEYS = {
    "anthropic": "ANTHROPIC_AGENT_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
}


class ProviderRegistry:
    def __init__(self, providers: Optional[dict[str, ProviderConfig]] = None):
        self.providers: dict[str, ProviderConfig] = providers or {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    @staticmethod
    def split(model: str) -> tuple[str, str]:
        if "/" not in model:
            return "", model
        provider, name = model.split("/", 1)
        return provider, name

    def get(self, provider: str) -> ProviderConfig:
        if provider in self.providers:
            return self.providers[provider]
        # implicit provider from env var
        cfg = ProviderConfig(
            api_key=os.environ.get(_ENV_KEYS.get(provider, f"{provider.upper()}_API_KEY")),
        )
        self.providers[provider] = cfg
        return cfg

    def resolve(self, model: str) -> tuple[ProviderConfig, str, str]:
        """-> (config, provider, bare_model_name)"""
        provider, name = self.split(model)
        return self.get(provider), provider, name

    def semaphore(self, provider: str) -> asyncio.Semaphore:
        if provider not in self._semaphores:
            self._semaphores[provider] = asyncio.Semaphore(self.get(provider).max_concurrency)
        return self._semaphores[provider]

    def fallbacks(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for provider, cfg in self.providers.items():
            if cfg.fallback_models:
                # applied to every model of that provider lazily by RetryingClient callers
                out[provider] = cfg.fallback_models
        return out
