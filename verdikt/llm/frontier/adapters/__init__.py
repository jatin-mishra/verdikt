"""Registry of frontier wire-protocol adapters and their provider defaults."""
from __future__ import annotations

from .anthropic import AnthropicAdapter
from .anthropic import DEFAULT_BASE_URLS as _ANTHROPIC_URLS
from .anthropic import PROTOCOL as _ANTHROPIC
from .base import ProtocolAdapter
from .gemini import DEFAULT_BASE_URLS as _GEMINI_URLS
from .gemini import PROTOCOL as _GEMINI
from .gemini import GeminiAdapter

# classes, not shared instances: FrontierClient instantiates its own adapters
# so each client owns (and can aclose()) its own cached SDK clients.
PROTOCOL_ADAPTER_CLASSES: dict[str, type[ProtocolAdapter]] = {
    _ANTHROPIC: AnthropicAdapter,
    _GEMINI: GeminiAdapter,
}

PROVIDER_PROTOCOLS: dict[str, str] = {
    **dict.fromkeys(_ANTHROPIC_URLS, _ANTHROPIC),
    **dict.fromkeys(_GEMINI_URLS, _GEMINI),
}

PROVIDER_BASE_URLS: dict[str, str] = {
    **_ANTHROPIC_URLS,
    **_GEMINI_URLS,
}

__all__ = [
    "ProtocolAdapter",
    "PROTOCOL_ADAPTER_CLASSES",
    "PROVIDER_PROTOCOLS",
    "PROVIDER_BASE_URLS",
]
