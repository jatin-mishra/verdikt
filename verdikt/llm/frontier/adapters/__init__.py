"""Registry of frontier wire-protocol adapters and their provider defaults."""
from __future__ import annotations

from .anthropic import DEFAULT_BASE_URLS as _ANTHROPIC_URLS
from .anthropic import PROTOCOL as _ANTHROPIC
from .anthropic import AnthropicAdapter
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


def register_protocol(
    protocol: str,
    adapter_cls: type[ProtocolAdapter],
    *,
    providers: dict[str, str] | None = None,
) -> None:
    """Plug in a wire protocol / provider SDK verdikt doesn't ship.

    ``adapter_cls`` is a ``ProtocolAdapter`` subclass (see ``adapters/base.py``)
    that owns one provider's SDK request/response shapes entirely on its own
    -- ``FrontierClient`` never sees them, only the ``(text, input_tokens,
    output_tokens)`` tuple its ``complete()`` returns.

    ``providers`` optionally maps provider prefixes (the part before "/" in
    "myprovider/some-model") to a default base URL, so
    ``ProviderConfig(api_key=...)`` alone is enough for them; otherwise set
    ``protocol=`` (and usually ``base_url=``) explicitly per ProviderConfig.

    Call this before constructing ``Verdikt``/``FrontierClient`` -- adapters
    are resolved lazily on first use, so registering right after import is
    enough; a client instance already mid-call for that protocol won't pick
    up a later registration.
    """
    PROTOCOL_ADAPTER_CLASSES[protocol] = adapter_cls
    for name, base_url in (providers or {}).items():
        PROVIDER_PROTOCOLS[name] = protocol
        PROVIDER_BASE_URLS[name] = base_url


__all__ = [
    "ProtocolAdapter",
    "PROTOCOL_ADAPTER_CLASSES",
    "PROVIDER_PROTOCOLS",
    "PROVIDER_BASE_URLS",
    "register_protocol",
]
