"""Frontier LLM client — official provider SDKs, no litellm.

Each provider's wire protocol and SDK usage lives entirely in its own module
under ``.adapters``: ``anthropic.py`` (the ``anthropic`` SDK) and
``gemini.py`` (the ``google-genai`` SDK). ``FrontierClient`` never touches a
provider's SDK types directly — it only sees the generic ``(text,
input_tokens, output_tokens)`` tuple each adapter's ``complete()`` returns.
The protocol for a model is inferred from its provider prefix
("anthropic/claude-sonnet-4-5" -> anthropic protocol); custom/self-hosted
endpoints can set ``protocol`` and ``base_url`` in their ProviderConfig.
"""
from .adapters import ProtocolAdapter, register_protocol
from .client import FrontierClient

__all__ = ["FrontierClient", "ProtocolAdapter", "register_protocol"]
