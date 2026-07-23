"""Native HTTP client for frontier LLM providers — no litellm, no provider SDKs.

Each provider family's wire protocol lives in its own module under
``.adapters``: ``openai.py`` (OpenAI, plus OpenAI-compatible providers like
Kimi, Mistral, OpenRouter, xAI, DeepSeek), ``anthropic.py``, and ``gemini.py``.
The protocol for a model is inferred from its provider prefix
("anthropic/claude-sonnet-4-5" -> anthropic protocol); custom providers can
set ``protocol`` and ``base_url`` in their ProviderConfig.
"""
from .client import FrontierClient

__all__ = ["FrontierClient"]
