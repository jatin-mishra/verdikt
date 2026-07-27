"""Rough cost estimation. USD per 1M tokens: (input, output).

Override or extend via ``register_cost``.
"""
from __future__ import annotations

_COST_TABLE: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet": (3.0, 15.0),
    "anthropic/claude-haiku": (0.8, 4.0),
    "gemini/gemini-2.5-pro": (1.25, 10.0),
    "gemini/gemini-2.5-flash": (0.15, 0.6),
}


def register_cost(model: str, input_per_m: float, output_per_m: float) -> None:
    _COST_TABLE[model] = (input_per_m, output_per_m)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _COST_TABLE.get(model)
    if not rates:
        # prefix match (version suffixes)
        for key, val in _COST_TABLE.items():
            if model.startswith(key):
                rates = val
                break
    if not rates:
        return 0.0
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000
