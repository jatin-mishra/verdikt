"""Feature-flagged console logging of exact LLM request/response payloads.

Enabled by default (for now) via the ``VERDIKT_LOG_LLM_CALLS`` env var; flip
it off with ``VERDIKT_LOG_LLM_CALLS=0`` or ``set_llm_logging(False)``.
``FrontierClient(log_calls=...)`` overrides both for that one client.
"""
from __future__ import annotations

import itertools
import os
import sys
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .client import LLMResponse

_call_ids = itertools.count(1)
_override: Optional[bool] = None


def set_llm_logging(enabled: Optional[bool]) -> None:
    """Force LLM call logging on/off process-wide. Pass ``None`` to fall
    back to the ``VERDIKT_LOG_LLM_CALLS`` env var."""
    global _override
    _override = enabled


def is_llm_logging_enabled(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return explicit
    if _override is not None:
        return _override
    return os.environ.get("VERDIKT_LOG_LLM_CALLS", "1").strip().lower() not in ("0", "false", "")


def _use_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


class _Style:
    def __init__(self, on: bool):
        self.dim = "\033[2m" if on else ""
        self.bold = "\033[1m" if on else ""
        self.cyan = "\033[36m" if on else ""
        self.green = "\033[32m" if on else ""
        self.yellow = "\033[33m" if on else ""
        self.red = "\033[31m" if on else ""
        self.reset = "\033[0m" if on else ""


def _print_block(title: str, color: str, lines: list[str]) -> None:
    s = _Style(_use_color())
    width = max([len(title) + 4] + [len(line) for block in lines for line in block.splitlines()] + [40])
    width = min(width, 100)
    print(f"{color}┌─ {s.bold}{title}{s.reset}{color} " + "─" * max(0, width - len(title) - 3) + s.reset)
    for block in lines:
        for line in block.splitlines() or [""]:
            print(f"{color}│{s.reset} {line}")
    print(f"{color}└" + "─" * (width + 1) + s.reset)


def log_request(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    params: dict[str, Any],
    *,
    enabled: Optional[bool] = None,
) -> str:
    call_id = f"llm-{next(_call_ids)}"
    if not is_llm_logging_enabled(enabled):
        return call_id
    s = _Style(_use_color())
    lines = []
    for m in messages:
        role = m.get("role", "?")
        lines.append(f"{s.bold}{role}{s.reset}: {m.get('content', '')}")
    if temperature is not None:
        extra = f"temperature={temperature} max_tokens={max_tokens} json_mode={json_mode}"
        if params:
            extra += f" params={params}"
        lines.append(f"{s.dim}{extra}{s.reset}")
    _print_block(f"[{call_id}] REQUEST -> {model}", s.cyan, lines)
    return call_id


def log_response(
    call_id: str,
    model: str,
    response: "LLMResponse",
    *,
    enabled: Optional[bool] = None,
) -> None:
    if not is_llm_logging_enabled(enabled):
        return
    s = _Style(_use_color())
    stats = (
        f"input_tokens={response.input_tokens} output_tokens={response.output_tokens} "
        f"cost_usd={response.cost_usd:.6f} latency_ms={response.latency_ms:.1f}"
    )
    lines = [response.text, f"{s.dim}{stats}{s.reset}"]
    _print_block(f"[{call_id}] RESPONSE <- {model}", s.green, lines)


def log_error(
    call_id: str,
    model: str,
    error: Exception,
    *,
    enabled: Optional[bool] = None,
) -> None:
    if not is_llm_logging_enabled(enabled):
        return
    s = _Style(_use_color())
    _print_block(f"[{call_id}] ERROR <- {model}", s.red, [f"{type(error).__name__}: {error}"])
