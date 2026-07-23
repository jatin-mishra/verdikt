"""Jinja2 prompt templates. Judges load defaults from ``templates/`` and any
judge can override with an inline template string via ``prompt_template``.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TEMPLATE_DIR = Path(__file__).parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render(template_name: str, **ctx: object) -> str:
    return _env.get_template(template_name).render(**ctx)


def render_inline(template_source: str, **ctx: object) -> str:
    return _env.from_string(template_source).render(**ctx)
