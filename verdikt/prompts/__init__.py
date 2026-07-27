"""Jinja2 prompt templates. Judges load defaults from ``templates/``; a
custom judge can either override inline via ``prompt_template`` or ship its
own ``.j2`` file in a directory registered with ``add_template_dir``.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TEMPLATE_DIR = Path(__file__).parent / "templates"

_env = Environment(
    loader=FileSystemLoader([str(_TEMPLATE_DIR)]),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def add_template_dir(path: str | Path) -> None:
    """Register a directory of custom ``.j2`` templates, searched before the
    built-ins. A custom judge can then set ``template = "my_judge.j2"`` and
    resolve it from here -- no need to fork/edit the installed package.

    Call this before rendering any prompt that needs it (e.g. at import time
    of the module defining your custom judge).
    """
    searchpath: list = _env.loader.searchpath  # type: ignore[attr-defined]
    searchpath.insert(0, str(path))


def render(template_name: str, **ctx: object) -> str:
    return _env.get_template(template_name).render(**ctx)


def render_inline(template_source: str, **ctx: object) -> str:
    return _env.from_string(template_source).render(**ctx)
