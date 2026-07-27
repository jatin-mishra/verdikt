"""Judge type registry. Custom judges: subclass BaseJudge + @register("my_type")."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseJudge

_REGISTRY: dict[str, type[BaseJudge]] = {}


def register(type_name: str) -> Callable[[type[BaseJudge]], type[BaseJudge]]:
    def deco(cls: type[BaseJudge]) -> type[BaseJudge]:
        _REGISTRY[type_name] = cls
        return cls

    return deco


def get_judge_class(type_name: str) -> type[BaseJudge]:
    # ensure built-ins are registered
    from .. import judges  # noqa: F401

    if type_name not in _REGISTRY:
        raise KeyError(
            f"unknown judge type '{type_name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[type_name]


def available_types() -> list[str]:
    from .. import judges  # noqa: F401

    return sorted(_REGISTRY)
