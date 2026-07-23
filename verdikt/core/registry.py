"""Judge type registry. Custom judges: subclass BaseJudge + @register("my_type")."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Type

if TYPE_CHECKING:
    from .base import BaseJudge

_REGISTRY: dict[str, "Type[BaseJudge]"] = {}


def register(type_name: str) -> Callable[["Type[BaseJudge]"], "Type[BaseJudge]"]:
    def deco(cls: "Type[BaseJudge]") -> "Type[BaseJudge]":
        _REGISTRY[type_name] = cls
        return cls

    return deco


def get_judge_class(type_name: str) -> "Type[BaseJudge]":
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
