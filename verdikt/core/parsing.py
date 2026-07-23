"""Robust JSON extraction from LLM output."""
from __future__ import annotations

import json
import re
from typing import Any


class ParseError(ValueError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from ``text``.

    Tolerates markdown fences and prose around the object.
    """
    text = text.strip()
    # strip markdown fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # first balanced {...}
    start = text.find("{")
    if start == -1:
        raise ParseError(f"no JSON object found in: {text[:200]!r}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ParseError(f"malformed JSON: {candidate[:200]!r}") from exc
                if isinstance(obj, dict):
                    return obj
                raise ParseError("top-level JSON is not an object")
    raise ParseError(f"unbalanced JSON in: {text[:200]!r}")
