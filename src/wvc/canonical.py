"""Canonical JSON for hashed records (spec section 3.5).

UTF-8, keys sorted, separators ``,`` and ``:`` with no whitespace, ``ensure_ascii`` off, and
no floats anywhere: sizes are integers and times are strings. Ids are always computed over
this form; files on disk may be pretty-printed.
"""

from __future__ import annotations

import hashlib
import json


def canonical_bytes(obj: object) -> bytes:
    """Serialise ``obj`` canonically; raise ValueError on a float or a non-JSON value."""
    try:
        text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:  # unsupported type, unsortable keys, or a cycle
        raise ValueError(f"not JSON-serialisable: {exc}") from exc
    _reject_floats(obj, "$")
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:  # a lone surrogate in some string
        raise ValueError(f"not encodable as UTF-8: {exc}") from exc


def object_id(obj: object) -> str:
    """Lowercase hex SHA-256 of ``canonical_bytes(obj)``."""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def _reject_floats(value: object, path: str) -> None:
    """Raise ValueError on any float or non-string key; ``path`` locates the value.

    Depth-first in document order, as a recursive walk would report the first problem, but
    with an explicit stack, so a deeply nested record cannot overflow the interpreter's stack.
    """
    stack: list[tuple[bool, object, str]] = [(False, value, path)]  # (is a key, item, where)
    while stack:
        is_key, item, where = stack.pop()
        if is_key:
            if not isinstance(item, str):
                raise ValueError(f"canonical JSON needs string keys; found {item!r} at {where}")
            continue
        if isinstance(item, float):
            raise ValueError(f"canonical JSON forbids floats; found {item!r} at {where}")
        children: list[tuple[bool, object, str]] = []
        if isinstance(item, dict):
            for key, child in item.items():
                children += [(True, key, where), (False, child, f"{where}.{key}")]
        elif isinstance(item, list | tuple):
            children = [(False, child, f"{where}[{index}]") for index, child in enumerate(item)]
        stack.extend(reversed(children))
