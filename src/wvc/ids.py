"""Ids, counts, and raw bytes: the small predicates and types every layer shares.

A leaf module (standard library and numpy only), so that the record validators, the codecs,
and the store can all import it without a cycle.
"""

from __future__ import annotations

import re
from typing import TypeAlias, TypeGuard

import numpy as np

SHA256_HEX = re.compile(r"[0-9a-f]{64}")  # a SHA-256 id: 64 lowercase hex characters
SHORT_ID = 12  # hex characters of an id shown in messages, tables, and summaries

# A tensor's or a payload's stored bytes: a 1-D uint8 array, or anything bytes-like.
RawBytes: TypeAlias = np.ndarray | bytes | bytearray | memoryview


def is_id(value: object) -> TypeGuard[str]:
    """A SHA-256 id: 64 lowercase hex characters."""
    return isinstance(value, str) and SHA256_HEX.fullmatch(value) is not None


def is_count(value: object) -> TypeGuard[int]:
    """A non-negative int; bool is refused although it subclasses int (JSON true/false)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def short(object_id: str | None) -> str:
    """The first ``SHORT_ID`` characters of a commit or tensor id; "none" for no id."""
    return "none" if object_id is None else object_id[:SHORT_ID]
