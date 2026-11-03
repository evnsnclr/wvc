"""Received text in messages: escaped, and cut to a bounded length.

A leaf module (standard library only), so that every module raising an error can import it
without an import cycle.
"""

from __future__ import annotations

SHOWN_LIMIT = 40  # characters of a received value's repr kept in a message
ELLIPSIS = "…"


def printable(text: object) -> str:
    """Escape characters a terminal could act on (control characters, ESC, BEL) as \\xNN."""
    return "".join(ch if ch.isprintable() or ch == " " else f"\\x{ord(ch):02x}" for ch in str(text))


def shown(value: object, limit: int = SHOWN_LIMIT) -> str:
    """``repr(value)`` for a message: control characters escaped, at most ``limit`` kept.

    A longer repr is cut to ``limit`` characters and marked with an ellipsis and the original
    length: a string's character count (its closing quote kept), otherwise the repr's, as in
    ``'aaaa…' (5000 characters)``.
    """
    text = printable(repr(value))
    if len(text) <= limit:
        return text
    if isinstance(value, str):
        return f"{text[:limit]}{ELLIPSIS}{text[-1]} ({len(value)} characters)"
    return f"{text[:limit]}{ELLIPSIS} ({len(text)} characters)"
