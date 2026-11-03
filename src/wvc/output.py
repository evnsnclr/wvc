"""Standard output: documents as bytes, messages through a forgiving text layer.

A document (a manifest, gguf-hash lines, JSON, model-card YAML) must be the same bytes on
every platform, so ``emit_bytes`` writes past the text layer of standard output: UTF-8, the
LF line endings the document carries, no code-page or newline translation. Messages keep
going through ``print``; ``configure_streams`` makes that text layer escape what a narrow
console cannot encode instead of raising.
"""

from __future__ import annotations

import json
import sys


def emit_bytes(data: bytes) -> None:
    """Write ``data`` to standard output unchanged, after what the text layer still buffers."""
    stream = sys.stdout
    stream.flush()
    buffer = getattr(stream, "buffer", None)
    if buffer is None:  # a text stream without a byte layer, such as a StringIO
        stream.write(data.decode("utf-8"))
        return
    buffer.write(data)
    buffer.flush()


def emit_text(text: str) -> None:
    """Write ``text`` as UTF-8 bytes, with the line endings it carries."""
    emit_bytes(text.encode("utf-8"))


def emit_json(obj: object, *, allow_nan: bool = True) -> None:
    """Write ``obj`` as two-space-indented JSON with a trailing newline, as UTF-8 bytes."""
    emit_text(json.dumps(obj, indent=2, allow_nan=allow_nan) + "\n")


def configure_streams() -> None:
    """Make standard output and error escape unencodable characters instead of raising."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
