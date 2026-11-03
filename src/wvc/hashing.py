"""Content hashes for tensors and checkpoints.

A tensor's digest is the SHA-256 of its raw bytes exactly as stored, so it can be
cross-checked against hashes other people publish. A manifest lists every tensor with
its dtype, shape, byte count, and digest. The manifest root is the SHA-256 of the
name-sorted manifest lines, so two checkpoints holding identical tensors under identical
names share a root regardless of sharding or header order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import numpy as np

from wvc.errors import WvcError
from wvc.safetensors_io import Checkpoint, offset_order

MANIFEST_FORMAT = "wvc-manifest/0"
HASH_CHUNK_BYTES = 64 * 1024 * 1024
GGUF_HASH_TYPE = "sha256"  # the hash type column of a llama.cpp gguf-hash manifest line


class Hasher(Protocol):
    """What ``digest_bytes`` needs from a second hash object."""

    def update(self, data: memoryview | bytes, /) -> object: ...


def digest_bytes(view: np.ndarray, *, also: Hasher | None = None) -> str:
    """SHA-256 of a contiguous uint8 view, hashed in bounded chunks; ``also`` sees every chunk."""
    hasher = hashlib.sha256()
    for start in range(0, len(view), HASH_CHUNK_BYTES):
        chunk = np.ascontiguousarray(view[start : start + HASH_CHUNK_BYTES]).data
        hasher.update(chunk)
        if also is not None:
            also.update(chunk)
    return hasher.hexdigest()


ESCAPES = {"\\": "\\\\", "\t": "\\t", "\n": "\\n", "\r": "\\r"}


def escape_name(name: str) -> str:
    """Escape the four characters that could make a manifest line ambiguous."""
    return "".join(ESCAPES.get(ch, ch) for ch in name)


def manifest_line(name: str, dtype: str, shape: tuple[int, ...], sha256: str) -> str:
    return f"{escape_name(name)}\t{dtype}\t{'x'.join(str(dim) for dim in shape)}\t{sha256}\n"


def manifest_root(entries: Mapping[str, Mapping[str, object]]) -> str:
    """Hash the name-sorted manifest lines; raises ValueError on a malformed entry."""
    hasher = hashlib.sha256()
    for name in sorted(entries):
        entry = entries[name]
        dtype, shape, sha256 = entry.get("dtype"), entry.get("shape"), entry.get("sha256")
        if not (
            isinstance(dtype, str) and isinstance(shape, list | tuple) and isinstance(sha256, str)
        ):
            raise ValueError(f"malformed manifest entry for tensor {name!r}: {entry!r}")
        line = manifest_line(name, dtype, tuple(int(dim) for dim in shape), sha256)
        hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()


def tensor_digests(ckpt: Checkpoint) -> dict[str, str]:
    """Every tensor's digest, read file by file in offset order (one file mapped at a time).

    The dict lists the tensors in that reading order; manifests sort by name afterwards.
    """
    return {
        info.name: digest_bytes(ckpt.raw(info.name))
        for file in ckpt.files
        for info in offset_order(file)
    }


def hash_files(ckpt: Checkpoint) -> tuple[dict[str, str], dict[Path, str]]:
    """Per-tensor digests and, per file, the SHA-256 over its tensor bytes in offset order.

    Files are read one after another, so a checkpoint keeps only one file in use at a time.
    Every tensor's bytes are read once. The file hash is what gguf-hash prints for a whole
    model: the tensors' bytes concatenated in offset order, which equals the file's data
    buffer only when the tensors tile it without gaps or overlaps.
    """
    digests: dict[str, str] = {}
    files: dict[Path, str] = {}
    for file in ckpt.files:
        whole = hashlib.sha256()
        for info in offset_order(file):
            digests[info.name] = digest_bytes(ckpt.raw(info.name), also=whole)
        files[file.path] = whole.hexdigest()
    return digests, files


def check_gguf_name(text: str) -> None:
    """Refuse a file or tensor name that a whitespace-separated gguf-hash line cannot carry."""
    if any(ch.isspace() or not ch.isprintable() for ch in text):
        raise WvcError(
            f"cannot write {text!r} as a gguf-hash line: names with whitespace or "
            "non-printable characters have no place in the format"
        )


def gguf_hash_lines(ckpt: Checkpoint) -> list[str]:
    """The checkpoint as llama.cpp ``gguf-hash`` manifest lines.

    Per file, in name order: one ``sha256    <hex>  <file>:<tensor>`` line per tensor in offset
    order, then ``sha256    <hex>  <file>`` over the file's tensor bytes. The file column is the
    file's base name. ``gguf-hash --check`` splits a line on whitespace, so a file or tensor
    name containing whitespace or a non-printable character cannot be written and is refused.
    """
    for file in ckpt.files:
        for text in (file.path.name, *file.tensors):
            check_gguf_name(text)
    digests, files = hash_files(ckpt)
    lines = []
    for file in ckpt.files:
        name = file.path.name
        for info in offset_order(file):
            lines.append(f"{GGUF_HASH_TYPE:<8}  {digests[info.name]}  {name}:{info.name}")
        lines.append(f"{GGUF_HASH_TYPE:<8}  {files[file.path]}  {name}")
    return lines


def build_manifest(ckpt: Checkpoint, digests: Mapping[str, str] | None = None) -> dict:
    """Describe every tensor of a checkpoint and compute the manifest root."""
    digests = dict(digests) if digests is not None else tensor_digests(ckpt)
    entries = {
        name: {
            "dtype": info.dtype,
            "shape": list(info.shape),
            "nbytes": info.nbytes,
            "sha256": digests[name],
        }
        for name, info in ckpt.tensors.items()
    }
    return {
        "format": MANIFEST_FORMAT,
        "root": manifest_root(entries),
        "tensor_count": len(entries),
        "total_bytes": sum(info.nbytes for info in ckpt.tensors.values()),
        "tensors": dict(sorted(entries.items())),
    }
