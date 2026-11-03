"""Tensor byte codecs of spec section 3.2: zstd byte planes, XOR deltas, and sparse deltas.

Pure functions over contiguous 1-D ``uint8`` arrays holding a tensor's stored bytes. A payload
is a concatenation of zstd frames (level 3) whose sizes the object record keeps. Tensors are
never decoded to numbers: deltas compare and combine raw bytes. The ``delta-lowrank`` payload
(adapter factors A then B, uncompressed) is only split and joined here; ``wvc.lowrank`` merges
it. Integrity checks (SHA-256) belong to the store; decoders only reject malformed input,
raising CodecError.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate

import numpy as np
import zstandard

from wvc.errors import WvcError
from wvc.ids import RawBytes, is_count

ZSTD_LEVEL = 3
SPARSE_DENOMINATOR = 20  # delta-sparse is eligible when changed elements * 20 <= elements (5%)
UINT32_LIMIT = 2**32  # delta-sparse stores element indices as uint32
INDEX_BYTES = 4
SIZE_NOT_DECLARED = -1  # zstandard.frame_content_size when a frame header omits the size
UINT_VIEWS = {size: np.dtype(f"u{size}") for size in (1, 2, 4, 8)}  # compare elements as integers
# The forms of spec section 3.2, as object records name them.
FULL_FORM = "zstd-planes"
XOR_FORM = "delta-xor-planes"
SPARSE_FORM = "delta-sparse"
LOWRANK_FORM = "delta-lowrank"  # written only by merge-adapter (spec sections 3.2 and 5.3)


class CodecError(WvcError):
    """Codec input is malformed: bad arrays, inconsistent sizes, or undecodable frames."""


@dataclass(frozen=True)
class Encoded:
    """One encoded tensor: its form, payload, and record fields other than base and payload."""

    form: str  # FULL_FORM, XOR_FORM, or SPARSE_FORM
    payload: bytes
    fields: dict[str, int | list[int]]  # {"plane_sizes": [...]} or index_size and value_size


def encode_planes(raw: np.ndarray, itemsize: int) -> tuple[bytes, list[int]]:
    """Split ``raw`` into ``itemsize`` byte-position planes and compress each as one frame.

    Plane j holds the bytes at positions congruent to j modulo ``itemsize``. Returns the
    concatenated frames and their sizes in plane order.
    """
    rows = _check_raw(raw, itemsize, "tensor").reshape(-1, itemsize)
    compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    frames = [compressor.compress(_buffer(rows[:, j])) for j in range(itemsize)]
    return b"".join(frames), [len(frame) for frame in frames]


def decode_planes(
    payload: bytes, plane_sizes: Sequence[int], itemsize: int, nbytes: int
) -> np.ndarray:
    """Inverse of ``encode_planes``: decompress each plane and interleave ``nbytes`` bytes.

    Every frame's declared content size is checked against ``nbytes`` before the output is
    allocated, so a record whose frames do not match is refused without a large allocation.
    """
    _check_itemsize(itemsize)
    if not is_count(nbytes) or nbytes % itemsize:
        raise CodecError(f"nbytes must be a non-negative multiple of {itemsize}, got {nbytes!r}")
    if not isinstance(plane_sizes, list | tuple) or len(plane_sizes) != itemsize:
        raise CodecError(f"expected {itemsize} plane sizes, got {plane_sizes!r}")
    frames = _split(payload, plane_sizes, "planes")
    plane_bytes = nbytes // itemsize
    for j, frame in enumerate(frames):
        _check_declared(frame, plane_bytes, f"plane {j}", exact=True)
    rows = np.empty((plane_bytes, itemsize), dtype=np.uint8)
    for j, frame in enumerate(frames):
        plane = _decompress(frame, plane_bytes, f"plane {j}", exact=True)
        rows[:, j] = np.frombuffer(plane, dtype=np.uint8)
    return rows.reshape(-1)


def encode_xor_planes(
    base_raw: np.ndarray, raw: np.ndarray, itemsize: int
) -> tuple[bytes, list[int]]:
    """``encode_planes`` applied to ``base_raw XOR raw``; both must have the same length."""
    base, new = _check_pair(base_raw, raw, itemsize)
    return encode_planes(np.bitwise_xor(base, new), itemsize)


def decode_xor_planes(
    base_raw: np.ndarray, payload: bytes, plane_sizes: Sequence[int], itemsize: int
) -> np.ndarray:
    """Inverse of ``encode_xor_planes``: decode the XOR planes and XOR them with the base."""
    base = _check_raw(base_raw, itemsize, "base")
    return np.bitwise_xor(base, decode_planes(payload, plane_sizes, itemsize, base.size))


def sparse_eligible(base_raw: np.ndarray, raw: np.ndarray, itemsize: int) -> bool:
    """True when at most 5% of the elements differ and every changed index fits in uint32."""
    base, new = _check_pair(base_raw, raw, itemsize)
    return _sparse_ok(_changed_indices(base, new, itemsize), new.size // itemsize)


def encode_sparse(base_raw: np.ndarray, raw: np.ndarray, itemsize: int) -> tuple[bytes, int, int]:
    """Encode the elements of ``raw`` that differ from ``base_raw`` as a delta-sparse payload.

    The payload is one zstd frame of the changed element indices (uint32 little-endian,
    ascending) followed by one frame of those elements' bytes in index order. Returns the
    payload, the index frame size, and the value frame size.
    """
    base, new = _check_pair(base_raw, raw, itemsize)
    indices = _changed_indices(base, new, itemsize)
    if indices.size and int(indices[-1]) >= UINT32_LIMIT:
        raise CodecError(f"element index {int(indices[-1])} does not fit in uint32")
    return _sparse_payload(new, indices, itemsize)


def decode_sparse(
    base_raw: np.ndarray, payload: bytes, index_size: int, value_size: int, itemsize: int
) -> np.ndarray:
    """Inverse of ``encode_sparse``: copy the base and overwrite the listed elements."""
    base = _check_raw(base_raw, itemsize, "base")
    numel = base.size // itemsize
    index_frame, value_frame = _split(payload, (index_size, value_size), SPARSE_FORM)
    index_bytes = _decompress(index_frame, INDEX_BYTES * numel, "sparse indices", exact=False)
    if len(index_bytes) % INDEX_BYTES:
        raise CodecError(f"sparse indices: {len(index_bytes)} bytes is not a multiple of 4")
    indices = np.frombuffer(index_bytes, dtype="<u4")
    _check_indices(indices, numel)
    values = _decompress(value_frame, indices.size * itemsize, "sparse values", exact=True)
    rows = base.reshape(-1, itemsize).copy()
    rows[indices] = np.frombuffer(values, dtype=np.uint8).reshape(-1, itemsize)
    return rows.reshape(-1)


def encode_lowrank(a_raw: RawBytes, b_raw: RawBytes) -> bytes:
    """The ``delta-lowrank`` payload: A's bytes followed by B's, uncompressed (factors are small).

    Each factor is bytes-like or a 1-D uint8 array holding its raw stored bytes.
    """
    return _factor_bytes(a_raw, "A") + _factor_bytes(b_raw, "B")


def decode_lowrank(payload: bytes, a_size: int) -> tuple[bytes, bytes]:
    """Inverse of ``encode_lowrank``: the first ``a_size`` bytes are A's and the rest are B's."""
    if not is_count(a_size) or a_size > len(payload):
        raise CodecError(
            f"delta-lowrank: a_size {a_size!r} does not fit the {len(payload)}-byte payload"
        )
    return bytes(payload[:a_size]), bytes(payload[a_size:])


def choose_encoding(raw: np.ndarray, itemsize: int, base_raw: np.ndarray | None = None) -> Encoded:
    """Encode ``raw`` in the form with the smallest payload (spec section 3.2).

    ``zstd-planes`` is always a candidate. With a base of equal length, ``delta-xor-planes``
    and, when eligible, ``delta-sparse`` are candidates too. Ties keep the earlier candidate in
    that order, so a delta is kept only when its payload is strictly smaller than the
    ``zstd-planes`` payload.
    """
    payload, plane_sizes = encode_planes(raw, itemsize)
    candidates = [Encoded(FULL_FORM, payload, {"plane_sizes": plane_sizes})]
    if base_raw is not None and len(base_raw) == len(raw):
        candidates += _delta_candidates(base_raw, raw, itemsize)
    return min(candidates, key=lambda encoded: len(encoded.payload))  # the first minimum wins


def _delta_candidates(base_raw: np.ndarray, raw: np.ndarray, itemsize: int) -> list[Encoded]:
    """The delta-xor-planes candidate, plus delta-sparse when it is eligible."""
    base, new = _check_pair(base_raw, raw, itemsize)
    payload, plane_sizes = encode_planes(np.bitwise_xor(base, new), itemsize)
    candidates = [Encoded(XOR_FORM, payload, {"plane_sizes": plane_sizes})]
    indices = _changed_indices(base, new, itemsize)
    if _sparse_ok(indices, new.size // itemsize):
        payload, index_size, value_size = _sparse_payload(new, indices, itemsize)
        sizes: dict[str, int | list[int]] = {"index_size": index_size, "value_size": value_size}
        candidates.append(Encoded(SPARSE_FORM, payload, sizes))
    return candidates


def _changed_indices(base: np.ndarray, new: np.ndarray, itemsize: int) -> np.ndarray:
    """Ascending indices of the elements whose bytes differ."""
    view = UINT_VIEWS.get(itemsize)
    if view is not None:
        differs = base.view(view) != new.view(view)
    else:
        differs = (base.reshape(-1, itemsize) != new.reshape(-1, itemsize)).any(axis=1)
    return np.flatnonzero(differs)


def _sparse_ok(indices: np.ndarray, numel: int) -> bool:
    fits = indices.size == 0 or int(indices[-1]) < UINT32_LIMIT
    return indices.size * SPARSE_DENOMINATOR <= numel and fits


def _sparse_payload(new: np.ndarray, indices: np.ndarray, itemsize: int) -> tuple[bytes, int, int]:
    compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    index_frame = compressor.compress(indices.astype("<u4").tobytes())
    value_frame = compressor.compress(new.reshape(-1, itemsize)[indices].tobytes())
    return index_frame + value_frame, len(index_frame), len(value_frame)


def _check_indices(indices: np.ndarray, numel: int) -> None:
    """Sparse indices must be strictly ascending and address existing elements."""
    if indices.size and int(indices[-1]) >= numel:
        raise CodecError(f"sparse index {int(indices[-1])} is out of range for {numel} elements")
    if np.any(indices[1:] <= indices[:-1]):
        raise CodecError("sparse indices are not strictly ascending")


def _check_itemsize(itemsize: object) -> None:
    if not is_count(itemsize) or itemsize == 0:
        raise CodecError(f"itemsize must be a positive integer, got {itemsize!r}")


def _check_raw(raw: object, itemsize: int, what: str) -> np.ndarray:
    """Return ``raw`` as a contiguous 1-D uint8 array whose length ``itemsize`` divides."""
    _check_itemsize(itemsize)
    if not isinstance(raw, np.ndarray) or raw.dtype != np.uint8 or raw.ndim != 1:
        raise CodecError(f"{what} must be a 1-D uint8 array")
    if raw.size % itemsize:
        raise CodecError(f"{what} holds {raw.size} bytes, not a multiple of itemsize {itemsize}")
    return np.ascontiguousarray(raw)


def _check_pair(base_raw: object, raw: object, itemsize: int) -> tuple[np.ndarray, np.ndarray]:
    base, new = _check_raw(base_raw, itemsize, "base"), _check_raw(raw, itemsize, "tensor")
    if base.size != new.size:
        raise CodecError(f"base holds {base.size} bytes but the tensor holds {new.size}")
    return base, new


def _factor_bytes(raw: object, what: str) -> bytes:
    """A low-rank factor's raw bytes from bytes-like input or a 1-D uint8 array."""
    if isinstance(raw, bytes | bytearray | memoryview):
        return bytes(raw)
    return _check_raw(raw, 1, what).tobytes()


def _buffer(array: np.ndarray) -> memoryview:
    """A zero-copy byte view of an array (copied first only when it is not contiguous)."""
    return np.ascontiguousarray(array).data.cast("B")  # .data is a memoryview on every numpy


def _split(payload: bytes, sizes: Sequence[int], what: str) -> list[memoryview]:
    """Cut ``payload`` into consecutive frames of the given sizes, which must cover it exactly."""
    if not all(is_count(size) for size in sizes):
        raise CodecError(f"{what}: frame sizes must be non-negative integers, got {sizes!r}")
    view = memoryview(payload)
    if sum(sizes) != len(view):
        raise CodecError(
            f"{what}: frame sizes sum to {sum(sizes)} but the payload holds {len(view)} bytes"
        )
    return [view[end - size : end] for size, end in zip(sizes, accumulate(sizes), strict=True)]


def _decompress(frame: memoryview, limit: int, what: str, *, exact: bool) -> bytes:
    """Decompress one zstd frame holding at most ``limit`` bytes (exactly ``limit`` if exact)."""
    _check_declared(frame, limit, what, exact=exact)
    try:
        data = zstandard.ZstdDecompressor().decompress(
            frame, max_output_size=limit, allow_extra_data=False
        )
    except zstandard.ZstdError as exc:
        raise CodecError(f"{what}: not a valid zstd frame ({exc})") from exc
    if exact and len(data) != limit:
        raise CodecError(f"{what}: decoded {len(data)} bytes, expected {limit}")
    return data


def _check_declared(frame: memoryview, limit: int, what: str, *, exact: bool) -> None:
    """A frame's header must declare at most ``limit`` bytes (exactly ``limit`` if exact) or
    no size at all; read from the header alone, without decompressing."""
    try:
        declared = zstandard.frame_content_size(frame)
    except zstandard.ZstdError as exc:
        raise CodecError(f"{what}: not a valid zstd frame ({exc})") from exc
    if declared > limit or (exact and declared not in (limit, SIZE_NOT_DECLARED)):
        raise CodecError(f"{what}: frame declares {declared} bytes, expected {limit}")
