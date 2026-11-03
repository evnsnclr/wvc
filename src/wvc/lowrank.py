"""The ``lowrank-merge/0`` procedure of spec section 3.2: W' = W + scale * (B @ A), byte-exact.

W is a base tensor of shape [out, in]; A ([rank, in]) and B ([out, rank]) are the factors of a
LoRA adapter. Each of the three is F32, F16, or BF16 and is decoded to float64. D = B @ A is
accumulated in float64 from +0.0 as the outer products of B[:, k] and A[k, :] for k in increasing
order, with numpy elementwise operations only (each one correctly rounded IEEE arithmetic, never a
BLAS product), so every machine computes the same bits. W + scale * D is rounded to float32
(round-to-nearest-even) and then to the base dtype: BF16 by the round-to-nearest-even bit trick,
F16 and F32 by numpy casts. W is processed in blocks of rows to bound memory; every element goes
through the same sequence of operations whatever the block size, so the blocks change no bit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from wvc.errors import WvcError
from wvc.ids import RawBytes
from wvc.safetensors_io import ITEMSIZES, to_float32

PROCEDURE = "lowrank-merge/0"
FLOAT_DTYPES = ("F32", "F16", "BF16")  # the dtypes W, A, and B may have
NUMPY_CASTS = {"F32": "<f4", "F16": "<f2"}  # BF16 is rounded by f32_to_bf16_bits instead
BLOCK_ELEMENTS = 1 << 20  # elements of W merged at a time; any value gives the same bytes


class LowRankError(WvcError):
    """Inputs of the merge are malformed: an unsupported dtype, or inconsistent shapes or sizes."""


@dataclass(frozen=True, eq=False)  # arrays have no single truth value to compare by
class LowRankFactors:
    """One adapter target's factors exactly as stored, and the scale they are merged with."""

    a_raw: RawBytes  # A's bytes (bytes-like or 1-D uint8); A has shape a_shape = [rank, in]
    a_dtype: str
    a_shape: Sequence[int]
    b_raw: RawBytes  # B's bytes (bytes-like or 1-D uint8); B has shape b_shape = [out, rank]
    b_dtype: str
    b_shape: Sequence[int]
    scale: float


def merge_lowrank(
    base_raw: RawBytes, base_dtype: str, shape: Sequence[int], factors: LowRankFactors
) -> np.ndarray:
    """The bytes (1-D uint8) of W + scale * (B @ A) in ``base_dtype`` under ``lowrank-merge/0``.

    W is ``base_raw`` with ``shape``; A, B, and the scale come from ``factors``. Raw inputs are
    bytes-like or 1-D uint8 arrays holding little-endian values. LowRankError when a dtype is
    not F32, F16, or BF16, the shapes are not [out, in], [rank, in], and [out, rank] with
    rank >= 1, a byte length does not match its shape, or the scale is not a finite number.
    """
    out_dim, in_dim, rank = check_factor_shapes(shape, factors.a_shape, factors.b_shape)
    factor = check_scale(factors.scale)
    w = _decode(base_raw, base_dtype, (out_dim, in_dim), "W")
    a = _decode(factors.a_raw, factors.a_dtype, (rank, in_dim), "A").astype(np.float64)
    b = _decode(factors.b_raw, factors.b_dtype, (out_dim, rank), "B").astype(np.float64)
    merged = np.empty((out_dim, in_dim), dtype=np.float32)
    rows = max(1, BLOCK_ELEMENTS // max(1, in_dim))
    with np.errstate(over="ignore", invalid="ignore"):  # IEEE overflow and NaN are the result
        for start in range(0, out_dim, rows):
            block = slice(start, start + rows)
            merged[block] = _merge_block(w[block], a, b[block], factor)
        return _encode(merged, base_dtype)


def check_factor_shapes(
    shape: Sequence[int], a_shape: Sequence[int], b_shape: Sequence[int]
) -> tuple[int, int, int]:
    """Check W [out, in], A [rank, in], and B [out, rank] with rank >= 1; return (out, in, rank)."""
    out_dim, in_dim = _pair(shape, "W")
    rank, a_in = _pair(a_shape, "A")
    b_out, b_rank = _pair(b_shape, "B")
    if rank < 1 or (a_in, b_out, b_rank) != (in_dim, out_dim, rank):
        raise LowRankError(
            f"A {list(a_shape)} and B {list(b_shape)} do not factor W {list(shape)}: A must be "
            f"[rank, {in_dim}] and B [{out_dim}, rank] with the same rank of at least 1"
        )
    return out_dim, in_dim, rank


def check_scale(scale: object) -> float:
    """``scale`` as a finite float; LowRankError for anything else, bool included."""
    if isinstance(scale, bool) or not isinstance(scale, int | float):
        raise LowRankError(f"scale must be a finite number, got {scale!r}")
    try:
        value = float(scale)
    except OverflowError:  # an int beyond the float range
        value = math.inf
    if not math.isfinite(value):
        raise LowRankError(f"scale must be a finite number, got {scale!r}")
    return value


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 values to bfloat16 (round-to-nearest-even) and return the 16-bit patterns."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype(np.uint16)


def _merge_block(w: np.ndarray, a: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    """Rows of W + scale * D as float32, with D accumulated in float64 for k ascending."""
    delta = np.zeros(w.shape, dtype=np.float64)
    term = np.empty(w.shape, dtype=np.float64)
    for k in range(a.shape[0]):
        np.outer(b[:, k], a[k], out=term)  # numpy computes outer as a broadcast elementwise product
        delta += term
    return (w.astype(np.float64) + scale * delta).astype(np.float32)


def _decode(raw: object, dtype: object, shape: tuple[int, int], what: str) -> np.ndarray:
    """``raw`` as a float32 array of ``shape``; LowRankError on a dtype or byte-length mismatch."""
    if not isinstance(dtype, str) or dtype not in FLOAT_DTYPES:
        raise LowRankError(f"{what}: dtype {dtype!r} is not one of {', '.join(FLOAT_DTYPES)}")
    data = _uint8(raw, what)
    expected = math.prod(shape) * ITEMSIZES[dtype]
    if data.size != expected:
        raise LowRankError(
            f"{what}: {data.size} bytes cannot hold {dtype}{list(shape)} ({expected} bytes)"
        )
    return to_float32(data, dtype, shape)


def _encode(values: np.ndarray, dtype: str) -> np.ndarray:
    """float32 values as the little-endian bytes (1-D uint8) of F32, F16, or BF16."""
    if dtype == "BF16":
        encoded = f32_to_bf16_bits(values).astype("<u2", copy=False)
    else:
        encoded = values.astype(NUMPY_CASTS[dtype])
    return np.ascontiguousarray(encoded).reshape(-1).view(np.uint8)


def _uint8(raw: object, what: str) -> np.ndarray:
    """Bytes-like input or a 1-D uint8 array as a contiguous 1-D uint8 array."""
    if isinstance(raw, bytes | bytearray | memoryview):
        return np.frombuffer(raw, dtype=np.uint8)
    if isinstance(raw, np.ndarray) and raw.dtype == np.uint8 and raw.ndim == 1:
        return np.ascontiguousarray(raw)
    raise LowRankError(f"{what}: expected bytes or a 1-D uint8 array")


def _pair(shape: object, what: str) -> tuple[int, int]:
    """A shape of two non-negative ints (bool refused); LowRankError otherwise."""
    if not (
        isinstance(shape, list | tuple)
        and len(shape) == 2
        and all(isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape)
    ):
        raise LowRankError(f"{what}: a shape must be two non-negative integers, got {shape!r}")
    return shape[0], shape[1]
