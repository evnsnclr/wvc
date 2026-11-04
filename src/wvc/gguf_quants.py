"""Dequantise ggml block types to float32, exactly as llama.cpp's reference code does.

Each function takes the stored bytes of whole blocks (a uint8 array) and returns one float32
per element, in storage order. The block layouts and the arithmetic follow the scalar
reference functions ``dequantize_row_*`` in llama.cpp's ``ggml/src/ggml-quants.c``: every
product and sum is a single float32 operation in the order the reference writes it (scale
times quant, then minus or plus the offset), so the results are bit-identical to the reference
evaluated without fused multiply-add, and to gguf-py's NumPy implementation, which the tests
compare against block by block. Plain types (F32, F16, BF16, F64, the integers, F8_E4M3) are
decoded as their safetensors twins are.

Types outside ``DEQUANTIZERS`` are refused by name, never skipped.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wvc.errors import WvcError
from wvc.safetensors_io import decodable

QK_K = 256  # elements in a K-quant super-block


class DequantError(WvcError):
    """A tensor's type cannot be dequantised by wvc."""


def _f16(blocks: np.ndarray, start: int) -> np.ndarray:
    """The little-endian float16 at byte ``start`` of every block, as an (n, 1) float32 column."""
    raw = np.ascontiguousarray(blocks[:, start : start + 2])
    return raw.view("<f2").astype(np.float32)


def _nibbles(qs: np.ndarray) -> np.ndarray:
    """Low nibbles of every byte, then high nibbles: (n, k) bytes -> (n, 2k) values."""
    return np.concatenate([qs & 0x0F, qs >> 4], axis=1)


def _high_bits(qh: np.ndarray) -> np.ndarray:
    """Bit ``e`` of the little-endian u32 ``qh`` for e in 0..31, as an (n, 32) array of 0/1."""
    word = np.ascontiguousarray(qh).view("<u4").astype(np.uint32)
    return ((word >> np.arange(32, dtype=np.uint32)) & 1).astype(np.uint8)


def q4_0(blocks: np.ndarray) -> np.ndarray:
    d = _f16(blocks, 0)
    q = _nibbles(blocks[:, 2:18]).astype(np.int16) - 8
    return q.astype(np.float32) * d


def q4_1(blocks: np.ndarray) -> np.ndarray:
    d, m = _f16(blocks, 0), _f16(blocks, 2)
    q = _nibbles(blocks[:, 4:20]).astype(np.float32)
    return q * d + m


def q5_0(blocks: np.ndarray) -> np.ndarray:
    d = _f16(blocks, 0)
    q = _nibbles(blocks[:, 6:22]) | (_high_bits(blocks[:, 2:6]) << 4)
    return (q.astype(np.int16) - 16).astype(np.float32) * d


def q5_1(blocks: np.ndarray) -> np.ndarray:
    d, m = _f16(blocks, 0), _f16(blocks, 2)
    q = _nibbles(blocks[:, 8:24]) | (_high_bits(blocks[:, 4:8]) << 4)
    return q.astype(np.float32) * d + m


def q8_0(blocks: np.ndarray) -> np.ndarray:
    d = _f16(blocks, 0)
    return np.ascontiguousarray(blocks[:, 2:34]).view(np.int8).astype(np.float32) * d


def q8_k(blocks: np.ndarray) -> np.ndarray:
    d = np.ascontiguousarray(blocks[:, 0:4]).view("<f4").astype(np.float32)
    return np.ascontiguousarray(blocks[:, 4:260]).view(np.int8).astype(np.float32) * d


def _k4_scales(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The eight 6-bit scales and mins of a Q4_K or Q5_K block (``get_scale_min_k4``)."""
    low, mid, top = scales[:, 0:4], scales[:, 4:8], scales[:, 8:12]
    sc = np.concatenate([low & 63, (top & 0x0F) | ((low >> 6) << 4)], axis=1)
    mn = np.concatenate([mid & 63, (top >> 4) | ((mid >> 6) << 4)], axis=1)
    return sc.astype(np.float32), mn.astype(np.float32)


def _k_offsets(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per 32-element sub-block: ``d * scale`` and ``dmin * min``, each (n, 8, 1)."""
    d, dmin = _f16(blocks, 0), _f16(blocks, 2)
    sc, mn = _k4_scales(blocks[:, 4:16])
    return (d * sc)[:, :, None], (dmin * mn)[:, :, None]


def _k_nibbles(qs: np.ndarray) -> np.ndarray:
    """Q4_K/Q5_K quants: per 32-byte group the low nibbles, then the high ones: (n, 8, 32)."""
    groups = qs.reshape(len(qs), 4, 1, 32)
    return np.concatenate([groups & 0x0F, groups >> 4], axis=2).reshape(len(qs), 8, 32)


def q4_k(blocks: np.ndarray) -> np.ndarray:
    d1, m1 = _k_offsets(blocks)
    q = _k_nibbles(blocks[:, 16:144]).astype(np.float32)
    return (d1 * q - m1).reshape(len(blocks), QK_K)


def q5_k(blocks: np.ndarray) -> np.ndarray:
    d1, m1 = _k_offsets(blocks)
    qh = blocks[:, 16:48]  # bit 2c of qh[l] tops sub-block 2c, bit 2c + 1 tops 2c + 1
    high = (qh[:, None, :] >> np.arange(8, dtype=np.uint8)[None, :, None]) & 1
    q = (_k_nibbles(blocks[:, 48:176]) | (high << 4)).astype(np.float32)
    return (d1 * q - m1).reshape(len(blocks), QK_K)


def q6_k(blocks: np.ndarray) -> np.ndarray:
    n = len(blocks)
    ql = blocks[:, 0:128].reshape(n, 2, 2, 32)  # [half, low/high 32 bytes, l]
    qh = blocks[:, 128:192].reshape(n, 2, 1, 32)  # [half, -, l]
    scales = np.ascontiguousarray(blocks[:, 192:208]).view(np.int8).astype(np.float32)
    d = _f16(blocks, 208)
    low = np.concatenate([ql & 0x0F, ql >> 4], axis=2)  # q1, q2, q3, q4 low bits per half
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)[None, None, :, None]
    high = (qh >> shifts) & 3
    q = ((low | (high << 4)).astype(np.int16) - 32).astype(np.float32)  # (n, 2, 4, 32)
    scale = (d * scales).reshape(n, 2, 4, 2, 1)  # 16 scales: [half, quarter, 16-element run]
    return (scale * q.reshape(n, 2, 4, 2, 16)).reshape(n, QK_K)


DEQUANTIZERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Q4_0": q4_0,
    "Q4_1": q4_1,
    "Q5_0": q5_0,
    "Q5_1": q5_1,
    "Q8_0": q8_0,
    "Q4_K": q4_k,
    "Q5_K": q5_k,
    "Q6_K": q6_k,
    "Q8_K": q8_k,
}


def can_dequantize(dtype: str) -> bool:
    """True when ``dequantize`` decodes the dtype: a supported block type or a plain one."""
    return dtype in DEQUANTIZERS or decodable(dtype)


def supported_text() -> str:
    return ", ".join(["F32", "F16", "BF16", *DEQUANTIZERS])


def dequantize(raw: np.ndarray, dtype: str, block: int, size: int) -> np.ndarray:
    """Float32 values of whole ``dtype`` blocks held in the uint8 array ``raw``.

    ``block`` and ``size`` are the type's elements and bytes per block. A type with no
    dequantiser is refused by name.
    """
    function = DEQUANTIZERS.get(dtype)
    if function is None:
        raise DequantError(
            f"cannot dequantise ggml type {dtype}: wvc dequantises {supported_text()}"
        )
    if len(raw) % size:
        raise DequantError(f"{len(raw)} bytes are not a whole number of {dtype} blocks")
    blocks = np.asarray(raw, dtype=np.uint8).reshape(-1, size)
    with np.errstate(over="ignore", invalid="ignore"):  # NaN and infinite scales propagate
        values = function(blocks)
    return np.ascontiguousarray(values, dtype=np.float32).reshape(len(blocks) * block)
