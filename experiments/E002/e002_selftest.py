"""Self-test for E002 on small synthetic safetensors files with known answers.

Checks that S5 and S6 rebuild B exactly (and that the verifier rejects a corrupted payload),
that S4u equals a byte-by-byte brute force and a hand count, that the S4 chunker equals a
sequential rolling-hash reference, and that the P4 generator draws and patches as specified.
"""

from __future__ import annotations

import json
import struct
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import zstandard
from e002_strategies import (
    CDC_BLOCK,
    CDC_MASK,
    CDC_MAX,
    CDC_MIN,
    CDC_WINDOW,
    MIN_RUN,
    ZSTD_LEVEL,
    buzhash_table,
    cdc_chunks,
    counterpart,
    f32_to_bf16_bits,
    range_bytes,
    s0_all_tensors,
    s2_tensor_dedup,
    s4u_cdc_upper_bound,
    s5_xor_delta,
    s6_sparse_delta,
    verify_payloads,
    window_hashes,
)
from e002_synthetic import P4_RANK, P4_REL_NORM, P4_SEED, bf16_bits_to_f32, generate_p4

from wvc.hashing import tensor_digests
from wvc.safetensors_io import Checkpoint, load_checkpoint

NUMPY_FOR = {"F32": "<f4", "F16": "<f2", "I32": "<i4", "I8": "i1", "U8": "u1", "BF16": "<u2"}
# Byte offsets flipped in B. sparse.bf16 (514,514 bytes) leaves unchanged runs of 3,000,
# 16,384, 16,383, 46,149, 118,079, 99,899, and 214,513 bytes; sparse.i8 (70,000 bytes) leaves
# runs of 100, 49,899, and 19,998 bytes, the last ending inside the partial tail block.
SPARSE_BF16_FLIPS = [3000, 19385, 35769, 81919, 81920, 200000, *range(200001, 200101), 300000]
SPARSE_I8_FLIPS = [100, 50000, 69999]
# Unchanged tensors (40,000 + 20 bytes) plus the runs of at least 16 KiB listed above.
SELFTEST_DEDUPABLE = 40_000 + 20 + (16_384 + 46_149 + 118_079 + 99_899 + 214_513)
SELFTEST_DEDUPABLE += 49_899 + 19_998
EXPECTED_S6 = {
    "dense.f32": "xor",
    "sparse.bf16": "sparse",
    "sparse.i8": "sparse",
    "unchanged.f32": "sparse",
    "unchanged_small.f16": "sparse",
    "reshaped.f32": "zstd",
    "at_threshold.bf16": "sparse",
    "over_threshold.bf16": "xor",
    "only_b.i32": "zstd",
}


def write_safetensors(path: Path, tensors: dict, metadata: dict | None = None) -> Path:
    """Write ``name -> (dtype, array)``; BF16 arrays hold bit patterns. From tests/conftest.py."""
    header: dict[str, object] = {}
    buffer = bytearray()
    for name, (dtype, array) in tensors.items():
        raw = np.ascontiguousarray(array).astype(NUMPY_FOR[dtype], copy=False).tobytes()
        begin = len(buffer)
        buffer += raw
        offsets = [begin, len(buffer)]
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": offsets}
    if metadata:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(buffer))
    return path


def flipped(array: np.ndarray, offsets: Iterable[int]) -> np.ndarray:
    """Copy of the array with the bytes at the given offsets XOR 0xFF."""
    out = np.array(array, copy=True)
    out.view(np.uint8).reshape(-1)[np.asarray(list(offsets))] ^= np.uint8(0xFF)
    return out


def selftest_tensors(rng: np.random.Generator) -> tuple[dict, dict]:
    """A dense change, sparse changes, unchanged, reshaped, and one-sided tensors."""
    dense = rng.standard_normal((64, 300)).astype("<f4")
    sparse = f32_to_bf16_bits(rng.standard_normal((257, 1001)))
    small = rng.integers(-128, 128, 70_000).astype("i1")
    same = rng.standard_normal((100, 100)).astype("<f4")
    tiny = rng.standard_normal(10).astype("<f2")
    shaped = rng.standard_normal((16, 8)).astype("<f4")
    edge = f32_to_bf16_bits(rng.standard_normal(1000))
    a = {
        "dense.f32": ("F32", dense),
        "sparse.bf16": ("BF16", sparse),
        "sparse.i8": ("I8", small),
        "unchanged.f32": ("F32", same),
        "unchanged_small.f16": ("F16", tiny),
        "reshaped.f32": ("F32", shaped),
        "at_threshold.bf16": ("BF16", edge),
        "over_threshold.bf16": ("BF16", edge),
        "only_a.f32": ("F32", same[:3]),
    }
    b = {  # a different header order than A
        "only_b.i32": ("I32", rng.integers(-1000, 1000, (33, 7)).astype("<i4")),
        "over_threshold.bf16": ("BF16", flipped(edge, [38 * i + 1 for i in range(51)])),
        "at_threshold.bf16": ("BF16", flipped(edge, [38 * i + 1 for i in range(50)])),
        "reshaped.f32": ("F32", shaped.reshape(8, 16)),
        "unchanged_small.f16": ("F16", tiny),
        "unchanged.f32": ("F32", same),
        "sparse.i8": ("I8", flipped(small, SPARSE_I8_FLIPS)),
        "sparse.bf16": ("BF16", flipped(sparse, SPARSE_BF16_FLIPS)),
        "dense.f32": ("F32", flipped(dense, range(0, dense.nbytes, 4))),
    }
    return a, b


def brute_force_dedupable(a: Checkpoint, b: Checkpoint) -> int:
    """Byte-by-byte reference for S4u's dedupable bytes."""
    total = 0
    for name in b.tensors:
        if not counterpart(a, b, name):
            continue
        x, y = bytes(a.raw(name)), bytes(b.raw(name))
        if x == y:
            total += len(x)
            continue
        run = 0
        for p, q in zip(x, y, strict=True):
            if p == q:
                run += 1
                continue
            total, run = total + (run if run >= MIN_RUN else 0), 0
        total += run if run >= MIN_RUN else 0
    return total


def corrupted(payloads: dict, name: str) -> dict:
    kind, frames = payloads[name]
    plane = bytearray(zstandard.ZstdDecompressor().decompress(frames[0]))
    plane[0] ^= 1
    frame = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(bytes(plane))
    return {**payloads, name: (kind, [frame, *frames[1:]])}


def selftest_strategies(tmp: Path) -> dict[str, bool]:
    tensors_a, tensors_b = selftest_tensors(np.random.default_rng(0))
    a = load_checkpoint(write_safetensors(tmp / "a.safetensors", tensors_a))
    b = load_checkpoint(write_safetensors(tmp / "b.safetensors", tensors_b, {"format": "pt"}))
    db = tensor_digests(b)
    s5, s6 = s5_xor_delta(a, b), s6_sparse_delta(a, b)
    ranges, stats = s4u_cdc_upper_bound(a, b)
    brute = brute_force_dedupable(a, b)
    s0 = range_bytes(s0_all_tensors(a, b)[0])
    s4u_ok = stats["dedupable_bytes"] == brute == SELFTEST_DEDUPABLE
    kinds = {name: payload[0] for name, payload in s6.items()}
    kept = set(tensors_b) - {name for name, _, _ in s2_tensor_dedup(a, b)[0]}
    rejected = not verify_payloads(a, b, corrupted(s5, "sparse.bf16"), db)[0]
    return {
        "S5 reconstructs B exactly": verify_payloads(a, b, s5, db)[0],
        "S6 reconstructs B exactly": verify_payloads(a, b, s6, db)[0],
        "S6 picks the sparse, XOR, or zstd encoding as specified": kinds == EXPECTED_S6,
        "verifier rejects a corrupted payload": rejected,
        "S4u equals a byte-by-byte brute force": s4u_ok and range_bytes(ranges) == s0 - brute,
        "S2 keeps exactly the unchanged tensors": kept == {"unchanged.f32", "unchanged_small.f16"},
    }


def naive_buzhash(data: bytes, table: np.ndarray) -> list[int]:
    """Rolling buzhash reference, h = rotl(h, 1) ^ T[in] ^ T[out]; valid from index 31."""
    t = [int(x) for x in table]
    h, out = 0, []
    for i, byte in enumerate(data):
        h = ((h << 1) | (h >> 31)) & 0xFFFFFFFF
        h ^= t[byte] ^ (t[data[i - CDC_WINDOW]] if i >= CDC_WINDOW else 0)
        out.append(h)
    return out


def naive_chunks(hashes: list[int], n: int, mask: int, sizes: tuple[int, int]) -> list:
    """Sequential chunker reference over precomputed rolling hashes."""
    cuts, start = [], 0
    for p in range(n):
        size = p + 1 - start
        boundary = size >= sizes[0] and p >= CDC_WINDOW - 1 and hashes[p] & mask == 0
        if boundary or size == sizes[1]:
            cuts.append(p + 1)
            start = p + 1
    if start < n:
        cuts.append(n)
    return list(zip([0, *cuts[:-1]], cuts, strict=True))


def selftest_cdc() -> dict[str, bool]:
    rng, table = np.random.default_rng(1), buzhash_table()
    data = rng.integers(0, 256, 600_000, dtype=np.uint8)
    data[200_000:260_000] = data[:60_000]
    zeros = np.zeros(600_000, dtype=np.uint8)
    hashes = naive_buzhash(data.tobytes(), table)
    plan = (CDC_MIN, CDC_MAX)
    cases = ((data, hashes), (zeros, naive_buzhash(zeros.tobytes(), table)))
    plan_ok = all(
        cdc_chunks(buf, table, block=block) == naive_chunks(h, len(buf), CDC_MASK, plan)
        for buf, h in cases
        for block in (100_000, CDC_BLOCK)
    )
    small = cdc_chunks(data[:60_000], table, 0x3F, (64, 256), 1000)
    hashes_ok = window_hashes(data, table).tolist() == hashes[CDC_WINDOW - 1 :]
    return {
        "S4 window hashes equal a rolling buzhash": hashes_ok,
        "S4 chunker equals a sequential reference (small sizes)": small
        == naive_chunks(hashes, 60_000, 0x3F, (64, 256)),
        "S4 chunker equals a sequential reference (plan sizes)": plan_ok,
    }


def p4_fixture(rng: np.random.Generator) -> dict:
    """Small Qwen-like tensors; header order differs from draw order; k_proj is not a target."""
    tensors = {"model.embed_tokens.weight": ("BF16", f32_to_bf16_bits(rng.normal(size=(10, 16))))}
    for layer in (10, 0, 2, 1):
        for proj, rows in (("q", 16), ("k", 4), ("v", 4)):
            prefix = f"model.layers.{layer}.self_attn.{proj}_proj"
            weight = f32_to_bf16_bits(rng.standard_normal((rows, 16)))
            tensors[f"{prefix}.weight"] = ("BF16", weight)
            tensors[f"{prefix}.bias"] = ("BF16", f32_to_bf16_bits(rng.standard_normal(rows)))
    return tensors


def selftest_p4(tmp: Path) -> dict[str, bool]:
    tensors = p4_fixture(np.random.default_rng(3))
    a_path = write_safetensors(tmp / "p4-a.safetensors", tensors, {"format": "pt"})
    b_path, factors, _ = generate_p4(a_path, tmp / "p4-b.safetensors", tmp / "p4-factors.npz")
    order = [f"model.layers.{n}.self_attn.{p}_proj.weight" for n in (0, 1, 2, 10) for p in "qv"]
    a, b, draw = load_checkpoint(a_path), load_checkpoint(b_path), np.random.default_rng(P4_SEED)
    values_ok = list(factors) == order
    for name in order:
        w = bf16_bits_to_f32(tensors[name][1]).astype(np.float64)
        u = draw.standard_normal((w.shape[0], P4_RANK))
        v = draw.standard_normal((P4_RANK, w.shape[1]))
        s = P4_REL_NORM * np.linalg.norm(w) / np.linalg.norm(u @ v)
        expected = f32_to_bf16_bits((w + s * (u @ v)).astype(np.float32)).tobytes()
        values_ok = values_ok and bytes(b.raw(name)) == expected
    others_ok = all(bytes(a.raw(n)) == bytes(b.raw(n)) for n in tensors if n not in order)
    start = a.files[0].data_start
    header_ok = a_path.read_bytes()[:start] == b_path.read_bytes()[:start]
    return {
        "P4 generator draws in layer order and merges as specified": values_ok,
        "P4 generator keeps every other byte of A": others_ok and header_ok,
    }


def selftest(scratch: Path) -> dict:
    """Run every check in a temporary directory under scratch (removed afterwards)."""
    t0 = time.perf_counter()
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="selftest-") as tmp:
        checks = {**selftest_strategies(Path(tmp)), **selftest_p4(Path(tmp)), **selftest_cdc()}
    seconds = round(time.perf_counter() - t0, 3)
    return {"passed": all(checks.values()), "checks": checks, "seconds": seconds}
