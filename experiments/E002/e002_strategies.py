"""E002 strategies: bytes needed to store checkpoint B when checkpoint A is already stored.

Each strategy is a pure function of two memory-mapped checkpoints. Selection strategies
(S0, S2, S3, S4u, S4) return the byte ranges of B that must be stored plus statistics; their
z variants compress those ranges with zstd_size. Delta strategies (S5, S6) return zstd
payloads that rebuild B from A, checked by verify_payloads. zstd level 3 everywhere; every
hash is the SHA-256 of raw bytes (wvc.hashing.digest_bytes). Definitions follow the frozen
plan in experiments/E002-storage-measurement.md.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator

import numpy as np
import zstandard

from wvc.hashing import digest_bytes, tensor_digests
from wvc.safetensors_io import ITEMSIZES, Checkpoint

ZSTD_LEVEL = 3
FIXED_CHUNK = 64 * 1024  # S3 chunk size
MIN_RUN = 16 * 1024  # S4u: shortest unchanged run a 16 KiB-minimum chunker can dedup
RUN_BLOCK = 8 * 1024  # S4u: aligned block of the clean-block scan
SCAN_SEGMENT = 64 * 1024 * 1024  # S4u: bytes compared per numpy call (multiple of RUN_BLOCK)
SPARSE_DENOM = 20  # S6: sparse when changed elements * 20 <= numel, i.e. at most 5%
CDC_WINDOW, CDC_MASK, CDC_SEED = 32, 0xFFFF, 0
CDC_MIN, CDC_MAX = 16 * 1024, 256 * 1024
CDC_BLOCK = 4 * 1024 * 1024  # S4: bytes hashed per numpy pass
INT_VIEWS = {1: np.dtype("u1"), 2: np.dtype("<u2"), 4: np.dtype("<u4"), 8: np.dtype("<u8")}

Range = tuple[str, int, int]  # tensor name, start, end: a byte range of one tensor buffer
Selection = tuple[list[Range], dict]
Payload = tuple[str, list[bytes]]  # encoding kind, zstd frames


# ---------------------------------------------------------------- shared helpers


def counterpart(a: Checkpoint, b: Checkpoint, name: str) -> bool:
    """True when A holds a tensor of this name with the same dtype and shape."""
    info_a, info_b = a.tensors.get(name), b.tensors[name]
    return info_a is not None and (info_a.dtype, info_a.shape) == (info_b.dtype, info_b.shape)


def unchanged_names(a: Checkpoint, b: Checkpoint, da: dict, db: dict) -> list[str]:
    return [n for n in b.tensors if counterpart(a, b, n) and da[n] == db[n]]


def whole(b: Checkpoint, names: Iterable[str]) -> list[Range]:
    return [(name, 0, b.tensors[name].nbytes) for name in names]


def range_bytes(ranges: Iterable[Range]) -> int:
    return sum(end - start for _, start, end in ranges)


def zstd_size(b: Checkpoint, ranges: list[Range], per_tensor: bool) -> int:
    """Sum of zstd frame sizes: one frame per range, or per tensor with its ranges joined."""
    zc = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    if not per_tensor:
        return sum(len(zc.compress(b.raw(name)[s:e])) for name, s, e in ranges)
    total = 0
    for name, group in itertools.groupby(ranges, key=lambda r: r[0]):
        raw = b.raw(name)
        parts = [raw[s:e] for _, s, e in group]
        total += len(zc.compress(parts[0] if len(parts) == 1 else np.concatenate(parts)))
    return total


# ---------------------------------------------------------------- S0 to S3


def s0_all_tensors(a: Checkpoint, b: Checkpoint) -> Selection:
    return whole(b, b.tensors), {"tensors": len(b.tensors)}


def s2_tensor_dedup(a: Checkpoint, b: Checkpoint) -> Selection:
    """B's tensors unless the same-named tensor of A has equal dtype, shape, and SHA-256."""
    da, db = tensor_digests(a), tensor_digests(b)
    new = [n for n in b.tensors if not (counterpart(a, b, n) and da[n] == db[n])]
    stats = {"new_tensors": len(new), "deduplicated_tensors": len(b.tensors) - len(new)}
    return whole(b, new), stats


def s2_content_addressed_bytes(a: Checkpoint, b: Checkpoint, da: dict, db: dict) -> int:
    """Variant ignoring names: B's tensors whose (dtype, shape, sha256) is absent from A."""
    known = {(info.dtype, info.shape, da[n]) for n, info in a.tensors.items()}
    new = [n for n, info in b.tensors.items() if (info.dtype, info.shape, db[n]) not in known]
    return range_bytes(whole(b, new))


def fixed_chunks(raw: np.ndarray) -> list[tuple[int, int]]:
    return [(s, min(s + FIXED_CHUNK, len(raw))) for s in range(0, len(raw), FIXED_CHUNK)]


def chunk_digests(ckpt: Checkpoint, chunker: Callable) -> Iterator[tuple[str, int, int, str]]:
    for name in ckpt.tensors:
        raw = ckpt.raw(name)
        for start, end in chunker(raw):
            yield name, start, end, digest_bytes(raw[start:end])


def new_chunks(a: Checkpoint, b: Checkpoint, chunker: Callable) -> Selection:
    """B's chunks whose SHA-256 is absent from A's chunk set (repeats within B all counted)."""
    known = {digest for *_, digest in chunk_digests(a, chunker)}
    ranges: list[Range] = []
    seen: set[str] = set()
    b_chunks = distinct_bytes = 0
    for name, start, end, digest in chunk_digests(b, chunker):
        b_chunks += 1
        if digest in known:
            continue
        ranges.append((name, start, end))
        if digest not in seen:
            seen.add(digest)
            distinct_bytes += end - start
    stats = {
        "a_distinct_chunks": len(known),
        "b_chunks": b_chunks,
        "new_chunks": len(ranges),
        "new_distinct_chunks": len(seen),
        "new_distinct_bytes": distinct_bytes,
    }
    return ranges, stats


def s3_fixed_chunk_dedup(a: Checkpoint, b: Checkpoint) -> Selection:
    return new_chunks(a, b, fixed_chunks)


# ---------------------------------------------------------------- S4u


def clean_blocks(a_raw: np.ndarray, b_raw: np.ndarray) -> np.ndarray:
    """For each full aligned RUN_BLOCK of the buffer: True when no byte differs."""
    full = len(b_raw) // RUN_BLOCK * RUN_BLOCK
    clean = np.empty(full // RUN_BLOCK, dtype=bool)
    for lo in range(0, full, SCAN_SEGMENT):
        hi = min(lo + SCAN_SEGMENT, full)
        dirty = (a_raw[lo:hi] != b_raw[lo:hi]).reshape(-1, RUN_BLOCK).any(axis=1)
        clean[lo // RUN_BLOCK : hi // RUN_BLOCK] = ~dirty
    return clean


def true_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, end) index spans where the boolean mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()
    return list(zip(starts, ends, strict=True))


def extend_left(a_raw: np.ndarray, b_raw: np.ndarray, k0: int) -> int:
    """Start of the unchanged run whose first clean block is k0 (block k0 - 1 is dirty)."""
    if k0 == 0:
        return 0
    lo = (k0 - 1) * RUN_BLOCK
    changed = np.flatnonzero(a_raw[lo : lo + RUN_BLOCK] != b_raw[lo : lo + RUN_BLOCK])
    return lo + int(changed[-1]) + 1


def extend_right(a_raw: np.ndarray, b_raw: np.ndarray, k1: int) -> int:
    """End of the unchanged run whose clean blocks stop before block k1 (dirty, or the tail)."""
    lo = k1 * RUN_BLOCK
    hi = min(lo + RUN_BLOCK, len(b_raw))
    changed = np.flatnonzero(a_raw[lo:hi] != b_raw[lo:hi])
    return lo + int(changed[0]) if changed.size else hi


def unchanged_runs(a_raw: np.ndarray, b_raw: np.ndarray) -> list[tuple[int, int]]:
    """Exact maximal unchanged byte runs of at least MIN_RUN bytes.

    Every such run contains a clean aligned RUN_BLOCK (RUN_BLOCK <= MIN_RUN / 2), so each
    maximal sequence of clean blocks is a candidate, extended byte-wise into the dirty
    neighbouring block on each side; the partial tail block never counts as clean.
    """
    runs = []
    for k0, k1 in true_spans(clean_blocks(a_raw, b_raw)):
        start, end = extend_left(a_raw, b_raw, k0), extend_right(a_raw, b_raw, k1)
        if end - start >= MIN_RUN:
            runs.append((start, end))
    return runs


def complement(runs: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    edges = [0, *itertools.chain.from_iterable(runs), n]
    return [(s, e) for s, e in zip(edges[::2], edges[1::2], strict=True) if e > s]


def s4u_cdc_upper_bound(a: Checkpoint, b: Checkpoint) -> Selection:
    """B's bytes outside unchanged runs >= 16 KiB of same-named, same-dtype-and-shape tensors."""
    da, db = tensor_digests(a), tensor_digests(b)
    ranges: list[Range] = []
    unchanged = scanned = runs_counted = 0
    for name, info in b.tensors.items():
        runs: list[tuple[int, int]] = []
        if counterpart(a, b, name) and da[name] == db[name]:
            runs, unchanged = [(0, info.nbytes)], unchanged + 1
        elif counterpart(a, b, name):
            runs, scanned = unchanged_runs(a.raw(name), b.raw(name)), scanned + 1
            runs_counted += len(runs)
        ranges += [(name, s, e) for s, e in complement(runs, info.nbytes)]
    stats = {
        "dedupable_bytes": sum(i.nbytes for i in b.tensors.values()) - range_bytes(ranges),
        "unchanged_tensors": unchanged,
        "scanned_tensors": scanned,
        "runs_counted_in_scanned_tensors": runs_counted,
        "unmatched_tensors": len(b.tensors) - unchanged - scanned,
    }
    return ranges, stats


# ---------------------------------------------------------------- S4 (content-defined chunking)


def buzhash_table(seed: int = CDC_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2**32, size=256, dtype=np.uint32)


def rotl32(x: np.ndarray, r: int) -> np.ndarray:
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def window_hashes(data: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Buzhash of every full 32-byte window; element j covers data[j : j + 32].

    H = XOR over k of rotl(T[byte at end - k], k), computed by doubling the span:
    g_2m[i] = g_m[i] ^ rotl(g_m[i - m], m). This equals the rolling update
    h = rotl(h, 1) ^ T[in] ^ T[out], since rotating by the window length is the identity.
    """
    h = table[data]
    span = 1
    while span < CDC_WINDOW:
        h = h[span:] ^ rotl32(h[:-span], span)
        span *= 2
    return h


def cdc_chunks(
    raw: np.ndarray,
    table: np.ndarray,
    mask: int = CDC_MASK,
    sizes: tuple[int, int] = (CDC_MIN, CDC_MAX),
    block: int = CDC_BLOCK,
) -> list[tuple[int, int]]:
    """Chunks of one buffer: cut after window end p when (H_p & mask) == 0 and the chunk holds
    at least sizes[0] bytes, or when it reaches sizes[1] bytes. Hashed block by block."""
    cuts: list[int] = []
    start, w = 0, CDC_WINDOW - 1
    for lo in range(w, len(raw), block):
        hi = min(lo + block, len(raw))
        h = window_hashes(np.asarray(raw[lo - w : hi]), table)
        candidates = lo + np.flatnonzero((h & np.uint32(mask)) == 0)
        start = advance_cuts(candidates, hi, start, sizes, cuts)
    if start < len(raw):
        cuts.append(len(raw))
    return list(zip([0, *cuts[:-1]], cuts, strict=True))


def advance_cuts(cand: np.ndarray, hi: int, start: int, sizes: tuple[int, int], cuts: list):
    """Consume sorted candidates below hi; append cuts; return the start of the open chunk."""
    while True:
        first, last = start + sizes[0] - 1, start + sizes[1] - 1
        i = int(np.searchsorted(cand, first))
        if i < len(cand) and cand[i] <= last:
            start = int(cand[i]) + 1
        elif last < hi:
            start = last + 1
        else:
            return start
        cuts.append(start)


def s4_cdc_dedup(a: Checkpoint, b: Checkpoint) -> Selection:
    table = buzhash_table()
    return new_chunks(a, b, lambda raw: cdc_chunks(raw, table))


# ---------------------------------------------------------------- S5, S6


def xor_planes(a_raw: np.ndarray, b_raw: np.ndarray, itemsize: int) -> list[np.ndarray]:
    x = np.bitwise_xor(a_raw, b_raw).reshape(-1, itemsize)
    return [np.ascontiguousarray(x[:, j]) for j in range(itemsize)]


def s5_xor_delta(a: Checkpoint, b: Checkpoint) -> dict[str, Payload]:
    zc = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    payloads: dict[str, Payload] = {}
    for name, info in b.tensors.items():
        if counterpart(a, b, name):
            planes = xor_planes(a.raw(name), b.raw(name), ITEMSIZES[info.dtype])
            payloads[name] = ("xor", [zc.compress(p) for p in planes])
        else:
            payloads[name] = ("zstd", [zc.compress(b.raw(name))])
    return payloads


def sparse_or_xor(a_raw: np.ndarray, b_raw: np.ndarray, itemsize: int, zc) -> Payload:
    view = INT_VIEWS[itemsize]
    old, new = a_raw.view(view), b_raw.view(view)
    differs = old != new
    if int(np.count_nonzero(differs)) * SPARSE_DENOM > old.size:
        return "xor", [zc.compress(p) for p in xor_planes(a_raw, b_raw, itemsize)]
    if old.size > 2**32:
        raise ValueError("uint32 element indices cannot address this tensor")
    idx = np.flatnonzero(differs)
    return "sparse", [zc.compress(idx.astype("<u4").tobytes()), zc.compress(new[idx].tobytes())]


def s6_sparse_delta(a: Checkpoint, b: Checkpoint) -> dict[str, Payload]:
    zc = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    payloads: dict[str, Payload] = {}
    for name, info in b.tensors.items():
        if counterpart(a, b, name):
            payloads[name] = sparse_or_xor(a.raw(name), b.raw(name), ITEMSIZES[info.dtype], zc)
        else:
            payloads[name] = ("zstd", [zc.compress(b.raw(name))])
    return payloads


def payload_bytes(payloads: Iterable[Payload]) -> int:
    return sum(len(frame) for _, frames in payloads for frame in frames)


def decode_payload(a: Checkpoint, name: str, payload: Payload, itemsize: int, dz) -> np.ndarray:
    kind, frames = payload
    planes = [np.frombuffer(dz.decompress(frame), dtype=np.uint8) for frame in frames]
    if kind == "zstd":
        return planes[0]
    if kind == "xor":
        return np.bitwise_xor(a.raw(name), np.stack(planes, axis=1).reshape(-1))
    if kind == "sparse":
        view = INT_VIEWS[itemsize]
        rebuilt = np.array(a.raw(name).view(view))
        rebuilt[planes[0].view("<u4")] = planes[1].view(view)
        return rebuilt.view(np.uint8)
    raise ValueError(f"unknown payload kind {kind!r}")


def verify_payloads(a: Checkpoint, b: Checkpoint, payloads: dict, db: dict) -> tuple[bool, list]:
    """Rebuild every tensor of B from A and the payloads; compare SHA-256 with B's digest."""
    dz = zstandard.ZstdDecompressor()
    failures = sorted(set(b.tensors) ^ set(payloads))
    for name in sorted(set(b.tensors) & set(payloads)):
        info = b.tensors[name]
        rebuilt = decode_payload(a, name, payloads[name], ITEMSIZES[info.dtype], dz)
        if len(rebuilt) != info.nbytes or digest_bytes(rebuilt) != db[name]:
            failures.append(name)
    return not failures, sorted(failures)


# ---------------------------------------------------------------- S7


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round to bfloat16 (round-to-nearest-even); 16-bit patterns. From tests/conftest.py."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype(np.uint16)


def s7_low_rank_bound(a: Checkpoint, b: Checkpoint, factors: dict) -> tuple[int, dict]:
    """bf16 bytes of the known U and V (scale folded into U) plus S2z of every other tensor."""
    new, _ = s2_tensor_dedup(a, b)
    others = [r for r in new if r[0] not in factors]
    others_z = zstd_size(b, others, per_tensor=False)
    uv = sum(
        f32_to_bf16_bits(s * u).nbytes + f32_to_bf16_bits(v).nbytes for u, v, s in factors.values()
    )
    details = {
        "factor_bytes_bf16": uv,
        "edited_tensors": len(factors),
        "other_new_tensors": len(others),
        "other_tensors_s2z_bytes": others_z,
        "lossless": False,
    }
    return uv + others_z, details
