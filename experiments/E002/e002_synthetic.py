"""P4 input for E002: A with a rank-8 update merged into every q_proj and v_proj weight.

B = bf16(W + s * (U @ V)) for each ``model.layers.<n>.self_attn.{q,v}_proj.weight`` (biases
untouched), U [out, 8] and V [8, in] standard normal from numpy.random.default_rng(2) drawn in
the order layer 0 q, layer 0 v, layer 1 q, ...; s = 0.01 ||W||_F / ||U @ V||_F. Every other
byte of A, including the header, __metadata__, tensor order, and offsets, is copied verbatim.
"""

from __future__ import annotations

import re
import shutil
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from e002_strategies import f32_to_bf16_bits

from wvc.safetensors_io import load_checkpoint

P4_RANK, P4_SEED, P4_REL_NORM = 8, 2, 0.01
P4_TARGET = re.compile(r"^model\.layers\.(\d+)\.self_attn\.([qv])_proj\.weight$")


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def p4_targets(names: Iterable[str]) -> list[str]:
    """q_proj and v_proj weights in draw order: layer 0 q, layer 0 v, layer 1 q, ..."""
    hits = sorted((int(m[1]), m[2], name) for name in names if (m := P4_TARGET.match(name)))
    return [name for *_, name in hits]


def merged_bits(w: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, float]:
    """bf16(W + s U V), s = 0.01 ||W||_F / ||U V||_F; float64 math, float32 cast, then RNE."""
    w64, uv = w.astype(np.float64), u @ v
    s = P4_REL_NORM * float(np.linalg.norm(w64)) / float(np.linalg.norm(uv))
    return f32_to_bf16_bits((w64 + s * uv).astype(np.float32)), s


def draw_and_merge(ckpt, names: list[str]) -> tuple[dict, dict, list[float]]:
    """Factors (U, V, s), merged bf16 bits, and the realised relative change per target."""
    rng = np.random.default_rng(P4_SEED)
    factors, patches, realized = {}, {}, []
    for name in names:
        info = ckpt.tensors[name]
        if info.dtype != "BF16" or len(info.shape) != 2:
            raise ValueError(f"P4 target {name} is {info.dtype}{list(info.shape)}, not 2-D BF16")
        w = ckpt.as_float32(name)
        u = rng.standard_normal((info.shape[0], P4_RANK))
        v = rng.standard_normal((P4_RANK, info.shape[1]))
        patches[name], s = merged_bits(w, u, v)
        factors[name] = (u, v, s)
        delta = bf16_bits_to_f32(patches[name]).reshape(w.shape) - w
        realized.append(float(np.linalg.norm(delta) / np.linalg.norm(w)))
    return factors, patches, realized


def generate_p4(a_path: Path, out_path: Path, factors_path: Path) -> tuple[Path, dict, dict]:
    """Write B next to the factors; return B's path, {name: (U, V, s)}, and a record."""
    t0 = time.perf_counter()
    ckpt = load_checkpoint(a_path)
    factors, patches, realized = draw_and_merge(ckpt, p4_targets(ckpt.tensors))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a_path, out_path)
    with out_path.open("r+b") as handle:
        for name, bits in patches.items():
            handle.seek(ckpt.files[0].data_start + ckpt.tensors[name].begin)
            handle.write(bits.astype("<u2").tobytes())
    arrays = {}
    for i, (u, v, _) in enumerate(factors.values()):
        arrays[f"U{i}"], arrays[f"V{i}"] = u, v
    scales = np.array([f[2] for f in factors.values()])
    np.savez(factors_path, names=np.array(list(factors)), s=scales, **arrays)
    record = {
        "rank": P4_RANK,
        "seed": P4_SEED,
        "relative_frobenius_norm": P4_REL_NORM,
        "targets": len(factors),
        "arithmetic": "W float32 -> float64; W + s*(U@V) in float64; cast to float32; RNE to bf16",
        "scale_min": float(scales.min()) if len(scales) else None,
        "scale_max": float(scales.max()) if len(scales) else None,
        "realized_relative_norm_after_bf16_min": min(realized, default=None),
        "realized_relative_norm_after_bf16_max": max(realized, default=None),
        "factors_path": str(factors_path),
        "seconds": round(time.perf_counter() - t0, 3),
    }
    return out_path, factors, record
