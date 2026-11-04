"""What bfloat16 rounding does to the diff record's rank (docs/classification.md).

A seeded 896 x 4864 matrix of N(0, 0.02) weights in bfloat16 receives a known change, is
rounded back to bfloat16, and the stored delta is measured three ways: the record's rank
(singular values above 1e-2 of the largest), the record's top-1 energy, and the rank above the
noise edge that ``wvc.features`` computes.

Run: ``uv run python experiments/E010/synthetic_noise.py``.
"""

from __future__ import annotations

import numpy as np

from wvc.diff import RANK_REL_TOL
from wvc.features import FEATURES_REL_TOL, spacing

ROWS, COLS, SEED = 896, 4864, 0


def bf16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, np.float32).view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return rounded.astype(np.uint32).view(np.float32).astype(np.float64)


def measure(base: np.ndarray, edited: np.ndarray) -> tuple[float, int, float, int]:
    delta = edited - base
    sigma = np.linalg.svd(delta, compute_uv=False)
    noise = spacing(edited, "BF16") ** 2 / 12
    edge = np.sqrt(noise.sum(axis=1).max()) + np.sqrt(noise.sum(axis=0).max())
    record_rank = int((sigma > RANK_REL_TOL * sigma[0]).sum())
    signal_rank = int((sigma > max(edge, FEATURES_REL_TOL * sigma[0])).sum())
    top1 = float(sigma[0] ** 2 / np.sum(sigma**2))
    rel = float(np.linalg.norm(delta) / np.linalg.norm(base))
    return rel, record_rank, top1, signal_rank


def main() -> None:
    rng = np.random.default_rng(SEED)
    base = bf16(rng.standard_normal((ROWS, COLS)) * 0.02)
    print("change | relative size | record rank | record top-1 | rank above noise edge")
    for rank, rel in ((8, 0.01), (16, 0.002), (64, 0.01)):
        delta = rng.standard_normal((ROWS, rank)) @ rng.standard_normal((rank, COLS))
        delta *= rel * np.linalg.norm(base) / np.linalg.norm(delta)
        print(f"LoRA r={rank}", *map(_fmt, measure(base, bf16(base + delta))), sep=" | ")
    r = rng.standard_normal(ROWS)
    r /= np.linalg.norm(r)
    for strength in (1.0, 0.3):
        edited = bf16(base - strength * np.outer(r, r @ base))
        print(f"rank-1 removal x{strength}", *map(_fmt, measure(base, edited)), sep=" | ")
    left, _ = np.linalg.qr(rng.standard_normal((ROWS, ROWS)))
    right, _ = np.linalg.qr(rng.standard_normal((COLS, ROWS)))
    delta = (left * np.arange(1, ROWS + 1) ** -0.7) @ right.T
    delta *= 0.01 * np.linalg.norm(base) / np.linalg.norm(delta)
    print("dense power-law", *map(_fmt, measure(base, bf16(base + delta))), sep=" | ")


def _fmt(value: float) -> str:
    return f"{value:.3g}" if isinstance(value, float) else str(value)


if __name__ == "__main__":
    main()
