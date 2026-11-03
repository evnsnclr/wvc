"""Post hoc robustness check for E002 (not part of the frozen plan).

The frozen S5 (XOR delta) compresses byte-position planes, while the standalone baseline
S1 and the chunk strategies compress raw bytes. This script separates the two effects:

  S1g  byte-grouped zstd of B alone (no A): the grouping effect without any delta
  S5p  XOR delta compressed without grouping: the delta effect without grouping
  S5g  XOR delta with grouping, recomputed here to check against the committed S5

Usage: uv run --with zstandard python experiments/E002/posthoc_grouping.py
Writes experiments/E002/results/posthoc_grouping.json and prints one table.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import zstandard

from wvc.hashing import tensor_digests
from wvc.safetensors_io import ITEMSIZES, Checkpoint, load_checkpoint

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LEVEL = 3
P4_B_OVERRIDE = HERE / "scratch" / "P4-B.safetensors"


def zsize(buf: np.ndarray) -> int:
    return len(zstandard.ZstdCompressor(level=LEVEL).compress(np.ascontiguousarray(buf)))


def planes(raw: np.ndarray, itemsize: int) -> list[np.ndarray]:
    if itemsize == 1:
        return [raw]
    matrix = raw.reshape(-1, itemsize)
    return [np.ascontiguousarray(matrix[:, j]) for j in range(itemsize)]


def same_layout(a: Checkpoint, b: Checkpoint, name: str) -> bool:
    ia, ib = a.tensors.get(name), b.tensors[name]
    return ia is not None and ia.dtype == ib.dtype and ia.shape == ib.shape


def measure(a: Checkpoint, b: Checkpoint) -> dict[str, int]:
    s1g = s5p = s5g = 0
    for name, info in b.tensors.items():
        raw_b = b.raw(name)
        size = ITEMSIZES[info.dtype]
        s1g += sum(zsize(p) for p in planes(raw_b, size))
        if same_layout(a, b, name):
            xor = np.bitwise_xor(a.raw(name), raw_b)
            s5p += zsize(xor)
            s5g += sum(zsize(p) for p in planes(xor, size))
        else:
            plain = zsize(raw_b)
            s5p += plain
            s5g += plain
    return {"S1g": s1g, "S5p": s5p, "S5g": s5g}


def committed(pair: dict, key: str) -> int:
    strategies = pair.get("strategies", pair)
    return int(strategies[key]["bytes"])


def main() -> None:
    out = {"note": __doc__.strip().splitlines()[0], "level": LEVEL, "pairs": {}}
    rows = []
    for pair_id in ("P1", "P2", "P3", "P4"):
        record = json.loads((RESULTS / f"{pair_id}.json").read_text())
        path_a = Path(record["inputs"]["a"]["path"])
        path_b = P4_B_OVERRIDE if pair_id == "P4" else Path(record["inputs"]["b"]["path"])
        a, b = load_checkpoint(path_a), load_checkpoint(path_b)
        tensor_digests(a)  # touch the files once so timings below are warm-cache
        start = time.perf_counter()
        sizes = measure(a, b)
        seconds = time.perf_counter() - start
        s0, s1, s5 = (committed(record, k) for k in ("S0", "S1", "S5"))
        entry = {
            "S0": s0, "S1": s1, "S5_committed": s5, **sizes, "seconds": round(seconds, 2),
            "ratio": {k: round(v / s0, 6) for k, v in {"S1": s1, **sizes, "S5": s5}.items()},
            "grouping_effect_S1g_over_S1": round(sizes["S1g"] / s1, 4),
            "delta_effect_grouped_S5g_over_S1g": round(sizes["S5g"] / sizes["S1g"], 4),
            "delta_effect_plain_S5p_over_S1": round(sizes["S5p"] / s1, 4),
            "S5g_matches_committed_S5_within_0.1pct": abs(sizes["S5g"] - s5) <= 0.001 * s5,
        }  # fmt: skip
        out["pairs"][pair_id] = entry
        r = entry["ratio"]
        rows.append(
            f"| {pair_id} | {r['S1']:.4f} | {r['S1g']:.4f} | {r['S5p']:.4f} | {r['S5g']:.4f} "
            f"| {r['S5']:.4f} | {entry['delta_effect_grouped_S5g_over_S1g']:.3f} "
            f"| {entry['grouping_effect_S1g_over_S1']:.3f} | {seconds:.1f} |"
        )
    (RESULTS / "posthoc_grouping.json").write_text(json.dumps(out, indent=2) + "\n")
    print("| Pair | S1 | S1g | S5p | S5g | S5 (committed) | S5g/S1g | S1g/S1 | s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
