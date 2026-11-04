"""Compare wvc's dequantisers with llama.cpp's compiled C reference, bit for bit (E011).

    uv run python experiments/E011/verify_dequant_c.py QREF [BLOCKS]

QREF is ``qref.c`` (next to this script) compiled against ``libggml-base`` from a llama.cpp
checkout. For every block type wvc dequantises, two inputs are compared: random block bytes
with finite scales (every quant bit pattern), and the blocks that ggml's reference quantiser
(``from_float_ref``) makes from standard normals (realistic scales and mins). Each is
dequantised by ggml (``to_float``) and by ``wvc.gguf_quants.dequantize``; the float32 bit
patterns must be identical. Prints one line per type and input.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from wvc.gguf_io import GGML_TYPES, TYPE_SIZES
from wvc.gguf_quants import DEQUANTIZERS, dequantize

SEED = 20261103
# Byte offsets of each type's float16 scale fields (Q8_K's scale is a float32 at 0).
F16_FIELDS = {
    "Q4_0": (0,), "Q4_1": (0, 2), "Q5_0": (0,), "Q5_1": (0, 2), "Q8_0": (0,),
    "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,), "Q8_K": (),
}  # fmt: skip
TYPE_IDS = {name: type_id for type_id, (name, _, _) in GGML_TYPES.items()}


def random_blocks(name: str, count: int, rng: np.random.Generator) -> np.ndarray:
    size = TYPE_SIZES[name][1]
    raw = rng.integers(0, 256, (count, size), dtype=np.uint8)
    for start in F16_FIELDS[name]:
        scale = (rng.standard_normal(count) * 0.01).astype(np.float16)
        raw[:, start : start + 2] = scale.view(np.uint8).reshape(count, 2)
    if name == "Q8_K":
        scale = (rng.standard_normal(count) * 0.01).astype(np.float32)
        raw[:, 0:4] = scale.view(np.uint8).reshape(count, 4)
    return raw.reshape(-1)


def run(qref: str, type_id: int, mode: str, data: bytes, work: Path) -> bytes:
    source, target = work / "in.bin", work / "out.bin"
    source.write_bytes(data)
    subprocess.run([qref, str(type_id), mode, str(source), str(target)], check=True)
    return target.read_bytes()


def compare(name: str, raw: np.ndarray, qref: str, work: Path) -> tuple[bool, int]:
    block, size = TYPE_SIZES[name]
    ref = np.frombuffer(run(qref, TYPE_IDS[name], "deq", raw.tobytes(), work), dtype="<f4")
    mine = dequantize(raw, name, block, size)
    return bool(np.array_equal(ref.view("<u4"), mine.view("<u4"))), len(mine)


def main(argv: list[str]) -> int:
    qref = argv[0]
    count = int(argv[1]) if len(argv) > 1 else 4096
    rng = np.random.default_rng(SEED)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in DEQUANTIZERS:
            block, _ = TYPE_SIZES[name]
            same, n = compare(name, random_blocks(name, count, rng), qref, work)
            print(f"{name:5} random blocks      {n:>8} values  bit-identical: {same}")
            floats = rng.standard_normal(count * block).astype("<f4")
            quantized = run(qref, TYPE_IDS[name], "quant", floats.tobytes(), work)
            same_q, n = compare(name, np.frombuffer(quantized, dtype=np.uint8), qref, work)
            print(f"{name:5} reference quantised {n:>8} values  bit-identical: {same_q}")
            failures += (not same) + (not same_q)
    print(f"failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
