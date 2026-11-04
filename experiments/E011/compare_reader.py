"""Compare wvc's GGUF reader with gguf-py's GGUFReader on whole files (E011).

    uv run python experiments/E011/compare_reader.py FILE.gguf ...

For each file: the number of tensors, whether every tensor's name, type, row-major shape,
offset, and byte count agree between the two readers, whether the order agrees, and the
architecture, alignment, and type counts wvc reads.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

from gguf import GGUFReader

from wvc.gguf_io import read_gguf


def main(paths: list[str]) -> int:
    failures = 0
    for path in paths:
        started = time.perf_counter()
        ours = read_gguf(path)
        seconds = time.perf_counter() - started
        theirs = GGUFReader(path)
        ref = {
            t.name: (
                t.tensor_type.name,
                tuple(int(x) for x in t.shape[::-1]),
                int(t.data_offset) - ours.data_start,
                int(t.n_bytes),
            )
            for t in theirs.tensors
        }
        mine = {n: (i.dtype, i.shape, i.begin, i.nbytes) for n, i in ours.tensors.items()}
        same, order = ref == mine, list(mine) == [t.name for t in theirs.tensors]
        failures += (not same) + (not order)
        types = dict(sorted(Counter(i.dtype for i in ours.tensors.values()).items()))
        print(
            f"{Path(path).name}: {len(mine)} tensors in {seconds:.2f} s; equal to GGUFReader: "
            f"{same}; same order: {order}; architecture {ours.architecture}; alignment "
            f"{ours.alignment}; types {types}"
        )
    print(f"failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
