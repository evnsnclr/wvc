"""Generate the wvc spec v0 test vectors.

Writes two small safetensors checkpoints (``a``, ``b``), their ``wvc-manifest/0``
manifests, and the diff record between them into OUT_DIR (default: this directory):

    uv run python spec/vectors/v0/make_vectors.py [OUT_DIR]

The checkpoint bytes depend only on the seed and NumPy's ``Generator`` stream. The two
writer helpers are copies of those in ``tests/conftest.py``, so this script does not
import from the test suite; the manifests and the diff record come from ``wvc`` itself.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from wvc.cli import main as wvc_main
from wvc.hashing import build_manifest
from wvc.safetensors_io import load_checkpoint

SEED = 20261102
METADATA = {"format": "pt"}
EDITED_COLUMN = 2  # checkpoint b adds 1.0 to this column of tensor "w" in every row
DIFF_ARGS = ("diff", "--rank", "--json", "--all")

TensorMap = dict[str, tuple[str, np.ndarray]]

# Copied from tests/conftest.py: NUMPY_FOR, f32_to_bf16_bits, write_safetensors.
NUMPY_FOR = {"F32": "<f4", "F16": "<f2", "I32": "<i4", "I8": "i1", "U8": "u1", "BF16": "<u2"}


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 values to bfloat16 (round-to-nearest-even) and return the 16-bit patterns."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype(np.uint16)


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, np.ndarray]],
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write ``name -> (dtype, array)``; BF16 arrays must already hold uint16 bit patterns."""
    header: dict[str, object] = {}
    buffer = bytearray()
    for name, (dtype, array) in tensors.items():
        raw = np.ascontiguousarray(array).astype(NUMPY_FOR[dtype], copy=False).tobytes()
        begin = len(buffer)
        buffer += raw
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [begin, len(buffer)],
        }
    if metadata:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(buffer))
    return path


def make_checkpoints(seed: int = SEED) -> tuple[TensorMap, TensorMap]:
    """Return the tensors of checkpoints a and b, in file order."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((4, 6)).astype(np.float32)
    b = f32_to_bf16_bits(rng.standard_normal(8).astype(np.float32))
    h = rng.standard_normal((2, 2, 2)).astype(np.float32).astype(np.float16)
    # A rank-1 edit: 1.0 added to one column in every row (4 rows, 1 column touched).
    w_edited = np.where(np.arange(w.shape[1]) == EDITED_COLUMN, w + np.float32(1.0), w)
    x = np.array([1, -2, 3], dtype=np.int8)
    tensors_a = {"w": ("F32", w), "b": ("BF16", b), "h": ("F16", h)}
    tensors_b = {"w": ("F32", w_edited), "b": ("BF16", b), "x": ("I8", x)}
    return tensors_a, tensors_b


def diff_record(path_a: Path, path_b: Path) -> dict:
    """Run ``wvc diff --rank --json --all`` and reduce the two path fields to basenames."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        status = wvc_main([*DIFF_ARGS, str(path_a), str(path_b)])
    if status != 1:
        raise RuntimeError(f"wvc diff exited with {status}; expected 1 (the checkpoints differ)")
    payload = json.loads(printed.getvalue())
    return {
        **payload,
        "a": {**payload["a"], "path": path_a.name},
        "b": {**payload["b"], "path": path_b.name},
    }


def write_json(path: Path, value: object) -> None:
    """Write ``value`` as indent-2 JSON plus a newline, with LF line endings on every OS."""
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_vectors(out_dir: Path) -> None:
    """Write a.safetensors, b.safetensors, their manifests, and diff.json into ``out_dir``."""
    tensors_a, tensors_b = make_checkpoints()
    path_a = write_safetensors(out_dir / "a.safetensors", tensors_a, METADATA)
    path_b = write_safetensors(out_dir / "b.safetensors", tensors_b, METADATA)
    write_json(out_dir / "a.manifest.json", build_manifest(load_checkpoint(path_a)))
    write_json(out_dir / "b.manifest.json", build_manifest(load_checkpoint(path_b)))
    write_json(out_dir / "diff.json", diff_record(path_a, path_b))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the wvc spec v0 test vectors.")
    parser.add_argument(
        "out_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory to write into (default: the directory of this script)",
    )
    out_dir = parser.parse_args(argv).out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_vectors(out_dir)


if __name__ == "__main__":
    main()
