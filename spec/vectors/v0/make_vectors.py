"""Generate the wvc spec v0 test vectors.

Writes two small safetensors checkpoints (``a``, ``b``), their ``wvc-manifest/0``
manifests, and the diff record between them into OUT_DIR (default: this directory), and the
GGUF vectors: ``quant.gguf`` (one tensor per dequantised ggml type), its safetensors twin
``quant-base.safetensors``, the GGUF's manifest, gguf-hash lines, and dequantised digests,
and the quantization comparison between the two:

    uv run python spec/vectors/v0/make_vectors.py [OUT_DIR]

The checkpoint bytes depend only on the seeds and NumPy's ``Generator`` stream. The writer
helpers are copies of those in ``tests/helpers.py`` and ``tests/gguf_helpers.py``, so this
script does not import from the test suite; the twin's values, the manifests, the hash
lines, and the diff record come from ``wvc`` itself.
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
from wvc.gguf_io import GGML_TYPES, TYPE_SIZES, load_gguf
from wvc.gguf_map import rotary_rows
from wvc.gguf_quants import dequantize
from wvc.hashing import build_manifest, digest_bytes, gguf_hash_lines
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


# The GGUF vectors (spec section 1.1): a one-layer llama file with one tensor per type wvc
# dequantises, written as llama.cpp lays a file out, and its safetensors twin.
GGUF_SEED = 20261103
HEADS, KV_HEADS = 2, 1  # llama.attention.head_count and head_count_kv
# GGUF name, Hub name, ggml type, row-major shape. attn_q and attn_k are permuted.
GGUF_TENSORS = (
    ("token_embd.weight", "model.embed_tokens.weight", "Q8_0", (8, 256)),
    ("output_norm.weight", "model.norm.weight", "F32", (256,)),
    ("output.weight", "lm_head.weight", "Q6_K", (4, 256)),
    ("blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight", "BF16", (256,)),
    ("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight", "Q4_K", (8, 256)),
    ("blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight", "Q8_K", (4, 256)),
    ("blk.0.attn_v.weight", "model.layers.0.self_attn.v_proj.weight", "Q5_0", (4, 256)),
    ("blk.0.attn_output.weight", "model.layers.0.self_attn.o_proj.weight", "Q5_K", (4, 256)),
    ("blk.0.ffn_norm.weight", "model.layers.0.post_attention_layernorm.weight", "F16", (256,)),
    ("blk.0.ffn_gate.weight", "model.layers.0.mlp.gate_proj.weight", "Q4_0", (4, 256)),
    ("blk.0.ffn_up.weight", "model.layers.0.mlp.up_proj.weight", "Q4_1", (4, 256)),
    ("blk.0.ffn_down.weight", "model.layers.0.mlp.down_proj.weight", "Q5_1", (4, 256)),
)
GGUF_METADATA = (
    ("general.architecture", "llama"),
    ("general.name", "wvc spec v0 vector"),
    ("llama.attention.head_count", HEADS),
    ("llama.attention.head_count_kv", KV_HEADS),
)
GGUF_ALIGNMENT = 32
PERMUTED_HEADS = {"blk.0.attn_q.weight": HEADS, "blk.0.attn_k.weight": KV_HEADS}
# Byte offsets of each block type's float16 scales; Q8_K's scale is a float32 at 0.
SCALE_FIELDS = {"Q4_0": (0,), "Q4_1": (0, 2), "Q5_0": (0,), "Q5_1": (0, 2), "Q8_0": (0,),
                "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,)}  # fmt: skip
TYPE_IDS = {name: type_id for type_id, (name, _, _) in GGML_TYPES.items()}


def gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def gguf_value(value: str | int) -> bytes:
    """A STRING (type 8) or UINT32 (type 4) metadata value with its type id."""
    if isinstance(value, str):
        return struct.pack("<I", 8) + gguf_string(value)
    return struct.pack("<II", 4, value)


def write_gguf(path: Path, tensors: Sequence[tuple[str, str, tuple[int, ...], bytes]]) -> Path:
    """Write a version 3 GGUF: header, metadata, tensor infos, padding, padded tensor data."""
    head = b"GGUF" + struct.pack("<Iqq", 3, len(tensors), len(GGUF_METADATA))
    table = b"".join(gguf_string(key) + gguf_value(value) for key, value in GGUF_METADATA)
    infos, data = b"", b""
    for name, dtype, shape, raw in tensors:
        infos += gguf_string(name) + struct.pack("<I", len(shape))
        infos += b"".join(struct.pack("<q", dim) for dim in reversed(shape))
        infos += struct.pack("<IQ", TYPE_IDS[dtype], len(data))
        data += raw + bytes(-len(raw) % GGUF_ALIGNMENT)
    header = head + table + infos
    path.write_bytes(header + bytes(-len(header) % GGUF_ALIGNMENT) + data)
    return path


def random_blocks(dtype: str, count: int, rng: np.random.Generator) -> bytes:
    """``count`` blocks of random quant bytes with small finite scales."""
    size = TYPE_SIZES[dtype][1]
    raw = rng.integers(0, 256, (count, size), dtype=np.uint8)
    scale_dtype, starts = ("<f4", (0,)) if dtype == "Q8_K" else ("<f2", SCALE_FIELDS[dtype])
    for start in starts:
        scale = (rng.standard_normal(count) * 0.01).astype(scale_dtype)
        raw[:, start : start + scale.itemsize] = scale.view(np.uint8).reshape(count, -1)
    return raw.tobytes()


def make_gguf(seed: int = GGUF_SEED) -> tuple[list, TensorMap]:
    """The GGUF tensors and the twin's BF16 tensors under Hub names, in file order.

    Block types hold random blocks; the twin holds their dequantised values with the q and k
    permutation undone, rounded to BF16, so the comparison sees only that rounding. The
    plain tensors draw normals scaled by powers of ten from 1e-7 to 1: F32 holds the twin's
    BF16 values widened (equal), BF16 the same bits (identical), F16 their nearest F16.
    """
    rng = np.random.default_rng(seed)
    gguf, twin = [], {}
    for gguf_name, hub_name, dtype, shape in GGUF_TENSORS:
        if dtype in SCALE_FIELDS or dtype == "Q8_K":
            block, size = TYPE_SIZES[dtype]
            raw = random_blocks(dtype, int(np.prod(shape)) // block, rng)
            values = dequantize(np.frombuffer(raw, dtype=np.uint8), dtype, block, size)
            values = values.reshape(shape)
            heads = PERMUTED_HEADS.get(gguf_name)
            if heads is not None:
                values = values[rotary_rows(shape[0], heads, gguf_name)]
            bits = f32_to_bf16_bits(values)
        else:
            scales = 10.0 ** rng.integers(-7, 1, shape)
            bits = f32_to_bf16_bits((rng.standard_normal(shape) * scales).astype(np.float32))
            widened = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
            stored = {"F32": widened, "BF16": bits, "F16": widened.astype(np.float16)}[dtype]
            raw = np.ascontiguousarray(stored).tobytes()
        gguf.append((gguf_name, dtype, shape, raw))
        twin[hub_name] = ("BF16", bits)
    return gguf, twin


def dequantised_digests(path: Path) -> dict:
    """Per GGUF tensor, in file order: its type and the SHA-256 of its float32 values."""
    digests = {}
    with load_gguf(path) as ckpt:
        for name, info in ckpt.tensors.items():
            raw = ckpt.raw(name)
            if info.dtype in SCALE_FIELDS or info.dtype == "Q8_K":
                values = dequantize(raw, info.dtype, *TYPE_SIZES[info.dtype])
            else:
                values = load_gguf_plain(raw, info.dtype)
            digests[name] = {
                "dtype": info.dtype,
                "float32_sha256": digest_bytes(values.view(np.uint8)),
            }
    return digests


def load_gguf_plain(raw: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << np.uint32(16)).view(np.float32)
    return np.frombuffer(raw, dtype={"F32": "<f4", "F16": "<f2"}[dtype]).astype("<f4")


def write_gguf_vectors(out_dir: Path) -> None:
    """Write quant.gguf, quant-base.safetensors, and their records into ``out_dir``."""
    gguf, twin = make_gguf()
    path = write_gguf(out_dir / "quant.gguf", gguf)
    base = write_safetensors(out_dir / "quant-base.safetensors", twin, METADATA)
    with load_gguf(path) as ckpt:
        write_json(out_dir / "quant.manifest.json", build_manifest(ckpt))
        lines = gguf_hash_lines(ckpt)
    (out_dir / "quant.gguf-hash").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    write_json(out_dir / "quant.dequant.json", dequantised_digests(path))
    write_json(out_dir / "quant-diff.json", diff_record(base, path))


def write_vectors(out_dir: Path) -> None:
    """Write a.safetensors, b.safetensors, their manifests, diff.json, and the GGUF vectors."""
    tensors_a, tensors_b = make_checkpoints()
    path_a = write_safetensors(out_dir / "a.safetensors", tensors_a, METADATA)
    path_b = write_safetensors(out_dir / "b.safetensors", tensors_b, METADATA)
    write_json(out_dir / "a.manifest.json", build_manifest(load_checkpoint(path_a)))
    write_json(out_dir / "b.manifest.json", build_manifest(load_checkpoint(path_b)))
    write_json(out_dir / "diff.json", diff_record(path_a, path_b))
    write_gguf_vectors(out_dir)


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
