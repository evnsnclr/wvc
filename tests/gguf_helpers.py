"""A small GGUF writer for tests, with knobs for writing hostile headers.

``gguf_bytes`` lays a file out as llama.cpp's writer does: header, key-value table, tensor
infos (dimensions fastest first), padding to the alignment, then each tensor's bytes padded to
the alignment. Its keyword arguments override single fields so that tests can write the files
the reader must refuse. ``llama_permute`` is llama.cpp's q/k conversion, copied from
``convert_hf_to_gguf.py`` so that the tests check wvc's inverse against the original.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wvc.gguf_io import GGML_TYPES, TYPE_SIZES

TYPE_IDS = {name: type_id for type_id, (name, _, _) in GGML_TYPES.items()}
PLAIN_FORMATS = {"F32": "<f4", "F16": "<f2", "F64": "<f8", "I8": "i1", "I16": "<i2", "I32": "<i4"}
STRING, ARRAY, UINT16, UINT32, INT32, FLOAT32, BOOL, UINT64 = 8, 9, 2, 4, 5, 6, 7, 10
SCALAR_FORMATS = {
    UINT16: "<H",
    UINT32: "<I",
    INT32: "<i",
    FLOAT32: "<f",
    BOOL: "<B",
    UINT64: "<Q",
    0: "<B",
}


@dataclass(frozen=True)
class Value:
    """A metadata value with an explicit GGUF value type id."""

    type_id: int
    value: object


@dataclass(frozen=True)
class Tensor:
    """One tensor to write: dtype, row-major shape, and its stored bytes."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes


def plain(dtype: str, array: np.ndarray) -> Tensor:
    """A plain-typed tensor; BF16 arrays hold uint16 bit patterns."""
    fmt = "<u2" if dtype == "BF16" else PLAIN_FORMATS[dtype]
    return Tensor(dtype, tuple(array.shape), np.ascontiguousarray(array).astype(fmt).tobytes())


def gguf_string(text: str | bytes) -> bytes:
    raw = text.encode("utf-8") if isinstance(text, str) else text
    return struct.pack("<Q", len(raw)) + raw


def _value(value: object) -> tuple[int, bytes]:
    if isinstance(value, Value):
        if value.type_id == STRING:
            return STRING, gguf_string(value.value)  # type: ignore[arg-type]
        return value.type_id, struct.pack(SCALAR_FORMATS[value.type_id], value.value)
    if isinstance(value, str):
        return STRING, gguf_string(value)
    if isinstance(value, bool):
        return BOOL, struct.pack("<B", value)
    if isinstance(value, int):
        return UINT32, struct.pack("<I", value)
    if isinstance(value, float):
        return FLOAT32, struct.pack("<f", value)
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            items = b"".join(gguf_string(item) for item in value)
            return ARRAY, struct.pack("<IQ", STRING, len(value)) + items
        items = b"".join(struct.pack("<i", item) for item in value)
        return ARRAY, struct.pack("<IQ", INT32, len(value)) + items
    raise TypeError(f"no GGUF encoding for {value!r}")


def kv_bytes(key: str | bytes, value: object) -> bytes:
    type_id, payload = _value(value)
    return gguf_string(key) + struct.pack("<I", type_id) + payload


def _pad(size: int, alignment: int) -> int:
    return -(-size // alignment) * alignment


def gguf_bytes(
    tensors: Mapping[str, Tensor],
    metadata: Mapping[str, object] | None = None,
    *,
    alignment: int = 32,
    magic: bytes = b"GGUF",
    version: int = 3,
    n_tensors: int | None = None,
    n_kv: int | None = None,
    offsets: Mapping[str, int] | None = None,
    type_ids: Mapping[str, int] | None = None,
    dims: Mapping[str, Sequence[int]] | None = None,
    names: Mapping[str, bytes] | None = None,
) -> bytes:
    """The bytes of a GGUF file; each keyword replaces one field of a well-formed file."""
    metadata = dict(metadata or {})
    head = magic + struct.pack("<Iqq", version, len(tensors) if n_tensors is None else n_tensors,
                               len(metadata) if n_kv is None else n_kv)  # fmt: skip
    table = b"".join(kv_bytes(key, value) for key, value in metadata.items())
    infos, data, offset = b"", b"", 0
    for name, tensor in tensors.items():
        ne = list(tensor.shape[::-1]) if dims is None or name not in dims else list(dims[name])
        type_id = (type_ids or {}).get(name, TYPE_IDS[tensor.dtype])
        start = offset if offsets is None else offsets.get(name, offset)
        raw_name = name.encode("utf-8") if names is None else names.get(name, name.encode())
        infos += gguf_string(raw_name) + struct.pack("<I", len(ne))
        infos += b"".join(struct.pack("<q", dim) for dim in ne)
        infos += struct.pack("<IQ", type_id, start)
        data += tensor.data + bytes(_pad(len(tensor.data), alignment) - len(tensor.data))
        offset = len(data)
    header = head + table + infos
    if tensors:
        header += bytes(_pad(len(header), alignment) - len(header))
    return header + data


def write_gguf(
    path: Path, tensors: Mapping[str, Tensor], metadata: Mapping | None = None, **fields: object
) -> Path:
    path.write_bytes(gguf_bytes(tensors, metadata, **fields))  # type: ignore[arg-type]
    return path


def llama_permute(weights: np.ndarray, n_head: int, n_head_kv: int | None) -> np.ndarray:
    """``LlamaModel.permute`` from llama.cpp's convert_hf_to_gguf.py, in NumPy."""
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (
        weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
        .swapaxes(1, 2)
        .reshape(weights.shape)
    )


def random_blocks(dtype: str, count: int, rng: np.random.Generator) -> bytes:
    """``count`` blocks of random quant bytes whose scales are finite and small."""
    size = TYPE_SIZES[dtype][1]
    raw = rng.integers(0, 256, (count, size), dtype=np.uint8)
    fields = {"Q4_1": (0, 2), "Q5_1": (0, 2), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,)}
    if dtype == "Q8_K":
        scale = (rng.standard_normal(count) * 0.01).astype("<f4")
        raw[:, 0:4] = scale.view(np.uint8).reshape(count, 4)
    else:
        for start in fields.get(dtype, (0,)):
            scale = (rng.standard_normal(count) * 0.01).astype("<f2")
            raw[:, start : start + 2] = scale.view(np.uint8).reshape(count, 2)
    return raw.tobytes()
