"""Read GGUF files (llama.cpp's container) without llama.cpp.

Layout (version 3, little-endian; ``ggml/src/gguf.cpp`` is the reference reader): the magic
``GGUF``, a u32 version, an i64 tensor count and an i64 key-value count, the key-value
table (each key a u64-length string, a u32 value type, then the value; an array is an item
type, a u64 count, and the items), the tensor-info table (each a name, a u32 dimension count,
that many i64 dimensions with the fastest-varying first, a u32 ggml type, and a u64 offset),
padding to ``general.alignment`` (32 by default), then the data section, where every tensor
starts at its offset and is followed by padding to the alignment.

The parser follows the reference's rules and refuses everything they refuse, before anything
is allocated from a count or a length: a version other than 3, a count or a length that the
bytes left in the file cannot hold, a duplicate key or tensor name, a nested array, an
alignment that is not a power of two, more than four dimensions, a negative dimension, an
unknown or retired ggml type, a row that is not a whole number of blocks, and an offset other
than the one the tensors before it leave (so tensors can neither overlap nor leave gaps), and
it adds the safetensors reader's rules: names are UTF-8 without control characters and every
tensor's bytes lie inside the file. The header is read through a read-only memory map that is
closed before the function returns; tensor bytes are read later through the same
``TensorReader`` lifecycle as safetensors shards.

Shapes are given in row-major order (the reverse of ggml's ``ne``), so a ggml ``[in, out]``
weight is ``[out, in]`` here, as in the safetensors file it was converted from; the stored
bytes are then exactly the row-major bytes of that shape for every plain type. The dtype of a
tensor is its ggml type name in upper case: ``F32``, ``F16``, and ``BF16`` keep the names they
have in safetensors (the bytes are the same), and block types are named as llama.cpp's Python
package names them (``Q4_K``, ``Q8_0``, ...).
"""

from __future__ import annotations

import mmap
import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from wvc.errors import WvcError
from wvc.safetensors_io import Checkpoint, TensorInfo, element_count, shape_text

MAGIC = b"GGUF"
VERSION = 3
DEFAULT_ALIGNMENT = 32
MAX_DIMS = 4  # GGML_MAX_DIMS
MAX_NAME_BYTES = 63  # GGML_MAX_NAME is 64 with the terminating zero
MAX_KEY_BYTES = 65_535  # keys are short dotted names; a longer one is refused, not allocated
MAX_STRING_BYTES = 16 * 1024 * 1024  # a decoded metadata string (chat templates are ~10 KB)
HEADER_FIELDS_BYTES = 4 + 4 + 8 + 8  # magic, version, tensor count, key-value count
MIN_KV_BYTES = 8 + 1 + 4 + 1  # a one-byte key, a type, and a one-byte value
MIN_TENSOR_INFO_BYTES = 8 + 1 + 4 + 4 + 8  # a one-byte name, no dims, a type, an offset
ARCHITECTURE_KEY = "general.architecture"
ALIGNMENT_KEY = "general.alignment"
FILE_TYPE_KEY = "general.file_type"

# ggml type id -> (name, elements per block, bytes per block), from gguf-py's GGML_QUANT_SIZES.
# Ids 4, 5, 31-33, and 36-38 were retired and are refused like unknown ids.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18), 43: ("F8_E4M3", 1, 1),
}  # fmt: skip
TYPE_SIZES: dict[str, tuple[int, int]] = {
    name: (block, size) for name, block, size in GGML_TYPES.values()
}
# The block types by name: dtypes whose elements are not stored one per fixed-size item.
BLOCK_TYPES: dict[str, tuple[int, int]] = {
    name: sizes for name, sizes in TYPE_SIZES.items() if sizes[0] > 1
}

# general.file_type -> the quantization a file was written as: llama.cpp's ``llama_ftype``
# (gguf-py's ``LlamaFileType``) without its ``MOSTLY_`` prefix; retired ids are not listed.
FILE_TYPES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
    17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0",
}  # fmt: skip

# GGUF value type id -> (name, struct format or None for strings and arrays).
VALUE_TYPES: dict[int, tuple[str, str | None]] = {
    0: ("UINT8", "<B"), 1: ("INT8", "<b"), 2: ("UINT16", "<H"), 3: ("INT16", "<h"),
    4: ("UINT32", "<I"), 5: ("INT32", "<i"), 6: ("FLOAT32", "<f"), 7: ("BOOL", "<B"),
    8: ("STRING", None), 9: ("ARRAY", None), 10: ("UINT64", "<Q"), 11: ("INT64", "<q"),
    12: ("FLOAT64", "<d"),
}  # fmt: skip
STRING_TYPE, ARRAY_TYPE, BOOL_TYPE, UINT32_TYPE = 8, 9, 7, 4


class GgufError(WvcError):
    """The file does not follow the GGUF layout, or uses what wvc does not read."""


@dataclass(frozen=True)
class GgufArray:
    """A metadata array, kept as its item type and length (its items are checked, not kept)."""

    item_type: str
    count: int


@dataclass(frozen=True)
class GgufFile:
    """One parsed GGUF file: the ``SafetensorsFile`` fields ``Checkpoint`` reads, and more.

    ``tensors`` lists the tensors in tensor-info order, which is also their offset order.
    ``metadata`` maps each key to an int, float, bool, str, or ``GgufArray``, and
    ``value_types`` each key to its GGUF value type name (``UINT16``, ``STRING``, ...).
    """

    path: Path
    data_start: int
    metadata: dict[str, object]
    tensors: dict[str, TensorInfo]
    version: int
    alignment: int
    value_types: dict[str, str] = field(default_factory=dict)

    @property
    def architecture(self) -> str | None:
        value = self.metadata.get(ARCHITECTURE_KEY)
        return value if isinstance(value, str) else None

    @property
    def file_type(self) -> str | None:
        """The quantization the file says it was written as (``Q4_K_M``), or None if unsaid.

        An id wvc has no name for is given as ``file_type N``; a value that is not an
        integer is refused.
        """
        if FILE_TYPE_KEY not in self.metadata:
            return None
        value = self.metadata[FILE_TYPE_KEY]
        if type(value) is not int:
            raise GgufError(f"{self.path}: {FILE_TYPE_KEY} must be an integer")
        return FILE_TYPES.get(value, f"file_type {value}")


def tensor_nbytes(dtype: str, shape: Sequence[int], limit: int) -> int | None:
    """The stored size of a block-type tensor, or None when its elements exceed ``limit``.

    Raises ValueError when the last (fastest) dimension is not a whole number of blocks.
    """
    block, size = BLOCK_TYPES[dtype]
    if not shape or shape[-1] % block != 0:
        raise ValueError(f"a {dtype} row must hold a whole number of {block}-element blocks")
    count = element_count(shape, limit * block)
    return None if count is None else count // block * size


class _Cursor:
    """Bounds-checked little-endian reads from the header bytes of one file."""

    def __init__(self, path: Path, data: memoryview) -> None:
        self.path = path
        self.data = data
        self.pos = 0

    @property
    def remaining(self) -> int:
        return len(self.data) - self.pos

    def fail(self, message: str) -> GgufError:
        return GgufError(f"{self.path}: {message}")

    def take(self, count: int, what: str) -> memoryview:
        if count > self.remaining:
            raise self.fail(f"{what} needs {count} bytes but only {self.remaining} remain")
        view = self.data[self.pos : self.pos + count]
        self.pos += count
        return view

    def unpack(self, fmt: str, what: str) -> int | float:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt), what))[0]

    def u32(self, what: str) -> int:
        return int(self.unpack("<I", what))

    def u64(self, what: str) -> int:
        return int(self.unpack("<Q", what))

    def i64(self, what: str) -> int:
        return int(self.unpack("<q", what))

    def string_length(self, what: str, limit: int) -> int:
        length = self.u64(f"the length of {what}")
        if length > self.remaining:
            raise self.fail(f"{what} claims {length} bytes but only {self.remaining} remain")
        if length > limit:
            raise self.fail(f"{what} is {length} bytes long, more than the {limit} allowed")
        return length

    def text(self, what: str, limit: int) -> str:
        raw = bytes(self.take(self.string_length(what, limit), what))
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise self.fail(f"{what} is not valid UTF-8 text") from exc

    def skip_string(self, what: str) -> None:
        self.take(self.string_length(what, self.remaining), what)


def _read_value(cur: _Cursor, key: str, type_id: int) -> object:
    """One scalar value of type ``type_id``; strings are decoded, bools must be 0 or 1."""
    if type_id == STRING_TYPE:
        return cur.text(f"the value of {key!r}", MAX_STRING_BYTES)
    fmt = VALUE_TYPES[type_id][1]
    if fmt is None:  # pragma: no cover - ARRAY is read by _read_array, STRING above
        raise cur.fail(f"key {key!r} has no scalar value type")
    value = cur.unpack(fmt, f"the value of {key!r}")
    if type_id == BOOL_TYPE:
        if value not in (0, 1):
            raise cur.fail(f"key {key!r} holds the bool byte {value}, not 0 or 1")
        return bool(value)
    return value


def _value_type(cur: _Cursor, what: str) -> int:
    type_id = cur.u32(f"the value type of {what}")
    if type_id not in VALUE_TYPES:
        raise cur.fail(f"{what} has unknown value type {type_id}")
    return type_id


def _read_array(cur: _Cursor, key: str) -> GgufArray:
    """Check an array's items against the file and return its item type and length."""
    item_type = _value_type(cur, f"the items of {key!r}")
    if item_type == ARRAY_TYPE:
        raise cur.fail(f"key {key!r} holds an array of arrays, which GGUF does not allow")
    count = cur.u64(f"the length of {key!r}")
    name, fmt = VALUE_TYPES[item_type]
    item_bytes = 8 if fmt is None else struct.calcsize(fmt)  # a string takes 8 bytes at least
    if count > cur.remaining // item_bytes:
        raise cur.fail(f"key {key!r} claims {count} items but only {cur.remaining} bytes remain")
    if fmt is None:
        for _ in range(count):
            cur.skip_string(f"an item of {key!r}")
    else:
        items = cur.take(count * item_bytes, f"the items of {key!r}")
        if item_type == BOOL_TYPE and any(byte > 1 for byte in bytes(items)):
            raise cur.fail(f"key {key!r} holds a bool byte that is not 0 or 1")
    return GgufArray(item_type=name, count=count)


def _read_metadata(cur: _Cursor, count: int) -> tuple[dict[str, object], dict[str, int]]:
    """The key-value table as values by key and the value type ids by key."""
    metadata: dict[str, object] = {}
    types: dict[str, int] = {}
    for _ in range(count):
        key = cur.text("a metadata key", MAX_KEY_BYTES)
        if not key:
            raise cur.fail("a metadata key is empty")
        if key in metadata:
            raise cur.fail(f"duplicate metadata key {key!r}")
        type_id = _value_type(cur, f"key {key!r}")
        types[key] = type_id
        metadata[key] = (
            _read_array(cur, key) if type_id == ARRAY_TYPE else _read_value(cur, key, type_id)
        )
    return metadata, types


def _alignment(cur: _Cursor, metadata: dict[str, object], types: dict[str, int]) -> int:
    if ALIGNMENT_KEY not in metadata:
        return DEFAULT_ALIGNMENT
    value = metadata[ALIGNMENT_KEY]
    if types[ALIGNMENT_KEY] != UINT32_TYPE or not isinstance(value, int):
        raise cur.fail(f"{ALIGNMENT_KEY} must be a UINT32")
    if value == 0 or value & (value - 1):
        raise cur.fail(f"{ALIGNMENT_KEY} is {value}, not a power of two")
    return value


@dataclass(frozen=True)
class _RawInfo:
    name: str
    ne: tuple[int, ...]  # ggml order: fastest-varying first
    type_name: str
    offset: int


def _read_tensor_info(cur: _Cursor, seen: set[str]) -> _RawInfo:
    name = cur.text("a tensor name", MAX_NAME_BYTES)
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        raise cur.fail("a tensor name contains a control character")
    if not name:
        raise cur.fail("a tensor name is empty")
    if name in seen:
        raise cur.fail(f"duplicate tensor name {name!r}")
    seen.add(name)
    n_dims = cur.u32(f"the dimension count of {name!r}")
    if n_dims > MAX_DIMS:
        raise cur.fail(f"tensor {name!r} has {n_dims} dimensions, more than {MAX_DIMS}")
    ne = tuple(cur.i64(f"a dimension of {name!r}") for _ in range(n_dims))
    if any(dim < 0 for dim in ne):
        raise cur.fail(f"tensor {name!r} has a negative dimension {shape_text(ne[::-1])}")
    type_id = cur.u32(f"the type of {name!r}")
    if type_id not in GGML_TYPES:
        raise cur.fail(f"tensor {name!r} has unknown ggml type id {type_id}")
    offset = cur.u64(f"the offset of {name!r}")
    return _RawInfo(name=name, ne=ne, type_name=GGML_TYPES[type_id][0], offset=offset)


def _padded(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _place(path: Path, raw: _RawInfo, expected: int, data_len: int) -> TensorInfo:
    """Validate one tensor's type, size, and offset against the data section."""
    shape = raw.ne[::-1]
    block, size = TYPE_SIZES[raw.type_name]
    if raw.ne and raw.ne[0] % block != 0:
        raise GgufError(
            f"{path}: tensor {raw.name!r} of type {raw.type_name} has rows of {raw.ne[0]} "
            f"elements, not a whole number of {block}-element blocks"
        )
    if not raw.ne and block != 1:
        raise GgufError(f"{path}: tensor {raw.name!r} of block type {raw.type_name} has no rows")
    count = element_count(shape, data_len * block)
    if count is None or count // block * size > data_len:
        raise GgufError(
            f"{path}: tensor {raw.name!r} of shape {shape_text(shape)} holds more bytes than "
            f"the {data_len}-byte data section"
        )
    nbytes = count // block * size
    if raw.offset != expected:
        raise GgufError(
            f"{path}: tensor {raw.name!r} starts at offset {raw.offset}, but the tensors before "
            f"it end at {expected}: tensors must follow one another in order, without overlap"
        )
    if raw.offset + nbytes > data_len:
        raise GgufError(
            f"{path}: tensor {raw.name!r} spans [{raw.offset}, {raw.offset + nbytes}) past the "
            f"end of the {data_len}-byte data section"
        )
    return TensorInfo(
        name=raw.name, dtype=raw.type_name, shape=shape, begin=raw.offset, end=raw.offset + nbytes
    )


def _counts(cur: _Cursor) -> tuple[int, int]:
    magic = bytes(cur.take(4, "the magic"))
    if magic != MAGIC:
        raise cur.fail("not a GGUF file (the magic is not 'GGUF')")
    version = cur.u32("the version")
    if version != VERSION:
        if version & 0xFFFF == 0:
            raise cur.fail(
                f"version field {version:#010x}: a big-endian GGUF, which wvc cannot read"
            )
        raise cur.fail(f"GGUF version {version}; wvc reads version {VERSION} only")
    n_tensors = cur.i64("the tensor count")
    n_kv = cur.i64("the key-value count")
    for what, count, least in (
        ("tensor count", n_tensors, MIN_TENSOR_INFO_BYTES),
        ("key-value count", n_kv, MIN_KV_BYTES),
    ):
        if count < 0:
            raise cur.fail(f"the {what} {count} is negative")
        if count > cur.remaining // least:
            raise cur.fail(f"the {what} {count} is more than the file's bytes can hold")
    return n_tensors, n_kv


def parse_gguf(path: Path, data: memoryview) -> GgufFile:
    """Parse the GGUF header held at the start of ``data``, the whole file's bytes."""
    cur = _Cursor(path, data)
    n_tensors, n_kv = _counts(cur)
    metadata, types = _read_metadata(cur, n_kv)
    alignment = _alignment(cur, metadata, types)
    seen: set[str] = set()
    raws = [_read_tensor_info(cur, seen) for _ in range(n_tensors)]
    data_start = _padded(cur.pos, alignment) if raws else cur.pos
    if data_start > len(data):
        raise cur.fail("the file ends before its data section starts")
    data_len = len(data) - data_start
    tensors: dict[str, TensorInfo] = {}
    expected = 0
    for raw in raws:
        info = _place(path, raw, expected, data_len)
        tensors[info.name] = info
        expected = _padded(info.end, alignment)
    return GgufFile(
        path=path,
        data_start=data_start,
        metadata=metadata,
        tensors=tensors,
        version=VERSION,
        alignment=alignment,
        value_types={key: VALUE_TYPES[type_id][0] for key, type_id in types.items()},
    )


def is_gguf(path: str | Path) -> bool:
    """True when ``path`` is a file whose first four bytes are the GGUF magic."""
    path = Path(path)
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(len(MAGIC)) == MAGIC


def read_gguf(path: str | Path) -> GgufFile:
    """Parse the header of one GGUF file through a read-only map that is closed on return."""
    from wvc.fsio import open_input  # here: wvc.fsio imports wvc.records, which imports this

    path = Path(path)
    with open(open_input(path), "rb") as handle:
        if os.fstat(handle.fileno()).st_size < HEADER_FIELDS_BYTES:
            raise GgufError(f"{path}: file is shorter than the {HEADER_FIELDS_BYTES}-byte header")
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        view = memoryview(mapped)
        try:
            return parse_gguf(path, view)
        finally:
            view.release()
    finally:
        mapped.close()


def load_gguf(path: str | Path) -> Checkpoint:
    """A one-file ``Checkpoint`` over a GGUF file; only the header is read here."""
    return Checkpoint([read_gguf(path)])


def gguf_files(ckpt: Checkpoint) -> list[GgufFile]:
    """The GGUF files of a checkpoint (empty for a safetensors checkpoint)."""
    return [file for file in ckpt.files if isinstance(file, GgufFile)]
