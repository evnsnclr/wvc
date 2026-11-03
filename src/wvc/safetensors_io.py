"""Read safetensors checkpoints without PyTorch.

Layout (https://github.com/huggingface/safetensors): an 8-byte little-endian header
size, a JSON header mapping tensor names to ``{dtype, shape, data_offsets}`` plus an
optional ``__metadata__`` string map, then one byte buffer holding every tensor.
Tensors are exposed as memory-mapped byte views and, on request, float32 arrays.

Lifecycle: ``load_checkpoint`` reads only the headers. A ``Checkpoint`` maps a file when one
of its tensors is first read, through a ``TensorReader`` that holds one mapping and one
descriptor, and keeps at most ``MAX_OPEN_READERS`` of them open, closing the least recently
used. A raw view is valid until its reader is closed, so consumers finish with one file's
views before reading another's (``hashing`` and ``diff`` walk file by file). ``Checkpoint``
is a context manager; ``close`` releases every reader. A reader closed while a view is still
alive releases its mapping when that view is dropped, so a view never points at unmapped
memory. ``raw`` refuses a tensor whose bytes the mapped file no longer holds.
"""

from __future__ import annotations

import contextlib
import json
import math
import mmap
import os
import struct
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wvc.errors import WvcError

MAX_OPEN_READERS = 16  # files a Checkpoint keeps mapped at once (one descriptor each)
MAX_HEADER_BYTES = 100_000_000  # the reference implementation caps the header at 100 MB
SIZE_FIELD_BYTES = 8  # the little-endian u64 header length that starts every file
MAX_DIMS = 64  # longer shapes are refused before anything multiplies them
# A tensor entry takes at least about 50 bytes for its one '{' and two '['. A header holding more
# '{' and '[' than header_size // 16 + 64 is refused unparsed, which bounds the objects it makes.
BYTES_PER_BRACKET = 16
BRACKET_SLACK = 64
SHAPE_TEXT_DIMS = 4  # the dims of a long shape that a message quotes

ITEMSIZES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E4M3FNUZ": 1, "F8_E5M2FNUZ": 1, "F8_E8M0": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8, "C64": 8,
}  # fmt: skip

# FP8 formats as (exponent bits, mantissa bits, bias, kind): "fn" has no infinities and NaN at
# exponent and mantissa all ones; "ieee" has IEEE infinities and NaNs; "fnuz" has one NaN at
# 0x80, no negative zero, and no infinities; "e8m0" is an unsigned power-of-two scale.
FP8_FORMATS = {
    "F8_E4M3": (4, 3, 7, "fn"),
    "F8_E5M2": (5, 2, 15, "ieee"),
    "F8_E4M3FNUZ": (4, 3, 8, "fnuz"),
    "F8_E5M2FNUZ": (5, 2, 16, "fnuz"),
    "F8_E8M0": (8, 0, 127, "e8m0"),
}
_FP8_TABLES: dict[str, np.ndarray] = {}

NUMPY_DTYPES = {
    "BOOL": np.bool_, "U8": np.uint8, "I8": np.int8,
    "I16": np.int16, "U16": np.uint16, "F16": np.float16,
    "I32": np.int32, "U32": np.uint32, "F32": np.float32,
    "I64": np.int64, "U64": np.uint64, "F64": np.float64,
}  # fmt: skip


class SafetensorsError(WvcError):
    """The file does not follow the safetensors layout."""


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """``object_pairs_hook`` that refuses duplicate keys instead of keeping the last one."""
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r} in the header")
        result[key] = value
    return result


def _no_constant(name: str) -> object:
    """``parse_constant`` that refuses NaN, Infinity, and -Infinity, which JSON does not have."""
    raise ValueError(f"{name} is not JSON")


def element_count(shape: Sequence[int], limit: int) -> int | None:
    """The product of ``shape``'s dimensions, or None when it exceeds ``limit``.

    Multiplies one dimension at a time and stops once the running product passes ``limit``, so
    no huge intermediate integer is built; any zero dimension makes the count 0.
    """
    if 0 in shape:
        return 0
    count = 1
    for dim in shape:
        count *= dim
        if count > limit:
            return None
    return count


def shape_text(shape: Sequence[object]) -> str:
    """A shape for a message: a short one whole, a long one as its first dims and its length."""
    head = ", ".join(str(dim) for dim in shape[:SHAPE_TEXT_DIMS])
    if len(shape) <= SHAPE_TEXT_DIMS:
        return f"[{head}]"
    return f"[{head}, ...] ({len(shape)} dims)"


def bracket_problem(header_bytes: bytes) -> str | None:
    """Why a header holds too many '{' and '[' to be parsed, or None when it may be parsed."""
    count = header_bytes.count(b"{") + header_bytes.count(b"[")
    limit = len(header_bytes) // BYTES_PER_BRACKET + BRACKET_SLACK
    if count > limit:
        return f"the header holds {count} '{{' and '[', more than the {limit} its size allows"
    return None


def check_name(path: Path, name: str) -> None:
    """Tensor names must be UTF-8 text without control characters."""
    try:
        name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SafetensorsError(f"{path}: a tensor name is not valid UTF-8 text") from exc
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        raise SafetensorsError(f"{path}: a tensor name contains a control character")


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int  # offsets are relative to the start of the byte buffer
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class SafetensorsFile:
    path: Path
    data_start: int
    metadata: dict[str, str]
    tensors: dict[str, TensorInfo]


def header_size_from(path: Path, size_field: bytes) -> int:
    """The header length from a file's first eight bytes, capped as the reference caps it."""
    if len(size_field) < SIZE_FIELD_BYTES:
        raise SafetensorsError(f"{path}: file is shorter than the 8-byte header size field")
    (header_size,) = struct.unpack("<Q", size_field[:SIZE_FIELD_BYTES])
    if header_size > MAX_HEADER_BYTES:
        raise SafetensorsError(f"{path}: header size {header_size} exceeds {MAX_HEADER_BYTES}")
    return header_size


def parse_header(
    path: Path, header_bytes: bytes, buffer_len: int
) -> tuple[dict[str, str], dict[str, TensorInfo]]:
    """Parse a header's JSON bytes and validate every entry against a ``buffer_len``-byte buffer.

    The one header parser: ``read_header`` (hash, diff, verify, commit), ``parse_payload``
    (stored header payloads: show, checkout, fsck, received commits), and Hub range reads all
    use it. The JSON must be strict (no duplicate key; no NaN, Infinity, or -Infinity, which
    the reference loader refuses too), every tensor name must pass ``check_name``, and every
    entry ``parse_entry``. ``path`` names the file in error messages. Returns the
    ``__metadata__`` map and the tensors.
    """
    if not header_bytes.startswith(b"{"):
        raise SafetensorsError(f"{path}: header does not start with '{{'")
    problem = bracket_problem(header_bytes)
    if problem is not None:
        raise SafetensorsError(f"{path}: {problem}")
    try:
        header = json.loads(
            header_bytes.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_no_constant,
        )
    except (ValueError, RecursionError) as exc:  # UnicodeDecodeError and JSONDecodeError too
        raise SafetensorsError(f"{path}: header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise SafetensorsError(f"{path}: header must be a JSON object")
    metadata = header.pop("__metadata__", None)
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
    ):
        raise SafetensorsError(f"{path}: __metadata__ must be an object mapping strings to strings")
    for name in header:
        check_name(path, name)
    tensors = {name: parse_entry(path, name, entry, buffer_len) for name, entry in header.items()}
    return dict(metadata), tensors


def read_header(path: str | Path) -> SafetensorsFile:
    """Parse the header of one safetensors file and validate every tensor entry."""
    from wvc.fsio import open_input  # here: wvc.fsio imports wvc.records, which imports this

    path = Path(path)
    with open(open_input(path), "rb") as handle:  # a regular file, or a link to one
        file_size = os.fstat(handle.fileno()).st_size
        header_size = header_size_from(path, handle.read(SIZE_FIELD_BYTES))
        header_bytes = handle.read(header_size)
    if len(header_bytes) != header_size:
        raise SafetensorsError(
            f"{path}: truncated header ({len(header_bytes)} of {header_size} bytes)"
        )
    data_start = SIZE_FIELD_BYTES + header_size
    metadata, tensors = parse_header(path, header_bytes, file_size - data_start)
    return SafetensorsFile(path=path, data_start=data_start, metadata=metadata, tensors=tensors)


def parse_payload(path: Path, data: bytes, file_nbytes: int) -> SafetensorsFile:
    """A header payload (size field and JSON header, verbatim) parsed as the start of a file.

    ``data`` is the first bytes of a ``file_nbytes``-byte file. Its size field must give its
    exact length and the file must hold at least the payload; the JSON header is then parsed
    by ``parse_header``, as ``read_header`` parses it. ``path`` names the payload in messages.
    """
    header_size = header_size_from(path, data[:SIZE_FIELD_BYTES])
    if len(data) != SIZE_FIELD_BYTES + header_size:
        raise SafetensorsError(f"{path}: the size field does not give the payload's length")
    buffer_len = file_nbytes - len(data)
    if buffer_len < 0:
        raise SafetensorsError(f"{path}: the file is shorter than its header")
    metadata, tensors = parse_header(path, data[SIZE_FIELD_BYTES:], buffer_len)
    return SafetensorsFile(path=path, data_start=len(data), metadata=metadata, tensors=tensors)


def offset_order(file: SafetensorsFile) -> list[TensorInfo]:
    """A file's tensors in the order their bytes sit in the buffer; ties broken by name."""
    return sorted(file.tensors.values(), key=lambda info: (info.begin, info.end, info.name))


def parse_entry(path: Path, name: str, entry: object, buffer_len: int) -> TensorInfo:
    """Validate one header entry against the buffer length; ``path`` names it in errors."""
    if not isinstance(entry, dict):
        raise SafetensorsError(f"{path}: malformed entry for tensor {name!r}: {entry!r}")
    dtype, shape_raw, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
    if isinstance(shape_raw, list) and len(shape_raw) > MAX_DIMS:
        raise SafetensorsError(
            f"{path}: tensor {name!r} has {len(shape_raw)} dimensions, more than {MAX_DIMS}"
        )
    if (
        not isinstance(shape_raw, list)
        or not all(_is_exact_int(dim) for dim in shape_raw)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(_is_exact_int(x) for x in offsets)
    ):
        raise SafetensorsError(
            f"{path}: malformed entry for tensor {name!r}: shape must be a list of integers "
            "and data_offsets a list of two integers"
        )
    shape = tuple(shape_raw)
    begin, end = offsets
    if not isinstance(dtype, str) or dtype not in ITEMSIZES:
        raise SafetensorsError(f"{path}: tensor {name!r} has unsupported dtype {dtype!r}")
    if not all(_is_flat_value(value) for value in entry.values()):
        raise SafetensorsError(
            f"{path}: malformed entry for tensor {name!r}: its values must be strings or lists "
            "of integers"
        )
    if any(dim < 0 for dim in shape):
        raise SafetensorsError(
            f"{path}: tensor {name!r} has a negative dimension {shape_text(shape)}"
        )
    if begin < 0 or end < begin or end > buffer_len:
        raise SafetensorsError(
            f"{path}: tensor {name!r} offsets [{begin}, {end}) fall outside "
            f"the {buffer_len}-byte buffer"
        )
    count = element_count(shape, buffer_len)
    if count is None:
        raise SafetensorsError(
            f"{path}: tensor {name!r} of shape {shape_text(shape)} has more elements than "
            f"the {buffer_len}-byte buffer holds"
        )
    expected = count * ITEMSIZES[dtype]
    if end - begin != expected:
        raise SafetensorsError(
            f"{path}: tensor {name!r} spans {end - begin} bytes "
            f"but {dtype}{shape_text(shape)} needs {expected}"
        )
    return TensorInfo(name=name, dtype=dtype, shape=shape, begin=begin, end=end)


def _is_exact_int(value: object) -> bool:
    return type(value) is int


def _is_flat_value(value: object) -> bool:
    """A string or a list of integers: no value of a tensor entry nests deeper."""
    return isinstance(value, str) or (
        isinstance(value, list) and all(_is_exact_int(item) for item in value)
    )


def _fp8_value(code: int, exp_bits: int, mant_bits: int, bias: int, kind: str) -> float:
    """The float value of one FP8 code under the given format."""
    if kind == "e8m0":
        return float("nan") if code == 0xFF else float(2.0 ** (code - bias))
    sign = -1.0 if code & 0x80 else 1.0
    exponent = (code >> mant_bits) & ((1 << exp_bits) - 1)
    mantissa = code & ((1 << mant_bits) - 1)
    all_ones = (1 << exp_bits) - 1
    if kind == "fnuz":
        if code == 0x80:
            return float("nan")
    elif kind == "fn":
        if exponent == all_ones and mantissa == (1 << mant_bits) - 1:
            return float("nan")
    elif exponent == all_ones:
        return sign * float("inf") if mantissa == 0 else float("nan")
    if exponent == 0:
        return sign * (mantissa / (1 << mant_bits)) * 2.0 ** (1 - bias)
    return sign * (1 + mantissa / (1 << mant_bits)) * 2.0 ** (exponent - bias)


def fp8_table(dtype: str) -> np.ndarray:
    """A 256-entry float32 lookup table for an FP8 dtype (exact; cached)."""
    if dtype not in _FP8_TABLES:
        exp_bits, mant_bits, bias, kind = FP8_FORMATS[dtype]
        values = [_fp8_value(code, exp_bits, mant_bits, bias, kind) for code in range(256)]
        table = np.array(values, dtype=np.float32)
        table.flags.writeable = False  # one cached copy serves every caller
        _FP8_TABLES[dtype] = table
    return _FP8_TABLES[dtype]


def decodable(dtype: str) -> bool:
    """True when ``to_float32`` can decode the dtype (C64 and unknown dtypes cannot)."""
    return dtype == "BF16" or dtype in FP8_FORMATS or dtype in NUMPY_DTYPES


def to_float32(raw: np.ndarray, dtype: str, shape: Sequence[int]) -> np.ndarray:
    """Decode raw little-endian bytes of the given safetensors dtype into a float32 array.

    F32 data is not copied: the result is then a view of ``raw``, read-only when ``raw`` is (as
    a mapped tensor is), so callers treat every result as read-only.
    """
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << np.uint32(16)
        values = bits.view(np.float32)
    elif dtype in FP8_FORMATS:
        values = fp8_table(dtype)[np.frombuffer(raw, dtype=np.uint8)]
    elif dtype in NUMPY_DTYPES:
        values = np.frombuffer(raw, dtype=np.dtype(NUMPY_DTYPES[dtype]).newbyteorder("<"))
        values = values.astype(np.float32, copy=False)
    else:
        raise SafetensorsError(f"cannot decode dtype {dtype!r} to float32")
    return values.reshape(tuple(shape))


class TensorReader:
    """Memory-mapped access to the tensors of one safetensors file, until ``close``.

    The file is opened through ``fsio.open_input`` and mapped read-only. The map keeps its own
    duplicate of the descriptor (``mmap`` duplicates it), so the reader holds one descriptor and
    one mapping; ``close`` closes the map, which unmaps the file and closes that descriptor.
    A view returned by ``raw`` belongs to the map: use it before ``close``. If one is still
    alive at ``close`` (a caller kept it, or a traceback holds it), the bytes stay readable and
    the map and its descriptor are released when the last such view is dropped instead.
    """

    def __init__(self, file: SafetensorsFile) -> None:
        from wvc.fsio import open_input  # here: wvc.fsio imports wvc.records, which imports this

        self.file = file
        self._map: mmap.mmap | None = None
        with open(open_input(file.path), "rb") as handle:  # the map keeps its own descriptor
            try:
                self._map = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            except ValueError:  # an empty file cannot be mapped: it was truncated to nothing
                self._map = None
        self._bytes: np.ndarray | None = (
            np.zeros(0, dtype=np.uint8)
            if self._map is None
            else np.frombuffer(self._map, dtype=np.uint8)
        )

    @property
    def closed(self) -> bool:
        return self._bytes is None

    def close(self) -> None:
        """Unmap the file and close the map's descriptor (deferred while a view is alive)."""
        self._bytes = None
        mapped, self._map = self._map, None
        if mapped is None:
            return
        with contextlib.suppress(BufferError):  # a view is alive; dropping it releases the map
            mapped.close()

    def raw(self, name: str) -> np.ndarray:
        """The stored bytes of tensor ``name``; SafetensorsError when the file is shorter."""
        if self._bytes is None:
            raise ValueError(f"{self.file.path}: the reader is closed")
        info = self.file.tensors[name]
        start = self.file.data_start + info.begin
        view = self._bytes[start : start + info.nbytes]
        if len(view) != info.nbytes:
            raise SafetensorsError(
                f"tensor {name!r} in {self.file.path} needs {info.nbytes} bytes but the file "
                f"holds only {len(view)} of them: it was truncated after its header was read"
            )
        return view

    def as_float32(self, name: str) -> np.ndarray:
        info = self.file.tensors[name]
        return to_float32(self.raw(name), info.dtype, info.shape)


class Checkpoint:
    """One or more safetensors shards exposed as a single tensor namespace.

    Readers are opened lazily, one per file, and at most ``MAX_OPEN_READERS`` stay open: the
    least recently used one is closed when another file is needed. A view returned by ``raw``
    is valid only until its file's reader is closed, so finish with it before asking for
    tensors of other files. ``close`` (or leaving a ``with`` block) closes every reader; a
    closed checkpoint refuses further reads.
    """

    def __init__(self, files: Sequence[SafetensorsFile]) -> None:
        self.files = tuple(files)
        self.tensors: dict[str, TensorInfo] = {}
        self._owner: dict[str, SafetensorsFile] = {}
        for file in self.files:
            for name, info in file.tensors.items():
                if name in self.tensors:
                    raise SafetensorsError(
                        f"tensor {name!r} appears in both {self._owner[name].path} and {file.path}"
                    )
                self.tensors[name] = info
                self._owner[name] = file
        self._readers: OrderedDict[Path, TensorReader] = OrderedDict()  # least recent first
        self._closed = False

    def __enter__(self) -> Checkpoint:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close every open reader; later reads raise ValueError."""
        self._closed = True
        while self._readers:
            _, reader = self._readers.popitem(last=False)
            reader.close()

    def file_of(self, name: str) -> SafetensorsFile:
        """The file that holds tensor ``name``."""
        return self._owner[name]

    def _reader(self, name: str) -> TensorReader:
        if self._closed:
            raise ValueError("the checkpoint is closed")
        file = self._owner[name]
        reader = self._readers.get(file.path)
        if reader is not None:
            self._readers.move_to_end(file.path)
            return reader
        while self._readers and len(self._readers) >= MAX_OPEN_READERS:
            _, oldest = self._readers.popitem(last=False)
            oldest.close()
        reader = TensorReader(file)
        self._readers[file.path] = reader
        return reader

    def raw(self, name: str) -> np.ndarray:
        return self._reader(name).raw(name)

    def as_float32(self, name: str) -> np.ndarray:
        return self._reader(name).as_float32(name)


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Open a single ``.safetensors`` file or a directory of shards.

    Only the headers are read here. Use the result as a context manager (or call ``close``) to
    release its mapped files promptly; one that is never closed is released when collected.
    """
    path = Path(path)
    if path.is_dir():
        shards = sorted(p for p in path.glob("*.safetensors") if not p.name.startswith("."))
        if not shards:
            raise SafetensorsError(f"{path}: no *.safetensors files found")
    elif path.is_file():
        shards = [path]
    else:
        raise SafetensorsError(f"{path}: not a file or directory")
    return Checkpoint([read_header(shard) for shard in shards])
