"""Property-based fuzzing of the two safetensors header parsers.

``read_header`` parses the start of a file on disk; ``file_layout.parse_header_payload`` is the
strict checker a store applies to a recorded header payload. For any input, each parser either
accepts it or raises a ``WvcError``: no other exception, no ``MemoryError``, and no call that
runs for seconds. Whatever the strict checker accepts, ``read_header`` accepts with the same
tensors, and those tensors tile the data buffer exactly. A well-formed header round-trips: both
parsers accept it and the manifest root does not depend on the parse or the header's key order.

This module found one divergence on 2026-11-03: the strict checker did not apply
``read_header``'s tensor-name rule, so it accepted names holding a control character or a lone
surrogate that ``read_header`` refuses. The parsers were unified the same day, and
``test_strict_checker_refuses_names_read_header_refuses`` keeps the minimal counterexample.

Each property runs 200 examples, so the module finishes in a few seconds.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from wvc.errors import WvcError
from wvc.file_layout import parse_header_payload
from wvc.hashing import build_manifest, manifest_root
from wvc.safetensors_io import (
    ITEMSIZES,
    MAX_DIMS,
    SIZE_FIELD_BYTES,
    SafetensorsError,
    SafetensorsFile,
    TensorInfo,
    check_name,
    load_checkpoint,
    read_header,
)

FUZZ = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
CALL_SECONDS = 5.0  # generous for a slow CI runner; a parse of these inputs takes milliseconds
MAX_BUFFER = 4096  # the largest data buffer an arbitrary-header example writes
DTYPES = sorted(ITEMSIZES)
BAD_DTYPES = ["", "F128", "f32", "F32 ", "__metadata__"]
JSON_ALPHABET = '{}[]":,.-+0123456789eEaflnrstuINy_ \\\t\n'

Parsed = tuple[tuple[str, str, tuple[int, ...], int, int], ...]


@pytest.fixture(scope="module")
def fuzz_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("header-fuzz")


def bounded(call: Callable[[], object]) -> object:
    """The call's result, or the WvcError it raised; any other exception propagates."""
    start = time.perf_counter()
    try:
        result = call()
    except WvcError as exc:
        result = exc
    elapsed = time.perf_counter() - start
    assert elapsed < CALL_SECONDS, f"a header parse took {elapsed:.2f} s"
    return result


def described(tensors: object) -> Parsed:
    """Tensors (a name map or a sequence of TensorInfo) as comparable tuples, sorted by name."""
    infos = tensors.values() if isinstance(tensors, dict) else tensors
    assert all(isinstance(info, TensorInfo) for info in infos)
    return tuple(
        sorted((info.name, info.dtype, info.shape, info.begin, info.end) for info in infos)
    )


def assert_tiles(ordered: tuple[TensorInfo, ...], buffer_len: int) -> None:
    cursor = 0
    for info in ordered:
        assert info.begin == cursor
        cursor = info.end
    assert cursor == buffer_len


def check_parsers_agree(path: Path, data: bytes, size_field: bytes | None = None) -> None:
    """Write ``data`` to ``path``, run both parsers, and check that they agree.

    The strict checker sees the prefix ``read_header`` would read; with ``size_field`` it also
    sees the header under that size field, which may not describe it.
    """
    path.write_bytes(data)
    loose = bounded(lambda: read_header(path))
    if len(data) >= SIZE_FIELD_BYTES:
        (declared,) = struct.unpack_from("<Q", data)
        prefix = data[: SIZE_FIELD_BYTES + declared]
    else:
        prefix = data
    strict = bounded(lambda: parse_header_payload(prefix, len(data), "fuzz"))
    event(f"read_header {'accepts' if isinstance(loose, SafetensorsFile) else 'refuses'}")
    event(f"strict checker {'accepts' if isinstance(strict, tuple) else 'refuses'}")
    if isinstance(strict, tuple):
        assert_tiles(strict, len(data) - len(prefix))
        assert isinstance(loose, SafetensorsFile), f"strict accepted, read_header refused: {loose}"
        assert described(strict) == described(loose.tensors)
    if size_field is not None:
        other = size_field + data[SIZE_FIELD_BYTES:]
        bounded(lambda: parse_header_payload(other, len(data), "fuzz"))


def name_allowed(name: str) -> bool:
    """Whether ``read_header``'s tensor-name rule accepts ``name``."""
    try:
        check_name(Path("fuzz"), name)
    except SafetensorsError:
        return False
    return True


def framed(header: bytes, buffer_len: int) -> bytes:
    """A file: the size field of ``header``, the header, and ``buffer_len`` zero bytes."""
    return struct.pack("<Q", len(header)) + header + bytes(buffer_len)


# Arbitrary JSON values: nested containers, non-finite floats, huge integers, any text.
json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**70), max_value=2**70),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=12),
)
json_values = st.recursive(
    json_scalars,
    lambda inner: st.one_of(
        st.lists(inner, max_size=4), st.dictionaries(st.text(max_size=6), inner, max_size=4)
    ),
    max_leaves=16,
)


def mostly(good: st.SearchStrategy, bad: st.SearchStrategy) -> st.SearchStrategy:
    """``good`` nine times in ten, else ``bad``: one defect at a time keeps headers near valid."""
    return st.integers(min_value=0, max_value=9).flatmap(lambda k: good if k else bad)


small_dims = st.integers(min_value=0, max_value=6)
dims = st.one_of(
    small_dims,
    st.integers(min_value=-3, max_value=2**70),
    st.sampled_from([0, 1, 2**31, 2**32, 2**63 - 1, 2**63, 2**64, 10**30]),
)
odd_dims = st.one_of(dims, st.floats(allow_nan=True), st.booleans(), st.text(max_size=3))
shapes = mostly(
    st.lists(small_dims, max_size=3),
    st.one_of(
        st.lists(dims, max_size=8),  # huge, negative, and zero dimensions
        st.lists(  # many dimensions, around MAX_DIMS
            st.integers(min_value=0, max_value=2), min_size=MAX_DIMS - 2, max_size=MAX_DIMS + 30
        ),
        st.lists(odd_dims, max_size=4),  # non-integers, NaN, and Infinity
        json_values,  # nested values
    ),
)
dtypes = mostly(st.sampled_from(DTYPES), st.one_of(st.sampled_from(BAD_DTYPES), json_values))
names = mostly(
    st.text(max_size=10),  # includes control characters now and then
    st.sampled_from(["a", "__metadata__", "", "\x00", "a\x7f", "\ud800"]),
)
bad_offsets = st.one_of(
    st.lists(st.integers(min_value=-5, max_value=2**65), max_size=3), json_values, st.just([])
)
metadata_values = mostly(
    st.dictionaries(st.text(max_size=5), st.text(max_size=5), max_size=3), json_values
)


def as_json(value: object) -> str:
    """JSON text of ``value``; Python's encoder writes NaN and Infinity for non-finite floats."""
    return json.dumps(value)


def small_shape(shape: object) -> bool:
    """A list of small non-negative ints, whose byte count is cheap to compute."""
    return isinstance(shape, list) and all(type(d) is int and 0 <= d <= 6 for d in shape)


def nbytes_of(dtype: object, shape: object) -> int:
    """The bytes a valid entry of this dtype and shape spans; 0 when the entry is not valid."""
    if small_shape(shape) and isinstance(dtype, str) and dtype in ITEMSIZES:
        return int(np.prod(shape, dtype=np.int64)) * ITEMSIZES[dtype]
    return 0


@st.composite
def fuzzed_entry(draw: st.DrawFn, cursor: int) -> tuple[str, int]:
    """One ``"name": {...}`` pair whose offsets usually start at ``cursor``; and its span."""
    dtype, shape = draw(dtypes), draw(shapes)
    nbytes = nbytes_of(dtype, shape)
    offsets = draw(mostly(st.just([cursor, cursor + nbytes]), bad_offsets))
    entry: dict[str, object] = {"dtype": dtype, "shape": shape, "data_offsets": offsets}
    if draw(st.integers(min_value=0, max_value=9)) == 0:
        entry[draw(st.text(max_size=6))] = draw(json_values)  # an extra key, maybe nested
    return f"{as_json(draw(names))}: {as_json(entry)}", nbytes


def excess_brackets(draw: st.DrawFn, text: str) -> str:
    """Sometimes replace or extend the header with more brackets than its size allows."""
    kind = draw(st.integers(min_value=0, max_value=12))
    if kind == 1:
        return text + "[" * draw(st.integers(min_value=1, max_value=400))
    if kind == 2:
        depth = draw(st.integers(min_value=1, max_value=120))
        return '{"x": ' + "[" * depth + "]" * depth + "}"
    if kind == 3:
        return "{" + '"e": {}, ' * draw(st.integers(min_value=1, max_value=200)) + '"z": 0}'
    return text


@st.composite
def fuzzed_headers(draw: st.DrawFn) -> tuple[bytes, int]:
    """A header near the safetensors shape, usually with a defect, and a buffer length.

    Entries are written as text so names may repeat. Offsets usually tile a buffer for the
    declared shapes; values may nest, hold non-finite numbers, or name unknown dtypes; the
    buffer may be a byte short or long; brackets may run past what the header's size allows.
    """
    pairs: list[str] = []
    cursor = 0
    for _ in range(draw(st.integers(min_value=0, max_value=5))):
        pair, nbytes = draw(fuzzed_entry(cursor))
        pairs.append(pair)
        cursor += nbytes
    if draw(st.booleans()):
        pairs.append(f'"__metadata__": {as_json(draw(metadata_values))}')
    if pairs and draw(st.integers(min_value=0, max_value=9)) == 0:
        pairs.append(pairs[0])  # a duplicate key
    text = excess_brackets(draw, "{" + ", ".join(pairs) + "}")
    text += " " * draw(st.integers(min_value=0, max_value=7))
    slack = draw(mostly(st.just(0), st.sampled_from([1, -1, 8])))
    buffer_len = max(0, min(cursor, MAX_BUFFER) + slack)
    return text.encode("utf-8", errors="surrogatepass"), buffer_len


@FUZZ
@given(raw=st.binary(max_size=300))
def test_random_bytes_are_accepted_or_refused_as_wvc_errors(fuzz_dir: Path, raw: bytes) -> None:
    check_parsers_agree(fuzz_dir / "raw.safetensors", raw)


@FUZZ
@given(
    body=st.one_of(
        st.text(alphabet=JSON_ALPHABET, max_size=300),
        st.dictionaries(st.text(max_size=8), json_values, max_size=5).map(as_json),
    ),
    buffer_len=st.integers(min_value=0, max_value=64),
)
def test_json_ish_headers_are_accepted_or_refused(
    fuzz_dir: Path, body: str, buffer_len: int
) -> None:
    header = ("{" + body if not body.startswith("{") else body).encode("utf-8", "surrogatepass")
    check_parsers_agree(fuzz_dir / "jsonish.safetensors", framed(header, buffer_len))


@FUZZ
@given(
    case=fuzzed_headers(),
    size_delta=mostly(
        st.just(0),
        st.one_of(
            st.integers(min_value=-3, max_value=3),
            st.sampled_from([2**32, 2**63, 2**64 - 1, 100_000_001]),
        ),
    ),
)
def test_fuzzed_headers_are_accepted_or_refused(
    fuzz_dir: Path, case: tuple[bytes, int], size_delta: int
) -> None:
    header, buffer_len = case
    data = framed(header, buffer_len)
    declared = (len(header) + size_delta) % 2**64
    mismatched = struct.pack("<Q", declared)
    check_parsers_agree(fuzz_dir / "fuzzed.safetensors", data, size_field=mismatched)
    if size_delta:
        check_parsers_agree(fuzz_dir / "mismatch.safetensors", mismatched + data[SIZE_FIELD_BYTES:])


tensor_names = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")), min_size=1, max_size=16
).filter(lambda name: name != "__metadata__")


@st.composite
def well_formed(draw: st.DrawFn) -> tuple[dict[str, dict], list[str], dict[str, str], bytes]:
    """Header entries that tile a buffer, two header key orders, metadata, and the buffer."""
    keys = draw(st.lists(tensor_names, unique=True, max_size=6))
    specs = {
        name: (
            draw(st.sampled_from(DTYPES)),
            draw(st.lists(st.integers(min_value=0, max_value=5), max_size=4)),
        )
        for name in keys
    }
    entries: dict[str, dict] = {}
    cursor = 0
    for name in draw(st.permutations(keys)):  # the order of the bytes in the buffer
        dtype, shape = specs[name]
        nbytes = int(np.prod(shape, dtype=np.int64)) * ITEMSIZES[dtype]
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + nbytes]}
        cursor += nbytes
    other_order = draw(st.permutations(keys))
    metadata = draw(st.dictionaries(st.text(max_size=8), st.text(max_size=8), max_size=3))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    buffer = np.random.default_rng(seed).bytes(cursor)
    return entries, list(other_order), metadata, buffer


def header_bytes(entries: dict[str, dict], order: list[str], metadata: dict[str, str]) -> bytes:
    header: dict[str, object] = {name: entries[name] for name in order}
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode("utf-8")
    return raw + b" " * ((8 - len(raw) % 8) % 8)


def root_of(path: Path) -> str:
    with load_checkpoint(path) as ckpt:
        return build_manifest(ckpt)["root"]


def strict_root(data: bytes) -> str:
    """The manifest root from the strict checker's tensors and the buffer bytes they cover."""
    (declared,) = struct.unpack_from("<Q", data)
    start = SIZE_FIELD_BYTES + declared
    ordered = parse_header_payload(data[:start], len(data), "fuzz")
    return manifest_root(
        {
            info.name: {
                "dtype": info.dtype,
                "shape": list(info.shape),
                "sha256": hashlib.sha256(data[start + info.begin : start + info.end]).hexdigest(),
            }
            for info in ordered
        }
    )


@FUZZ
@given(case=well_formed())
def test_well_formed_headers_round_trip(
    fuzz_dir: Path, case: tuple[dict[str, dict], list[str], dict[str, str], bytes]
) -> None:
    entries, other_order, metadata, buffer = case
    first = header_bytes(entries, list(entries), metadata)
    second = header_bytes(entries, other_order, metadata)
    path_a, path_b = fuzz_dir / "wf-a.safetensors", fuzz_dir / "wf-b.safetensors"
    path_a.write_bytes(framed(first, 0) + buffer)
    path_b.write_bytes(framed(second, 0) + buffer)
    data = path_a.read_bytes()

    loose = read_header(path_a)
    assert loose.metadata == metadata
    strict = parse_header_payload(data[: loose.data_start], len(data), "fuzz")
    assert described(strict) == described(loose.tensors)
    assert_tiles(strict, len(buffer))
    assert sum(info.nbytes for info in strict) == len(buffer)

    root = root_of(path_a)
    assert root_of(path_a) == root  # deterministic across two parses
    assert root_of(path_b) == root  # independent of the header's key order
    assert strict_root(data) == root


NAMES_READ_HEADER_REFUSES = ["\x00", "a\x7f", "\ud800"]


def name_case(fuzz_dir: Path, name: str) -> tuple[Path, bytes, bytes]:
    """A file with one valid BF16 scalar named ``name``; its path, bytes, and header payload."""
    entry = {"dtype": "BF16", "shape": [], "data_offsets": [0, 2]}
    header = json.dumps({name: entry}).encode("utf-8")
    data = framed(header, 2)
    path = fuzz_dir / "names.safetensors"
    path.write_bytes(data)
    return path, data, data[: SIZE_FIELD_BYTES + len(header)]


@pytest.mark.parametrize("name", NAMES_READ_HEADER_REFUSES)
def test_read_header_refuses_control_and_surrogate_names(fuzz_dir: Path, name: str) -> None:
    path, _, _ = name_case(fuzz_dir, name)
    with pytest.raises(SafetensorsError):
        read_header(path)


@pytest.mark.parametrize("name", NAMES_READ_HEADER_REFUSES)
def test_strict_checker_refuses_names_read_header_refuses(fuzz_dir: Path, name: str) -> None:
    """Hypothesis's minimal counterexample (a NUL name), with DEL and a lone surrogate."""
    _, data, payload = name_case(fuzz_dir, name)
    with pytest.raises(WvcError):
        parse_header_payload(payload, len(data), "fuzz")
