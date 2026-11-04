"""Property-based fuzzing of the GGUF header parser.

For any bytes, ``read_gguf`` either accepts them or raises a ``WvcError``: no other exception,
no ``MemoryError``, and no call that runs for seconds. Whatever it accepts tiles the data
section as the reference loader requires: tensors in order, each at the padded end of the one
before, inside the file. Inputs are well-formed files with bytes overwritten, truncated, or
extended, and whole-header field substitutions, since random bytes rarely pass the magic.

Each property runs 200 examples, so the module finishes in a few seconds.
"""

from __future__ import annotations

import struct
import time

import numpy as np
from gguf_helpers import Tensor, gguf_bytes, plain, random_blocks
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from wvc.errors import WvcError
from wvc.gguf_io import TYPE_SIZES, read_gguf

FUZZ = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
CALL_SECONDS = 5.0
RNG = np.random.default_rng(11)
SEED_FILE = gguf_bytes(
    {
        "a": plain("F32", RNG.standard_normal((2, 3)).astype(np.float32)),
        "q": Tensor("Q4_K", (1, 256), random_blocks("Q4_K", 1, RNG)),
        "b": plain("BF16", RNG.integers(0, 2**15, 7).astype(np.uint16)),
    },
    {"general.architecture": "llama", "tokens": ["x", "yz"], "n": 3, "f": 0.25},
)


def _parse(tmp_path, data: bytes):
    path = tmp_path / "f.gguf"
    path.write_bytes(data)
    started = time.perf_counter()
    try:
        file = read_gguf(path)
    except WvcError:
        file = None
    assert time.perf_counter() - started < CALL_SECONDS
    return file, len(data)


def _check_tiling(file, size: int) -> None:
    expected = 0
    for info in file.tensors.values():
        block, nbytes = TYPE_SIZES[info.dtype]
        assert info.begin == expected
        assert info.nbytes == info.numel // block * nbytes
        assert file.data_start + info.end <= size
        expected = -(-info.end // file.alignment) * file.alignment


@FUZZ
@given(
    edits=st.lists(st.tuples(st.integers(0, len(SEED_FILE) - 1), st.integers(0, 255)), max_size=8),
    cut=st.integers(0, len(SEED_FILE)),
    tail=st.binary(max_size=64),
)
def test_mutated_files_are_accepted_or_refused(tmp_path, edits, cut, tail):
    data = bytearray(SEED_FILE)
    for index, value in edits:
        data[index] = value
    file, size = _parse(tmp_path, bytes(data[:cut]) + tail)
    if file is not None:
        _check_tiling(file, size)


@FUZZ
@given(
    n_tensors=st.integers(-(2**63), 2**63 - 1),
    n_kv=st.integers(-(2**63), 2**63 - 1),
    dims=st.lists(st.integers(-(2**63), 2**63 - 1), max_size=6),
    type_id=st.integers(0, 2**32 - 1),
    offset=st.integers(0, 2**64 - 1),
)
def test_hostile_fields_are_refused_without_allocating(
    tmp_path, n_tensors, n_kv, dims, type_id, offset
):
    data = bytearray(SEED_FILE)
    struct.pack_into("<qq", data, 8, n_tensors, n_kv)
    file, size = _parse(tmp_path, bytes(data))
    if file is not None:
        _check_tiling(file, size)
    one = gguf_bytes(
        {"a": plain("F32", np.zeros(4, np.float32))},
        dims={"a": dims},
        type_ids={"a": type_id},
        offsets={"a": offset},
    )
    file, size = _parse(tmp_path, one)
    if file is not None:
        _check_tiling(file, size)
