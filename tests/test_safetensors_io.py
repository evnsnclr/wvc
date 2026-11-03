import json
import os
import re
import struct
import time

import numpy as np
import pytest
from helpers import bf16_bits_to_f32, f32_to_bf16_bits, write_safetensors

from wvc import hashing
from wvc.cli import main
from wvc.file_layout import data_order, parse_header_payload
from wvc.records import IntegrityError, StoreError
from wvc.safetensors_io import (
    FP8_FORMATS,
    SafetensorsError,
    element_count,
    fp8_table,
    load_checkpoint,
    offset_order,
    read_header,
    to_float32,
)


def test_read_header_parses_entries_and_metadata(tmp_path, rng):
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = f32_to_bf16_bits(rng.standard_normal(5).astype(np.float32))
    path = write_safetensors(
        tmp_path / "m.safetensors", {"a": ("F32", a), "b": ("BF16", b)}, {"k": "v"}
    )
    st = read_header(path)
    assert st.metadata == {"k": "v"}
    assert list(st.tensors) == ["a", "b"]
    assert st.tensors["a"].dtype == "F32"
    assert st.tensors["a"].shape == (3, 4)
    assert st.tensors["a"].nbytes == 48
    assert st.tensors["a"].numel == 12
    assert st.tensors["b"].begin == 48
    assert st.tensors["b"].end == 58


def test_float32_and_bf16_round_trip(tmp_path, rng):
    a = rng.standard_normal((2, 3)).astype(np.float32)
    values = np.array([1.0, -2.5, 3.14159, 1e-3, 65504.0, -1e-30], dtype=np.float32)
    bits = f32_to_bf16_bits(values)
    path = write_safetensors(tmp_path / "m.safetensors", {"a": ("F32", a), "b": ("BF16", bits)})
    ckpt = load_checkpoint(path)
    np.testing.assert_array_equal(ckpt.as_float32("a"), a)
    np.testing.assert_array_equal(ckpt.as_float32("b"), bf16_bits_to_f32(bits))
    assert ckpt.raw("a").tobytes() == a.tobytes()
    assert ckpt.as_float32("b").dtype == np.float32


def test_f16_and_int_decode():
    half = np.array([0.5, -1.25], dtype="<f2")
    np.testing.assert_array_equal(
        to_float32(np.frombuffer(half.tobytes(), np.uint8), "F16", (2,)), half
    )
    ints = np.array([[1, -2], [3, 4]], dtype="<i4")
    np.testing.assert_array_equal(
        to_float32(np.frombuffer(ints.tobytes(), np.uint8), "I32", (2, 2)), ints
    )


def test_directory_of_shards_merges_namespaces(tmp_path, rng):
    a = rng.standard_normal(4).astype(np.float32)
    b = rng.standard_normal(2).astype(np.float32)
    write_safetensors(tmp_path / "model-00001-of-00002.safetensors", {"a": ("F32", a)})
    write_safetensors(tmp_path / "model-00002-of-00002.safetensors", {"b": ("F32", b)})
    ckpt = load_checkpoint(tmp_path)
    assert sorted(ckpt.tensors) == ["a", "b"]
    np.testing.assert_array_equal(ckpt.as_float32("b"), b)


def test_duplicate_tensor_across_shards_is_an_error(tmp_path, rng):
    a = rng.standard_normal(4).astype(np.float32)
    write_safetensors(tmp_path / "x.safetensors", {"a": ("F32", a)})
    write_safetensors(tmp_path / "y.safetensors", {"a": ("F32", a)})
    with pytest.raises(SafetensorsError, match="appears in both"):
        load_checkpoint(tmp_path)


def test_rejects_header_not_starting_with_brace(tmp_path):
    body = b"[]      "
    (tmp_path / "bad.safetensors").write_bytes(struct.pack("<Q", len(body)) + body)
    with pytest.raises(SafetensorsError, match="does not start"):
        read_header(tmp_path / "bad.safetensors")


def test_rejects_offsets_outside_buffer(tmp_path):
    header = b'{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}'
    header += b" " * ((8 - len(header) % 8) % 8)
    (tmp_path / "bad.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
    with pytest.raises(SafetensorsError, match="outside"):
        read_header(tmp_path / "bad.safetensors")


def test_rejects_size_shape_mismatch(tmp_path):
    header = b'{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,16]}}'
    header += b" " * ((8 - len(header) % 8) % 8)
    (tmp_path / "bad.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 16)
    with pytest.raises(SafetensorsError, match="needs 12"):
        read_header(tmp_path / "bad.safetensors")


def test_missing_path_is_an_error(tmp_path):
    with pytest.raises(SafetensorsError, match="not a file or directory"):
        load_checkpoint(tmp_path / "nope")


def test_rejects_entry_that_is_not_an_object(tmp_path):
    header = b'{"a": 5}'
    header += b" " * ((8 - len(header) % 8) % 8)
    (tmp_path / "bad.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(SafetensorsError, match="malformed entry"):
        read_header(tmp_path / "bad.safetensors")


def _write_raw(path, header, buffer=b""):
    """A file of ``header`` (JSON text) padded to 8 bytes, then ``buffer``."""
    body = header.encode() + b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(body)) + body + buffer)
    return path


def _entry(shape, dtype="U8", nbytes=0):
    return f'{{"a":{{"dtype":"{dtype}","shape":{json.dumps(shape)},"data_offsets":[0,{nbytes}]}}}}'


def test_a_100000_dim_shape_is_refused_quickly(tmp_path):
    path = _write_raw(tmp_path / "long.safetensors", _entry([2] * 100_000))
    start = time.perf_counter()
    with pytest.raises(SafetensorsError, match="'a' has 100000 dimensions, more than 64"):
        read_header(path)
    assert time.perf_counter() - start < 2


def test_shape_dimension_cap_and_zero_dims(tmp_path):
    assert read_header(_write_raw(tmp_path / "a.safetensors", _entry([1] * 64, nbytes=1), b"\0"))
    path = _write_raw(tmp_path / "b.safetensors", _entry([1] * 65, nbytes=1), b"\0")
    with pytest.raises(SafetensorsError, match="65 dimensions, more than 64"):
        read_header(path)
    path = _write_raw(tmp_path / "c.safetensors", _entry([3, 0, 10**100], dtype="F32"))
    assert read_header(path).tensors["a"].nbytes == 0  # a zero dim holds no elements


def test_shape_whose_product_exceeds_the_buffer_stops_early(tmp_path):
    assert element_count([10**1000] * 64, 8) is None
    assert element_count([2, 3, 0], 1) == 0
    assert element_count([2, 3], 6) == 6
    path = _write_raw(tmp_path / "big.safetensors", _entry([10**1000] * 64, nbytes=8), b"\0" * 8)
    with pytest.raises(
        SafetensorsError, match=r"\.\.\.\] \(64 dims\) has more elements than the 8"
    ):
        read_header(path)


def test_header_with_excess_brackets_is_refused_before_parsing(tmp_path):
    amplified = '{"a":[' + "{}," * 10_000 + "{}]}"
    braced_name = '{"' + "{" * 1_000 + '":{"dtype":"U8","shape":[],"data_offsets":[0,1]}}'
    for header in (amplified, braced_name):
        path = _write_raw(tmp_path / "amp.safetensors", header, b"\0")
        with pytest.raises(SafetensorsError, match=r"'\{' and '\[', more than the \d+ its size"):
            read_header(path)
        body = path.read_bytes()
        with pytest.raises(IntegrityError, match="more than the"):
            parse_header_payload(body[: len(body) - 1], len(body), "payload")
    # The densest honest header, compact scalar entries with short names, stays under the bound.
    entries = {
        str(i): {"dtype": "U8", "shape": [], "data_offsets": [i, i + 1]} for i in range(1000)
    }
    compact = json.dumps(entries, separators=(",", ":"))
    assert len(read_header(_write_raw(tmp_path / "ok.safetensors", compact, b"\0" * 1000)).tensors)


def test_header_values_nested_deeper_than_an_entry_are_refused(tmp_path):
    nested = '{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"x":{"y":"z"}}}'
    path = _write_raw(tmp_path / "nested.safetensors", nested, b"\0")
    with pytest.raises(SafetensorsError, match="values must be strings or lists of integers"):
        read_header(path)
    path = _write_raw(tmp_path / "meta.safetensors", '{"__metadata__":{"k":{"a":"b"}}}')
    with pytest.raises(SafetensorsError, match="__metadata__ must be an object mapping strings"):
        read_header(path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are POSIX only")
def test_a_fifo_shard_is_refused_without_blocking(tmp_path, capsys):
    write_safetensors(tmp_path / "a.safetensors", {"w": ("F32", np.zeros(4, np.float32))})
    os.mkfifo(tmp_path / "b.safetensors")
    with pytest.raises(StoreError, match=re.escape("b.safetensors is not a regular file")):
        load_checkpoint(tmp_path)
    assert main(["hash", str(tmp_path), "--root-only"]) == 2
    assert "is not a regular file" in capsys.readouterr().err


def test_raw_refuses_a_file_truncated_after_its_header_was_read(tmp_path, rng):
    a = rng.standard_normal(4).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)
    path = write_safetensors(tmp_path / "m.safetensors", {"a": ("F32", a), "b": ("F32", b)})
    ckpt = load_checkpoint(path)  # the header promises 48 data bytes
    with path.open("r+b") as handle:
        handle.truncate(ckpt.files[0].data_start + 20)  # a whole, b cut short
    assert ckpt.raw("a").tobytes() == a.tobytes()
    with pytest.raises(SafetensorsError, match=r"tensor 'b' .*m\.safetensors.* holds only 4"):
        ckpt.raw("b")
    ckpt.close()
    write_safetensors(path, {"a": ("F32", a), "b": ("F32", b)})
    ckpt = load_checkpoint(path)
    with path.open("r+b") as handle:
        handle.truncate(0)
    with pytest.raises(SafetensorsError, match="tensor 'a'"):
        ckpt.raw("a")
    ckpt.close()


ENTRY_W = '"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}'
HEADER_CORPUS = {  # each header is followed by a 4-byte data buffer
    "valid": ("{" + ENTRY_W + "}", None),
    "valid with metadata": ('{"__metadata__":{"k":"v"},' + ENTRY_W + "}", None),
    "NaN as an entry": ('{"x":NaN,' + ENTRY_W + "}", "NaN is not JSON"),
    "NaN in metadata": ('{"__metadata__":{"k":NaN},' + ENTRY_W + "}", "NaN is not JSON"),
    "NaN inside an entry": (
        '{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"x":NaN}}',
        "NaN is not JSON",
    ),
    "Infinity in a shape": (
        '{"w":{"dtype":"F32","shape":[Infinity],"data_offsets":[0,4]}}',
        "Infinity is not JSON",
    ),
    "-Infinity in offsets": (
        '{"w":{"dtype":"F32","shape":[1],"data_offsets":[-Infinity,4]}}',
        "-Infinity is not JSON",
    ),
    "overflowing float": (
        '{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"x":1e999}}',
        "values must be strings or lists of integers",
    ),
    "duplicate key": ("{" + ENTRY_W + "," + ENTRY_W + "}", "duplicate key"),
    "control character in a name": (
        '{"w\\u0001":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
        "control character",
    ),
    "lone surrogate in a name": (
        '{"w\\ud800":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
        "not valid UTF-8",
    ),
    "not an object": ("[1]", "does not start with"),
    "truncated JSON": ('{"w":{"dtype":"F32"', "not valid JSON"),
    "metadata not strings": ('{"__metadata__":{"k":1},' + ENTRY_W + "}", "__metadata__ must"),
}


@pytest.mark.parametrize("case", sorted(HEADER_CORPUS))
def test_read_header_and_header_payloads_agree(tmp_path, case):
    """``hash`` (read_header) and ``commit``/``show``/``checkout`` (payloads) share one parser."""
    header, refusal = HEADER_CORPUS[case]
    path = _write_raw(tmp_path / "m.safetensors", header, b"\0" * 4)
    body = path.read_bytes()
    payload = body[: len(body) - 4]
    if refusal is None:
        expected = read_header(path)
        ordered = parse_header_payload(payload, len(body), "payload")
        assert ordered == tuple(expected.tensors.values())
        return
    with pytest.raises(SafetensorsError, match=refusal):
        read_header(path)
    with pytest.raises(IntegrityError, match=refusal):
        parse_header_payload(payload, len(body), "payload")


def test_header_payload_framing_is_checked(tmp_path):
    path = _write_raw(tmp_path / "m.safetensors", "{" + ENTRY_W + "}", b"\0" * 4)
    body = path.read_bytes()
    payload = body[: len(body) - 4]
    with pytest.raises(IntegrityError, match="shorter than the 8-byte header size field"):
        parse_header_payload(payload[:7], len(body), "payload")
    with pytest.raises(IntegrityError, match="does not give the payload's length"):
        parse_header_payload(payload + b" ", len(body) + 1, "payload")
    with pytest.raises(IntegrityError, match="shorter than its header"):
        parse_header_payload(payload, len(payload) - 1, "payload")


def test_offset_order_is_one_key_shared_by_hashing_and_file_layout(tmp_path):
    header = (
        '{"b":{"dtype":"U8","shape":[0],"data_offsets":[0,0]},'
        '"a":{"dtype":"U8","shape":[0],"data_offsets":[0,0]},'
        '"c":{"dtype":"U8","shape":[2],"data_offsets":[2,4]},'
        '"d":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}'
    )
    path = _write_raw(tmp_path / "m.safetensors", header, b"\0" * 4)
    file = read_header(path)
    assert hashing.offset_order is offset_order
    assert [info.name for info in offset_order(file)] == ["a", "b", "d", "c"]
    assert data_order(file, path.stat().st_size) == tuple(offset_order(file))


def test_f32_decode_is_a_read_only_view_not_a_copy(tmp_path, rng):
    a = rng.standard_normal((2, 3)).astype(np.float32)
    raw = np.frombuffer(a.tobytes(), np.uint8)
    values = to_float32(raw, "F32", (2, 3))
    assert np.shares_memory(values, raw)
    assert not values.flags.writeable
    np.testing.assert_array_equal(values, a)
    path = write_safetensors(tmp_path / "m.safetensors", {"a": ("F32", a)})
    with load_checkpoint(path) as ckpt:
        mapped = ckpt.as_float32("a")
        assert np.shares_memory(mapped, ckpt.raw("a"))
        assert not mapped.flags.writeable
        np.testing.assert_array_equal(mapped, a)
        del mapped


def test_the_cached_fp8_table_is_read_only():
    for dtype in FP8_FORMATS:
        table = fp8_table(dtype)
        assert not table.flags.writeable
        assert fp8_table(dtype) is table
        with pytest.raises(ValueError, match="read-only"):
            table[0] = 1.0
