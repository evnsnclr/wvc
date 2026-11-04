"""GGUF reading: the container, identity, gguf-hash lines, manifests, and hostile headers."""

import hashlib
import json
import re
import struct
import tracemalloc

import numpy as np
import pytest
from gguf_helpers import (
    ARRAY,
    STRING,
    UINT64,
    Tensor,
    Value,
    gguf_bytes,
    gguf_string,
    plain,
    random_blocks,
    write_gguf,
)
from helpers import write_safetensors

from wvc.cli import main
from wvc.errors import WvcError
from wvc.gguf_io import GgufArray, GgufError, is_gguf, load_gguf, read_gguf
from wvc.hashing import build_manifest, gguf_hash_lines
from wvc.records import ManifestError, validate_manifest
from wvc.safetensors_io import SafetensorsError, load_checkpoint

LINE = re.compile(r"^sha256    ([0-9a-f]{64})  ([^:]+)(?::(.+))?$")
META = {"general.architecture": "llama", "general.name": "tiny", "llama.block_count": 1}


def _model(rng):
    """Three tensors: F32 [2, 3], Q8_0 [2, 32], and BF16 [5], in that file order."""
    return {
        "a.weight": plain("F32", rng.standard_normal((2, 3)).astype(np.float32)),
        "q.weight": Tensor("Q8_0", (2, 32), random_blocks("Q8_0", 2, rng)),
        "n.weight": plain("BF16", rng.integers(0, 2**15, 5).astype(np.uint16)),
    }


def test_header_and_tensors_are_read(tmp_path, rng):
    tensors = _model(rng)
    metadata = {**META, "tokens": ["a", "b", "c"], "flag": True, "eps": 0.5}
    path = write_gguf(tmp_path / "m.gguf", tensors, metadata)
    file = read_gguf(path)
    assert file.architecture == "llama"
    assert file.alignment == 32
    assert file.version == 3
    assert file.metadata["tokens"] == GgufArray("STRING", 3)
    assert file.metadata["flag"] is True
    assert file.metadata["eps"] == 0.5
    assert list(file.tensors) == ["a.weight", "q.weight", "n.weight"]
    q = file.tensors["q.weight"]
    assert (q.dtype, q.shape, q.nbytes) == ("Q8_0", (2, 32), 68)
    assert file.tensors["a.weight"].shape == (2, 3)  # ne [3, 2] reversed
    assert file.data_start % 32 == 0
    with load_gguf(path) as ckpt:
        for name, tensor in tensors.items():
            assert ckpt.raw(name).tobytes() == tensor.data


def test_a_custom_alignment_places_the_tensors(tmp_path, rng):
    tensors = _model(rng)
    path = write_gguf(tmp_path / "m.gguf", tensors, {"general.alignment": 64}, alignment=64)
    file = read_gguf(path)
    assert file.alignment == 64
    assert all(info.begin % 64 == 0 for info in file.tensors.values())
    with load_gguf(path) as ckpt:
        assert ckpt.raw("n.weight").tobytes() == tensors["n.weight"].data


def test_gguf_hash_lines_agree_with_independent_hashes(tmp_path, rng, capsys):
    tensors = _model(rng)
    path = write_gguf(tmp_path / "m.gguf", tensors, META)
    assert main(["hash", "--format", "gguf-hash", str(path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    # llama.cpp's gguf-hash: one line per tensor in tensor-info order, then the whole model,
    # which is the SHA-256 of the tensors' bytes concatenated without the alignment padding.
    expected = [
        (hashlib.sha256(tensor.data).hexdigest(), "m.gguf", name)
        for name, tensor in tensors.items()
    ]
    whole = hashlib.sha256(b"".join(tensor.data for tensor in tensors.values())).hexdigest()
    expected.append((whole, "m.gguf", None))
    assert [LINE.match(line).groups() for line in lines] == expected


def test_gguf_ids_equal_safetensors_ids_for_the_same_bytes(tmp_path, rng):
    values = rng.standard_normal((4, 8)).astype(np.float32)
    gguf = write_gguf(tmp_path / "m.gguf", {"w": plain("F32", values)}, META)
    st = write_safetensors(tmp_path / "m.safetensors", {"w": ("F32", values)})
    with load_gguf(gguf) as a, load_checkpoint(st) as b:
        assert build_manifest(a)["tensors"] == build_manifest(b)["tensors"]
        assert build_manifest(a)["root"] == build_manifest(b)["root"]
        assert gguf_hash_lines(a)[0].split()[1] == gguf_hash_lines(b)[0].split()[1]


def test_manifest_records_the_quantization_type_and_verifies(tmp_path, rng, capsys):
    path = write_gguf(tmp_path / "m.gguf", _model(rng), META)
    assert main(["hash", str(path)]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["tensors"]["q.weight"] == {
        "dtype": "Q8_0",
        "shape": [2, 32],
        "nbytes": 68,
        "sha256": manifest["tensors"]["q.weight"]["sha256"],
    }
    assert validate_manifest(manifest) is manifest
    written = tmp_path / "m.manifest.json"
    written.write_text(json.dumps(manifest), encoding="utf-8")
    assert main(["verify", str(path), str(written)]) == 0


@pytest.mark.parametrize(
    ("entry", "rule"),
    [
        ({"dtype": "Q8_0", "shape": [2, 30], "nbytes": 68}, "whole number of 32-element blocks"),
        ({"dtype": "Q8_0", "shape": [2, 32], "nbytes": 64}, "holds 68 bytes"),
        ({"dtype": "Q4_K", "shape": [], "nbytes": 144}, "whole number"),
        ({"dtype": "Q4_K", "shape": [2**40, 256], "nbytes": 144}, "more elements"),
        ({"dtype": "IQ9_Z", "shape": [32], "nbytes": 18}, "unsupported dtype"),
    ],
)
def test_manifest_block_entries_are_validated(entry, rule):
    manifest = {
        "format": "wvc-manifest/0",
        "root": "0" * 64,
        "tensors": {"t": {**entry, "sha256": "0" * 64}},
    }
    with pytest.raises(ManifestError, match=rule):
        validate_manifest(manifest)


def test_commit_refuses_a_gguf_file_by_name(tmp_path, rng):
    path = write_gguf(tmp_path / "m.gguf", _model(rng), META)
    with pytest.raises(SafetensorsError, match="a GGUF file; this command reads safetensors"):
        load_checkpoint(path)


def test_is_gguf_reads_the_magic(tmp_path, rng):
    assert is_gguf(write_gguf(tmp_path / "x.bin", _model(rng), META))
    assert not is_gguf(write_safetensors(tmp_path / "m.gguf", {"w": ("F32", np.zeros(2))}))
    assert not is_gguf(tmp_path)


def _tensors():
    rng = np.random.default_rng(7)
    return _model(rng)


def _hostile_cases():
    t = _tensors()
    huge = 2**62
    bad_utf8 = {"a.weight": b"\xff\xfe"}
    return {
        "bad magic": gguf_bytes(t, magic=b"GGUG"),
        "version 2": gguf_bytes(t, version=2),
        "big-endian": gguf_bytes(t, version=3 << 24),
        "tensor count past the file": gguf_bytes(t, n_tensors=huge),
        "negative tensor count": gguf_bytes(t, n_tensors=-1),
        "key-value count past the file": gguf_bytes(t, META, n_kv=huge),
        "missing tensor infos": gguf_bytes(t, n_tensors=len(t) + 1),
        "overlapping offsets": gguf_bytes(t, offsets={"q.weight": 0}),
        "a gap between tensors": gguf_bytes(t, offsets={"q.weight": 64}),
        "absurd dimensions": gguf_bytes(t, dims={"a.weight": [2**31, 2**31, 2**31, 2**31]}),
        "five dimensions": gguf_bytes(t, dims={"a.weight": [3, 2, 1, 1, 1]}),
        "a negative dimension": gguf_bytes(t, dims={"a.weight": [-3, -2]}),
        "an unknown type": gguf_bytes(t, type_ids={"a.weight": 99}),
        "a retired type": gguf_bytes(t, type_ids={"a.weight": 4}),
        "a partial block": gguf_bytes(t, dims={"q.weight": [30, 2]}),
        "a tensor past the end": gguf_bytes(t, dims={"n.weight": [10_000]}),
        "a non-UTF-8 name": gguf_bytes(t, names=bad_utf8),
        "a control character in a name": gguf_bytes(t, names={"a.weight": b"a\nb"}),
        "a long name": gguf_bytes(t, names={"a.weight": b"x" * 64}),
        "an empty name": gguf_bytes(t, names={"a.weight": b""}),
        "a duplicate name": gguf_bytes(t, names={"q.weight": b"a.weight"}),
        "a duplicate key": _with_kv(t, _entry("k") + _entry("k"), n_kv=2),
        "an alignment of 3": gguf_bytes(t, {"general.alignment": 3}),
        "an alignment of the wrong type": gguf_bytes(t, {"general.alignment": Value(UINT64, 32)}),
        "a nested array": _with_kv(t, gguf_string("k") + struct.pack("<IIQ", ARRAY, ARRAY, 1)),
        "an array longer than the file": _with_kv(
            t, gguf_string("k") + struct.pack("<IIQ", ARRAY, STRING, 2**40)
        ),
        "a string longer than the file": _with_kv(
            t, gguf_string("k") + struct.pack("<IQ", STRING, 2**40)
        ),
        "an unknown value type": _with_kv(t, gguf_string("k") + struct.pack("<I", 13)),
        "a bool that is not 0 or 1": _with_kv(t, gguf_string("k") + struct.pack("<IB", 7, 2)),
        "an empty key": _with_kv(t, gguf_string("") + struct.pack("<IB", 0, 1)),
        "truncated data": gguf_bytes(t)[:-40],
    }


def _entry(key):
    """A UINT8 key-value entry holding 1."""
    return gguf_string(key) + struct.pack("<IB", 0, 1)


def _with_kv(tensors, kv, n_kv=1):
    """A file whose key-value table is the raw bytes ``kv`` holding ``n_kv`` entries."""
    good = gguf_bytes(tensors, {"k": Value(0, 1)}, n_kv=n_kv)
    return good.replace(_entry("k"), kv, 1)


@pytest.mark.parametrize("case", sorted(_hostile_cases()))
def test_hostile_headers_are_refused(tmp_path, case):
    data = _hostile_cases()[case]
    path = tmp_path / "m.gguf"
    path.write_bytes(data)
    tracemalloc.start()
    try:
        with pytest.raises(GgufError):
            read_gguf(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1 << 20, f"{case}: {peak} bytes allocated before refusing"
    assert main(["hash", str(path)]) == 2


def test_hostile_header_messages_name_the_rule(tmp_path):
    cases = _hostile_cases()
    expected = {
        "version 2": "GGUF version 2; wvc reads version 3 only",
        "big-endian": "big-endian GGUF",
        "overlapping offsets": "must follow one another in order, without overlap",
        "an unknown type": "unknown ggml type id 99",
        "a partial block": "not a whole number of 32-element blocks",
        "absurd dimensions": "holds more bytes than",
        "tensor count past the file": "more than the file's bytes can hold",
    }
    for case, message in expected.items():
        path = tmp_path / f"{case}.gguf"
        path.write_bytes(cases[case])
        with pytest.raises(GgufError, match=re.escape(message)):
            read_gguf(path)


def test_a_fifo_is_refused(tmp_path):
    if not hasattr(__import__("os"), "mkfifo"):
        pytest.skip("no FIFOs on this platform")
    fifo = tmp_path / "m.gguf"
    __import__("os").mkfifo(fifo)
    with pytest.raises(WvcError, match="not a regular file"):
        read_gguf(fifo)


def test_a_file_shorter_than_the_header_is_refused(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF\x03\x00")
    with pytest.raises(GgufError, match="shorter than"):
        read_gguf(path)


def test_an_empty_gguf_has_no_tensors(tmp_path):
    path = write_gguf(tmp_path / "m.gguf", {}, META)
    file = read_gguf(path)
    assert file.tensors == {}
    with load_gguf(path) as ckpt:
        assert build_manifest(ckpt)["tensor_count"] == 0
