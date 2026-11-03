import hashlib
import json
import os
import time

import numpy as np
import pytest
from helpers import write_safetensors

from wvc.cli import main
from wvc.hashing import build_manifest
from wvc.quote import SHOWN_LIMIT, shown
from wvc.records import StoreError
from wvc.safetensors_io import load_checkpoint
from wvc.verify import ManifestError, _check, load_manifest, verify_checkpoint

NO_TENSORS = {"ok": 0, "mismatch": 0, "reshaped": 0, "missing": 0, "extra": 0}


def _write_manifest(path, manifest):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return path


def _hash_to(path, ckpt_path):
    """Save the manifest of ``ckpt_path`` the way ``wvc hash`` writes it."""
    return _write_manifest(path, build_manifest(load_checkpoint(ckpt_path)))


def _base(tmp_path, rng):
    """Write a two-tensor checkpoint and its manifest; return both arrays and both paths."""
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    ckpt = write_safetensors(tmp_path / "base.safetensors", {"a": ("F32", a), "b": ("F32", b)})
    return a, b, ckpt, _hash_to(tmp_path / "manifest.json", ckpt)


def _flip_last_byte(src, dst):
    """Copy ``src`` to ``dst`` with the final byte inverted (the last byte of the last tensor)."""
    data = bytearray(src.read_bytes())
    data[-1] ^= 0xFF
    dst.write_bytes(bytes(data))
    return dst


def _flipped_digest(array):
    raw = bytearray(array.tobytes())
    raw[-1] ^= 0xFF
    return hashlib.sha256(raw).hexdigest()


def _verify(ckpt_path, manifest_path):
    report = verify_checkpoint(load_checkpoint(ckpt_path), load_manifest(manifest_path))
    return report, {check.name: check for check in report.tensors}


def _rows(out):
    """Split the table rows of the text output by tensor name (skips summary, blank, header)."""
    return {line.split()[1]: line.split() for line in out.splitlines()[3:]}


def test_matching_checkpoint_verifies(tmp_path, rng, capsys):
    _, _, ckpt, manifest = _base(tmp_path, rng)
    report, by_name = _verify(ckpt, manifest)
    assert report.passed
    assert report.root_match
    assert report.summary == {**NO_TENSORS, "ok": 2}
    assert by_name["a"].expected == by_name["a"].actual
    assert main(["verify", str(ckpt), str(manifest)]) == 0
    out = capsys.readouterr().out
    assert out == "verify: 2 ok, 0 mismatch, 0 reshaped, 0 missing, 0 extra; root ok\n"


def test_flipped_byte_is_a_mismatch(tmp_path, rng, capsys):
    _, b, ckpt, manifest = _base(tmp_path, rng)
    corrupt = _flip_last_byte(ckpt, tmp_path / "corrupt.safetensors")
    report, by_name = _verify(corrupt, manifest)
    assert report.summary == {**NO_TENSORS, "ok": 1, "mismatch": 1}
    assert by_name["a"].status == "ok"
    assert by_name["b"].status == "mismatch"
    assert not report.root_match
    assert not report.passed
    expected = hashlib.sha256(b.tobytes()).hexdigest()
    actual = _flipped_digest(b)
    assert (by_name["b"].expected.sha256, by_name["b"].actual.sha256) == (expected, actual)
    assert main(["verify", str(corrupt), str(manifest)]) == 1
    out = capsys.readouterr().out
    assert out.splitlines()[0] == (
        "verify: 1 ok, 1 mismatch, 0 reshaped, 0 missing, 0 extra; root MISMATCH"
    )
    assert _rows(out) == {"b": ["mismatch", "b", expected[:12], actual[:12]]}


def test_dtype_or_shape_change_is_reshaped(tmp_path, rng, capsys):
    v = rng.standard_normal(4).astype(np.float32)
    base = write_safetensors(
        tmp_path / "base.safetensors", {"cast": ("F32", v), "shape": ("F32", v)}
    )
    manifest = _hash_to(tmp_path / "manifest.json", base)
    # Same stored bytes under a new dtype or shape: reshaped depends on dtype and shape alone.
    changed = write_safetensors(
        tmp_path / "changed.safetensors",
        {"cast": ("I32", v.view(np.int32)), "shape": ("F32", v.reshape(2, 2))},
    )
    report, by_name = _verify(changed, manifest)
    assert report.summary == {**NO_TENSORS, "reshaped": 2}
    assert not report.passed
    shape = by_name["shape"]
    assert (shape.expected.shape, shape.actual.shape) == ((4,), (2, 2))
    assert shape.expected.sha256 == shape.actual.sha256
    assert (by_name["cast"].expected.dtype, by_name["cast"].actual.dtype) == ("F32", "I32")
    assert main(["verify", str(changed), str(manifest)]) == 1
    assert _rows(capsys.readouterr().out) == {
        "cast": ["reshaped", "cast", "F32[4]", "I32[4]"],
        "shape": ["reshaped", "shape", "F32[4]", "F32[2,2]"],
    }


def test_missing_and_extra_tensors(tmp_path, rng, capsys):
    a, b, c = (rng.standard_normal(3).astype(np.float32) for _ in range(3))
    base = write_safetensors(tmp_path / "base.safetensors", {"a": ("F32", a), "b": ("F32", b)})
    manifest = _hash_to(tmp_path / "manifest.json", base)
    other = write_safetensors(tmp_path / "other.safetensors", {"a": ("F32", a), "c": ("F32", c)})
    report, by_name = _verify(other, manifest)
    assert report.summary == {**NO_TENSORS, "ok": 1, "missing": 1, "extra": 1}
    assert by_name["b"].status == "missing"
    assert by_name["b"].actual is None
    assert by_name["c"].status == "extra"
    assert by_name["c"].expected is None
    sha_b = hashlib.sha256(b.tobytes()).hexdigest()
    sha_c = hashlib.sha256(c.tobytes()).hexdigest()
    assert by_name["c"].actual.sha256 == sha_c
    # Extra tensors are hashed too, so the computed root covers the whole checkpoint.
    assert report.computed_root == build_manifest(load_checkpoint(other))["root"]
    assert main(["verify", str(other), str(manifest)]) == 1
    assert _rows(capsys.readouterr().out) == {
        "b": ["missing", "b", sha_b[:12], "-"],
        "c": ["extra", "c", "-", sha_c[:12]],
    }
    assert main(["verify", str(other), str(manifest), "--json"]) == 1
    entries = {t["name"]: t for t in json.loads(capsys.readouterr().out)["tensors"]}
    assert entries["b"]["actual"] is None
    assert entries["b"]["expected"]["sha256"] == sha_b
    assert entries["c"]["expected"] is None
    assert entries["c"]["actual"]["sha256"] == sha_c


def test_tampered_root_fails_although_every_tensor_is_ok(tmp_path, rng, capsys):
    _, _, ckpt, _ = _base(tmp_path, rng)
    tampered = {**build_manifest(load_checkpoint(ckpt)), "root": "0" * 64}
    manifest = _write_manifest(tmp_path / "tampered.json", tampered)
    report, _ = _verify(ckpt, manifest)
    assert report.summary == {**NO_TENSORS, "ok": 2}
    assert report.manifest_root == "0" * 64
    assert not report.root_match
    assert not report.passed
    assert main(["verify", str(ckpt), str(manifest)]) == 1
    out = capsys.readouterr().out
    assert out == "verify: 2 ok, 0 mismatch, 0 reshaped, 0 missing, 0 extra; root MISMATCH\n"


def test_json_output(tmp_path, rng, capsys):
    _, b, ckpt, manifest = _base(tmp_path, rng)
    corrupt = _flip_last_byte(ckpt, tmp_path / "corrupt.safetensors")
    assert main(["verify", str(corrupt), str(manifest), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert list(payload) == [
        "checkpoint", "manifest_root", "computed_root", "root_match", "summary", "tensors",
    ]  # fmt: skip
    assert payload["checkpoint"] == str(corrupt)
    assert payload["manifest_root"] == json.loads(manifest.read_text())["root"]
    assert payload["computed_root"] == build_manifest(load_checkpoint(corrupt))["root"]
    assert payload["root_match"] is False
    assert payload["summary"] == {**NO_TENSORS, "ok": 1, "mismatch": 1}
    [entry] = payload["tensors"]
    assert entry == {
        "name": "b",
        "status": "mismatch",
        "expected": {
            "dtype": "F32",
            "shape": [4],
            "sha256": hashlib.sha256(b.tobytes()).hexdigest(),
        },
        "actual": {"dtype": "F32", "shape": [4], "sha256": _flipped_digest(b)},
    }
    assert list(entry) == ["name", "status", "expected", "actual"]


def test_all_lists_ok_tensors(tmp_path, rng, capsys):
    a, _, ckpt, manifest = _base(tmp_path, rng)
    assert main(["verify", str(ckpt), str(manifest), "--all"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == ""
    assert lines[2].split() == ["status", "tensor", "expected", "actual"]
    sha_a = hashlib.sha256(a.tobytes()).hexdigest()[:12]
    assert lines[3].split() == ["ok", "a", sha_a, sha_a]
    assert [line.split()[:2] for line in lines[3:]] == [["ok", "a"], ["ok", "b"]]
    assert all(line == line.rstrip() for line in lines)
    assert main(["verify", str(ckpt), str(manifest), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tensors"] == []
    assert main(["verify", str(ckpt), str(manifest), "--json", "--all"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [(t["name"], t["status"]) for t in payload["tensors"]] == [("a", "ok"), ("b", "ok")]


def test_sharded_directory_checkpoint(tmp_path, rng):
    a, b, single, manifest = _base(tmp_path, rng)
    shards = tmp_path / "shards"
    shards.mkdir()
    write_safetensors(shards / "model-00001-of-00002.safetensors", {"a": ("F32", a)})
    write_safetensors(shards / "model-00002-of-00002.safetensors", {"b": ("F32", b)})
    report, _ = _verify(shards, manifest)
    assert report.passed
    assert report.summary == {**NO_TENSORS, "ok": 2}
    assert main(["verify", str(shards), str(manifest)]) == 0
    sharded_manifest = _hash_to(tmp_path / "sharded.json", shards)
    assert main(["verify", str(single), str(sharded_manifest)]) == 0


def _with_entry(manifest, **fields):
    """Copy ``manifest`` with fields of tensor ``w`` replaced."""
    entry = {**manifest["tensors"]["w"], **fields}
    return {**manifest, "tensors": {**manifest["tensors"], "w": entry}}


MALFORMED = [
    pytest.param(lambda m: {**m, "format": "wvc-manifest/1"}, "'format' must be", id="format"),
    pytest.param(
        lambda m: _with_entry(m, sha256=m["tensors"]["w"]["sha256"][:-1]),
        "'sha256' must be 64 lowercase hex",
        id="sha256-length",
    ),
    pytest.param(lambda m: _with_entry(m, shape="2x2"), "'shape' must be a list", id="shape-str"),
    pytest.param(
        lambda m: _with_entry(m, sha256=m["tensors"]["w"]["sha256"].upper()),
        "'sha256' must be 64 lowercase hex",
        id="sha256-case",
    ),
    pytest.param(lambda m: _with_entry(m, shape=[2, -2]), "non-negative", id="shape-negative"),
    pytest.param(lambda m: _with_entry(m, shape=[True, 4]), "non-negative", id="shape-bool"),
    pytest.param(lambda m: _with_entry(m, nbytes="16"), "'nbytes' must be", id="nbytes"),
    pytest.param(lambda m: _with_entry(m, dtype=None), "'dtype' must be a string", id="dtype"),
    pytest.param(lambda m: {**m, "root": 5}, "'root' must be a string", id="root"),
    pytest.param(lambda m: {**m, "tensors": []}, "'tensors' must be a JSON object", id="tensors"),
    pytest.param(lambda m: {**m, "tensors": {"w": 5}}, "must be a JSON object", id="entry"),
    pytest.param(lambda m: [m], "expected a JSON object", id="top-level"),
]


@pytest.mark.parametrize(("corrupt", "message"), MALFORMED)
def test_malformed_manifest_is_rejected(tmp_path, rng, capsys, corrupt, message):
    w = rng.standard_normal((2, 2)).astype(np.float32)
    ckpt = write_safetensors(tmp_path / "m.safetensors", {"w": ("F32", w)})
    bad = corrupt(build_manifest(load_checkpoint(ckpt)))
    path = _write_manifest(tmp_path / "manifest.json", bad)
    with pytest.raises(ManifestError, match=message):
        load_manifest(path)
    assert main(["verify", str(ckpt), str(path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"wvc: {path}: ")
    assert message in err


def test_manifest_that_is_not_json_is_rejected(tmp_path, rng, capsys):
    _, _, ckpt, _ = _base(tmp_path, rng)
    path = tmp_path / "bad.json"
    for raw in (b"{not json", b"\xff\xfe\xfd"):
        path.write_bytes(raw)
        with pytest.raises(ManifestError, match="not valid JSON"):
            load_manifest(path)
    assert main(["verify", str(ckpt), str(path)]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_manifest_with_bom_or_utf16_is_accepted(tmp_path, rng):
    _, _, ckpt, manifest = _base(tmp_path, rng)
    text = manifest.read_text(encoding="utf-8")
    path = tmp_path / "encoded.json"
    for encoding in ("utf-8-sig", "utf-16"):
        path.write_bytes(text.encode(encoding))
        report, _ = _verify(ckpt, path)
        assert report.passed


def test_unreadable_inputs_return_2(tmp_path, rng, capsys):
    _, _, ckpt, manifest = _base(tmp_path, rng)
    assert main(["verify", str(ckpt), str(tmp_path / "missing.json")]) == 2
    assert main(["verify", str(tmp_path / "missing.safetensors"), str(manifest)]) == 2
    assert capsys.readouterr().err.count("wvc: ") == 2


def test_verify_checkpoint_validates_its_manifest(tmp_path, rng):
    _, _, ckpt, _ = _base(tmp_path, rng)
    with pytest.raises(ManifestError, match="'format' must be"):
        verify_checkpoint(load_checkpoint(ckpt), {"format": "nope"})


def test_check_needs_a_record_on_at_least_one_side():
    with pytest.raises(ValueError, match="neither"):
        _check("x", None, None)


def test_manifest_nbytes_must_match_dtype_and_shape():
    from wvc.verify import ManifestError, validate_manifest

    good = {"dtype": "F32", "shape": [2, 3], "nbytes": 24, "sha256": "00" * 32}
    validate_manifest({"format": "wvc-manifest/0", "root": "x", "tensors": {"a": good}})
    for bad in ({**good, "nbytes": 23}, {**good, "dtype": "Q4"}):
        with pytest.raises(ManifestError):
            validate_manifest({"format": "wvc-manifest/0", "root": "x", "tensors": {"a": bad}})


def _manifest_with(**entry):
    """A one-tensor manifest whose entry has ``entry``'s fields (the root is not checked)."""
    tensor = {"dtype": "U8", "shape": [1], "nbytes": 1, "sha256": "00" * 32, **entry}
    return {"format": "wvc-manifest/0", "root": "x", "tensors": {"w": tensor}}


def test_manifest_shape_of_100000_dims_is_refused_quickly():
    from wvc.verify import validate_manifest

    start = time.perf_counter()
    with pytest.raises(ManifestError, match="'shape' has 100000 dimensions, more than 64"):
        validate_manifest(_manifest_with(shape=[2] * 100_000, nbytes=4))
    assert time.perf_counter() - start < 2


def test_manifest_shape_bounds():
    from wvc.verify import validate_manifest

    validate_manifest(_manifest_with(shape=[1] * 64))
    with pytest.raises(ManifestError, match="65 dimensions, more than 64"):
        validate_manifest(_manifest_with(shape=[1] * 65))
    validate_manifest(_manifest_with(shape=[3, 0, 10**100], nbytes=0))  # a zero dim: no elements
    big = str(10**100)
    with pytest.raises(ManifestError, match=rf"shape \[{big}, {big}, {big}, {big}, \.\.\.\] "):
        validate_manifest(_manifest_with(shape=[10**100] * 64, nbytes=4))


def test_deeply_nested_manifest_is_refused_with_exit_2(tmp_path, rng, capsys):
    _, _, ckpt, _ = _base(tmp_path, rng)
    path = tmp_path / "deep.json"
    path.write_text("[" * 100_000 + "]" * 100_000)
    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest(path)
    assert main(["verify", str(ckpt), str(path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"wvc: {path}: manifest is not valid JSON")
    assert "Traceback" not in err


def test_oversized_manifest_is_refused_unread(tmp_path, rng, monkeypatch):
    _, _, _, manifest = _base(tmp_path, rng)
    monkeypatch.setattr("wvc.store.RECORD_LIMIT", 16)
    with pytest.raises(StoreError, match="more than the 16 bytes allowed"):
        load_manifest(manifest)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs and /dev/zero are POSIX only")
def test_manifest_that_is_a_fifo_or_a_device_is_refused(tmp_path, rng, capsys):
    _, _, ckpt, _ = _base(tmp_path, rng)
    fifo, device = tmp_path / "fifo.json", tmp_path / "zero.json"
    os.mkfifo(fifo)
    device.symlink_to("/dev/zero")
    for path in (fifo, device):
        with pytest.raises(StoreError, match="not a regular file"):
            load_manifest(path)
        assert main(["verify", str(ckpt), str(path)]) == 2
    assert capsys.readouterr().err.count("is not a regular file") == 2


def test_shown_keeps_a_short_value_whole():
    assert shown("F32") == "'F32'"
    assert shown("a" * (SHOWN_LIMIT - 2)) == repr("a" * (SHOWN_LIMIT - 2))  # repr fits the limit


def test_shown_cuts_a_long_value_and_states_its_length():
    assert shown("a" * 5000) == "'" + "a" * (SHOWN_LIMIT - 1) + "…' (5000 characters)"
    assert shown("b" * 5000, limit=10) == "'bbbbbbbbb…' (5000 characters)"


def test_shown_escapes_control_characters():
    text = shown("x\x1b[31m\x07\nred\x85")
    assert text.isprintable()
    assert "\\x1b" in text
    long = shown("\x1b" * 5000)
    assert long.isprintable()
    assert long.endswith("…' (5000 characters)")

    class Raw:
        def __repr__(self) -> str:
            return "raw\x1b[2J"

    assert shown(Raw()) == "raw\\x1b[2J"


def test_shown_quotes_a_non_string_by_its_repr():
    assert (shown(12), shown(None), shown([1, 2])) == ("12", "None", "[1, 2]")
    numbers = list(range(2000))
    text = shown(numbers)
    assert text == repr(numbers)[:SHOWN_LIMIT] + f"… ({len(repr(numbers))} characters)"


def test_a_long_manifest_value_is_not_echoed_in_the_error(tmp_path, rng):
    """H2: a received dtype, name, sha256, or format is cut in the message."""
    good = json.loads(_base(tmp_path, rng)[3].read_text())
    name, entry = next(iter(good["tensors"].items()))
    source = str(tmp_path / "long.json")
    cases = {
        "D": {**good, "tensors": {name: {**entry, "dtype": "D" * 5000}}},
        "N": {**good, "tensors": {"N" * 5000: {**entry, "sha256": "S" * 5000}}},
        "F": {**good, "format": "F" * 5000},
    }
    for letter, manifest in cases.items():
        with pytest.raises(ManifestError) as raised:
            load_manifest(_write_manifest(tmp_path / "long.json", manifest))
        message = str(raised.value).removeprefix(source)
        assert len(message) < 200
        assert "…" in message
        assert "(5000 characters)" in message
        assert letter * (SHOWN_LIMIT + 1) not in message
        assert "S" * (SHOWN_LIMIT + 1) not in message
