"""``wvc hash --format gguf-hash``: llama.cpp's per-tensor manifest lines for safetensors."""

import hashlib
import re

import numpy as np
import pytest
from helpers import write_safetensors

from wvc.cli import main
from wvc.hashing import gguf_hash_lines, hash_files, tensor_digests
from wvc.safetensors_io import load_checkpoint

TENSOR_LINE = re.compile(r"^sha256    ([0-9a-f]{64})  ([^:]+):(.+)$")
FILE_LINE = re.compile(r"^sha256    ([0-9a-f]{64})  ([^:]+)$")


def _two_tensors(tmp_path, rng, name="m.safetensors"):
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    # The header lists b first while the buffer holds a first: lines follow the buffer.
    path = write_safetensors(
        tmp_path / name, {"a": ("F32", a), "b": ("F32", b)}, header_order=["b", "a"]
    )
    return a, b, path


def test_lines_follow_the_gguf_hash_format(tmp_path, rng, capsys):
    a, b, path = _two_tensors(tmp_path, rng)
    assert main(["hash", "--format", "gguf-hash", str(path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    tensors = [TENSOR_LINE.match(line) for line in lines[:2]]
    assert [(m.group(2), m.group(3)) for m in tensors] == [
        ("m.safetensors", "a"),
        ("m.safetensors", "b"),
    ]
    assert tensors[0].group(1) == hashlib.sha256(a.tobytes()).hexdigest()
    assert tensors[1].group(1) == hashlib.sha256(b.tobytes()).hexdigest()
    whole = FILE_LINE.match(lines[2])
    assert whole.group(2) == "m.safetensors"
    assert whole.group(1) == hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest()
    # gguf-hash --check reads three whitespace-separated tokens; the type is "%-8s" then two spaces.
    assert all(len(line.split()) == 3 and line.startswith("sha256    ") for line in lines)


def test_tensor_lines_agree_with_the_json_manifest(tmp_path, rng):
    _, _, path = _two_tensors(tmp_path, rng)
    ckpt = load_checkpoint(path)
    digests, files = hash_files(ckpt)
    assert digests == tensor_digests(ckpt)
    assert list(files) == [path]
    hexes = {m.group(3): m.group(1) for m in map(TENSOR_LINE.match, gguf_hash_lines(ckpt)) if m}
    assert hexes == digests


def test_a_directory_gives_one_block_per_shard_in_name_order(tmp_path, rng, capsys):
    directory = tmp_path / "model"
    directory.mkdir()
    x = rng.standard_normal(3).astype(np.float32)
    y = rng.standard_normal(5).astype(np.float32)
    write_safetensors(directory / "model-00002-of-00002.safetensors", {"y": ("F32", y)})
    write_safetensors(directory / "model-00001-of-00002.safetensors", {"x": ("F32", x)})
    assert main(["hash", "--format", "gguf-hash", str(directory)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[2] for line in lines] == [
        "model-00001-of-00002.safetensors:x",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors:y",
        "model-00002-of-00002.safetensors",
    ]
    assert lines[1].split()[1] == hashlib.sha256(x.tobytes()).hexdigest()  # one tensor: same hash


def test_names_with_whitespace_cannot_be_written(tmp_path, rng, capsys):
    path = write_safetensors(tmp_path / "m.safetensors", {"a b": ("F32", np.zeros(2, np.float32))})
    assert main(["hash", "--format", "gguf-hash", str(path)]) == 2
    assert "gguf-hash" in capsys.readouterr().err
    spaced = write_safetensors(
        tmp_path / "my model.safetensors", {"a": ("F32", np.zeros(2, np.float32))}
    )
    assert main(["hash", "--format", "gguf-hash", str(spaced)]) == 2
    with pytest.raises(ValueError, match="gguf-hash"):
        gguf_hash_lines(load_checkpoint(spaced))


def test_empty_tensors_colons_and_empty_files(tmp_path, rng):
    empty = write_safetensors(
        tmp_path / "e.safetensors",
        {"a:b": ("F32", np.zeros(0, np.float32)), "c": ("F32", np.ones(1, np.float32))},
    )
    lines = gguf_hash_lines(load_checkpoint(empty))
    assert lines[0].split() == ["sha256", hashlib.sha256(b"").hexdigest(), "e.safetensors:a:b"]
    assert lines[1].split()[2] == "e.safetensors:c"
    assert len(lines) == 3
    nothing = write_safetensors(tmp_path / "n.safetensors", {})
    assert gguf_hash_lines(load_checkpoint(nothing)) == [
        f"sha256    {hashlib.sha256(b'').hexdigest()}  n.safetensors"
    ]


def test_root_only_applies_to_json_only(tmp_path, rng, capsys):
    _, _, path = _two_tensors(tmp_path, rng)
    with pytest.raises(SystemExit) as exit_info:  # a usage error, as argparse reports one
        main(["hash", "--format", "gguf-hash", "--root-only", str(path)])
    err = capsys.readouterr().err
    assert exit_info.value.code == 2
    assert err.startswith("usage: wvc hash")
    assert "--root-only" in err
    assert main(["hash", "--format", "json", "--root-only", str(path)]) == 0
    assert len(capsys.readouterr().out.strip()) == 64
