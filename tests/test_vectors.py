"""Check wvc against the committed spec v0 test vectors in spec/vectors/v0."""

import json
import runpy
from pathlib import Path

import pytest

from wvc.cli import main
from wvc.gguf_io import load_gguf
from wvc.gguf_quants import DEQUANTIZERS
from wvc.hashing import build_manifest
from wvc.safetensors_io import load_checkpoint

VECTORS = Path(__file__).resolve().parents[1] / "spec" / "vectors" / "v0"
FLOAT_FIELDS = ("max_abs_delta", "delta_norm", "rel_delta_norm", "top1_energy")
REL_TOL = 1e-6


@pytest.mark.parametrize("stem", ["a", "b"])
def test_manifest_matches_vector(stem):
    manifest = build_manifest(load_checkpoint(VECTORS / f"{stem}.safetensors"))
    committed = (VECTORS / f"{stem}.manifest.json").read_text(encoding="utf-8")
    assert json.dumps(manifest, indent=2) + "\n" == committed


def _basename_paths(payload: dict) -> dict:
    return {
        **payload,
        "a": {**payload["a"], "path": Path(payload["a"]["path"]).name},
        "b": {**payload["b"], "path": Path(payload["b"]["path"]).name},
    }


def _with_tolerance(record: dict) -> dict:
    """Let the floating statistics of one tensor record match within REL_TOL (relative)."""
    return {
        key: pytest.approx(value, rel=REL_TOL, abs=0.0)
        if key in FLOAT_FIELDS and value is not None
        else value
        for key, value in record.items()
    }


def test_diff_matches_vector(capsys):
    args = ["diff", "--rank", "--json", "--all"]
    assert main([*args, str(VECTORS / "a.safetensors"), str(VECTORS / "b.safetensors")]) == 1
    actual = _basename_paths(json.loads(capsys.readouterr().out))
    expected = json.loads((VECTORS / "diff.json").read_text(encoding="utf-8"))
    assert {**actual, "tensors": None} == {**expected, "tensors": None}
    assert [t["name"] for t in actual["tensors"]] == [t["name"] for t in expected["tensors"]]
    for got, want in zip(actual["tensors"], expected["tensors"], strict=True):
        assert got == _with_tolerance(want), got["name"]


def test_generator_reproduces_checkpoints(tmp_path):
    generator = runpy.run_path(str(VECTORS / "make_vectors.py"))
    generator["main"]([str(tmp_path)])
    for name in ("a.safetensors", "b.safetensors"):
        assert (tmp_path / name).read_bytes() == (VECTORS / name).read_bytes(), name


def test_gguf_manifest_matches_vector():
    with load_gguf(VECTORS / "quant.gguf") as ckpt:
        manifest = build_manifest(ckpt)
    committed = (VECTORS / "quant.manifest.json").read_text(encoding="utf-8")
    assert json.dumps(manifest, indent=2) + "\n" == committed


def test_gguf_hash_lines_match_vector(capsys):
    assert main(["hash", "--format", "gguf-hash", str(VECTORS / "quant.gguf")]) == 0
    assert capsys.readouterr().out == (VECTORS / "quant.gguf-hash").read_text(encoding="utf-8")


def test_gguf_dequantised_values_match_vector():
    committed = json.loads((VECTORS / "quant.dequant.json").read_text(encoding="utf-8"))
    generator = runpy.run_path(str(VECTORS / "make_vectors.py"))
    assert generator["dequantised_digests"](VECTORS / "quant.gguf") == committed
    with load_gguf(VECTORS / "quant.gguf") as ckpt:
        types = [info.dtype for info in ckpt.tensors.values()]
    assert sorted(types) == sorted(["F32", "F16", "BF16", *DEQUANTIZERS])  # one per type


def test_gguf_diff_matches_vector(capsys):
    args = ["diff", "--rank", "--json", "--all"]
    pair = [str(VECTORS / "quant-base.safetensors"), str(VECTORS / "quant.gguf")]
    assert main([*args, *pair]) == 1
    actual = _basename_paths(json.loads(capsys.readouterr().out))
    expected = json.loads((VECTORS / "quant-diff.json").read_text(encoding="utf-8"))
    assert {**actual, "tensors": None} == {**expected, "tensors": None}
    assert [t["name"] for t in actual["tensors"]] == [t["name"] for t in expected["tensors"]]
    for got, want in zip(actual["tensors"], expected["tensors"], strict=True):
        assert got == _with_tolerance(want), got["name"]


def test_generator_reproduces_the_gguf_pair(tmp_path):
    generator = runpy.run_path(str(VECTORS / "make_vectors.py"))
    generator["write_gguf_vectors"](tmp_path)
    for name in ("quant.gguf", "quant-base.safetensors"):
        assert (tmp_path / name).read_bytes() == (VECTORS / name).read_bytes(), name
