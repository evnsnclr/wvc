"""Check wvc against the committed spec v0 test vectors in spec/vectors/v0."""

import json
import runpy
from pathlib import Path

import pytest

from wvc.cli import main
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
