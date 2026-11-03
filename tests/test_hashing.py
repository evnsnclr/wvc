import hashlib

import numpy as np
import pytest
from helpers import write_safetensors

from wvc.hashing import MANIFEST_FORMAT, build_manifest, manifest_root, tensor_digests
from wvc.safetensors_io import load_checkpoint


def test_tensor_digest_is_sha256_of_raw_bytes(tmp_path, rng):
    a = rng.standard_normal((8, 8)).astype(np.float32)
    ckpt = load_checkpoint(write_safetensors(tmp_path / "m.safetensors", {"a": ("F32", a)}))
    assert tensor_digests(ckpt)["a"] == hashlib.sha256(a.tobytes()).hexdigest()


def test_manifest_root_ignores_header_order_and_sharding(tmp_path, rng):
    a = rng.standard_normal(4).astype(np.float32)
    b = rng.standard_normal(3).astype(np.float32)
    one = write_safetensors(tmp_path / "one.safetensors", {"b": ("F32", b), "a": ("F32", a)})
    shards = tmp_path / "shards"
    shards.mkdir()
    write_safetensors(shards / "s1.safetensors", {"a": ("F32", a)})
    write_safetensors(shards / "s2.safetensors", {"b": ("F32", b)})
    m1 = build_manifest(load_checkpoint(one))
    m2 = build_manifest(load_checkpoint(shards))
    assert m1["format"] == MANIFEST_FORMAT
    assert m1["root"] == m2["root"]
    assert m1["tensor_count"] == 2
    assert m1["total_bytes"] == 28
    assert list(m1["tensors"]) == ["a", "b"]


def test_manifest_root_changes_with_one_byte(tmp_path, rng):
    a = rng.standard_normal(16).astype(np.float32)
    m1 = build_manifest(
        load_checkpoint(write_safetensors(tmp_path / "x.safetensors", {"a": ("F32", a)}))
    )
    a2 = a.copy()
    a2.view(np.uint32)[0] ^= 1
    m2 = build_manifest(
        load_checkpoint(write_safetensors(tmp_path / "y.safetensors", {"a": ("F32", a2)}))
    )
    assert m1["root"] != m2["root"]
    assert m1["tensors"]["a"]["sha256"] != m2["tensors"]["a"]["sha256"]


def test_manifest_root_depends_on_name_and_shape():
    entry = {"dtype": "F32", "shape": [2, 2], "sha256": "00" * 32}
    assert manifest_root({"a": entry}) != manifest_root({"b": entry})
    assert manifest_root({"a": entry}) != manifest_root({"a": {**entry, "shape": [4]}})


def test_manifest_root_rejects_malformed_entry():
    with pytest.raises(ValueError, match="malformed manifest entry"):
        manifest_root({"a": {"dtype": "F32", "shape": "2x2", "sha256": "00" * 32}})
