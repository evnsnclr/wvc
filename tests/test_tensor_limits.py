"""Size ceilings against hostile stores: a record may not claim more bytes than the store's
tensor limit, frames are checked before the output is allocated, a checkout needs the free
space first, and running out of memory is an error, not a traceback (review H1)."""

import hashlib
import json
import struct
from collections import namedtuple

import numpy as np
import pytest
import zstandard
from helpers import commit, plant_commit, write_safetensors

from wvc import checkout as checkout_module
from wvc import codec as codec_module
from wvc.checkout import checkout_commit
from wvc.cli import main
from wvc.codec import CodecError, decode_planes
from wvc.fsck import fsck
from wvc.hashing import MANIFEST_FORMAT, manifest_root
from wvc.records import check_tensor_bytes, validate_object
from wvc.store import TENSOR_BYTES_CEILING, IntegrityError, ObjectStore, StoreError

LIMIT = 1024  # bytes: the lowered max_tensor_bytes of these tests
DiskUsage = namedtuple("DiskUsage", "total used free")


def set_max_tensor_bytes(store, limit):
    store.write_config({**store.config(), "max_tensor_bytes": limit})


def reopened(store):
    """A new instance of the same store, which reads config.json afresh."""
    return ObjectStore(store.repo)


def plant_object(store, record):
    """Write an object record straight into the store, as a received .wvc may hold it."""
    path = store.dir / "objects" / record["id"][:2] / f"{record['id']}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def tensor_bytes(numel, seed=0):
    values = np.random.default_rng(seed).standard_normal(numel).astype(np.float32)
    return values.view(np.uint8)


def frame(nbytes):
    """One zstd frame of ``nbytes`` zero bytes; its header declares that size."""
    return zstandard.ZstdCompressor(level=3).compress(bytes(nbytes))


def small_commit(store, tmp_path):
    """Commit a one-file checkpoint of two 64-byte F32 tensors; return the commit id."""
    values = np.arange(32, dtype=np.float32)
    tensors = {"a": ("F32", values[:16]), "b": ("F32", values[16:])}
    path = write_safetensors(tmp_path / "model.safetensors", tensors)
    return commit(store, path).commit_id


def lying_commit(store, nbytes):
    """A commit whose one file holds one U8 tensor said to be ``nbytes`` long, stored as a
    10-byte frame: the scenario of review H1 (S04). The records are planted, not put."""
    tensor_id = hashlib.sha256(b"x").hexdigest()
    payload = frame(10)
    plant_object(
        store,
        {
            "format": "wvc-object/0",
            "id": tensor_id,
            "nbytes": nbytes,
            "itemsize": 1,
            "form": "zstd-planes",
            "payload": store.put_payload(payload),
            "depth": 0,
            "plane_sizes": [len(payload)],
        },
    )
    header = json.dumps({"t0": {"dtype": "U8", "shape": [nbytes], "data_offsets": [0, nbytes]}})
    header += " " * (-len(header) % 8)
    prefix = struct.pack("<Q", len(header)) + header.encode()
    tensors = {"t0": {"dtype": "U8", "shape": [nbytes], "nbytes": nbytes, "sha256": tensor_id}}
    manifest = {
        "format": MANIFEST_FORMAT,
        "root": manifest_root(tensors),
        "tensor_count": 1,
        "total_bytes": nbytes,
        "tensors": tensors,
    }
    file = {
        "name": "lie.safetensors",
        "header": store.put_payload(prefix),
        "tensors": [tensor_id],
        "sha256": hashlib.sha256(b"y").hexdigest(),
        "nbytes": len(prefix) + nbytes,
    }
    commit_id = plant_commit(store, commit_record(store.put_manifest(manifest), [file]))
    store.write_ref("main", commit_id)
    return tensor_id


def commit_record(manifest, files):
    return {
        "format": "wvc-commit/0",
        "manifest": manifest,
        "parents": [],
        "operation": {"kind": "unknown"},
        "message": "x",
        "author": "x",
        "time": "2026-11-03T00:00:00Z",
        "files": files,
    }


def test_init_writes_the_default_limit_and_old_configs_still_read(store):
    assert store.config()["max_tensor_bytes"] == 1 << 34
    store.write_config({"format": "wvc-repo/0", "max_delta_depth": 8})  # made before the limit
    assert reopened(store).put_tensor(tensor_bytes(16), 4)["nbytes"] == 64
    for bad in (0, -1, "16", 1.5, True):
        with pytest.raises(StoreError, match="max_tensor_bytes must be a positive integer"):
            store.write_config(
                {"format": "wvc-repo/0", "max_delta_depth": 8, "max_tensor_bytes": bad}
            )


def test_put_refuses_a_tensor_above_the_limit_and_stores_nothing(store):
    record = store.put_tensor(tensor_bytes(512), 4)  # 2048 bytes, under the default limit
    set_max_tensor_bytes(store, LIMIT)
    before = sorted(path for path in store.dir.rglob("*") if path.is_file())
    with pytest.raises(StoreError, match=f"tensor limit of {LIMIT} bytes") as raised:
        store.put_tensor(tensor_bytes(512, seed=1), 4)
    assert "2048" not in str(raised.value)
    with pytest.raises(StoreError, match=f"object {record['id']}: .*tensor limit of {LIMIT}"):
        reopened(store).put_object(record)  # as a pull would store a received record
    assert sorted(path for path in store.dir.rglob("*") if path.is_file()) == before
    assert store.put_tensor(tensor_bytes(256), 4)["nbytes"] == LIMIT  # at the limit is allowed


def test_a_stored_record_above_the_limit_is_refused_on_read_and_reported_by_fsck(store):
    record = store.put_tensor(tensor_bytes(512), 4)
    set_max_tensor_bytes(store, LIMIT)
    store = reopened(store)
    for read in (store.get_object, store.read_tensor):
        with pytest.raises(StoreError, match=f"object {record['id']}: .*tensor limit of {LIMIT}"):
            read(record["id"])
    report = fsck(store)
    assert report.tensors_decoded == 0
    assert any(
        problem.startswith(f"object {record['id']}:") and f"tensor limit of {LIMIT}" in problem
        for problem in report.problems
    )


def test_the_hard_ceiling_applies_whatever_the_config_says(store):
    set_max_tensor_bytes(store, 1 << 50)
    store = reopened(store)
    payload = frame(10)
    record = {
        "format": "wvc-object/0",
        "id": hashlib.sha256(b"x").hexdigest(),
        "nbytes": TENSOR_BYTES_CEILING + 1,
        "itemsize": 1,
        "form": "zstd-planes",
        "payload": store.put_payload(payload),
        "depth": 0,
        "plane_sizes": [len(payload)],
    }
    with pytest.raises(StoreError, match=f"tensor limit of {TENSOR_BYTES_CEILING} bytes"):
        store.put_object(record)
    plant_object(store, record)
    with pytest.raises(StoreError, match=f"tensor limit of {TENSOR_BYTES_CEILING} bytes"):
        store.read_tensor(record["id"])
    assert any(f"tensor limit of {TENSOR_BYTES_CEILING}" in p for p in fsck(store).problems)


def test_lowrank_factors_count_against_the_limit():
    rank = 64  # factors of 256 bytes each for a 4-byte tensor
    record = {
        "format": "wvc-object/0",
        "id": "a" * 64,
        "nbytes": 4,
        "itemsize": 4,
        "form": "delta-lowrank",
        "payload": "b" * 64,
        "depth": 1,
        "base": "c" * 64,
        "base_dtype": "F32",
        "shape": [1, 1],
        "rank": rank,
        "a_dtype": "F32",
        "a_shape": [rank, 1],
        "b_dtype": "F32",
        "b_shape": [1, rank],
        "a_size": 4 * rank,
        "scale": "1.0",
        "procedure": "lowrank-merge/0",
    }
    validate_object(record)
    check_tensor_bytes(record, 4 * rank, "object")
    with pytest.raises(StoreError, match="tensor limit of 255 bytes"):
        check_tensor_bytes(record, 4 * rank - 1, "object")


def test_a_delta_must_have_its_base_size(store):
    base = store.put_tensor(tensor_bytes(256), 4)
    other = tensor_bytes(512, seed=1)
    delta = {**store.put_tensor(other, 4), "form": "delta-xor-planes", "depth": 1}
    plant_object(store, {**delta, "base": base["id"]})
    with pytest.raises(IntegrityError, match="nbytes differs from its base's"):
        store.read_tensor(delta["id"])


def test_frames_are_checked_before_the_output_is_allocated(monkeypatch):
    allocations = []
    real_empty = np.empty

    def counting_empty(*args, **kwargs):
        allocations.append(args)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(codec_module.np, "empty", counting_empty)
    small = frame(10)
    with pytest.raises(
        CodecError, match="plane 0: frame declares 10 bytes, expected 70368744177664"
    ):
        decode_planes(small, [len(small)], 1, 1 << 46)
    good, bad = frame(1 << 20), frame(10)  # plane 0 matches; plane 1 does not
    with pytest.raises(CodecError, match="plane 1: frame declares 10 bytes"):
        decode_planes(good + bad, [len(good), len(bad)], 2, 2 << 20)
    assert allocations == []
    assert decode_planes(frame(8) * 2, [len(frame(8))] * 2, 2, 16).tolist() == [0] * 16
    assert len(allocations) == 1


@pytest.mark.parametrize(
    ("nbytes", "refusal"),
    [
        (1 << 52, f"a file record claims more than the ceiling of {TENSOR_BYTES_CEILING} bytes"),
        (1 << 39, f"tensor limit of {1 << 34} bytes"),  # a file under the ceiling
    ],
)
def test_a_huge_record_is_refused_by_checkout_and_fsck(store, tmp_path, capsys, nbytes, refusal):
    tensor_id = lying_commit(store, nbytes)
    out = tmp_path / "out"
    assert main(["checkout", "HEAD", "--repo", str(store.repo), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("wvc: ")
    assert refusal in err
    assert "Traceback" not in err
    assert not out.exists()
    assert main(["fsck", "--repo", str(store.repo)]) == 1  # problems are reported, not raised
    output = capsys.readouterr()
    assert f"object {tensor_id}: " in output.out
    assert "tensor limit of" in output.out
    assert "Traceback" not in output.err


def test_out_of_memory_is_an_error_for_checkout_and_a_problem_for_fsck(
    store, tmp_path, capsys, monkeypatch
):
    small_commit(store, tmp_path)

    def exhausted(self, tensor_id):
        raise MemoryError

    monkeypatch.setattr(ObjectStore, "read_tensor", exhausted)
    out = tmp_path / "out"
    assert main(["checkout", "HEAD", "--repo", str(store.repo), "--out", str(out)]) == 2
    assert capsys.readouterr().err == "wvc: out of memory while running checkout\n"
    assert list(out.iterdir()) == []
    report = fsck(store)
    assert report.problems
    assert all(
        problem.endswith(": out of memory rebuilding 64 bytes") for problem in report.problems
    )
    assert main(["fsck", "--repo", str(store.repo)]) == 1
    assert "out of memory rebuilding 64 bytes" in capsys.readouterr().out


def test_checkout_needs_the_free_space_before_it_writes(store, tmp_path, monkeypatch):
    commit_id = small_commit(store, tmp_path)
    needed = store.get_commit(commit_id)["files"][0]["nbytes"]
    monkeypatch.setattr(checkout_module.shutil, "disk_usage", lambda path: DiskUsage(0, 0, 100))
    out = tmp_path / "out"
    with pytest.raises(StoreError, match=f"need {needed} bytes but only 100 bytes are free"):
        checkout_commit(store, "HEAD", out)
    assert list(out.iterdir()) == []
    monkeypatch.setattr(checkout_module.shutil, "disk_usage", lambda path: DiskUsage(0, 0, needed))
    assert checkout_commit(store, "HEAD", out).nbytes == needed


def test_a_file_record_cannot_claim_more_than_the_ceiling(store, tmp_path):
    record = store.get_commit(small_commit(store, tmp_path))
    files = [{**record["files"][0], "nbytes": TENSOR_BYTES_CEILING + 1}]
    commit_id = plant_commit(store, {**record, "files": files})
    with pytest.raises(StoreError, match="a file record claims more than the ceiling"):
        checkout_commit(store, commit_id, tmp_path / "out")
    assert any("ceiling" in problem for problem in fsck(store).problems)
