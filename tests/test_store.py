import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path

import numpy as np
import pytest
from helpers import store_checkpoint_files, write_safetensors

from wvc import store as store_module
from wvc.canonical import object_id
from wvc.cli import main
from wvc.codec import choose_encoding
from wvc.hashing import build_manifest
from wvc.safetensors_io import load_checkpoint
from wvc.store import DeltaDepthError, IntegrityError, ObjectStore, StoreError, find_repo

LAYOUT = ("refs/heads", "commits", "manifests", "objects", "payloads")
NUMEL = 4096
TWO_HOURS = 2 * 3600


def f32_bytes(values):
    return np.ascontiguousarray(values, dtype=np.float32).view(np.uint8)


def sparse_edit(values, rng, fraction=0.01):
    """Copy ``values`` with a random ``fraction`` of the elements redrawn."""
    edited = values.copy()
    changed = rng.choice(values.size, size=int(values.size * fraction), replace=False)
    edited[changed] = rng.standard_normal(changed.size).astype(np.float32)
    return edited


def dense_noise(values, rng):
    return values * (1 + 1e-3 * rng.standard_normal(values.size)).astype(np.float32)


def object_path(store, tensor_id):
    return store.dir / "objects" / tensor_id[:2] / f"{tensor_id}.json"


def payload_path(store, pid):
    return store.dir / "payloads" / pid[:2] / pid


def plant_object(store, record):
    """Write an object record straight into the store, as a received .wvc may hold it."""
    path = object_path(store, record["id"])
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def set_max_delta_depth(store, depth):
    store.write_config({"format": "wvc-repo/0", "max_delta_depth": depth})


def stored_checkpoint(store, tmp_path, rng, name="m"):
    """Store a small synthetic checkpoint's objects, header, and manifest; return root, files."""
    tensors = {"w": ("F32", rng.standard_normal((4, 4)).astype(np.float32))}
    path = write_safetensors(tmp_path / f"{name}.safetensors", tensors)
    return store_checkpoint_files(store, path)


def commit_record(manifest, files, parents=(), message="first", kind="finetune"):
    return {
        "format": "wvc-commit/0",
        "manifest": manifest,
        "parents": list(parents),
        "operation": {"kind": kind, "note": "extra keys are kept"},
        "message": message,
        "author": "Évan",
        "time": "2026-11-02T12:00:00Z",
        "files": files,
    }


def age(path, seconds=TWO_HOURS):
    """Set a file's times ``seconds`` into the past."""
    then = time.time() - seconds
    os.utime(path, (then, then))
    return path


def temporary(directory, stale):
    path = directory / f".wvc-{os.urandom(16).hex()}.tmp"
    path.write_bytes(b"left over")
    return age(path) if stale else path


def test_init_creates_the_layout_and_refuses_to_reinit(tmp_path):
    store = ObjectStore(tmp_path / "repo")
    assert not store.exists()
    store.init_layout()
    assert store.exists()
    assert all((store.dir / sub).is_dir() for sub in LAYOUT)
    assert store.config() == {
        "format": "wvc-repo/0",
        "max_delta_depth": 8,
        "max_tensor_bytes": 1 << 34,
    }
    assert (store.dir / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    assert store.current_branch() == "main"
    assert store.head() is None
    assert sorted(path.name for path in store.repo.iterdir()) == [".wvc"]  # no temporaries
    with pytest.raises(StoreError, match="already exists"):
        ObjectStore(tmp_path / "repo").init_layout()


def test_an_interrupted_init_leaves_nothing_and_stale_builds_are_removed(tmp_path, monkeypatch):
    """L3: the layout is built in a temporary sibling that is renamed to .wvc last."""
    repo = tmp_path / "repo"
    repo.mkdir()
    stale = repo / f".wvc-{'a' * 32}.tmp"
    (stale / "refs").mkdir(parents=True)
    age(stale)
    fresh = temporary(repo, stale=False)
    real = store_module.write_atomic

    def crash_on_head(path, data):
        if path.name == "HEAD":
            raise OSError("simulated crash")
        real(path, data)

    monkeypatch.setattr(store_module, "write_atomic", crash_on_head)
    with pytest.raises(OSError, match="simulated crash"):
        ObjectStore(repo).init_layout()
    assert sorted(path.name for path in repo.iterdir()) == [fresh.name]  # stale build removed
    monkeypatch.undo()
    ObjectStore(repo).init_layout()
    assert ObjectStore(repo).current_branch() == "main"


def test_payloads_are_content_addressed_and_verified(store):
    data = b"weights \x00\xff"
    pid = store.put_payload(data)
    assert pid == hashlib.sha256(data).hexdigest()
    assert store.put_payload(np.frombuffer(data, dtype=np.uint8)) == pid
    assert store.get_payload(pid) == data
    payload_path(store, pid).write_bytes(b"weights \x00\xfe")
    with pytest.raises(IntegrityError, match="hash to"):
        store.get_payload(pid)
    with pytest.raises(IntegrityError, match="holds 10 bytes, not the 9 its record implies"):
        store.get_payload(pid, size=9)
    with pytest.raises(IntegrityError, match="more than the 9 allowed"):
        store.get_payload(pid, limit=9)
    with pytest.raises(StoreError, match="not stored"):
        store.get_payload("0" * 64)
    with pytest.raises(StoreError, match="64 lowercase hex"):
        store.get_payload("../config.json")


def test_put_payload_repairs_a_damaged_payload_and_counts_only_writes(store):
    """C3, S-M1, L4: an existing payload is re-hashed and replaced when wrong."""
    data = b"payload bytes " * 10
    pid = store.put_payload(data)
    assert store.payload_bytes_written == len(data)
    assert store.put_payload(data) == pid
    assert store.payload_bytes_written == len(data)
    path = payload_path(store, pid)
    for damage in (b"", data[:-1], data[:-1] + b"!"):  # emptied by a crash, truncated, rotten
        path.write_bytes(damage)
        assert store.put_payload(data) == pid
        assert path.read_bytes() == data
        assert store.verify_payload(pid) == len(data)
    assert store.payload_bytes_written == 4 * len(data)
    assert not list(path.parent.glob(".wvc-*.tmp"))


def test_writes_fsync_the_file_before_the_rename_and_the_directory_after(store, monkeypatch):
    """C3, S-L10: _write_atomic is durable."""
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append(("fsync", "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"))
        real_fsync(fd)

    def replace(source, target):
        events.append(("replace", Path(target).name))
        real_replace(source, target)

    monkeypatch.setattr("wvc.fsio.os.fsync", fsync)
    monkeypatch.setattr("wvc.fsio.os.replace", replace)
    pid = store.put_payload(b"durable")
    directory_sync = [] if os.name == "nt" else [("fsync", "directory")]  # none on Windows
    assert events == [("fsync", "file"), ("replace", pid), *directory_sync]


def test_stale_temporaries_are_removed_where_the_store_writes(store, tmp_path, rng):
    """S-L10, L2: .wvc-*.tmp files older than an hour go; fresh ones may belong to a writer."""
    data = b"a payload"
    shard = payload_path(store, hashlib.sha256(data).hexdigest()).parent
    shard.mkdir()
    commits = store.dir / "commits"
    old = [temporary(shard, stale=True), temporary(commits, stale=True)]
    new = [temporary(shard, stale=False), temporary(commits, stale=False)]
    store.put_payload(data)
    root, files = stored_checkpoint(store, tmp_path, rng)
    store.put_commit(commit_record(root, files))
    assert not any(path.exists() for path in old)
    assert all(path.exists() for path in new)


def test_put_tensor_deduplicates(store, rng):
    raw = f32_bytes(rng.standard_normal(NUMEL))
    record = store.put_tensor(raw, 4)
    assert record["id"] == hashlib.sha256(raw.tobytes()).hexdigest()
    assert (record["form"], record["depth"], record["nbytes"]) == ("zstd-planes", 0, raw.size)
    assert store.put_tensor(raw.copy(), 4, base_id=record["id"]) == record
    assert store.get_object(record["id"]) == record
    assert len(list((store.dir / "objects").rglob("*.json"))) == 1
    assert np.array_equal(store.read_tensor(record["id"]), raw)


def test_put_tensor_takes_a_vouched_id_without_hashing_again(store, rng, monkeypatch):
    """M1: the caller's id is used; the store does not hash the tensor bytes."""
    raw = f32_bytes(rng.standard_normal(NUMEL))
    tensor_id = hashlib.sha256(raw.tobytes()).hexdigest()
    hashed = []
    real = store_module.digest_bytes

    def spy(view):
        hashed.append(bytes(view) == raw.tobytes())
        return real(view)

    monkeypatch.setattr(store_module, "digest_bytes", spy)
    record = store.put_tensor(raw, 4, tensor_id=tensor_id)
    assert record["id"] == tensor_id
    assert hashed == [False]  # only the payload was hashed
    with pytest.raises(StoreError, match="64 lowercase hex"):
        store.put_tensor(raw, 4, tensor_id="not an id")


def test_chain_of_three_deltas_reads_back_exactly(store, rng):
    v0 = rng.standard_normal(NUMEL).astype(np.float32)
    v1 = sparse_edit(v0, rng)
    v2 = dense_noise(v1, rng)
    v3 = sparse_edit(v2, rng)
    records = [store.put_tensor(f32_bytes(v0), 4)]
    for values in (v1, v2, v3):
        records.append(store.put_tensor(f32_bytes(values), 4, base_id=records[-1]["id"]))
    forms = [record["form"] for record in records]
    assert forms == ["zstd-planes", "delta-sparse", "delta-xor-planes", "delta-sparse"]
    assert [record["depth"] for record in records] == [0, 1, 2, 3]
    assert [record.get("base") for record in records[1:]] == [r["id"] for r in records[:-1]]
    for record, values in zip(records, (v0, v1, v2, v3), strict=True):
        data = store.read_tensor(record["id"])
        assert data.dtype == np.uint8
        assert data.flags.c_contiguous
        assert np.array_equal(data, f32_bytes(values))
    assert not list(store.dir.rglob("*.tmp"))


def test_depth_limit_forces_a_full_object(store, rng):
    set_max_delta_depth(store, 1)
    values = [rng.standard_normal(NUMEL).astype(np.float32)]
    for _ in range(3):
        values.append(sparse_edit(values[-1], rng))
    records = [store.put_tensor(f32_bytes(values[0]), 4)]
    for v in values[1:]:
        records.append(store.put_tensor(f32_bytes(v), 4, base_id=records[-1]["id"]))
    summary = [(record["form"], record["depth"], "base" in record) for record in records]
    assert summary == [
        ("zstd-planes", 0, False),
        ("delta-sparse", 1, True),
        ("zstd-planes", 0, False),  # a delta here would have depth 2
        ("delta-sparse", 1, True),
    ]
    assert np.array_equal(store.read_tensor(records[-1]["id"]), f32_bytes(values[-1]))


def test_chains_must_count_down_by_one_and_stay_within_the_depth_limit(store, rng):
    """S-M4: depths are checked, ids may not come back, and the depth is capped."""
    values = [rng.standard_normal(NUMEL).astype(np.float32)]
    for _ in range(3):
        values.append(sparse_edit(values[-1], rng))
    records = [store.put_tensor(f32_bytes(values[0]), 4)]
    for v in values[1:]:
        records.append(store.put_tensor(f32_bytes(v), 4, base_id=records[-1]["id"]))
    set_max_delta_depth(store, 2)  # local policy now forbids the depth-3 delta
    with pytest.raises(IntegrityError, match="delta chain is 3 deep, over 2"):
        store.read_tensor(records[3]["id"])
    assert np.array_equal(store.read_tensor(records[2]["id"]), f32_bytes(values[2]))
    with pytest.raises(DeltaDepthError, match="depth 3 is over the limit 2"):
        store.put_object({**records[3], "id": hashlib.sha256(b"other").hexdigest()})
    skipping = {**records[3], "id": hashlib.sha256(b"skips").hexdigest(), "depth": 2}
    plant_object(store, {**skipping, "base": records[0]["id"]})  # depth 2 on a depth-0 base
    with pytest.raises(IntegrityError, match="depth 2 is not its base's depth 0 plus one"):
        store.read_tensor(skipping["id"])
    set_max_delta_depth(store, 1000)  # a received config cannot lift the cap of 64
    deep = {**records[3], "id": hashlib.sha256(b"deep").hexdigest(), "depth": 65}
    plant_object(store, deep)
    with pytest.raises(IntegrityError, match="65 deep, over 64"):
        store.read_tensor(deep["id"])


def test_unusable_or_missing_bases(store, rng):
    base = store.put_tensor(f32_bytes(rng.standard_normal(NUMEL)), 4)
    shorter = f32_bytes(rng.standard_normal(NUMEL // 2))
    assert store.put_tensor(shorter, 4, base_id=base["id"])["form"] == "zstd-planes"
    with pytest.raises(StoreError, match="not stored"):
        store.put_tensor(f32_bytes(rng.standard_normal(8)), 4, base_id="0" * 64)
    with pytest.raises(StoreError, match="1-D uint8"):
        store.put_tensor(rng.standard_normal(8).astype(np.float32), 4)


def test_read_verifies_every_level_of_a_chain(store, rng):
    v0 = rng.standard_normal(NUMEL).astype(np.float32)
    real = store.put_tensor(f32_bytes(v0), 4)
    fake_id = hashlib.sha256(b"not these bytes").hexdigest()
    store.put_object({**real, "id": fake_id})  # claims to hold other bytes than its payload
    delta = store.put_tensor(f32_bytes(sparse_edit(v0, rng)), 4, base_id=real["id"])
    on_fake = hashlib.sha256(b"a delta on the fake").hexdigest()
    store.put_object({**delta, "id": on_fake, "base": fake_id})
    with pytest.raises(IntegrityError, match=f"object {fake_id}: decoded bytes"):
        store.read_tensor(on_fake)
    path = payload_path(store, real["payload"])
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))  # same length, other bytes
    with pytest.raises(IntegrityError, match="hash to"):
        store.read_tensor(delta["id"])
    path.write_bytes(b"corrupt")  # another length: refused before it is read
    with pytest.raises(IntegrityError, match="holds 7 bytes, not the"):
        store.read_tensor(delta["id"])


def test_reuse_needs_the_payload_at_the_size_its_record_implies(store, rng):
    """C3: a payload emptied by a crash is no longer silently reused (the caller stores the
    tensor again since the second-pass M3 fix), nor is a record of another size."""
    raw = f32_bytes(rng.standard_normal(NUMEL))
    record = store.put_tensor(raw, 4)
    assert store.check_reusable(record["id"], raw.size) == record
    assert store.check_reusable(record["id"], raw.size + 4) is None
    path = payload_path(store, record["payload"])
    for damage in (b"", b"short"):
        path.write_bytes(damage)
        assert store.check_reusable(record["id"], raw.size) is None
    path.unlink()
    assert store.check_reusable(record["id"], raw.size) is None


def test_reuse_decodes_unmarked_objects_and_replacement_clears_every_marker(
    store, rng, monkeypatch
):
    """Second-pass M3: reuse decodes an object unless a marker says this store already did;
    replacing a record or a damaged payload removes every marker."""
    raw = f32_bytes(rng.standard_normal(NUMEL))
    edited = f32_bytes(sparse_edit(raw.view(np.float32), rng))
    record = store.put_tensor(raw, 4)
    delta = store.put_tensor(edited, 4, base_id=record["id"])
    assert delta["form"] != "zstd-planes"
    assert store.is_verified(record["id"])
    assert store.is_verified(delta["id"])  # encoded here
    decoded = []
    real_read = ObjectStore.read_tensor
    monkeypatch.setattr(
        ObjectStore, "read_tensor", lambda self, tid: decoded.append(tid) or real_read(self, tid)
    )
    assert store.check_reusable(delta["id"], edited.size) == delta
    assert decoded == []
    shutil.rmtree(store.dir / "verified")
    assert store.check_reusable(delta["id"], edited.size) == delta
    assert decoded == [delta["id"]]
    assert store.is_verified(delta["id"])
    replaced = store.replace_tensor(edited, 4, tensor_id=delta["id"])
    assert (replaced["form"], replaced["depth"], store.get_object(delta["id"])) == (
        "zstd-planes",
        0,
        replaced,
    )
    assert store.is_verified(delta["id"])
    assert not store.is_verified(record["id"])
    assert np.array_equal(real_read(store, delta["id"]), edited)
    store.mark_verified(record["id"])
    path = payload_path(store, record["payload"])
    path.write_bytes(b"\0" * path.stat().st_size)  # damaged in place, then repaired by a put
    assert store.put_payload(choose_encoding(raw, 4).payload) == record["payload"]
    assert not store.is_verified(record["id"])
    assert not store.is_verified(delta["id"])


def test_corrupt_records_and_delta_cycles_are_reported(store, rng):
    record = store.put_tensor(f32_bytes(rng.standard_normal(16)), 4)
    a, b = (hashlib.sha256(name).hexdigest() for name in (b"a", b"b"))
    for tensor_id, base_id, depth in ((a, b, 2), (b, a, 1)):  # each claims the other as its base
        plant_object(
            store,
            {
                **record,
                "id": tensor_id,
                "form": "delta-xor-planes",
                "base": base_id,
                "depth": depth,
            },
        )
    with pytest.raises(IntegrityError, match="revisits"):
        store.read_tensor(a)
    plant_object(store, {**record, "id": b, "form": "delta-xor-planes", "base": a, "depth": 1})
    with pytest.raises(IntegrityError, match=r"revisits|depth"):  # a cycle cannot count down
        store.read_tensor(b)
    object_path(store, record["id"]).write_text("{not json")
    with pytest.raises(IntegrityError, match="not UTF-8 JSON"):
        store.get_object(record["id"])


def test_object_records_are_validated(store, rng):
    record = store.put_tensor(f32_bytes(rng.standard_normal(16)), 4)
    other = hashlib.sha256(b"other").hexdigest()
    bad = {
        "format": {**record, "id": other, "format": "wvc-object/1"},
        "keys": {**record, "id": other, "extra": 1},
        "form": {**record, "id": other, "form": "zstd"},
        "payload": {**record, "id": other, "payload": "0" * 64},
        "depth": {**record, "id": other, "depth": 1},
    }
    for name, value in bad.items():
        with pytest.raises(StoreError, match=name):
            store.put_object(value)


def test_put_object_refuses_another_record_for_a_stored_tensor(store, rng):
    """S-M1: an object record already stored must equal the new one."""
    record = store.put_tensor(f32_bytes(rng.standard_normal(NUMEL)), 4)
    store.put_object(dict(record))  # the same record again: nothing to do
    other = {**record, "plane_sizes": list(reversed(record["plane_sizes"]))}
    with pytest.raises(IntegrityError, match="another record is stored for it"):
        store.put_object(other)
    assert store.get_object(record["id"]) == record


def test_incomplete_lowrank_record_is_rejected(store, rng):
    base = store.put_tensor(f32_bytes(rng.standard_normal(16)), 4)
    lowrank_id = hashlib.sha256(b"low rank").hexdigest()
    lowrank = {**base, "id": lowrank_id, "form": "delta-lowrank", "base": base["id"], "rank": 1}
    with pytest.raises(StoreError, match="delta-lowrank needs the keys"):
        store.put_object(lowrank)
    plant_object(store, lowrank)
    with pytest.raises(StoreError, match="delta-lowrank needs the keys"):
        store.read_tensor(lowrank_id)


def test_manifest_round_trip_matches_wvc_hash(store, tmp_path, rng, capsys):
    tensors = {"b": ("F32", rng.standard_normal(3).astype(np.float32))}
    tensors["a"] = ("BF16", rng.integers(0, 2**16, size=(2, 2), dtype=np.uint16))
    ckpt = write_safetensors(tmp_path / "m.safetensors", tensors)
    manifest = build_manifest(load_checkpoint(ckpt))
    root = store.put_manifest(manifest)
    assert root == manifest["root"]
    assert store.put_manifest(manifest) == root
    assert store.get_manifest(root) == manifest
    assert main(["hash", str(ckpt)]) == 0
    path = store.dir / "manifests" / f"{root}.json"
    assert path.read_text(encoding="utf-8") == capsys.readouterr().out
    tampered = {**manifest, "tensors": {**manifest["tensors"], "a": {**manifest["tensors"]["a"]}}}
    tampered["tensors"]["a"]["sha256"] = "0" * 64
    with pytest.raises(IntegrityError, match="does not match its tensors"):
        store.put_manifest(tampered)
    path.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="does not match its tensors"):
        store.get_manifest(root)


def test_put_manifest_refuses_other_tensors_under_a_stored_root(store, tmp_path, rng, monkeypatch):
    """H2: a manifest already stored under a root must list the same tensors."""
    root, _ = stored_checkpoint(store, tmp_path, rng, "x")
    other = write_safetensors(tmp_path / "y.safetensors", {"v": ("F32", np.ones(3, np.float32))})
    colliding = build_manifest(load_checkpoint(other))
    monkeypatch.setattr(store_module, "check_manifest", lambda manifest, where: root)
    with pytest.raises(IntegrityError, match="other tensors is stored under this root"):
        store.put_manifest(colliding)


def test_commit_round_trip_verifies_its_id(store, tmp_path, rng):
    record = commit_record(*stored_checkpoint(store, tmp_path, rng))
    commit_id = store.put_commit(record)
    assert commit_id == object_id(record)
    assert store.put_commit(record) == commit_id
    assert store.get_commit(commit_id) == record
    text = (store.dir / "commits" / f"{commit_id}.json").read_text(encoding="utf-8")
    assert text.startswith('{\n  "author": "Évan",')
    child = store.put_commit(commit_record(record["manifest"], record["files"], [commit_id], "2"))
    assert store.get_commit(child)["parents"] == [commit_id]


def test_a_damaged_commit_file_is_replaced_when_the_record_is_stored_again(store, tmp_path, rng):
    record = commit_record(*stored_checkpoint(store, tmp_path, rng))
    commit_id = store.put_commit(record)
    path = store.dir / "commits" / f"{commit_id}.json"
    path.write_text("", encoding="utf-8")  # emptied by a crash
    assert store.put_commit(record) == commit_id
    assert store.get_commit(commit_id) == record


def test_commit_records_are_validated(store, tmp_path, rng):
    record = commit_record(*stored_checkpoint(store, tmp_path, rng))
    bad = {
        "must not contain 'id'": {**record, "id": "0" * 64},
        "format": {**record, "format": "wvc-commit/1"},
        "manifest must be": {**record, "manifest": "HEAD"},
        "parents must be": {**record, "parents": "none"},
        "operation.kind": {**record, "operation": {"kind": "retrain"}},
        "files must be": {**record, "files": {}},
        "forbids floats": {**record, "operation": {"kind": "finetune", "lr": 1e-4}},
        "manifest 0+ is not stored": {**record, "manifest": "0" * 64},
        "parent 0+ is not stored": {**record, "parents": ["0" * 64]},
    }
    for message, value in bad.items():
        with pytest.raises(StoreError, match=message):
            store.put_commit(value)


def with_file(record, **changes):
    """``record`` with its one file record changed."""
    return {**record, "files": [{**record["files"][0], **changes}]}


def python_source_record(store, record):
    """The record's file with Python source as its header payload."""
    source = b"print('attacker code ran on import')\n"
    return with_file(record, header=store.put_payload(source), tensors=[], nbytes=len(source))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s, r, o: {**r, "files": []}, "do not hold the tensors of manifest"),
        (lambda s, r, o: {**r, "files": [1]}, "exactly the keys"),
        (lambda s, r, o: {**r, "files": o["files"]}, "do not hold the tensors of manifest"),
        (
            lambda s, r, o: with_file(r, name="safetensors.py"),
            "'safetensors.py' is not an allowed file name",
        ),
        (lambda s, r, o: python_source_record(s, r), "not a safetensors header"),
        (lambda s, r, o: with_file(r, header="0" * 64), "payload 0+ is not stored"),
        (lambda s, r, o: with_file(r, tensors=["0" * 64]), "object 0+ is not stored"),
        (
            lambda s, r, o: with_file(r, tensors=o["files"][0]["tensors"]),
            "spans 64 bytes but its object",
        ),
        (
            lambda s, r, o: with_file(r, nbytes=r["files"][0]["nbytes"] + 4),
            "not a safetensors header",
        ),
    ],
    ids=[
        "no-files", "not-a-record", "other-commit-files", "module-name", "python-header",
        "missing-header", "missing-object", "object-size", "file-size",
    ],
)  # fmt: skip
def test_commit_files_must_hold_the_manifest_tensors(store, tmp_path, rng, change, message):
    """S-H1, S-H2, H5: put_commit rebuilds the manifest from the header payloads and ids."""
    record = commit_record(*stored_checkpoint(store, tmp_path, rng, "a"))
    other_path = write_safetensors(
        tmp_path / "o.safetensors", {"v": ("F32", rng.standard_normal(32).astype(np.float32))}
    )
    other_root, other_files = store_checkpoint_files(store, other_path)
    other = commit_record(other_root, other_files)
    before = sorted((store.dir / "commits").iterdir())
    with pytest.raises(StoreError, match=message):
        store.put_commit(change(store, record, other))
    assert sorted((store.dir / "commits").iterdir()) == before


def test_tampered_commit_raises_integrity_error(store, tmp_path, rng):
    commit_id = store.put_commit(commit_record(*stored_checkpoint(store, tmp_path, rng)))
    path = store.dir / "commits" / f"{commit_id}.json"
    path.write_text(path.read_text(encoding="utf-8").replace("first", "edited"), encoding="utf-8")
    with pytest.raises(IntegrityError, match="hashes to"):
        store.get_commit(commit_id)


def test_config_is_read_once_per_store_and_kept_current_by_write_config(store, rng, monkeypatch):
    """L1: config.json is not read again for every tensor that has a base."""
    reads = []
    real = store_module.read_json_object
    monkeypatch.setattr(
        store_module, "read_json_object", lambda *args: reads.append(args[1]) or real(*args)
    )
    fresh = ObjectStore(store.repo)
    values = [rng.standard_normal(NUMEL).astype(np.float32)]
    for _ in range(4):
        values.append(sparse_edit(values[-1], rng))
    record = fresh.put_tensor(f32_bytes(values[0]), 4)
    for v in values[1:]:
        record = fresh.put_tensor(f32_bytes(v), 4, base_id=record["id"])
    fresh.read_tensor(record["id"])
    assert reads.count("repository config") == 1
    fresh.write_config({"format": "wvc-repo/0", "max_delta_depth": 3})
    assert fresh.config()["max_delta_depth"] == 3
    assert reads.count("repository config") == 1


def test_ambiguous_prefix_is_an_error(store):
    for suffix in ("1", "2"):
        (store.dir / "commits" / f"{'abcdef0' + suffix * 57}.json").write_text("{}")
    with pytest.raises(StoreError, match="ambiguous: 2 commits match"):
        store.resolve("abcdef0")


def test_find_repo_walks_up_from_a_nested_directory(store, tmp_path):
    nested = store.repo / "a" / "b"
    nested.mkdir(parents=True)
    assert find_repo(nested) == store.repo.resolve()
    assert find_repo(store.repo) == store.repo.resolve()
    with pytest.raises(StoreError, match="not inside a wvc repository"):
        find_repo(tmp_path / "elsewhere")
