"""Commit and checkout integrity from the code and security reviews of 2026-11-03.

Commit: one open per file (C1, M1), no reuse of damaged payloads (C3), branches that do not lose
commits (H3, H4, S-L7), the "nothing changed" rule (H7), and file names (H5, H6, C2). Checkout:
file records checked against the manifest (S-H1, S-H2), all-or-nothing writes (S-M7, S-L10),
and targets checked after the renames (C2).
"""

import hashlib
import os

import numpy as np
import pytest
from helpers import NEW, SAME, SHARDS, commit, plant_commit, replace_or_skip, write_safetensors

from wvc import checkout as checkout_module
from wvc import repo as repo_module
from wvc import store as store_module
from wvc.checkout import _check_targets, checkout_commit
from wvc.fsck import fsck
from wvc.records import is_id
from wvc.repo import (
    commit_manifest,
    log_entries,
    show_commit,
)
from wvc.safetensors_io import load_checkpoint
from wvc.store import IntegrityError, ObjectStore, StoreError

VIEW_AS = {"F32": "<f4", "BF16": "<u2", "I8": "i1"}


def stored_files(store):
    return sorted(path for path in store.dir.rglob("*") if path.is_file())


def tensors_of(path):
    """``name -> (dtype, array)`` of a checkpoint, to rewrite it with ``write_safetensors``."""
    ckpt = load_checkpoint(path)
    return {
        name: (info.dtype, np.asarray(ckpt.raw(name)).view(VIEW_AS[info.dtype]).reshape(info.shape))
        for name, info in ckpt.tensors.items()
    }


def single_file(tmp_path, rng, name="model.safetensors", values=None):
    """A one-file checkpoint of three F32 tensors."""
    values = rng.standard_normal(3 * 256).astype(np.float32) if values is None else values
    tensors = {f"t{i}": ("F32", values[i * 256 : (i + 1) * 256]) for i in range(3)}
    tmp_path.mkdir(parents=True, exist_ok=True)
    return write_safetensors(tmp_path / name, tensors)


def same_size_other_bytes(path):
    """The file's bytes with the last byte of its data flipped: same header, same size."""
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    return bytes(data)


def replace_by_rename(path):
    """Save another version over ``path`` the way trainers do: write a sibling, then rename."""
    sibling = path.with_name(path.name + ".partial")
    sibling.write_bytes(same_size_other_bytes(path))
    replace_or_skip(sibling, path)


def single_file_or_skip(directory, rng, name):
    """``single_file``, or a skip on a Windows that cannot create the name (long paths off)."""
    try:
        return single_file(directory, rng, name=name)
    except OSError as exc:
        if os.name != "nt":
            raise
        pytest.skip(f"this Windows cannot create the file: {exc.strerror}")


NOT_ON_WINDOWS = pytest.mark.skipif(
    os.name == "nt", reason="Windows cannot hold a file of this name"
)


def rewrite_first_tensor_in_place(path):
    """Change bytes the commit already read, in the same inode."""
    file = load_checkpoint(path).files[0]
    with path.open("r+b") as handle:
        handle.seek(file.data_start)
        first = handle.read(1)
        handle.seek(file.data_start)
        handle.write(bytes([first[0] ^ 0xFF]))


# Commit: one open per file (C1, M1)


@pytest.mark.parametrize("change", [replace_by_rename, rewrite_first_tensor_in_place])
def test_a_file_changed_during_the_tensor_pass_is_refused(
    store, tmp_path, rng, monkeypatch, change
):
    """C1: replaced by rename or rewritten in place after a tensor was stored."""
    path = single_file(tmp_path / "ckpt", rng)
    real = ObjectStore.put_tensor
    calls = []

    def put_then_change(self, raw, itemsize, **options):
        record = real(self, raw, itemsize, **options)
        if not calls:
            change(path)
        calls.append(record["id"])
        return record

    monkeypatch.setattr(ObjectStore, "put_tensor", put_then_change)
    with pytest.raises(StoreError, match="changed while it was being committed"):
        commit(store, path)
    assert store.head() is None
    assert not list((store.dir / "commits").iterdir())
    report = fsck(store)  # every object stored so far holds the bytes of its id
    assert report.ok
    assert report.objects == len(calls)
    assert report.tensors_decoded == len(calls)


def test_a_file_replaced_after_the_tensor_pass_is_refused(store, tmp_path, rng, monkeypatch):
    """C1 as reproduced (R11): the trainer's rename lands between the tensors and the file hash."""
    path = single_file(tmp_path / "ckpt", rng)
    real_digest = repo_module.file_digest

    def rename_then_digest(target):
        replace_by_rename(target)
        return real_digest(target)

    monkeypatch.setattr(repo_module, "file_digest", rename_then_digest)
    with pytest.raises(StoreError, match="no longer hashes to the bytes that were stored"):
        commit(store, path)
    assert store.head() is None


def test_a_header_rewritten_after_parsing_is_refused(store, tmp_path, rng, monkeypatch):
    path = single_file(tmp_path / "ckpt", rng)
    real_plan = repo_module._plan_files

    def plan_then_rewrite(ckpt, where):
        plans = real_plan(ckpt, where)
        reversed_order = dict(reversed(list(tensors_of(path).items())))  # other offsets
        data = write_safetensors(tmp_path / "reversed.safetensors", reversed_order).read_bytes()
        assert len(data) == path.stat().st_size
        with path.open("r+b") as handle:  # in place: Windows forbids truncating a mapped file
            handle.write(data)
        return plans

    monkeypatch.setattr(repo_module, "_plan_files", plan_then_rewrite)
    with pytest.raises(StoreError, match="changed"):
        commit(store, path)
    assert store.head() is None
    assert not list((store.dir / "objects").iterdir())


def test_each_tensor_is_hashed_once_for_its_id_and_the_store_trusts_it(
    store, revisions, monkeypatch
):
    """M1: the store gets the id; it never hashes a tensor's mapped bytes again."""
    vouched, store_hashed_views = [], []
    real_put, real_digest = ObjectStore.put_tensor, store_module.digest_bytes

    def put_spy(self, raw, itemsize, **options):
        vouched.append(options.get("tensor_id"))
        return real_put(self, raw, itemsize, **options)

    def digest_spy(view):
        store_hashed_views.append(isinstance(view, np.memmap))
        return real_digest(view)

    monkeypatch.setattr(ObjectStore, "put_tensor", put_spy)
    monkeypatch.setattr(store_module, "digest_bytes", digest_spy)
    result = commit(store, revisions.base)
    assert len(vouched) == 4
    assert all(map(is_id, vouched))
    assert store_hashed_views
    assert not any(store_hashed_views)  # payloads only
    for entry in store.get_commit(result.commit_id)["files"]:
        data = (revisions.base / entry["name"]).read_bytes()
        assert (entry["sha256"], entry["nbytes"]) == (hashlib.sha256(data).hexdigest(), len(data))


def test_a_damaged_payload_is_not_reused(store, revisions, tmp_path):
    """C3 as reproduced (R10): an emptied payload made a later commit report success. Since the
    second-pass M3 fix the commit stores the tensor again, which repairs the earlier commit."""
    first = commit(store, revisions.base)
    same = commit_manifest(store, first.commit_id)["tensors"][SAME]["sha256"]  # kept by the edit
    pid = store.get_object(same)["payload"]
    (store.dir / "payloads" / pid[:2] / pid).write_bytes(b"")
    second = commit(store, revisions.edited, message="edit")
    assert second.reused == 0
    assert store.get_object(same)["payload"] == pid  # same full encoding
    assert fsck(store).ok
    for result, source in ((first, revisions.base), (second, revisions.edited)):
        out = tmp_path / "out" / result.commit_id[:12]
        checkout_commit(store, result.commit_id, out)
        assert all((out / name).read_bytes() == (source / name).read_bytes() for name in SHARDS)


# Commit: reuse decodes what it reuses (second-pass security review M3, triage item 13)


def one_file(directory, w, v):
    """A checkpoint of the F32 tensors w and v; returns its directory."""
    directory.mkdir(parents=True, exist_ok=True)
    write_safetensors(directory / "m.safetensors", {"w": ("F32", w), "v": ("F32", v)})
    return directory


def plant_poison(store, tensor_bytes):
    """F15: an object record for a tensor id the victim will commit later (ids are public),
    pointing at junk of exactly the size the record implies; returns the junk's payload id."""
    junk = b"not the tensor" + b"\0" * 2
    pid = store.put_payload(junk)
    tensor_id = hashlib.sha256(tensor_bytes).hexdigest()
    store.put_object(
        {
            "format": "wvc-object/0",
            "id": tensor_id,
            "nbytes": len(tensor_bytes),
            "itemsize": 4,
            "form": "zstd-planes",
            "payload": pid,
            "depth": 0,
            "plane_sizes": [4, 4, 4, 4],
        }
    )
    assert store.has_object(tensor_id)
    assert not store.is_verified(tensor_id)
    return pid


def count_decodes(monkeypatch, tensor_id):
    """A list that records each ``read_tensor`` call for ``tensor_id``."""
    calls = []
    real = ObjectStore.read_tensor

    def spy(self, tid):
        if tid == tensor_id:
            calls.append(tid)
        return real(self, tid)

    monkeypatch.setattr(ObjectStore, "read_tensor", spy)
    return calls


def test_a_record_planted_for_a_future_tensor_is_replaced_not_reused(store, tmp_path, rng):
    """The F15 reproduction: the planted record was 'reused by hash' and the bytes lost."""
    w = np.arange(16, dtype=np.float32)
    v = rng.standard_normal(64).astype(np.float32)
    tensor_id = hashlib.sha256(w.tobytes()).hexdigest()
    junk = plant_poison(store, w.tobytes())
    source = one_file(tmp_path / "ckpt", w, v)
    result = commit(store, source)
    assert (result.tensors, result.reused, result.stored_full) == (2, 0, 2)
    record = store.get_object(tensor_id)
    assert record["payload"] != junk
    assert store.is_verified(tensor_id)
    checkout_commit(store, result.commit_id, tmp_path / "out")
    assert (tmp_path / "out" / "m.safetensors").read_bytes() == (
        source / "m.safetensors"
    ).read_bytes()
    assert fsck(store).ok


def test_reuse_decodes_once_then_trusts_the_marker_until_it_is_removed(
    store, tmp_path, rng, monkeypatch
):
    w = rng.standard_normal(256).astype(np.float32)
    tensor_id = hashlib.sha256(w.tobytes()).hexdigest()
    plant_poison(store, w.tobytes())
    decodes = count_decodes(monkeypatch, tensor_id)
    first = commit(store, one_file(tmp_path / "a", w, rng.standard_normal(8).astype(np.float32)))
    assert (first.reused, len(decodes)) == (0, 1)  # decoded, refused, stored afresh
    second = commit(store, one_file(tmp_path / "b", w, rng.standard_normal(8).astype(np.float32)))
    assert (second.reused, len(decodes)) == (1, 1)  # the marker skips the decode
    marker = store.dir / "verified" / tensor_id[:2] / tensor_id
    assert marker.is_file()
    assert marker.stat().st_size == 0
    marker.unlink()
    third = commit(store, one_file(tmp_path / "c", w, rng.standard_normal(8).astype(np.float32)))
    assert (third.reused, len(decodes)) == (1, 2)
    assert marker.is_file()


def test_a_marker_does_not_hide_a_damaged_payload_from_checkout(store, tmp_path, rng):
    """Markers only skip the decode on reuse: checkout and fsck verify everything, fsck drops
    the stale marker, and the next commit of the tensor stores it again."""
    w = rng.standard_normal(256).astype(np.float32)
    tensor_id = hashlib.sha256(w.tobytes()).hexdigest()
    commit(store, one_file(tmp_path / "a", w, rng.standard_normal(8).astype(np.float32)))
    pid = store.get_object(tensor_id)["payload"]
    path = store.dir / "payloads" / pid[:2] / pid
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))  # same length: the cheap reuse checks pass
    assert store.is_verified(tensor_id)
    damaged = commit(store, one_file(tmp_path / "b", w, rng.standard_normal(8).astype(np.float32)))
    assert damaged.reused == 1
    with pytest.raises(IntegrityError, match="hash to"):
        checkout_commit(store, damaged.commit_id, tmp_path / "out")
    report = fsck(store)
    assert not report.ok
    assert not store.is_verified(tensor_id)
    source = one_file(tmp_path / "c", w, rng.standard_normal(8).astype(np.float32))
    repaired = commit(store, source)
    assert repaired.reused == 0
    assert fsck(store).ok
    for result in (damaged, repaired):
        out = tmp_path / "out" / result.commit_id[:12]
        checkout_commit(store, result.commit_id, out)
        written = load_checkpoint(out / "m.safetensors").raw("w")
        assert bytes(written) == w.tobytes()


def test_payload_bytes_count_what_reached_the_disk(store, tmp_path, rng):
    """L4: two tensors with the same delta payload write it once."""
    base = rng.standard_normal((2, 1024)).astype(np.float32)
    edited = (base.view(np.uint32) ^ np.uint32(1)).view(np.float32)  # every element, bit 0
    path = tmp_path / "ckpt" / "model.safetensors"
    path.parent.mkdir()
    write_safetensors(path, {"x": ("F32", base[0]), "y": ("F32", base[1])})
    commit(store, path)
    write_safetensors(path, {"x": ("F32", edited[0]), "y": ("F32", edited[1])})
    second = commit(store, path, message="flip bit 0")
    assert second.stored_delta == {"delta-xor-planes": 2}
    record = store.get_commit(second.commit_id)
    payloads = {store.get_object(tid)["payload"] for tid in record["files"][0]["tensors"]}
    assert len(payloads) == 1
    (pid,) = payloads
    payload = (store.dir / "payloads" / pid[:2] / pid).stat().st_size
    assert second.payload_bytes == payload  # the header is the parent's


# Commit: branches (H3, H4, S-L7)


def test_a_parent_that_is_not_the_head_needs_a_branch(store, revisions, tmp_path, rng):
    """H3 as reproduced (R03): --parent no longer moves the current branch off its history."""
    a = commit(store, revisions.base, "A").commit_id
    b = commit(store, revisions.edited, "B").commit_id
    other = single_file(tmp_path / "c", rng)
    before = stored_files(store)
    with pytest.raises(StoreError, match="not the head of branch 'main'"):
        commit(store, other, "C", parent_ref=a[:7])
    assert stored_files(store) == before
    assert store.head() == b
    side = commit(store, other, "C", parent_ref=a[:7], branch="side")
    assert (side.branch, side.parent_id, store.current_branch()) == ("side", a, "side")
    assert store.read_ref("main") == b
    assert [cid for cid, _ in log_entries(store, "main")] == [b, a]
    assert [cid for cid, _ in log_entries(store)] == [side.commit_id, a]


def test_a_branch_only_fast_forwards(store, revisions, tmp_path, rng):
    a = commit(store, revisions.base, "A").commit_id
    b = commit(store, revisions.edited, "B").commit_id
    other = single_file(tmp_path / "c", rng)
    with pytest.raises(StoreError, match=r"branch 'main' is at .* not an ancestor"):
        commit(store, other, "C", parent_ref=a, branch="main")
    store.write_ref("dev", a)
    moved = commit(store, other, "C", parent_ref=b, branch="dev")  # a is an ancestor of b
    assert (store.read_ref("dev"), store.current_branch()) == (moved.commit_id, "dev")
    again = commit(store, revisions.base, "D", branch="dev")  # the branch HEAD is already on
    assert again.parent_id == moved.commit_id
    assert store.read_ref("dev") == again.commit_id
    with pytest.raises(StoreError, match="invalid branch name"):
        commit(store, revisions.edited, "E", branch="abcdef01")


def test_a_commit_racing_another_on_its_branch_leaves_it(
    store, revisions, tmp_path, rng, monkeypatch
):
    """H4 as reproduced (R12): process B's commit lands between A's read and A's ref write."""
    base = commit(store, revisions.base).commit_id
    other = single_file(tmp_path / "b", rng)
    real = ObjectStore.put_commit
    raced = {}

    def put_while_b_commits(self, record):
        commit_id = real(self, record)
        if not raced:
            raced["running"] = True
            raced["b"] = commit(ObjectStore(store.repo), other, "process B").commit_id
        return commit_id

    monkeypatch.setattr(ObjectStore, "put_commit", put_while_b_commits)
    with pytest.raises(StoreError, match=r"moved to .* on no branch"):
        commit(store, revisions.edited, "process A")
    assert [cid for cid, _ in log_entries(store)] == [raced["b"], base]


def test_a_head_naming_a_case_twin_of_a_branch_is_refused(store, revisions):
    """S-L7: HEAD on MAIN where main exists would write main's file on case-insensitive disks."""
    first = commit(store, revisions.base).commit_id
    (store.dir / "HEAD").write_text("ref: refs/heads/MAIN\n", encoding="utf-8", newline="\n")
    before = stored_files(store)
    with pytest.raises(StoreError, match="only in case"):
        commit(ObjectStore(store.repo), revisions.edited, "edit")
    assert stored_files(store) == before
    assert store.read_ref("main") == first


# Commit: the nothing-changed rule (H7) and file names (H5, H6, C2)


def test_resharded_or_reannotated_checkpoints_are_new_commits(store, revisions, tmp_path):
    """H7 as reproduced (R04): only the same root and the same file layout is refused."""
    first = commit(store, revisions.base)
    single = write_safetensors(tmp_path / "single.safetensors", tensors_of(revisions.base))
    resharded = commit(store, single, "resharded")
    header = load_checkpoint(single).files[0].data_start
    assert (resharded.manifest_root, resharded.reused) == (first.manifest_root, 4)
    assert resharded.payload_bytes == header  # one new header, no tensor
    annotated = tmp_path / "annotated" / single.name  # the same name; only metadata differs
    annotated.parent.mkdir()
    write_safetensors(annotated, tensors_of(single), metadata={"license": "x"})
    licensed = commit(store, annotated, "add a licence to __metadata__")
    assert licensed.manifest_root == first.manifest_root
    with pytest.raises(StoreError, match="nothing changed"):
        commit(store, annotated, "again")
    for result, source in ((resharded, single), (licensed, annotated)):
        out = tmp_path / f"out-{result.commit_id[:8]}"
        checkout_commit(store, result.commit_id, out)
        assert (out / source.name).read_bytes() == source.read_bytes()


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("shard\\1.safetensors", marks=NOT_ON_WINDOWS),  # a separator there
        "m" * 228 + ".safetensors",
        pytest.param("CON.safetensors", marks=NOT_ON_WINDOWS),  # the console device
        pytest.param("nul.part.safetensors", marks=NOT_ON_WINDOWS),  # the null device
        "modèle.safetensors",
        pytest.param("a:b.safetensors", marks=NOT_ON_WINDOWS),  # an NTFS stream of a
        "weights.bin",
        "-x.safetensors",
    ],
)
def test_file_names_are_checked_before_anything_is_stored(store, tmp_path, rng, name):
    """H5, H6: names a checkout could not write back, or would write elsewhere, are refused."""
    path = single_file_or_skip(tmp_path / "ckpt", rng, name)
    with pytest.raises(StoreError, match="is not an allowed file name"):
        commit(store, path if name == "weights.bin" else path.parent)
    assert store.head() is None
    assert not list((store.dir / "objects").iterdir())


# Checkout: file records against the manifest (S-H1, S-H2, C2)


def test_a_commit_whose_files_are_not_its_manifest_is_refused(store, revisions, tmp_path):
    """S-H2 as reproduced: an audited commit's record with another commit's files."""
    audited = commit(store, revisions.base, "audited").commit_id
    other = commit(store, revisions.edited, "other").commit_id
    forged = {
        **store.get_commit(audited),
        "parents": [audited],
        "operation": {"kind": "convert"},
        "message": "re-save, no weight changes",
        "files": store.get_commit(other)["files"],
    }
    with pytest.raises(IntegrityError, match="do not hold the tensors of manifest"):
        store.put_commit(forged)
    planted = plant_commit(store, forged)
    for action in (
        lambda: show_commit(store, planted),
        lambda: checkout_commit(store, planted, tmp_path / "out"),
    ):
        with pytest.raises(IntegrityError, match="do not hold the tensors of manifest"):
            action()
    assert not (tmp_path / "out").exists()


def test_a_python_module_cannot_be_checked_out(store, revisions, tmp_path):
    """S-H1 as reproduced: an extra file record whose 'header' is Python source."""
    audited = commit(store, revisions.base, "audited").commit_id
    record = store.get_commit(audited)
    source = b"print('attacker code ran on import')\n"
    module = {
        "name": "safetensors.py",
        "header": store.put_payload(source),
        "tensors": [],
        "sha256": hashlib.sha256(source).hexdigest(),
        "nbytes": len(source),
    }
    for name, message in (
        ("safetensors.py", "not an allowed file name"),
        ("safetensors.safetensors", "not a safetensors header"),
    ):
        forged = {
            **record,
            "parents": [audited],
            "files": [*record["files"], {**module, "name": name}],
        }
        with pytest.raises(StoreError, match=message):
            store.put_commit(forged)
        out = tmp_path / f"out-{name}"
        with pytest.raises(StoreError, match=message):
            checkout_commit(store, plant_commit(store, forged), out)
        assert not out.exists()


def test_names_that_collide_on_case_insensitive_disks_are_refused(store, revisions, tmp_path):
    """C2 as reproduced (R08): a.safetensors and A.safetensors would be one file on APFS."""
    record = store.get_commit(commit(store, revisions.base).commit_id)
    first, second = record["files"]
    forged = {
        **record,
        "files": [{**first, "name": "a.safetensors"}, {**second, "name": "A.safetensors"}],
    }
    with pytest.raises(StoreError, match="differ only in case"):
        store.put_commit(forged)
    with pytest.raises(StoreError, match="differ only in case"):
        checkout_commit(store, plant_commit(store, forged), tmp_path / "out")
    assert not (tmp_path / "out").exists()


# Checkout: all or nothing (S-M7, S-L10), and the targets after the renames (C2)


def test_a_failed_checkout_changes_no_file(store, revisions, tmp_path):
    """S-M7 as reproduced: a damaged payload of v2's second shard left v1 and v2 mixed."""
    v1 = commit(store, revisions.base, "v1")
    v2 = commit(store, revisions.edited, "v2")
    deploy = tmp_path / "deploy"
    first = checkout_commit(store, v1.commit_id, deploy)
    assert first.manifest_root == v1.manifest_root
    before = {name: (deploy / name).read_bytes() for name in SHARDS}
    pid = store.get_object(commit_manifest(store, v2.commit_id)["tensors"][NEW]["sha256"])[
        "payload"
    ]
    payload = store.dir / "payloads" / pid[:2] / pid  # NEW: only v2 holds it
    data = bytearray(payload.read_bytes())
    data[-1] ^= 0xFF
    payload.write_bytes(bytes(data))
    with pytest.raises(IntegrityError, match="hash to"):
        checkout_commit(store, v2.commit_id, deploy, force=True)
    assert {name: (deploy / name).read_bytes() for name in SHARDS} == before
    assert sorted(p.name for p in deploy.iterdir()) == list(SHARDS)  # no temporaries left


def fail_replace(monkeypatch, call, *, and_later=False):
    """Make the checkout module's ``os.replace`` raise on its ``call``-th use (and later ones)."""
    real, calls = os.replace, []

    def failing(source, target):
        calls.append((source, target))
        if len(calls) == call or (and_later and len(calls) > call):
            raise PermissionError(13, "injected failure", str(target))
        real(source, target)

    monkeypatch.setattr(checkout_module.os, "replace", failing)
    return calls


@pytest.mark.parametrize("call", [1, 2, 3, 4])  # 1, 2 move the old shards aside; 3, 4 rename
def test_a_failed_rename_restores_the_previous_files(store, revisions, tmp_path, monkeypatch, call):
    """C-H1: a rename that fails after earlier ones succeeded no longer leaves v1 and v2 mixed."""
    v1 = commit(store, revisions.base, "v1")
    v2 = commit(store, revisions.edited, "v2")
    deploy = tmp_path / "deploy"
    checkout_commit(store, v1.commit_id, deploy)
    before = {name: (deploy / name).read_bytes() for name in SHARDS}
    calls = fail_replace(monkeypatch, call)
    with pytest.raises(StoreError, match="the directory was restored, nothing was changed"):
        checkout_commit(store, v2.commit_id, deploy, force=True)
    monkeypatch.undo()
    assert {name: (deploy / name).read_bytes() for name in SHARDS} == before
    assert sorted(p.name for p in deploy.iterdir()) == list(SHARDS)  # no temporaries left
    assert len(calls) >= call  # the injected failure was reached
    assert checkout_commit(store, v2.commit_id, deploy, force=True).replaced == SHARDS


def test_a_failed_rename_into_an_empty_directory_leaves_it_empty(
    store, revisions, tmp_path, monkeypatch
):
    """The second rename fails after the first shard is in place: that shard is removed."""
    commit(store, revisions.base)
    out = tmp_path / "out"
    fail_replace(monkeypatch, 2)
    with pytest.raises(StoreError, match="directory was restored") as caught:
        checkout_commit(store, "HEAD", out)
    assert isinstance(caught.value.__cause__, PermissionError)
    assert list(out.iterdir()) == []


def test_a_restore_that_fails_names_where_the_previous_file_is(
    store, revisions, tmp_path, monkeypatch
):
    """When putting a moved-aside file back fails too, the error says where it is left."""
    v1 = commit(store, revisions.base, "v1")
    v2 = commit(store, revisions.edited, "v2")
    deploy = tmp_path / "deploy"
    checkout_commit(store, v1.commit_id, deploy)
    before = {name: (deploy / name).read_bytes() for name in SHARDS}
    fail_replace(monkeypatch, 3, and_later=True)  # both shards moved aside; then all fail
    with pytest.raises(StoreError, match="could not be fully restored") as caught:
        checkout_commit(store, v2.commit_id, deploy, force=True)
    message = str(caught.value)
    for name in SHARDS:
        backup = deploy / message.split(f"the previous {name} is left at ")[1].split(" (")[0]
        assert backup.read_bytes() == before[name]
        assert not (deploy / name).exists()


def test_checkout_removes_stale_temporaries_and_reports_the_root(store, revisions, tmp_path):
    """S-L10, L2: a temporary left by a killed checkout no longer blocks the next one."""
    result = commit(store, revisions.base)
    out = tmp_path / "out"
    out.mkdir()
    stale = out / f".wvc-{'b' * 32}.tmp"
    stale.write_bytes(b"partial")
    then = os.stat(stale).st_mtime - 7200
    os.utime(stale, (then, then))
    checked = checkout_commit(store, "HEAD", out)
    assert checked.manifest_root == result.manifest_root
    assert sorted(p.name for p in out.iterdir()) == list(SHARDS)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / f".wvc-{'c' * 32}.tmp").write_bytes(b"a checkout in progress")
    with pytest.raises(StoreError, match="non-empty"):
        checkout_commit(store, "HEAD", fresh)


def test_targets_are_checked_after_the_renames(store, revisions, tmp_path, monkeypatch):
    """C2: a file system that folds two names into one file is caught after the renames."""
    commit(store, revisions.base)
    real = os.replace

    def fold(source, target):  # every target lands on the first shard's name
        real(source, os.path.join(os.path.dirname(target), SHARDS[0]))

    monkeypatch.setattr(checkout_module.os, "replace", fold)
    with pytest.raises(IntegrityError, match="not a regular file of"):
        checkout_commit(store, "HEAD", tmp_path / "out")
    monkeypatch.undo()
    out = tmp_path / "linked"
    out.mkdir()
    (out / SHARDS[0]).write_bytes(b"12345678")
    os.link(out / SHARDS[0], out / SHARDS[1])
    entries = [{"name": name, "nbytes": 8} for name in SHARDS]
    with pytest.raises(IntegrityError, match="are one file"):
        _check_targets(out, entries)


def test_a_long_allowed_name_checks_out(store, tmp_path, rng):
    """L9: a 200-byte name commits and checks out; temporary names do not grow with it."""
    name = "m" * (200 - len(".safetensors")) + ".safetensors"
    path = single_file_or_skip(tmp_path / "ckpt", rng, name)
    commit(store, path.parent)
    checkout_commit(store, "HEAD", tmp_path / "out")
    assert (tmp_path / "out" / name).read_bytes() == path.read_bytes()
