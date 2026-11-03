import hashlib
import json
import re
import struct
import tracemalloc

import numpy as np
import pytest
from helpers import (
    DENSE,
    GONE,
    NEW,
    SAME,
    SHARDS,
    SPARSE,
    commit,
    dense_noise,
    plant_commit,
    sparse_edit,
    write_safetensors,
)

from wvc.checkout import checkout_commit
from wvc.file_layout import validate_file_records
from wvc.hashing import build_manifest
from wvc.repo import (
    ManifestComparison,
    commit_checkpoint,
    commit_manifest,
    compare_manifests,
    default_author,
    init_repository,
    log_entries,
    open_repository,
    show_commit,
    tensor_count,
)
from wvc.safetensors_io import SafetensorsError, load_checkpoint
from wvc.store import IntegrityError, ObjectStore, StoreError

TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
COMMIT_KEYS = {"format", "manifest", "parents", "operation", "message", "author", "time", "files"}
VIEW_AS = {"F32": "<f4", "BF16": "<u2", "I8": "i1"}  # numpy views of stored tensor bytes


def entries(store, commit_id):
    """The manifest entries of a stored commit, by tensor name."""
    return commit_manifest(store, commit_id)["tensors"]


def payload_path(store, pid):
    return store.dir / "payloads" / pid[:2] / pid


def stored_files(store):
    return sorted(path for path in store.dir.rglob("*") if path.is_file())


def tensors_of(path):
    """``name -> (dtype, array)`` of a checkpoint, to rewrite it with ``write_safetensors``."""
    ckpt = load_checkpoint(path)
    return {
        name: (info.dtype, np.asarray(ckpt.raw(name)).view(VIEW_AS[info.dtype]).reshape(info.shape))
        for name, info in ckpt.tensors.items()
    }


def by_offset(file):
    return sorted(file.tensors.values(), key=lambda info: info.begin)


def write_raw(path, offsets, buffer_len):
    """A file whose F32 tensors sit at the given byte ranges of a ``buffer_len``-byte buffer."""
    header = {
        name: {"dtype": "F32", "shape": [(end - begin) // 4], "data_offsets": [begin, end]}
        for name, (begin, end) in offsets.items()
    }
    text = json.dumps(header).encode("utf-8")
    text += b" " * ((8 - len(text) % 8) % 8)
    path.write_bytes(
        struct.pack("<Q", len(text)) + text + bytes(i % 256 for i in range(buffer_len))
    )
    return path


def test_init_creates_a_repository_and_refuses_to_reinit(tmp_path):
    repo = init_repository(tmp_path / "new" / "repo")
    assert repo == (tmp_path / "new" / "repo").resolve()
    store = ObjectStore(repo)
    assert store.exists()
    assert store.current_branch() == "main"
    assert store.head() is None
    with pytest.raises(StoreError, match="already exists"):
        init_repository(repo)


def test_open_repository_needs_a_valid_store(store, tmp_path, monkeypatch):
    assert open_repository(store.repo).dir == store.dir
    nested = store.repo / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert open_repository().repo == store.repo.resolve()
    with pytest.raises(StoreError, match="not a wvc repository"):
        open_repository(tmp_path)
    (store.dir / "config.json").write_text('{"format": "wvc-repo/9"}', encoding="utf-8")
    with pytest.raises(StoreError, match="expected format"):
        open_repository(store.repo)


def test_first_commit_stores_every_tensor_full_and_sets_head(store, revisions):
    result = commit(store, revisions.base, "first", kind="pretrain-checkpoint")
    assert store.head() == store.read_ref("main") == result.commit_id
    assert (result.branch, result.parent_id, result.tensors, result.reused) == ("main", None, 4, 0)
    assert (result.stored_full, result.stored_delta) == (4, {})
    record = store.get_commit(result.commit_id)
    assert set(record) == COMMIT_KEYS
    assert record["format"] == "wvc-commit/0"
    assert (record["parents"], record["operation"]) == ([], {"kind": "pretrain-checkpoint"})
    assert (record["message"], record["author"]) == ("first", "tester")
    assert TIME.fullmatch(record["time"])
    ckpt = load_checkpoint(revisions.base)
    assert record["manifest"] == result.manifest_root == build_manifest(ckpt)["root"]
    assert [entry["name"] for entry in record["files"]] == list(SHARDS)
    for file, entry in zip(ckpt.files, record["files"], strict=True):
        data = file.path.read_bytes()
        assert (entry["sha256"], entry["nbytes"]) == (hashlib.sha256(data).hexdigest(), len(data))
        assert store.get_payload(entry["header"]) == data[: file.data_start]
        order = by_offset(file)
        assert [info.name for info in order] != list(file.tensors)  # header order differs
        start = file.data_start
        expected = [
            hashlib.sha256(data[start + i.begin : start + i.end]).hexdigest() for i in order
        ]
        assert entry["tensors"] == expected
        assert all(store.get_object(tid)["form"] == "zstd-planes" for tid in entry["tensors"])
    headers = sum(file.data_start for file in ckpt.files)
    objects = [store.get_object(tid) for entry in record["files"] for tid in entry["tensors"]]
    payloads = sum(payload_path(store, obj["payload"]).stat().st_size for obj in objects)
    assert result.payload_bytes == headers + payloads


def test_second_commit_reuses_unchanged_tensors_and_stores_deltas(store, revisions):
    first = commit(store, revisions.base)
    second = commit(store, revisions.edited, message="edit", kind="direct-edit")
    assert (second.parent_id, store.head()) == (first.commit_id, second.commit_id)
    assert (second.tensors, second.reused, second.stored_full) == (4, 1, 1)
    assert second.stored_delta == {"delta-sparse": 1, "delta-xor-planes": 1}
    old, new = entries(store, first.commit_id), entries(store, second.commit_id)
    assert new[SAME] == old[SAME]
    assert GONE not in new
    sparse, dense = (store.get_object(new[name]["sha256"]) for name in (SPARSE, DENSE))
    assert (sparse["form"], sparse["base"]) == ("delta-sparse", old[SPARSE]["sha256"])
    assert (dense["form"], dense["base"]) == ("delta-xor-planes", old[DENSE]["sha256"])
    assert store.get_object(new[NEW]["sha256"])["form"] == "zstd-planes"
    records = [store.get_commit(c.commit_id) for c in (first, second)]
    assert records[1]["parents"] == [first.commit_id]
    assert records[1]["operation"] == {"kind": "direct-edit"}
    # Shard 1 kept its layout, so its header payload is shared and was not written again.
    assert records[1]["files"][0]["header"] == records[0]["files"][0]["header"]
    new_objects = [store.get_object(new[name]["sha256"]) for name in (SPARSE, DENSE, NEW)]
    tensor_bytes = sum(payload_path(store, obj["payload"]).stat().st_size for obj in new_objects)
    header_2 = load_checkpoint(revisions.edited).files[1].data_start
    assert second.payload_bytes == tensor_bytes + header_2


def test_identical_checkpoint_is_refused(store, revisions):
    """H7 narrowed the rule: only the same root *and* the same file layout is "nothing changed";
    a re-sharded checkpoint is a new commit (tests/test_repo_integrity.py)."""
    first = commit(store, revisions.base)
    before = stored_files(store)
    with pytest.raises(StoreError, match="nothing changed"):
        commit(store, revisions.base, message="again")
    assert stored_files(store) == before
    assert store.head() == first.commit_id


def test_a_reused_tensor_with_a_corrupt_record_stops_the_commit(store, revisions):
    first = commit(store, revisions.base)
    same = entries(store, first.commit_id)[SAME]["sha256"]
    (store.dir / "objects" / same[:2] / f"{same}.json").write_text("{not json")
    with pytest.raises(IntegrityError, match="not UTF-8 JSON"):
        commit(store, revisions.edited, message="edit")
    assert store.head() == first.commit_id


def test_files_that_change_during_a_commit_are_reported(store, revisions, monkeypatch):
    """The final read of each file must hash to the bytes stored (more in test_repo_integrity)."""
    monkeypatch.setattr("wvc.repo.file_digest", lambda path: ("0" * 64, 1))
    with pytest.raises(StoreError, match="changed while it was being committed"):
        commit(store, revisions.base)
    assert store.head() is None
    with pytest.raises(StoreError, match="files must be a list"):
        validate_file_records({}, "record")


def test_explicit_parent(store, revisions):
    first = commit(store, revisions.base)
    second = commit(store, revisions.edited, message="edit")
    revert = commit(store, revisions.base, message="revert", parent_ref=second.commit_id[:7])
    assert revert.parent_id == second.commit_id
    assert store.head() == revert.commit_id
    assert (revert.reused, revert.stored_full, revert.stored_delta) == (4, 0, {})
    assert revert.payload_bytes == 0  # every tensor and both headers were stored already
    # H3: a parent other than the branch head needs --branch; then the rules apply as usual.
    with pytest.raises(StoreError, match=r"not the head of branch 'main'.*--branch"):
        commit(store, revisions.base, message="again", parent_ref=first.commit_id)
    with pytest.raises(StoreError, match="nothing changed"):
        commit(store, revisions.base, message="again", parent_ref=first.commit_id, branch="b")
    with pytest.raises(StoreError, match="no commit matches"):
        commit(store, revisions.edited, parent_ref="0" * 7)
    for suffix in ("1", "2"):
        (store.dir / "commits" / f"{'abcdef0' + suffix * 57}.json").write_text("{}")
    with pytest.raises(StoreError, match="ambiguous"):
        commit(store, revisions.edited, parent_ref="abcdef0")


def test_explicit_head_parent_needs_a_commit(store, revisions):
    with pytest.raises(StoreError, match="no commits yet"):
        commit(store, revisions.base, parent_ref="HEAD")
    assert stored_files(store) == sorted([store.dir / "HEAD", store.dir / "config.json"])


def test_bad_commit_text_is_refused_before_anything_is_stored(store, revisions):
    cases = {
        "unknown operation kind 'retrain'": {"kind": "retrain"},
        "message is empty": {"message": " \n"},
        "message is not valid UTF-8": {"message": "bad \udcff"},
    }
    for message, options in cases.items():
        with pytest.raises(StoreError, match=message):
            commit(store, revisions.base, **options)
    with pytest.raises(StoreError, match="author is empty"):
        commit_checkpoint(store, revisions.base, message="m", operation_kind="unknown", author="")
    assert store.head() is None
    assert not any((store.dir / "payloads").iterdir())


def test_default_author(monkeypatch):
    """S-L5: the author comes from WVC_AUTHOR or --author, never silently from the OS."""
    monkeypatch.setenv("WVC_AUTHOR", "Évan")
    assert default_author() == "Évan"
    monkeypatch.setattr("getpass.getuser", lambda: "os-user")
    for value in ("", " "):
        monkeypatch.setenv("WVC_AUTHOR", value)
        with pytest.raises(StoreError, match="pass --author NAME or set WVC_AUTHOR"):
            default_author()
    monkeypatch.delenv("WVC_AUTHOR")
    with pytest.raises(StoreError, match="pass --author NAME or set WVC_AUTHOR"):
        default_author()


def test_log_follows_first_parents(store, revisions):
    first = commit(store, revisions.base, "first").commit_id
    second = commit(store, revisions.edited, message="edit").commit_id
    third = commit(store, revisions.base, message="revert").commit_id
    log = list(log_entries(store))
    assert [commit_id for commit_id, _ in log] == [third, second, first]
    assert [record["message"] for _, record in log] == ["revert", "edit", "first"]
    assert [commit_id for commit_id, _ in log_entries(store, limit=2)] == [third, second]
    assert list(log_entries(store, limit=0)) == []
    assert [commit_id for commit_id, _ in log_entries(store, second[:7])] == [second, first]
    with pytest.raises(ValueError, match="limit"):
        log_entries(store, limit=-1)


def test_log_of_an_empty_branch_is_an_error(store):
    with pytest.raises(StoreError, match="no commits yet"):
        log_entries(store)


def test_show_compares_manifests_with_the_first_parent(store, revisions, monkeypatch):
    first = commit(store, revisions.base)
    second = commit(store, revisions.edited, message="edit", kind="direct-edit")

    def no_decoding(*args, **kwargs):
        raise AssertionError("show must not decode tensors")

    monkeypatch.setattr(ObjectStore, "read_tensor", no_decoding)
    shown = show_commit(store, "HEAD")
    assert (shown.commit_id, shown.parent_id, shown.tensor_count) == (
        second.commit_id,
        first.commit_id,
        4,
    )
    assert shown.record == store.get_commit(second.commit_id)
    assert shown.comparison == ManifestComparison(
        unchanged=1, changed=(DENSE, SPARSE), added=(NEW,), removed=(GONE,)
    )
    payload = shown.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["comparison"] == {
        "counts": {"changed": 2, "unchanged": 1, "added": 1, "removed": 1},
        "changed": [DENSE, SPARSE],
        "added": [NEW],
        "removed": [GONE],
    }
    root = show_commit(store, first.commit_id[:7])
    assert (root.parent_id, root.comparison, root.to_dict()["comparison"]) == (None, None, None)


def test_compare_manifests_counts_dtype_and_shape_changes_as_changed():
    sha = "a" * 64
    old = {"tensors": {"w": {"dtype": "F32", "shape": [4], "sha256": sha}}}
    for entry in (
        {"dtype": "F32", "shape": [2, 2], "sha256": sha},
        {"dtype": "I32", "shape": [4], "sha256": sha},
    ):
        comparison = compare_manifests(old, {"tensors": {"w": entry}})
        assert comparison == ManifestComparison(0, ("w",), (), ())
    assert compare_manifests(old, old) == ManifestComparison(1, (), (), ())


def test_tensor_count_is_checked():
    manifest = {"root": "r", "tensor_count": 2, "tensors": {"w": {}}}
    with pytest.raises(IntegrityError, match="does not match its 1 tensors"):
        tensor_count(manifest)
    assert tensor_count({**manifest, "tensor_count": 1}) == 1


def test_checkout_reproduces_every_shard_byte_for_byte(store, revisions, tmp_path):
    first = commit(store, revisions.base)
    second = commit(store, revisions.edited, message="edit")
    for result, source in ((first, revisions.base), (second, revisions.edited)):
        out = tmp_path / f"out-{result.commit_id[:7]}"
        checked = checkout_commit(store, result.commit_id, out)
        assert (checked.commit_id, checked.out_dir) == (result.commit_id, out)
        assert [file.name for file in checked.files] == list(SHARDS)
        assert sorted(path.name for path in out.iterdir()) == list(SHARDS)  # no temporaries
        for file in checked.files:
            data = (source / file.name).read_bytes()
            assert (out / file.name).read_bytes() == data
            assert file.path == out / file.name
            assert (file.sha256, file.nbytes) == (hashlib.sha256(data).hexdigest(), len(data))
        assert checked.nbytes == sum(file.nbytes for file in checked.files)


def test_checkout_refuses_a_non_empty_directory_unless_forced(store, revisions, tmp_path):
    commit(store, revisions.base)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    with pytest.raises(StoreError, match="non-empty"):
        checkout_commit(store, "HEAD", out)
    assert [path.name for path in out.iterdir()] == ["keep.txt"]
    (out / SHARDS[0]).write_bytes(b"stale")
    checkout_commit(store, "HEAD", out, force=True)
    assert sorted(path.name for path in out.iterdir()) == sorted([*SHARDS, "keep.txt"])
    for name in SHARDS:
        assert (out / name).read_bytes() == (revisions.base / name).read_bytes()
    assert (out / "keep.txt").read_text() == "mine"


def test_a_forced_checkout_reports_the_files_it_replaced(store, revisions, tmp_path):
    """S-M2: replacing an existing file is never silent; a fresh checkout replaces none."""
    commit(store, revisions.base)
    out = tmp_path / "out"
    fresh = checkout_commit(store, "HEAD", out)
    assert fresh.replaced == ()
    assert not any(file.replaced for file in fresh.files)
    (out / SHARDS[0]).unlink()
    (out / "keep.txt").write_text("mine")
    forced = checkout_commit(store, "HEAD", out, force=True)
    assert forced.replaced == (SHARDS[1],)
    assert [file.replaced for file in forced.files] == [False, True]
    assert (out / "keep.txt").read_text() == "mine"
    assert sorted(path.name for path in out.iterdir()) == sorted([*SHARDS, "keep.txt"])


def test_a_symlink_at_a_target_name_is_refused(store, revisions, tmp_path):
    """A checkout replaces only regular files; a link and the file it names stay untouched."""
    commit(store, revisions.base)
    victim = tmp_path / "victim.txt"
    victim.write_text("mine")
    out = tmp_path / "out"
    out.mkdir()
    (out / SHARDS[0]).symlink_to(victim)
    with pytest.raises(StoreError, match="is a symbolic link, and a checkout replaces only"):
        checkout_commit(store, "HEAD", out, force=True)
    assert (out / SHARDS[0]).is_symlink()
    assert victim.read_text() == "mine"
    assert [path.name for path in out.iterdir()] == [SHARDS[0]]


def test_checkout_refuses_a_file_or_the_store_as_output(store, revisions, tmp_path):
    commit(store, revisions.base)
    target = tmp_path / "file"
    target.write_text("x")
    with pytest.raises(StoreError, match="not a directory"):
        checkout_commit(store, "HEAD", target, force=True)
    with pytest.raises(StoreError, match="store itself"):
        checkout_commit(store, "HEAD", store.dir / "out", force=True)


def test_tampered_payload_fails_checkout_and_leaves_no_partial_file(store, revisions, tmp_path):
    result = commit(store, revisions.base)
    entry = store.get_commit(result.commit_id)["files"][0]
    last = store.get_object(entry["tensors"][-1])  # the header and first tensor come before it
    payload_path(store, last["payload"]).write_bytes(b"tampered")
    out = tmp_path / "out"
    with pytest.raises(IntegrityError, match="holds 8 bytes, not the"):
        checkout_commit(store, "HEAD", out)
    assert list(out.iterdir()) == []


def test_checkout_checks_the_rebuilt_file_against_the_record(store, revisions, tmp_path):
    record = store.get_commit(commit(store, revisions.base).commit_id)
    files = [{**record["files"][0], "sha256": "0" * 64}, record["files"][1]]
    wrong = store.put_commit({**record, "message": "wrong sha256", "files": files})
    out = tmp_path / "out"
    with pytest.raises(IntegrityError, match="rebuilt file"):
        checkout_commit(store, wrong, out)
    assert list(out.iterdir()) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda f: {**f, "name": "../escape.safetensors"}, "not an allowed file name"),
        (lambda f: {**f, "name": ""}, "not an allowed file name"),
        (lambda f: {**f, "tensors": ["HEAD"]}, "64-hex ids"),
        (lambda f: {**f, "nbytes": -1}, "nbytes"),
        (lambda f: {**f, "extra": 1}, "exactly the keys"),
    ],
)
def test_checkout_rejects_malformed_file_records(store, revisions, tmp_path, change, message):
    """put_commit refuses such records (H5); a received store may still hold them."""
    record = store.get_commit(commit(store, revisions.base).commit_id)
    bad_record = {**record, "message": "malformed"}
    bad_record["files"] = [change(record["files"][0]), record["files"][1]]
    with pytest.raises(StoreError, match=message):
        store.put_commit(bad_record)
    bad = plant_commit(store, bad_record)
    out = tmp_path / "nested" / "out"
    with pytest.raises(StoreError, match=message):
        checkout_commit(store, bad, out)
    with pytest.raises(StoreError, match=message):
        show_commit(store, bad)
    assert not out.exists()
    assert not (tmp_path / "nested" / "escape.safetensors").exists()


def test_checkout_rejects_repeated_file_names(store, revisions, tmp_path):
    record = store.get_commit(commit(store, revisions.base).commit_id)
    twice = {**record, "message": "repeated", "files": [record["files"][0]] * 2}
    with pytest.raises(StoreError, match="share a name"):
        store.put_commit(twice)
    with pytest.raises(StoreError, match="share a name"):
        checkout_commit(store, plant_commit(store, twice), tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("offsets", "buffer_len", "message"),
    [
        ({"a": (0, 8), "b": (16, 24)}, 24, "hole or an overlap"),
        ({"a": (0, 8), "b": (4, 12)}, 12, "hole or an overlap"),
        ({"a": (4, 12), "b": (12, 20)}, 20, "hole or an overlap"),
        ({"a": (0, 8), "b": (8, 16)}, 20, "end at byte 16 of the 20-byte"),
    ],
    ids=["hole", "overlap", "leading-gap", "trailing-bytes"],
)
def test_files_whose_tensors_do_not_fill_the_buffer_are_refused(
    store, tmp_path, offsets, buffer_len, message
):
    path = write_raw(tmp_path / "odd.safetensors", offsets, buffer_len)
    assert set(load_checkpoint(path).tensors) == set(offsets)  # the reader accepts it
    with pytest.raises(SafetensorsError, match=message):
        commit(store, path)
    assert store.head() is None
    assert not any((store.dir / "objects").iterdir())


def test_zero_size_tensors_round_trip(store, tmp_path, rng):
    tensors = {
        "empty.a": ("F32", np.zeros(0, dtype=np.float32)),
        "w": ("F32", rng.standard_normal(4).astype(np.float32)),
        "empty.b": ("I32", np.zeros((0, 3), dtype=np.int32)),
    }
    path = write_safetensors(
        tmp_path / "z.safetensors", tensors, header_order=["w", "empty.b", "empty.a"]
    )
    result = commit(store, path)
    assert result.tensors == 3
    checkout_commit(store, "HEAD", tmp_path / "out")
    assert (tmp_path / "out" / "z.safetensors").read_bytes() == path.read_bytes()


def test_symlinked_shards_check_out_as_regular_files(store, revisions, tmp_path):
    """A Hugging Face snapshot holds links named like shards that point at content blobs."""
    blobs, snapshot = tmp_path / "blobs", tmp_path / "snapshot"
    blobs.mkdir()
    snapshot.mkdir()
    for name in SHARDS:
        data = (revisions.base / name).read_bytes()
        blob = blobs / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        (snapshot / name).symlink_to(blob)
    record = store.get_commit(commit(store, snapshot).commit_id)
    assert [entry["name"] for entry in record["files"]] == list(SHARDS)
    out = tmp_path / "out"
    checkout_commit(store, "HEAD", out)
    for name in SHARDS:
        assert not (out / name).is_symlink()
        assert (out / name).read_bytes() == (revisions.base / name).read_bytes()


def test_large_checkpoint_commits_through_memory_maps(store, tmp_path, rng, monkeypatch):
    values = {f"block.{i}": rng.standard_normal(64 * 1024).astype(np.float32) for i in range(16)}
    base = write_safetensors(
        tmp_path / "base.safetensors", {k: ("F32", v) for k, v in values.items()}
    )
    edited_values = {
        k: sparse_edit(v, rng) if i % 2 else dense_noise(v, rng)
        for i, (k, v) in enumerate(values.items())
    }
    edited = write_safetensors(
        tmp_path / "edited.safetensors", {k: ("F32", v) for k, v in edited_values.items()}
    )
    size = base.stat().st_size
    assert size > 4_000_000
    views = []
    put_tensor = ObjectStore.put_tensor

    def spy(self, raw, itemsize, **options):
        views.append(isinstance(raw, np.memmap) and options["tensor_id"] is not None)
        return put_tensor(self, raw, itemsize, **options)

    monkeypatch.setattr(ObjectStore, "put_tensor", spy)
    for path, message in ((base, "base"), (edited, "edit")):
        tracemalloc.start()
        try:
            result = commit(store, path, message=message)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # Reading a whole file into memory would need at least ``size`` bytes at once.
        assert peak < size // 2, (message, peak, size)
    assert views == [True] * 32
    assert result.stored_delta == {"delta-sparse": 8, "delta-xor-planes": 8}
    checkout_commit(store, "HEAD", tmp_path / "out")
    assert (tmp_path / "out" / edited.name).read_bytes() == edited.read_bytes()
