import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np
import pytest
from helpers import (
    DENSE,
    REAL_MAKE_HUB_CLIENT,
    SHARDS,
    SPARSE,
    commit,
    dense_noise,
    sparse_edit,
    write_safetensors,
)

from wvc import remote as remote_module
from wvc.canonical import object_id
from wvc.checkout import checkout_commit
from wvc.cli import main
from wvc.file_layout import MAX_PREFIX_BYTES
from wvc.records import expected_payload_size
from wvc.refs import REF_LIMIT
from wvc.remote import (
    PullResult,
    PushResult,
    add_remote,
    clone,
    list_remotes,
    pull,
    push,
    remote_url,
)
from wvc.repo import init_repository
from wvc.safetensors_io import load_checkpoint
from wvc.store import OBJECT_LIMIT, RECORD_LIMIT, IntegrityError, ObjectStore, StoreError
from wvc.transport import DirTransport

KINDS = ("payloads", "objects", "manifests", "commits", "refs")  # the order push writes in


@dataclass(frozen=True)
class History:
    store: ObjectStore
    ids: list[str]  # oldest first
    sources: list[Path]  # the checkpoint directory of each commit


def write_third(edited: Path, rng: np.random.Generator, directory: Path) -> Path:
    """A third revision: SPARSE and DENSE edited again (deltas on deltas), shard 2 kept."""
    ckpt = load_checkpoint(edited)
    sparse = np.asarray(ckpt.raw(SPARSE)).view("<f4").reshape(64, 64)
    dense = np.asarray(ckpt.raw(DENSE)).view("<f4")
    tensors = {SPARSE: ("F32", sparse_edit(sparse, rng)), DENSE: ("F32", dense_noise(dense, rng))}
    directory.mkdir(parents=True)
    write_safetensors(directory / SHARDS[0], tensors, header_order=[DENSE, SPARSE])
    shutil.copyfile(edited / SHARDS[1], directory / SHARDS[1])
    return directory


@pytest.fixture
def third(tmp_path, revisions, rng):
    return write_third(revisions.edited, rng, tmp_path / "checkpoints" / "third")


@pytest.fixture
def history(tmp_path, revisions, third):
    """A repository with three commits whose delta chains reach depth 2."""
    store = ObjectStore(init_repository(tmp_path / "origin"))
    sources = [revisions.base, revisions.edited, third]
    ids = [commit(store, path, f"revision {i}").commit_id for i, path in enumerate(sources)]
    return History(store, ids, sources)


def store_files(store):
    """The store's files as transport paths: everything under .wvc but config.json, HEAD, and
    the local verification markers under verified/."""
    return sorted(
        relpath
        for relpath in (path.relative_to(store.dir).as_posix() for path in store.dir.rglob("*"))
        if (store.dir / relpath).is_file()
        and relpath not in ("config.json", "HEAD")
        and not relpath.startswith("verified/")
    )


def count(relpaths, kind):
    return sum(relpath.startswith(f"{kind}/") for relpath in relpaths)


def stem(relpath):
    return Path(relpath).name.removesuffix(".json")


def check_order(store, written):
    """Every file comes after everything it refers to; the ref comes last, once."""
    position = {relpath: index for index, relpath in enumerate(written)}
    assert len(position) == len(written), "a file was written twice"
    assert written[-1] == "refs/heads/main"

    def before(relpath, index):
        assert position[relpath] < index, f"{relpath} must precede {written[index]}"

    for index, relpath in enumerate(written[:-1]):
        kind = relpath.split("/")[0]
        if kind == "objects":
            record = store.get_object(stem(relpath))
            before(f"payloads/{record['payload'][:2]}/{record['payload']}", index)
            if "base" in record:
                before(f"objects/{record['base'][:2]}/{record['base']}.json", index)
        elif kind == "manifests":
            for entry in store.get_manifest(stem(relpath))["tensors"].values():
                before(f"objects/{entry['sha256'][:2]}/{entry['sha256']}.json", index)
        elif kind == "commits":
            record = store.get_commit(stem(relpath))
            before(f"manifests/{record['manifest']}.json", index)
            for parent in record["parents"]:
                before(f"commits/{parent}.json", index)
            for entry in record["files"]:
                before(f"payloads/{entry['header'][:2]}/{entry['header']}", index)


def kind_order(relpaths):
    return [KINDS.index(relpath.split("/")[0]) for relpath in relpaths]


class Recording(DirTransport):
    """A directory transport that records each write and each write_many batch."""

    def __init__(self, root) -> None:
        super().__init__(root)
        self.written = []
        self.batches = []
        self.reads = []

    def read(self, relpath, **bounds):
        self.reads.append(relpath)
        return super().read(relpath, **bounds)

    def write(self, relpath, data):
        self.written.append(relpath)
        super().write(relpath, data)

    def write_many(self, pairs):
        self.batches.append([relpath for relpath, _ in pairs])
        super().write_many(pairs)


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    """Every transport_for call in wvc.remote returns this one recording transport."""
    recording = Recording(tmp_path / "recorded")
    monkeypatch.setattr("wvc.remote.transport_for", lambda url: recording)
    return recording


def test_push_then_clone_reproduces_every_commit_byte_for_byte(tmp_path, history):
    store = history.store
    remote_dir = tmp_path / "remotes" / "shared"  # missing: push creates it
    add_remote(store, "origin", str(remote_dir))
    depths = [store.get_object(stem(p))["depth"] for p in store_files(store) if "objects/" in p]
    assert max(depths) == 2  # the third commit stores deltas on deltas
    result = push(store)
    files = store_files(store)
    assert isinstance(result, PushResult)
    assert (result.remote, result.branch, result.previous, result.head) == (
        "origin",
        "main",
        None,
        history.ids[-1],
    )
    assert (result.commits, result.objects, result.payloads) == (
        3,
        count(files, "objects"),
        count(files, "payloads"),
    )
    assert DirTransport(remote_dir).list() == files
    for relpath in files:  # the remote holds the store's files exactly
        assert (remote_dir / relpath).read_bytes() == (store.dir / relpath).read_bytes()
    stored = [relpath for relpath in files if not relpath.startswith("refs/")]
    assert result.nbytes == sum((remote_dir / relpath).stat().st_size for relpath in stored)
    cloned = clone(str(remote_dir), tmp_path / "copy")
    assert isinstance(cloned, PullResult)
    assert cloned.updated
    assert (cloned.previous, cloned.head, cloned.commits) == (None, history.ids[-1], 3)
    assert (cloned.objects, cloned.payloads, cloned.nbytes) == (
        result.objects,
        result.payloads,
        result.nbytes,
    )
    copy = ObjectStore(tmp_path / "copy")
    assert copy.head() == history.ids[-1]
    assert store_files(copy) == files
    assert list_remotes(copy) == {"origin": str(remote_dir.resolve())}
    for commit_id, source in zip(history.ids, history.sources, strict=True):
        out = tmp_path / f"out-{commit_id[:8]}"
        checkout_commit(copy, commit_id, out)
        for name in SHARDS:
            assert (out / name).read_bytes() == (source / name).read_bytes()


def test_push_writes_payloads_objects_manifest_and_commit_then_the_ref(history, recorder):
    store = history.store
    add_remote(store, "origin", str(recorder.root))
    push(store)
    check_order(store, recorder.written)
    assert len(recorder.batches) == 3  # one write_many per commit, then the ref on its own
    for batch in recorder.batches:
        assert kind_order(batch) == sorted(kind_order(batch))
        assert batch[-1].startswith("commits/")
    assert [stem(batch[-1]) for batch in recorder.batches] == history.ids  # parents first


def test_push_bounds_the_bytes_of_each_write_many_call(history, recorder, monkeypatch):
    monkeypatch.setattr("wvc.remote.MAX_BATCH_BYTES", 1)
    add_remote(history.store, "origin", str(recorder.root))
    push(history.store)
    assert all(len(batch) == 1 for batch in recorder.batches)
    check_order(history.store, recorder.written)


def test_a_second_push_transfers_nothing(history, recorder):
    add_remote(history.store, "origin", str(recorder.root))
    push(history.store)
    written = len(recorder.written)
    again = push(history.store)
    assert (again.commits, again.objects, again.payloads, again.nbytes) == (0, 0, 0, 0)
    assert (again.previous, again.head, again.updated) == (history.ids[-1], history.ids[-1], False)
    assert len(recorder.written) == written


def test_pull_into_a_repository_holding_the_first_commit_transfers_only_the_rest(
    tmp_path, revisions, third
):
    origin = ObjectStore(init_repository(tmp_path / "origin"))
    first = commit(origin, revisions.base, "base").commit_id
    remote_dir = tmp_path / "remote"
    add_remote(origin, "origin", str(remote_dir))
    push(origin)
    clone(str(remote_dir), tmp_path / "copy")
    copy = ObjectStore(tmp_path / "copy")
    before = set(store_files(origin))
    commit(origin, revisions.edited, "edit")
    commit(origin, third, "again")
    push(origin)
    new = sorted(set(store_files(origin)) - before)
    result = pull(copy)
    assert (result.previous, result.head, result.commits) == (first, origin.head(), 2)
    assert (result.objects, result.payloads) == (count(new, "objects"), count(new, "payloads"))
    assert result.nbytes == sum((origin.dir / relpath).stat().st_size for relpath in new)
    assert copy.head() == origin.head()
    assert store_files(copy) == store_files(origin)
    again = pull(copy)
    assert (again.commits, again.nbytes, again.updated) == (0, 0, False)


def test_pull_reads_nothing_it_already_holds(tmp_path, revisions, third, monkeypatch):
    origin = ObjectStore(init_repository(tmp_path / "origin"))
    first = commit(origin, revisions.base, "base").commit_id
    add_remote(origin, "origin", str(tmp_path / "remote"))
    push(origin)
    clone(str(tmp_path / "remote"), tmp_path / "copy")
    held = set(store_files(ObjectStore(tmp_path / "copy"))) - {"refs/heads/main"}
    commit(origin, revisions.edited, "edit")
    push(origin)
    recording = Recording(tmp_path / "remote")
    monkeypatch.setattr("wvc.remote.transport_for", lambda url: recording)
    pull(ObjectStore(tmp_path / "copy"))
    assert held.isdisjoint(recording.reads)
    assert f"commits/{first}.json" not in recording.reads


def test_divergent_histories_are_refused_on_push_and_pull(tmp_path, revisions, third):
    a = ObjectStore(init_repository(tmp_path / "a"))
    commit(a, revisions.base, "base")
    remote_dir = tmp_path / "remote"
    add_remote(a, "origin", str(remote_dir))
    push(a)
    clone(str(remote_dir), tmp_path / "b")
    b = ObjectStore(tmp_path / "b")
    a_head = commit(a, revisions.edited, "a's edit").commit_id
    push(a)
    b_head = commit(b, third, "b's edit").commit_id
    before = store_files(b)
    with pytest.raises(StoreError, match="not an ancestor"):
        push(b)
    assert (
        DirTransport(remote_dir).read("refs/heads/main", limit=REF_LIMIT) == f"{a_head}\n".encode()
    )
    assert not DirTransport(remote_dir).exists(f"commits/{b_head}.json")
    with pytest.raises(StoreError, match="diverge"):
        pull(b)
    assert b.head() == b_head
    assert not b.has_commit(a_head)
    assert store_files(b) == before


def test_push_refuses_when_the_remote_is_ahead_and_pull_keeps_a_local_branch_that_is(
    tmp_path, revisions, third
):
    a = ObjectStore(init_repository(tmp_path / "a"))
    commit(a, revisions.base, "base")
    add_remote(a, "origin", str(tmp_path / "remote"))
    push(a)
    clone(str(tmp_path / "remote"), tmp_path / "b")
    b = ObjectStore(tmp_path / "b")
    pushed = commit(a, revisions.edited, "edit").commit_id
    push(a)
    with pytest.raises(StoreError, match="remote is ahead"):
        push(b)
    ahead = commit(a, third, "not pushed yet").commit_id
    result = pull(a)  # the local branch already contains the remote's head
    assert (result.previous, result.head, result.updated, result.commits) == (
        ahead,
        ahead,
        False,
        0,
    )
    assert a.head() == ahead
    assert pull(b).head == pushed


@pytest.mark.parametrize("target", ["header", "delta"])
def test_a_tampered_remote_payload_stops_pull_before_anything_refers_to_it(
    tmp_path, history, target
):
    store = history.store
    remote_dir = tmp_path / "remote"
    add_remote(store, "origin", str(remote_dir))
    push(store)
    second = store.get_commit(history.ids[1])
    if target == "header":
        pid = second["files"][1]["header"]  # shard 2's header changed in the second commit
    else:
        pid = store.get_object(second["files"][0]["tensors"][0])["payload"]
    assert pid not in {entry["header"] for entry in store.get_commit(history.ids[0])["files"]}
    tampered = remote_dir / "payloads" / pid[:2] / pid
    data = tampered.read_bytes()  # same length: a wrong size is refused before hashing
    tampered.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(IntegrityError, match="received bytes hash to"):
        clone(str(remote_dir), tmp_path / "copy")
    copy = ObjectStore(tmp_path / "copy")
    assert not copy.has_payload(pid)
    assert copy.head() is None
    objects = (copy.dir / "objects").rglob("*.json")
    assert all(json.loads(path.read_text())["payload"] != pid for path in objects)
    assert copy.has_commit(history.ids[0])  # pulled completely before the bad payload
    assert not copy.has_commit(history.ids[1])
    assert not copy.has_commit(history.ids[2])
    assert not copy.has_manifest(second["manifest"])
    for path in (copy.dir / "commits").glob("*.json"):  # no stored commit names the header
        files = json.loads(path.read_text())["files"]
        assert all(entry["header"] != pid for entry in files)


def test_pull_refuses_an_object_that_does_not_decode_before_storing_it(tmp_path, history):
    """Second-pass M3: a record pointing at junk that hashes to its own name was stored, and
    a later commit of the tensor reused it. Pull now decodes each object before its record."""
    store = history.store
    remote_dir = tmp_path / "remote"
    add_remote(store, "origin", str(remote_dir))
    push(store)
    tensor_id = store.get_commit(history.ids[0])["files"][0]["tensors"][0]
    record = store.get_object(tensor_id)
    junk = bytes(expected_payload_size(record))  # the right size, but no zstd frame
    junk_pid = hashlib.sha256(junk).hexdigest()
    (remote_dir / "payloads" / junk_pid[:2]).mkdir(exist_ok=True)
    (remote_dir / "payloads" / junk_pid[:2] / junk_pid).write_bytes(junk)
    poisoned = json.dumps({**record, "payload": junk_pid})
    (remote_dir / "objects" / tensor_id[:2] / f"{tensor_id}.json").write_text(poisoned)
    marker = remote_dir / "verified" / tensor_id[:2] / tensor_id
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"")  # a received store's markers vouch for nothing
    with pytest.raises(IntegrityError, match=f"object {tensor_id}: "):
        clone(str(remote_dir), tmp_path / "copy")
    copy = ObjectStore(tmp_path / "copy")
    assert copy.has_payload(junk_pid)  # received and hashed, but nothing refers to it
    assert not copy.has_object(tensor_id)
    assert not copy.is_verified(tensor_id)
    assert copy.head() is None
    assert not any((copy.dir / "commits").iterdir())


def test_verification_markers_are_neither_pushed_nor_pulled(tmp_path, history, recorder):
    store = history.store
    local = {p.name for p in (store.dir / "verified").glob("*/*")}
    assert local == {stem(p) for p in store_files(store) if p.startswith("objects/")}
    add_remote(store, "origin", str(tmp_path / "unused"))  # the recorder stands in
    push(store)
    sent = [*recorder.written, *chain.from_iterable(recorder.batches)]
    assert sent
    assert not any(relpath.startswith("verified") for relpath in sent)
    assert not (recorder.root / "verified").exists()
    bogus = "ab" * 32
    (recorder.root / "verified" / "ab").mkdir(parents=True)
    (recorder.root / "verified" / "ab" / bogus).write_bytes(b"")
    clone(str(tmp_path / "unused"), tmp_path / "copy")
    assert not any(relpath.startswith("verified") for relpath in recorder.reads)
    copy = ObjectStore(tmp_path / "copy")
    assert not copy.is_verified(bogus)
    assert {p.name for p in (copy.dir / "verified").glob("*/*")} == local  # each decoded here


def _object_relpath(store, commit_id):
    """The object record of the second commit's first tensor (a delta stored by it)."""
    tensor_id = store.get_commit(commit_id)["files"][0]["tensors"][0]
    return f"objects/{tensor_id[:2]}/{tensor_id}.json"


def _commit_relpath(store, commit_id):
    return f"commits/{commit_id}.json"


def _manifest_relpath(store, commit_id):
    return f"manifests/{store.get_commit(commit_id)['manifest']}.json"


def _restate(key, value):
    """A rewrite of a JSON file that gives ``key`` another value."""
    return lambda data: json.dumps({**json.loads(data), key: value}).encode()


@pytest.mark.parametrize(
    ("relpath_of", "rewrite", "message"),
    [
        (_commit_relpath, lambda data: b'{"format": 1}', "hashes to"),
        (_commit_relpath, lambda data: b'{"format": 1.5}', "forbids floats"),
        (_commit_relpath, lambda data: b"\xff", "not UTF-8 JSON"),
        (_object_relpath, _restate("id", "0" * 64), "describes"),
        (_manifest_relpath, lambda data: b"[]", "JSON object"),
        (_manifest_relpath, _restate("root", "0" * 64), "the file states"),
    ],
    ids=["commit-id", "commit-float", "commit-bytes", "object-id", "manifest", "manifest-root"],
)
def test_tampered_remote_records_are_refused(tmp_path, history, relpath_of, rewrite, message):
    store = history.store
    remote_dir = tmp_path / "remote"
    add_remote(store, "origin", str(remote_dir))
    push(store)
    target = remote_dir / relpath_of(store, history.ids[1])
    target.write_bytes(rewrite(target.read_bytes()))
    with pytest.raises(IntegrityError, match=message):
        clone(str(remote_dir), tmp_path / "copy")
    copy = ObjectStore(tmp_path / "copy")
    assert copy.head() is None
    assert not copy.has_commit(history.ids[1])


def test_a_delta_chain_cycle_on_the_remote_is_refused(tmp_path, history):
    store = history.store
    remote_dir = tmp_path / "remote"
    add_remote(store, "origin", str(remote_dir))
    push(store)
    delta = store.get_object(store.get_commit(history.ids[1])["files"][0]["tensors"][0])
    base_id = delta["base"]
    looped = {**delta, "id": base_id, "base": delta["id"]}  # the base now rests on its delta
    (remote_dir / "objects" / base_id[:2] / f"{base_id}.json").write_text(json.dumps(looped))
    with pytest.raises(IntegrityError, match="revisits"):
        clone(str(remote_dir), tmp_path / "copy")
    assert not ObjectStore(tmp_path / "copy").has_object(base_id)


def test_histories_with_merges_send_each_commit_once_parents_first(
    tmp_path, revisions, third, recorder
):
    store = ObjectStore(init_repository(tmp_path / "origin"))
    root = commit(store, revisions.base, "base").commit_id
    left = commit(store, revisions.edited, "left").commit_id
    right = commit(
        store,
        third,
        "right",
        parent_ref=root,
        branch="right",  # H3: a parent other than main's head needs a branch of its own
    ).commit_id
    store.set_head("main")
    merged = {**store.get_commit(right), "parents": [left, right], "message": "merge"}
    merge = store.put_commit({**merged, "operation": {"kind": "model-merge"}})
    store.write_ref("main", merge)
    add_remote(store, "origin", str(recorder.root))
    assert push(store).commits == 4
    check_order(store, recorder.written)
    assert [stem(batch[-1]) for batch in recorder.batches] == [root, left, right, merge]
    cloned = clone(str(recorder.root), tmp_path / "copy")
    copy = ObjectStore(tmp_path / "copy")
    origin_files = [path for path in store_files(store) if path != "refs/heads/right"]
    assert cloned.commits == 4
    assert store_files(copy) == origin_files  # main only
    checkout_commit(copy, merge, tmp_path / "out")
    for name in SHARDS:
        assert (tmp_path / "out" / name).read_bytes() == (third / name).read_bytes()


def test_a_malformed_remote_ref_is_refused(tmp_path, history):
    add_remote(history.store, "origin", str(tmp_path / "remote"))
    push(history.store)
    (tmp_path / "remote" / "refs" / "heads" / "main").write_text("not a commit\n", newline="\n")
    with pytest.raises(StoreError, match="malformed ref"):
        push(history.store)
    with pytest.raises(StoreError, match="malformed ref"):
        clone(str(tmp_path / "remote"), tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


def test_push_and_pull_name_branches_and_refuse_what_is_missing(tmp_path, history):
    store = history.store
    remote_dir = tmp_path / "remote"
    with pytest.raises(StoreError, match="no remote named 'origin'"):
        push(store)
    add_remote(store, "origin", str(remote_dir))
    with pytest.raises(StoreError, match="no commits to push"):
        push(store, "origin", "dev")
    with pytest.raises(StoreError, match="invalid branch name"):
        push(store, "origin", "../main")
    with pytest.raises(StoreError, match="has no branch 'main'"):
        pull(store)
    with pytest.raises(StoreError, match="has no branch 'main' to clone"):
        clone(str(remote_dir), tmp_path / "copy")
    assert not (tmp_path / "copy").exists()
    store.write_ref("dev", history.ids[0])
    assert push(store, "origin", "dev").head == history.ids[0]
    assert DirTransport(remote_dir).list("refs/") == ["refs/heads/dev"]
    copy = ObjectStore(init_repository(tmp_path / "copy"))
    add_remote(copy, "origin", str(remote_dir))
    assert pull(copy, "origin", "dev").commits == 1
    assert copy.read_ref("dev") == history.ids[0]
    assert copy.read_ref("main") is None


def test_pull_from_another_repositorys_store(tmp_path, history):
    """A directory remote can be another repository's .wvc directory."""
    copy = ObjectStore(init_repository(tmp_path / "copy"))
    add_remote(copy, "upstream", str(history.store.dir))
    result = pull(copy, "upstream")
    assert (result.commits, result.head) == (3, history.ids[-1])
    assert store_files(copy) == store_files(history.store)


def test_remotes_are_recorded_in_the_config(tmp_path, monkeypatch):
    store = ObjectStore(init_repository(tmp_path / "repo"))
    assert list_remotes(store) == {}
    monkeypatch.chdir(tmp_path)
    stored = add_remote(store, "origin", "relative/dir")
    assert stored == str((tmp_path / "relative" / "dir").resolve())
    assert add_remote(store, "hub", "hf://owner/name@dev") == "hf://owner/name@dev"
    assert list_remotes(store) == {"hub": "hf://owner/name@dev", "origin": stored}
    assert list(list_remotes(store)) == ["hub", "origin"]
    assert remote_url(store, "origin") == stored
    config = json.loads((store.dir / "config.json").read_text(encoding="utf-8"))
    assert config == {
        "format": "wvc-repo/0",
        "max_delta_depth": 8,
        "max_tensor_bytes": 1 << 34,
        "remotes": {"hub": {"url": "hf://owner/name@dev"}, "origin": {"url": stored}},
    }
    assert store.config() == config  # the remotes key is tolerated


@pytest.mark.parametrize(
    ("name", "url", "message"),
    [
        ("origin", "/elsewhere", "already exists"),
        ("bad name", "/x", "invalid remote name"),
        ("-x", "/x", "invalid remote name"),
        ("a/b", "/x", "invalid remote name"),
        ("hub", "s3://bucket/key", "unsupported remote URL"),
        ("hub", "file:///tmp/x", "unsupported remote URL"),
        ("hub", "", "non-empty"),
        ("hub", "hf://gpt2", "OWNER/NAME"),
        ("hub", "hf://owner/name@", "OWNER/NAME"),
    ],
)
def test_add_remote_refusals(tmp_path, name, url, message):
    store = ObjectStore(init_repository(tmp_path / "repo"))
    add_remote(store, "origin", str(tmp_path / "remote"))
    before = (store.dir / "config.json").read_bytes()
    with pytest.raises(StoreError, match=message):
        add_remote(store, name, url)
    assert (store.dir / "config.json").read_bytes() == before


def test_config_validation_covers_remotes_and_writes(tmp_path):
    store = ObjectStore(init_repository(tmp_path / "repo"))
    before = (store.dir / "config.json").read_bytes()
    with pytest.raises(StoreError, match="expected format"):
        store.write_config({"format": "wvc-repo/9", "max_delta_depth": 8})
    with pytest.raises(StoreError, match="expected format"):
        store.write_config(["not", "a", "config"])
    assert (store.dir / "config.json").read_bytes() == before
    config = {"format": "wvc-repo/0", "max_delta_depth": 8, "remotes": {"origin": "/x"}}
    store.write_config(config)
    assert store.config() == config
    with pytest.raises(StoreError, match="remotes must map names"):
        list_remotes(store)


def test_push_and_pull_through_the_hub_transport(tmp_path, history, fake_hub):
    store = history.store
    add_remote(store, "hub", "hf://owner/name")
    result = push(store, "hub")
    assert (result.commits, result.head) == (3, history.ids[-1])
    # One Hub commit per wvc commit (payloads, objects, manifest, commit), then the ref.
    assert len(fake_hub.commits) == 4
    assert fake_hub.commits[-1] == ["wvc/refs/heads/main"]
    paths = [path for operations in fake_hub.commits for path in operations]
    assert all(path.startswith("wvc/") for path in paths)
    written = [path.removeprefix("wvc/") for path in paths]
    assert sorted(written) == store_files(store)
    check_order(store, written)
    for operations in fake_hub.commits[:3]:
        kinds = kind_order([path.removeprefix("wvc/") for path in operations])
        assert kinds == sorted(kinds)
    assert {path.split("/")[1] for path in fake_hub.commits[0]} == set(KINDS[:4])
    for relpath in store_files(store):
        assert fake_hub.files[f"wvc/{relpath}"] == (store.dir / relpath).read_bytes()
    again = push(store, "hub")
    assert (again.commits, again.nbytes, len(fake_hub.commits)) == (0, 0, 4)
    cloned = clone("hf://owner/name", tmp_path / "from-hub")
    assert (cloned.commits, cloned.objects, cloned.payloads, cloned.nbytes) == (
        3,
        result.objects,
        result.payloads,
        result.nbytes,
    )
    copy = ObjectStore(tmp_path / "from-hub")
    assert list_remotes(copy) == {"origin": "hf://owner/name"}
    for commit_id, source in zip(history.ids, history.sources, strict=True):
        out = tmp_path / f"hub-out-{commit_id[:8]}"
        checkout_commit(copy, commit_id, out)
        for name in SHARDS:
            assert (out / name).read_bytes() == (source / name).read_bytes()
    assert fake_hub.listings == 3  # one per transport: two pushes and the clone


# The command line


def run(capsys, *argv):
    """Run the CLI; return its exit code, stdout, and stderr."""
    code = main([str(arg) for arg in argv])
    out, err = capsys.readouterr()
    return code, out, err


TRANSFER = r"(pushed|pulled) main (to|from) origin: {}, [\d,]+ bytes in \d+\.\d\d s"


def test_cli_remote_push_clone_and_pull(tmp_path, revisions, capsys, monkeypatch):
    monkeypatch.setenv("WVC_AUTHOR", "Evan")
    repo, remote_dir, copy = tmp_path / "repo", tmp_path / "remote", tmp_path / "copy"
    run(capsys, "init", repo)
    run(capsys, "commit", revisions.base, "-m", "base", "--repo", repo)
    first = ObjectStore(repo).head()
    code, out, err = run(capsys, "remote", "add", "origin", remote_dir, "--repo", repo)
    assert (code, out, err) == (0, f"added remote origin: {remote_dir.resolve()}\n", "")
    run(capsys, "remote", "add", "hub", "hf://owner/name", "--repo", repo)
    code, out, _ = run(capsys, "remote", "list", "--repo", repo)
    assert (code, out) == (0, f"hub     hf://owner/name\norigin  {remote_dir.resolve()}\n")
    code, out, _ = run(capsys, "push", "--repo", repo)
    lines = out.splitlines()
    assert code == 0
    assert re.fullmatch(TRANSFER.format("1 commit, 4 objects, 6 payloads"), lines[0])
    assert lines[1:] == [f"origin main: none -> {first[:12]}"]
    code, out, _ = run(capsys, "clone", remote_dir, copy)
    lines = out.splitlines()
    assert code == 0
    assert lines[0] == f"cloned {remote_dir} into {copy}"
    assert re.fullmatch(TRANSFER.format("1 commit, 4 objects, 6 payloads"), lines[1])
    assert lines[2:] == [f"main: none -> {first[:12]}"]
    run(capsys, "commit", revisions.edited, "-m", "edit", "--repo", repo)
    second = ObjectStore(repo).head()
    code, out, _ = run(capsys, "push", "origin", "main", "--repo", repo)
    lines = out.splitlines()
    assert re.fullmatch(TRANSFER.format("1 commit, 3 objects, 4 payloads"), lines[0])
    assert lines[1:] == [f"origin main: {first[:12]} -> {second[:12]}"]
    code, out, _ = run(capsys, "pull", "--repo", copy)
    lines = out.splitlines()
    assert re.fullmatch(TRANSFER.format("1 commit, 3 objects, 4 payloads"), lines[0])
    assert lines[1:] == [f"main: {first[:12]} -> {second[:12]}"]
    code, out, _ = run(capsys, "pull", "origin", "--repo", copy)
    lines = out.splitlines()
    assert re.fullmatch(TRANSFER.format("0 commits, 0 objects, 0 payloads"), lines[0])
    assert lines[1:] == [f"main: up to date at {second[:12]}"]
    assert " 0 bytes " in lines[0]


def test_cli_remote_refusals_exit_2(tmp_path, revisions, capsys, monkeypatch):
    monkeypatch.setenv("WVC_AUTHOR", "Evan")
    repo = tmp_path / "repo"
    run(capsys, "init", repo)

    def refused(*argv):
        code, out, err = run(capsys, *argv)
        assert (code, out) == (2, ""), (argv, code, out, err)
        assert err.startswith("wvc: "), (argv, code, out, err)
        return err

    assert "no remote named 'origin'" in refused("push", "--repo", repo)
    assert "no remote named 'up'" in refused("pull", "up", "--repo", repo)
    run(capsys, "remote", "add", "origin", tmp_path / "remote", "--repo", repo)
    assert "already exists" in refused("remote", "add", "origin", tmp_path, "--repo", repo)
    assert "unsupported remote URL" in refused("remote", "add", "s3", "s3://b/k", "--repo", repo)
    assert "no commits to push" in refused("push", "--repo", repo)
    assert "has no branch 'main'" in refused("pull", "--repo", repo)
    assert "to clone" in refused("clone", tmp_path / "remote", tmp_path / "copy")
    assert not (tmp_path / "copy").exists()
    run(capsys, "commit", revisions.base, "-m", "base", "--repo", repo)
    assert run(capsys, "push", "--repo", repo)[0] == 0
    assert "already exists" in refused("clone", tmp_path / "remote", repo)
    assert "not a wvc repository" in refused("remote", "list", "--repo", tmp_path)


def test_cli_remote_list_escapes_names_and_urls(tmp_path, capsys):
    """S-L1: a hand-edited config.json cannot send ESC or BEL to the terminal."""
    store = ObjectStore(init_repository(tmp_path / "repo"))
    config = store.config()
    config["remotes"] = {
        "evil\x1b]0;x\x07": {"url": "/srv/\x1b[31mred\x07"},
        "origin": {"url": "/srv/plain"},
    }
    (store.dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    code, out, err = run(capsys, "remote", "list", "--repo", tmp_path / "repo")
    assert (code, err) == (0, "")
    assert "\x1b" not in out
    assert "\x07" not in out
    name = "evil\\x1b]0;x\\x07"
    assert out == f"{name}  /srv/\\x1b[31mred\\x07\n{'origin'.ljust(len(name))}  /srv/plain\n"


def test_cli_hf_remotes_without_huggingface_hub_name_the_extra(
    tmp_path, revisions, capsys, monkeypatch
):
    monkeypatch.setenv("WVC_AUTHOR", "Evan")
    monkeypatch.setattr("wvc.transport.make_hub_client", REAL_MAKE_HUB_CLIENT)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # the import now fails
    repo = tmp_path / "repo"
    run(capsys, "init", repo)
    run(capsys, "commit", revisions.base, "-m", "base", "--repo", repo)
    assert run(capsys, "remote", "add", "hub", "hf://owner/name", "--repo", repo)[0] == 0
    for argv in (("push", "hub", "--repo", repo), ("clone", "hf://owner/name", tmp_path / "c")):
        code, out, err = run(capsys, *argv)
        assert (code, out) == (2, "")
        assert "install wvc[hf]" in err
    assert not (tmp_path / "c").exists()


def test_pull_counts_delta_depth_in_the_receiving_store(tmp_path, revisions, history):
    """A tensor stored as a delta on the remote can be stored full here: depth is local."""
    add_remote(history.store, "origin", str(tmp_path / "remote"))
    push(history.store)
    local = ObjectStore(init_repository(tmp_path / "local"))
    (local.dir / "HEAD").write_text("ref: refs/heads/dev\n", encoding="utf-8", newline="\n")
    commit(local, revisions.edited, "the edited checkpoint on its own")  # stored full here
    (local.dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8", newline="\n")
    add_remote(local, "origin", str(tmp_path / "remote"))
    result = pull(local)
    assert (result.previous, result.head, result.commits) == (None, history.ids[-1], 3)
    third = local.get_commit(history.ids[-1])
    for tensor_id in third["files"][0]["tensors"]:  # deltas on tensors stored full here
        record = local.get_object(tensor_id)
        assert record["depth"] == local.get_object(record["base"])["depth"] + 1 == 1
    out = tmp_path / "out"
    checkout_commit(local, history.ids[-1], out)
    for name in SHARDS:
        assert (out / name).read_bytes() == (history.sources[-1] / name).read_bytes()


def test_pull_does_not_overwrite_a_branch_that_moved_meanwhile(
    tmp_path, revisions, third, monkeypatch
):
    """H4: pull fast-forwards the local branch only from the commit it read at the start."""
    origin = ObjectStore(init_repository(tmp_path / "origin"))
    commit(origin, revisions.base, "base")
    remote_dir = tmp_path / "remote"
    add_remote(origin, "origin", str(remote_dir))
    push(origin)
    clone(str(remote_dir), tmp_path / "copy")
    copy = ObjectStore(tmp_path / "copy")
    commit(origin, revisions.edited, "edit")
    push(origin)
    real = remote_module._pull_commit
    local = {}

    def pull_while_committing(store, remote, commit_id, record):
        moved = real(store, remote, commit_id, record)
        if not local:
            local["head"] = commit(
                ObjectStore(copy.repo), third, "a local commit meanwhile"
            ).commit_id
        return moved

    monkeypatch.setattr(remote_module, "_pull_commit", pull_while_committing)
    with pytest.raises(StoreError, match="moved to"):
        pull(copy)
    assert copy.read_ref("main") == local["head"]


def test_remote_delta_chains_are_bounded_and_may_not_cycle():
    """S-M4: a received chain is followed through a set of seen ids and at most 64 levels."""
    ids = [f"{index:064x}" for index in range(70)]
    records = {oid: {"id": oid, "base": ids[index + 1]} for index, oid in enumerate(ids[:-1])}
    records[ids[-1]] = {"id": ids[-1]}
    with pytest.raises(IntegrityError, match="over 64 deep"):
        remote_module._chain(ids[0], lambda oid: False, records.__getitem__)
    assert len(remote_module._chain(ids[-60], lambda oid: False, records.__getitem__)) == 60
    looped = {ids[0]: {"id": ids[0], "base": ids[1]}, ids[1]: {"id": ids[1], "base": ids[0]}}
    with pytest.raises(IntegrityError, match="revisits"):
        remote_module._chain(ids[0], lambda oid: False, looped.__getitem__)


def test_pull_refuses_a_commit_whose_files_are_not_its_manifest(tmp_path, history):
    """S-H2 for received stores: the remote's record names A's manifest with M's files."""
    store, (audited, other, _) = history.store, history.ids
    remote_dir = tmp_path / "remote"
    add_remote(store, "origin", str(remote_dir))
    push(store)
    forged = {
        **store.get_commit(audited),
        "parents": [audited],
        "message": "re-save, no weight changes",
        "files": store.get_commit(other)["files"],
    }
    forged_id = object_id(forged)
    (remote_dir / "commits" / f"{forged_id}.json").write_text(json.dumps(forged))
    (remote_dir / "refs" / "heads" / "main").write_text(f"{forged_id}\n", newline="\n")
    with pytest.raises(IntegrityError, match="do not hold the tensors of manifest"):
        clone(str(remote_dir), tmp_path / "copy")
    copy = ObjectStore(tmp_path / "copy")
    assert copy.head() is None
    assert not copy.has_commit(forged_id)


# Bounded reads from hostile remotes (S-H3, C-M1, P-M14)

SPARSE_BYTES = 64 << 20  # an oversized remote file costs no disk: it is sparse


def _delta_payload_relpath(store, commit_id):
    """The payload of the second commit's first tensor (a delta stored by it)."""
    pid = store.get_object(store.get_commit(commit_id)["files"][0]["tensors"][0])["payload"]
    return f"payloads/{pid[:2]}/{pid}"


def _header_relpath(store, commit_id):
    pid = store.get_commit(commit_id)["files"][1]["header"]
    return f"payloads/{pid[:2]}/{pid}"


def _ref_relpath(store, commit_id):
    return "refs/heads/main"


def _make_sparse(path, nbytes):
    """Replace ``path`` with a sparse file of ``nbytes`` zero bytes."""
    path.unlink()
    with path.open("wb") as handle:
        handle.truncate(nbytes)


def _implied(store, commit_id):
    record = store.get_object(store.get_commit(commit_id)["files"][0]["tensors"][0])
    return f"not the {expected_payload_size(record)} its record implies"


@pytest.mark.parametrize(
    ("relpath_of", "nbytes", "message"),
    [
        (_delta_payload_relpath, SPARSE_BYTES, _implied),
        (_header_relpath, MAX_PREFIX_BYTES + 1, f"more than the {MAX_PREFIX_BYTES} allowed"),
        (_object_relpath, SPARSE_BYTES, f"more than the {OBJECT_LIMIT} allowed"),
        (_manifest_relpath, RECORD_LIMIT + 1, f"more than the {RECORD_LIMIT} allowed"),
        (_commit_relpath, RECORD_LIMIT + 1, f"more than the {RECORD_LIMIT} allowed"),
        (_ref_relpath, SPARSE_BYTES, f"more than the {REF_LIMIT} allowed"),
    ],
    ids=["object-payload", "header-payload", "object", "manifest", "commit", "ref"],
)
def test_oversized_remote_files_are_refused_before_they_are_read(
    tmp_path, history, relpath_of, nbytes, message
):
    """The size check comes from fstat: the refusal names the limit, not a hash or a parse."""
    store = history.store
    remote_dir = tmp_path / "remote"
    add_remote(store, "origin", str(remote_dir))
    push(store)
    _make_sparse(remote_dir / relpath_of(store, history.ids[1]), nbytes)
    expected = message(store, history.ids[1]) if callable(message) else message
    with pytest.raises(IntegrityError, match=f"holds {nbytes} bytes, {re.escape(expected)}"):
        clone(str(remote_dir), tmp_path / "copy")
    copy = tmp_path / "copy"
    assert not copy.exists() or ObjectStore(copy).head() is None


def test_a_symlinked_remote_payload_is_refused_even_when_its_target_hashes_right(tmp_path, history):
    store = history.store
    remote_dir = tmp_path / "remote"
    add_remote(store, "origin", str(remote_dir))
    push(store)
    payload = remote_dir / _delta_payload_relpath(store, history.ids[1])
    target = tmp_path / "outside"
    target.write_bytes(payload.read_bytes())
    payload.unlink()
    payload.symlink_to(target)
    with pytest.raises(IntegrityError, match="not a regular file"):
        clone(str(remote_dir), tmp_path / "copy")
    assert not ObjectStore(tmp_path / "copy").has_commit(history.ids[1])


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_a_fifo_remote_payload_is_refused_without_blocking(tmp_path, history):
    store = history.store
    remote_dir = tmp_path / "remote"
    add_remote(store, "origin", str(remote_dir))
    push(store)
    payload = remote_dir / _delta_payload_relpath(store, history.ids[1])
    payload.unlink()
    os.mkfifo(payload)  # no writer: a blocking open or read would hang here
    with pytest.raises(IntegrityError, match="not a regular file"):
        clone(str(remote_dir), tmp_path / "copy")


def test_a_deeply_nested_remote_commit_is_refused_without_a_traceback(tmp_path):
    remote_dir = tmp_path / "remote"
    commit_id = "a" * 64
    (remote_dir / "refs" / "heads").mkdir(parents=True)
    (remote_dir / "commits").mkdir()
    (remote_dir / "refs" / "heads" / "main").write_bytes(f"{commit_id}\n".encode())
    (remote_dir / "commits" / f"{commit_id}.json").write_bytes(b"[" * 100_000)
    with pytest.raises(IntegrityError, match="not UTF-8 JSON"):
        clone(str(remote_dir), tmp_path / "copy")


def test_an_oversized_hub_file_is_refused(tmp_path, fake_hub):
    fake_hub.files["wvc/refs/heads/main"] = b"0" * 64 + b"\n" + b"x" * 4096
    message = f"holds {65 + 4096} bytes, more than the {REF_LIMIT} allowed"
    with pytest.raises(IntegrityError, match=message):
        clone("hf://owner/name", tmp_path / "copy")
    assert not (tmp_path / "copy").exists()
