"""The store trusts no file system object it did not check (security review S-H3, S-H4, S-M3,
S-L6): symlinked store directories, links and special files in place of store files, oversized
files, and repositories of other users found by walking up."""

import hashlib
import json
import os
import re
import shutil

import pytest
from helpers import SAME, commit, write_safetensors

from wvc.card import model_card
from wvc.checkout import checkout_commit
from wvc.fsck import fsck
from wvc.repo import commit_manifest, open_repository
from wvc.store import IntegrityError, ObjectStore, StoreError, find_repo

SECRET = "aws_secret_access_key = FAKE/SECRET/ONLY"  # noqa: S105  a planted fake, never quoted
LINKABLE = ("refs", "refs/heads", "commits", "manifests", "objects", "payloads")


def files_under(directory):
    return sorted(p.relative_to(directory) for p in directory.rglob("*"))


def test_a_symlinked_store_is_refused(store, tmp_path, revisions):
    other = tmp_path / "other"
    other.mkdir()
    (other / ".wvc").symlink_to(store.dir, target_is_directory=True)
    with pytest.raises(StoreError, match="symbolic link"):
        open_repository(other)
    nested = other / "nested"
    nested.mkdir()
    with pytest.raises(StoreError, match="symbolic link"):
        commit(ObjectStore(find_repo(nested)), revisions.base)
    assert files_under(store.dir / "objects") == []


@pytest.mark.parametrize("sub", LINKABLE)
def test_symlinked_store_directories_are_refused(store, tmp_path, revisions, sub):
    """S-H3: nothing is written through a store directory that is a link."""
    victim = tmp_path / "victim"
    shutil.move(store.dir / sub, victim)
    (store.dir / sub).symlink_to(victim, target_is_directory=True)
    before = files_under(victim)
    with pytest.raises(StoreError, match="symbolic link"):
        commit(ObjectStore(store.repo), revisions.base)
    assert files_under(victim) == before


def test_a_linked_refs_directory_cannot_overwrite_a_file_elsewhere(store, tmp_path, revisions):
    """S-H3 as reproduced: HEAD names authorized_keys in a refs/heads that links elsewhere."""
    first = commit(store, revisions.base)
    victim = tmp_path / "ssh"
    victim.mkdir()
    (victim / "authorized_keys").write_text("ssh-ed25519 AAAA mine\n")
    heads = store.dir / "refs" / "heads"
    heads.rename(store.dir / "refs" / "heads.real")
    heads.symlink_to(victim, target_is_directory=True)
    (store.dir / "HEAD").write_text("ref: refs/heads/authorized_keys\n", newline="\n")
    for options in ({"parent_ref": first.commit_id}, {}):
        with pytest.raises(StoreError, match="symbolic link"):
            commit(ObjectStore(store.repo), revisions.edited, **options)
    assert sorted(path.name for path in victim.iterdir()) == ["authorized_keys"]
    assert (victim / "authorized_keys").read_text() == "ssh-ed25519 AAAA mine\n"


def test_a_linked_shard_directory_is_refused_and_reported(store, tmp_path, revisions):
    result = commit(store, revisions.base)
    tensor_id = commit_manifest(store, result.commit_id)["tensors"][SAME]["sha256"]
    shard = store.dir / "objects" / tensor_id[:2]
    victim = tmp_path / "elsewhere"
    shutil.move(shard, victim)
    shard.symlink_to(victim, target_is_directory=True)
    with pytest.raises(StoreError, match="symbolic link"):
        checkout_commit(ObjectStore(store.repo), "HEAD", tmp_path / "out")
    report = fsck(ObjectStore(store.repo))
    assert not report.ok
    assert report.problems[0].startswith("layout: refusing")


def test_store_files_are_never_followed_or_quoted(store, tmp_path, revisions):
    """S-H4: links to credentials are refused, and no message quotes a file's contents."""
    commit(store, revisions.base)
    credentials = tmp_path / "credentials"
    credentials.write_text(f"[default]\n{SECRET}\n")
    keys = tmp_path / "kaggle.json"
    keys.write_text('{"username": "alice", "key": "FAKE0123"}')
    head, ref = store.dir / "HEAD", store.dir / "refs" / "heads" / "main"
    config = store.dir / "config.json"
    cases = [
        (head, credentials, lambda s: s.current_branch()),
        (ref, credentials, lambda s: s.read_ref("main")),
        (config, keys, lambda s: s.config()),
    ]
    for path, target, read in cases:
        saved = path.read_bytes()
        path.unlink()
        path.symlink_to(target)
        with pytest.raises(IntegrityError, match="not a regular file") as raised:
            read(ObjectStore(store.repo))
        assert "FAKE" not in str(raised.value)
        path.unlink()
        path.write_bytes(target.read_bytes()[:60])  # a regular file holding a secret
        with pytest.raises(StoreError) as raised:
            read(ObjectStore(store.repo))
        assert "FAKE" not in str(raised.value)
        assert "alice" not in str(raised.value)
        path.write_bytes(saved)


def assert_brief(message, letter):
    """H2: a message cuts a 5,000-character received value rather than echoing it."""
    assert len(message) < 200, message
    assert "…" in message
    assert "(5000 characters)" in message
    assert letter * 100 not in message


@pytest.mark.parametrize("field", ["dtype", "name"])
def test_a_stored_manifest_with_a_long_value_is_refused_briefly(store, revisions, field):
    """H2: get_manifest validates stored manifests; a 5,000-character value is not quoted."""
    root = commit_manifest(store, commit(store, revisions.base).commit_id)["root"]
    path = store.dir / "manifests" / f"{root}.json"
    manifest = json.loads(path.read_bytes())
    name, entry = sorted(manifest["tensors"].items())[0]
    del manifest["tensors"][name]
    if field == "dtype":
        manifest["tensors"][name] = {**entry, "dtype": "D" * 5000}
    else:  # a long name, refused for its entry's sha256
        manifest["tensors"]["D" * 5000] = {**entry, "sha256": "not hex"}
    path.write_text(json.dumps(manifest))
    with pytest.raises(StoreError) as raised:
        ObjectStore(store.repo).get_manifest(root)
    assert_brief(str(raised.value), "D")


def test_card_refuses_a_long_source_repo_briefly(store, revisions):
    """H2: a stored commit's source.repo is named by its key in the card error, not quoted."""
    record = store.get_commit(commit(store, revisions.base).commit_id)
    source = {"kind": "hf", "repo": "R" * 5000, "revision": "main"}
    forged = store.put_commit({**record, "message": "forged", "source": source})
    with pytest.raises(StoreError, match=re.escape("source.repo is not OWNER/NAME")) as raised:
        model_card(store, forged)
    assert len(str(raised.value)) < 200
    assert "R" * 100 not in str(raised.value)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_special_files_are_refused_without_blocking(store, revisions):
    """S-M3: a FIFO in place of HEAD, a record, or a payload must not hang wvc."""
    result = commit(store, revisions.base)
    record = store.get_commit(result.commit_id)
    tensor_id = record["files"][0]["tensors"][0]
    pid = store.get_object(tensor_id)["payload"]
    paths = {
        "HEAD": store.dir / "HEAD",
        "object": store.dir / "objects" / tensor_id[:2] / f"{tensor_id}.json",
        "payload": store.dir / "payloads" / pid[:2] / pid,
    }
    reads = {
        "HEAD": lambda s: s.current_branch(),
        "object": lambda s: s.get_object(tensor_id),
        "payload": lambda s: s.get_payload(pid),
    }
    for name, path in paths.items():
        saved = path.read_bytes()
        path.unlink()
        os.mkfifo(path)
        with pytest.raises(IntegrityError, match="not a regular file"):
            reads[name](ObjectStore(store.repo))
        path.unlink()
        path.write_bytes(saved)


def test_files_are_refused_by_size_before_they_are_read(store, revisions):
    """S-M3: HEAD, refs, records, and payloads have caps checked before reading."""
    result = commit(store, revisions.base)
    tensor_id = store.get_commit(result.commit_id)["files"][0]["tensors"][0]
    record = store.get_object(tensor_id)
    head = store.dir / "HEAD"
    head.write_text("ref: refs/heads/main\n" + " " * 200, newline="\n")
    with pytest.raises(IntegrityError, match="more than the 128 allowed"):
        ObjectStore(store.repo).current_branch()
    head.write_text("ref: refs/heads/main\n", newline="\n")
    ref = store.dir / "refs" / "heads" / "main"
    ref.write_text(f"{result.commit_id}\n" + " " * 40, newline="\n")
    with pytest.raises(IntegrityError, match="more than the 65 allowed"):
        ObjectStore(store.repo).read_ref("main")
    ref.write_text(f"{result.commit_id}\n", newline="\n")
    object_file = store.dir / "objects" / tensor_id[:2] / f"{tensor_id}.json"
    object_file.write_bytes(b" " * (1 << 20) + b"{}")
    with pytest.raises(IntegrityError, match="more than the 1048576 allowed"):
        ObjectStore(store.repo).get_object(tensor_id)
    object_file.write_text(json.dumps(record))
    payload = store.dir / "payloads" / record["payload"][:2] / record["payload"]
    os.truncate(payload, 1 << 31)  # a sparse 2 GiB file: refused by its size, not read
    with pytest.raises(IntegrityError, match="holds 2147483648 bytes, not the"):
        ObjectStore(store.repo).read_tensor(tensor_id)


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="needs POSIX uids")
def test_a_discovered_repository_of_another_user_is_refused(store, revisions, monkeypatch):
    """S-L6: walking up finds only repositories the current user owns; --repo names others."""
    nested = store.repo / "alice-work"
    nested.mkdir()
    owner = os.lstat(store.dir).st_uid
    monkeypatch.setattr(os, "getuid", lambda: owner + 1)
    with pytest.raises(StoreError, match="belongs to another user"):
        find_repo(nested)
    monkeypatch.chdir(nested)
    with pytest.raises(StoreError, match="--repo"):
        open_repository()
    assert open_repository(store.repo).dir == store.dir  # named explicitly


def test_a_linked_marker_directory_is_neither_read_nor_written(store, tmp_path, rng):
    """Second-pass M3: verification markers are hints; a verified/ that is a link vouches for
    nothing, gets nothing written through it, and does not stop a commit."""
    w = rng.standard_normal(16).astype("<f4")
    tensor_id = hashlib.sha256(w.tobytes()).hexdigest()
    junk = store.put_payload(b"\0" * 16)
    store.put_object(
        {
            "format": "wvc-object/0",
            "id": tensor_id,
            "nbytes": 64,
            "itemsize": 4,
            "form": "zstd-planes",
            "payload": junk,
            "depth": 0,
            "plane_sizes": [4, 4, 4, 4],
        }
    )
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / tensor_id[:2]).mkdir(parents=True)
    (elsewhere / tensor_id[:2] / tensor_id).write_bytes(b"")  # would vouch for the junk
    verified = store.dir / "verified"
    verified.symlink_to(elsewhere, target_is_directory=True)
    before = files_under(elsewhere)
    assert not store.is_verified(tensor_id)
    source = tmp_path / "ckpt"
    source.mkdir()
    write_safetensors(source / "m.safetensors", {"w": ("F32", w)})
    result = commit(store, source)
    assert (result.reused, result.stored_full) == (0, 1)
    assert files_under(elsewhere) == before
    assert fsck(store).ok
    assert store.get_object(tensor_id)["payload"] != junk
    assert not verified.is_symlink()  # the link was removed
    assert store.is_verified(tensor_id)
