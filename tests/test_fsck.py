import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import plant_commit, write_safetensors

from wvc.cli import main
from wvc.fsck import fsck
from wvc.repo import commit_checkpoint, init_repository
from wvc.store import ObjectStore


def _two_commits(tmp_path, rng):
    repo = init_repository(tmp_path / "repo")
    store = ObjectStore(repo)
    w = rng.standard_normal((8, 8)).astype(np.float32)
    v = rng.standard_normal(16).astype(np.float32)
    first = tmp_path / "v1"
    first.mkdir()
    write_safetensors(first / "m.safetensors", {"w": ("F32", w), "v": ("F32", v)})
    commit_checkpoint(store, first, message="one", operation_kind="unknown", author="t")
    second = tmp_path / "v2"
    second.mkdir()
    w2 = w.copy()
    w2[0, 0] += 1.0
    write_safetensors(second / "m.safetensors", {"w": ("F32", w2), "v": ("F32", v)})
    commit_checkpoint(store, second, message="two", operation_kind="direct-edit", author="t")
    return store


def test_clean_repository_passes(tmp_path, rng):
    report = fsck(_two_commits(tmp_path, rng))
    assert report.ok
    assert report.commits == 2
    assert report.manifests == 2
    assert report.refs == 1
    assert report.objects == 3
    assert report.tensors_decoded == 3
    assert report.payloads >= 4
    assert fsck(_two_commits(tmp_path / "again", rng), decode=False).tensors_decoded == 0
    assert json.loads(json.dumps(report.to_dict()))["ok"] is True  # serialisable


def test_tampered_payload_is_reported(tmp_path, rng):
    store = _two_commits(tmp_path, rng)
    payload = next(p for p in (store.dir / "payloads").glob("*/*") if p.stat().st_size > 64)
    data = bytearray(payload.read_bytes())
    data[-1] ^= 0xFF
    payload.write_bytes(bytes(data))
    report = fsck(store)
    assert not report.ok
    assert any(p.startswith("payload ") for p in report.problems)


def test_missing_manifest_and_bad_ref_are_reported(tmp_path, rng):
    store = _two_commits(tmp_path, rng)
    next((store.dir / "manifests").glob("*.json")).unlink()
    (store.dir / "refs" / "heads" / "main").write_text("0" * 64 + "\n", newline="\n")
    report = fsck(store, decode=False)
    kinds = {p.split(" ")[0] for p in report.problems}
    assert {"commit", "ref"} <= kinds


def test_corrupt_object_record_is_reported(tmp_path, rng):
    store = _two_commits(tmp_path, rng)
    record = next((store.dir / "objects").glob("*/*.json"))
    record.write_text("{not json")
    report = fsck(store)
    assert any(p.startswith("object ") for p in report.problems)


def test_layout_links_lock_files_and_odd_branches_are_reported(tmp_path, rng):
    """S-H3, H4, S-H5, S-L7: fsck re-runs the layout check and checks every ref file."""
    store = _two_commits(tmp_path, rng)
    heads = store.dir / "refs" / "heads"
    head = (heads / "main").read_text()
    (heads / "main.lock").write_text("")
    (heads / "abcdef0123").write_text(head, newline="\n")
    problems = fsck(store, decode=False).problems
    assert any(p.startswith("ref main.lock: a lock file") for p in problems)
    assert any(p.startswith("ref abcdef0123: the branch is named like a commit") for p in problems)
    for name in ("main.lock", "abcdef0123"):
        (heads / name).unlink()
    assert fsck(store, decode=False).ok
    shard = next((store.dir / "payloads").iterdir())
    shard.rename(tmp_path / "moved")
    shard.symlink_to(tmp_path / "moved", target_is_directory=True)
    report = fsck(ObjectStore(store.repo))
    assert len(report.problems) == 1
    assert "symbolic link" in report.problems[0]


@pytest.mark.skipif(os.name == "nt", reason="Windows cannot name a file with ESC or BEL in it")
def test_cli_fsck_escapes_ref_file_names(tmp_path, rng, capsys):
    """S-L1: a planted ref file name cannot send ESC or BEL to the terminal."""
    store = _two_commits(tmp_path, rng)
    heads = store.dir / "refs" / "heads"
    (heads / "evil\x1b[2J\x07").write_text((heads / "main").read_text(), newline="\n")
    assert main(["fsck", "--no-decode", "--repo", str(store.repo)]) == 1
    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert "\x07" not in out
    assert "\nref evil\\x1b[2J\\x07: " in out


def test_branch_names_equal_up_to_case_are_reported(tmp_path, rng, monkeypatch):
    """S-L7: on a case-sensitive volume a received store can hold main and MAIN."""
    store = _two_commits(tmp_path, rng)
    head, real_listdir = store.head(), os.listdir

    def listdir(directory):
        return ["MAIN", "main"] if Path(directory).name == "heads" else real_listdir(directory)

    monkeypatch.setattr("wvc.fsck.os", SimpleNamespace(listdir=listdir))
    monkeypatch.setattr(ObjectStore, "read_ref", lambda self, name: head)
    problems = fsck(store, decode=False).problems
    assert sorted(problems) == [
        "ref MAIN: another branch has the same name up to case",
        "ref main: another branch has the same name up to case",
    ]


def test_payload_sizes_and_commit_files_are_checked(tmp_path, rng):
    """C3, S-H2: a payload of the wrong length and a commit whose files are not its manifest."""
    store = _two_commits(tmp_path, rng)
    first, second = (store.get_commit(commit_id) for commit_id in _ids(store))
    forged = plant_commit(store, {**first, "files": second["files"], "message": "forged"})
    problems = fsck(store, decode=False).problems
    assert problems == [
        f"commit {forged}: record: the files do not hold the tensors of manifest "
        f"{first['manifest']}"
    ]
    tensor_id = first["files"][0]["tensors"][0]
    pid = store.get_object(tensor_id)["payload"]
    (store.dir / "payloads" / pid[:2] / pid).write_bytes(b"")
    problems = fsck(store, decode=False).problems
    assert any(p.startswith(f"object {tensor_id}: payload {pid} holds 0 bytes") for p in problems)


def _ids(store):
    """The two commit ids of ``_two_commits``, oldest first."""
    head = store.head()
    return [store.get_commit(head)["parents"][0], head]


def test_fsck_marks_what_it_decodes_and_unmarks_what_fails(tmp_path, rng):
    """Second-pass M3: fsck leaves a verification marker on every object it decoded, drops
    the marker of one that fails, and does not count verified/ as part of the layout."""
    store = _two_commits(tmp_path, rng)
    objects = sorted(p.name.removesuffix(".json") for p in (store.dir / "objects").glob("*/*"))
    shutil.rmtree(store.dir / "verified")  # as in a repository made before markers existed
    assert fsck(store, decode=False).ok
    assert not (store.dir / "verified").exists()
    assert fsck(store).ok
    assert all(map(store.is_verified, objects))
    (store.dir / "verified" / "ff").mkdir(exist_ok=True)
    (store.dir / "verified" / "ff" / "not an id").write_bytes(b"stray")
    assert fsck(store).ok  # the marker directory is not checked
    pid = store.get_object(objects[0])["payload"]
    path = store.dir / "payloads" / pid[:2] / pid
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    assert not fsck(store).ok
    assert not store.is_verified(objects[0])
