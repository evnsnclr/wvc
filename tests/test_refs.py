"""Branches, HEAD, and REF resolution (wvc.refs, through ObjectStore)."""

import numpy as np
import pytest
from helpers import store_checkpoint_files, write_safetensors

from wvc.store import StoreError


@pytest.fixture
def commits(store, tmp_path, rng):
    """Two stored commits, the second a child of the first; no branch points at either."""
    tensors = {"w": ("F32", rng.standard_normal(4).astype(np.float32))}
    root, files = store_checkpoint_files(
        store, write_safetensors(tmp_path / "m.safetensors", tensors)
    )
    record = {
        "format": "wvc-commit/0",
        "manifest": root,
        "parents": [],
        "operation": {"kind": "finetune"},
        "message": "first",
        "author": "tester",
        "time": "2026-11-02T12:00:00Z",
        "files": files,
    }
    first = store.put_commit(record)
    second = store.put_commit({**record, "parents": [first], "message": "second"})
    return first, second


def ref_path(store, name):
    return store.dir / "refs" / "heads" / name


def test_refs_head_and_resolve(store, commits):
    first, second = commits
    assert store.read_ref("main") is None
    with pytest.raises(StoreError, match="no commits yet"):
        store.resolve("HEAD")
    with pytest.raises(StoreError, match="branch 'main' has no commits yet"):  # L5
        store.resolve("main")
    store.write_ref("main", first)
    store.write_ref("dev", second)
    assert ref_path(store, "main").read_text(encoding="utf-8") == f"{first}\n"
    assert (store.head(), store.current_branch()) == (first, "main")
    assert store.resolve("HEAD") == store.resolve("main") == first
    assert store.resolve("dev") == store.resolve(second) == store.resolve(second[:7]) == second
    assert store.resolve(second[:9].upper()) == store.resolve(second.upper()) == second  # L5
    with pytest.raises(StoreError, match="no commit matches"):
        store.resolve("0" * 64)
    with pytest.raises(StoreError, match="unknown ref"):
        store.resolve(second[:6])
    with pytest.raises(StoreError, match="not stored"):
        store.write_ref("main", "0" * 64)
    with pytest.raises(StoreError, match="invalid branch name"):
        store.write_ref("../main", first)
    assert store.branches() == ["dev", "main"]
    assert not list(ref_path(store, "main").parent.glob("*.lock"))


def test_a_full_commit_id_resolves_as_the_commit_before_any_branch(store, commits):
    """S-H5: a branch named after an audited commit cannot redirect it."""
    first, second = commits
    # Planted, as a received store may hold them.
    ref_path(store, first).write_text(f"{second}\n", newline="\n")
    ref_path(store, first[:12]).write_text(f"{second}\n", newline="\n")
    assert store.resolve(first) == store.resolve(first.upper()) == first
    with pytest.raises(StoreError, match="ambiguous: it names a branch and prefixes 1 commit"):
        store.resolve(first[:12])
    missing = "e" * 64
    ref_path(store, missing).write_text(f"{second}\n", newline="\n")
    with pytest.raises(StoreError, match="no commit matches"):  # never through a branch
        store.resolve(missing)


@pytest.mark.parametrize(
    "name",
    ["abcdef0", "ABCDEF0123", "a" * 64, "main.lock", "-x", "HEAD", "a/b", "x" * 101, ""],
)
def test_branch_names_that_could_be_mistaken_are_refused(store, commits, name):
    """S-H5, H4: hex names would shadow commit prefixes; '.lock' names are lock files."""
    with pytest.raises(StoreError, match="invalid branch name"):
        store.write_ref(name, commits[0])
    with pytest.raises(StoreError, match="invalid branch name"):
        store.set_head(name)
    assert store.branches() == []
    assert store.current_branch() == "main"


def test_branch_names_that_differ_only_in_case_are_refused(store, commits):
    """S-L7: case-insensitive file systems would make them one ref file."""
    first, second = commits
    store.write_ref("main", first)
    for name in ("MAIN", "Main"):
        with pytest.raises(StoreError, match="differs from branch 'main' only in case"):
            store.write_ref(name, second)
        with pytest.raises(StoreError, match="only in case"):
            store.set_head(name)
    assert (store.read_ref("main"), store.current_branch()) == (first, "main")
    store.write_ref("main", second)  # the branch itself may move
    assert store.read_ref("main") == second


def test_write_ref_compares_and_swaps_under_a_lock_file(store, commits):
    """H4: a ref is updated only from the value the writer expects, under refs/heads/X.lock."""
    first, second = commits
    store.write_ref("main", first, expected=None)
    for expected in (None, second):
        with pytest.raises(StoreError, match="moved to"):
            store.write_ref("main", second, expected=expected)
    assert store.read_ref("main") == first
    lock = ref_path(store, "main.lock")
    assert not lock.exists()  # released after the refusals
    store.write_ref("main", second, expected=first)
    assert store.read_ref("main") == second
    lock.write_text("")  # another process holds the branch
    with pytest.raises(StoreError, match="is locked"):
        store.write_ref("main", first)
    assert store.read_ref("main") == second
    assert lock.exists()  # someone else's lock is kept
    assert store.branches() == ["main"]  # a lock file is not a branch
    lock.unlink()
    store.write_ref("main", first)
    assert store.read_ref("main") == first


def test_set_head_switches_the_current_branch(store, commits):
    store.write_ref("dev", commits[1])
    store.set_head("dev")
    assert (store.current_branch(), store.head()) == ("dev", commits[1])
    assert (store.dir / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/dev\n"
    store.set_head("fresh")  # a branch need not have commits yet
    assert store.current_branch() == "fresh"
    assert store.head() is None
