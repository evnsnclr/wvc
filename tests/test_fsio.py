"""File-system primitives (wvc.fsio): safe reads, durable writes, locks, stale temporaries."""

import os
import re
import time

import pytest
from helpers import replace_or_skip

from wvc import fsio
from wvc.fsio import (
    SourceFile,
    exclusive_lock,
    input_digest,
    open_input,
    read_regular,
    remove_stale_temporaries,
    temporary_name,
    write_atomic,
)
from wvc.records import IntegrityError, StoreError

STALE = time.time() - 7200


def test_temporary_names_are_short_and_fresh():
    """L9: the temporary name no longer embeds the target's name."""
    names = {temporary_name() for _ in range(100)}
    assert len(names) == 100
    assert all(len(name) == 41 for name in names)


def test_write_atomic_replaces_the_target_and_cleans_up_after_a_failure(tmp_path, monkeypatch):
    target = tmp_path / "file"
    write_atomic(target, b"first")
    write_atomic(target, memoryview(b"second"))
    assert target.read_bytes() == b"second"
    assert os.listdir(tmp_path) == ["file"]

    def fail(source, destination):
        raise OSError("disk full")

    monkeypatch.setattr("wvc.fsio.os.replace", fail)
    with pytest.raises(OSError, match="disk full"):
        write_atomic(target, b"third")
    assert target.read_bytes() == b"second"
    assert os.listdir(tmp_path) == ["file"]


def test_read_regular_checks_type_and_size_before_reading(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"0123456789")
    assert read_regular(path, "file", limit=10) == b"0123456789"
    assert read_regular(path, "file", limit=0, size=10) == b"0123456789"  # size decides
    with pytest.raises(IntegrityError, match="more than the 9 allowed"):
        read_regular(path, "file", limit=9)
    with pytest.raises(IntegrityError, match="holds 10 bytes, not the 11"):
        read_regular(path, "file", limit=100, size=11)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(IntegrityError, match="not a regular file"):
        read_regular(link, "link", limit=100)
    with pytest.raises(IntegrityError, match="not a regular file"):
        read_regular(tmp_path, "directory", limit=100)
    with pytest.raises(FileNotFoundError):
        read_regular(tmp_path / "missing", "missing", limit=100)


def test_without_o_nofollow_a_link_is_refused_after_the_open(tmp_path, monkeypatch):
    """Windows has no O_NOFOLLOW: the opened file is compared with lstat of the path."""
    path = tmp_path / "file"
    path.write_bytes(b"secret")
    link = tmp_path / "link"
    link.symlink_to(path)
    monkeypatch.setattr(fsio, "_NOFOLLOW", 0)
    monkeypatch.setattr(fsio, "READ_FLAGS", fsio.READ_FLAGS & ~getattr(os, "O_NOFOLLOW", 0))
    with pytest.raises(IntegrityError, match="symbolic link") as raised:
        read_regular(link, "link", limit=100)
    assert "secret" not in str(raised.value)
    assert read_regular(path, "file", limit=100) == b"secret"  # a plain file still reads


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_fifos_are_refused_at_once(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(IntegrityError, match="not a regular file"):
        read_regular(fifo, "fifo", limit=100)
    with pytest.raises(StoreError, match="not a regular file"):
        input_digest(fifo)
    with pytest.raises(StoreError, match="not a regular file"):
        SourceFile(fifo)


def test_exclusive_lock_is_held_once_and_released(tmp_path):
    lock = tmp_path / "main.lock"
    with exclusive_lock(lock, "branch 'main'"):
        assert lock.exists()
        with pytest.raises(StoreError, match="the ref is locked"), exclusive_lock(lock, "the ref"):
            pass
    assert not lock.exists()
    with pytest.raises(RuntimeError), exclusive_lock(lock, "branch 'main'"):
        raise RuntimeError("released anyway")
    assert not lock.exists()


def test_stale_temporaries_are_removed_and_fresh_ones_kept(tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep").write_text("never followed")
    entries = {
        "stale file": tmp_path / temporary_name(),
        "stale directory": tmp_path / temporary_name(),
        "fresh file": tmp_path / temporary_name(),
        "other name": tmp_path / ".wvc-notauuid.tmp",
    }
    entries["stale file"].write_bytes(b"x")
    (entries["stale directory"] / "refs").mkdir(parents=True)
    entries["fresh file"].write_bytes(b"x")
    entries["other name"].write_bytes(b"x")
    for name in ("stale file", "stale directory", "other name"):
        os.utime(entries[name], (STALE, STALE))
    assert remove_stale_temporaries(tmp_path) == 2
    left = {name for name, path in entries.items() if os.path.lexists(path)}
    assert left == {"fresh file", "other name"}
    # A link cannot be aged with utime on Windows, so here every entry counts as stale: the
    # link is removed and never followed.
    links = tmp_path / "links"
    links.mkdir()
    link = links / temporary_name()
    link.symlink_to(victim, target_is_directory=True)
    assert remove_stale_temporaries(links, max_age=-60) == 1
    assert not os.path.lexists(link)
    assert (victim / "keep").read_text() == "never followed"
    assert remove_stale_temporaries(tmp_path / "missing") == 0


def test_source_file_notices_a_rewrite_or_a_replacement(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"0123456789")
    with SourceFile(path) as source:
        assert bytes(source.view(2, 5)) == b"234"
        assert source.size == 10
        source.check_unchanged()
        with path.open("r+b") as handle:  # in place: same inode, a later mtime
            handle.write(b"x")
        os.utime(path, ns=(source.opened.st_atime_ns, source.opened.st_mtime_ns + 1_000_000))
        with pytest.raises(StoreError, match="changed while it was being committed"):
            source.check_unchanged()
    path.write_bytes(b"0123456789")
    with SourceFile(path) as source:
        sibling = tmp_path / "next"
        sibling.write_bytes(b"9876543210")
        replace_or_skip(sibling, path)  # same size, another inode
        with pytest.raises(StoreError, match="changed while it was being committed"):
            source.check_unchanged()
        assert bytes(source.view(0, 3)) == b"012"  # the map still shows the file that was opened
    with pytest.raises(StoreError, match="not a regular file"):
        SourceFile(tmp_path)


def test_source_file_closes_its_descriptor_when_mapping_fails(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"0123456789")
    opened: list[int] = []

    def recording_open_input(target):
        fd = open_input(target)
        opened.append(fd)
        return fd

    def failing_memmap(*args, **kwargs):
        raise OSError("mmap refused")

    monkeypatch.setattr(fsio, "open_input", recording_open_input)
    monkeypatch.setattr(fsio.np, "memmap", failing_memmap)
    with pytest.raises(OSError, match="mmap refused") as raised:
        SourceFile(path)
    assert len(opened) == 1
    # The traceback keeps the half-built object alive, so only an explicit close releases it.
    assert raised.traceback
    with pytest.raises(OSError, match=r"Bad file descriptor|handle is invalid"):
        os.fstat(opened[0])


def test_source_file_view_after_close_is_an_error(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"0123456789")
    with SourceFile(path) as source:
        early = source.view(0, 4)
    assert bytes(early) == b"0123"  # a view taken earlier keeps the mapping alive
    with pytest.raises(StoreError, match=re.escape("model.safetensors is closed")):
        source.view(0, 4)
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    with SourceFile(empty) as source:
        assert source.view(0, 0).size == 0
    with pytest.raises(StoreError, match="is closed"):
        source.view(0, 0)
