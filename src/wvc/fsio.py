"""File-system primitives of the store and of checkout: safe reads, durable writes, locks.

Store files are read through ``read_regular``: opened without following a final symlink and
without blocking (so a FIFO or a device returns at once), required by ``fstat`` to be a regular
file, and refused before a byte is read when larger than the caller's cap. Messages name the
path and what is wrong, never the file's contents. ``write_atomic`` writes a temporary file
named ``.wvc-<uuid>.tmp`` beside its target, flushes and fsyncs it, renames it into place, and
fsyncs the directory. ``exclusive_lock`` holds a lock file created with O_EXCL.
Where O_NOFOLLOW does not exist (Windows), ``open_regular`` refuses a symbolic link after
the open instead: the path's ``lstat`` must not be a link and must be the opened file.
``remove_stale_temporaries`` deletes the temporaries a killed process left behind.
``SourceFile`` opens a checkpoint file once for a commit: one descriptor, its ``fstat`` at
open, and one read-only memory map from which the header and every tensor are taken. Its
lifecycle: a constructor that fails closes the descriptor before re-raising; ``close`` (or
leaving its ``with`` block) closes the descriptor and drops the map, after which ``view``
raises StoreError; views taken before ``close`` keep the mapping alive until they are dropped.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import BinaryIO

import numpy as np

from wvc.errors import IntegrityError, StoreError

TEMP_NAME = re.compile(r"\.wvc-[0-9a-f]{32}\.tmp")
STALE_SECONDS = 3600  # temporaries older than this are left over from a killed process
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_BINARY = getattr(os, "O_BINARY", 0)  # Windows only
READ_FLAGS = os.O_RDONLY | _NOFOLLOW | _NONBLOCK | _BINARY
INPUT_FLAGS = os.O_RDONLY | _NONBLOCK | _BINARY  # inputs may be symlinks (Hub snapshots are)
CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _BINARY
UNSYNCABLE_DIRECTORY = (errno.EINVAL, errno.ENOTSUP, errno.EBADF)  # fsync on a directory refused


def temporary_name() -> str:
    """A fresh temporary file name, ``.wvc-<32 hex>.tmp`` (41 characters)."""
    return f".wvc-{uuid.uuid4().hex}.tmp"


@contextlib.contextmanager
def open_regular(path: Path, what: str) -> Generator[tuple[int, os.stat_result], None, None]:
    """A read-only descriptor of the regular file ``path`` and its ``fstat``.

    FileNotFoundError passes through; a symlink, a FIFO, a device, or a directory is an
    IntegrityError naming ``what`` and the path.
    """
    try:
        fd = os.open(path, READ_FLAGS)
    except FileNotFoundError:
        raise
    except OSError as exc:  # ELOOP for a symlink, EISDIR, ENXIO, ...
        raise IntegrityError(f"{what}: {path} is not a regular file ({exc.strerror})") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise IntegrityError(f"{what}: {path} is not a regular file")
        if not _NOFOLLOW:
            _refuse_link(path, info, what)
        yield fd, info
    finally:
        os.close(fd)


def _refuse_link(path: Path, opened: os.stat_result, what: str) -> None:
    """Without O_NOFOLLOW, refuse ``path`` unless its ``lstat`` is the file that was opened.

    A symbolic link, or a path whose own inode is not the opened file's, is an IntegrityError.
    """
    try:
        at_path = os.lstat(path)
    except OSError as exc:
        raise IntegrityError(f"{what}: {path} is not a regular file ({exc.strerror})") from exc
    if stat.S_ISLNK(at_path.st_mode) or at_path.st_ino != opened.st_ino:
        raise IntegrityError(f"{what}: {path} is not a regular file (a symbolic link)")


def read_regular(path: Path, what: str, *, limit: int, size: int | None = None) -> bytes:
    """The bytes of the regular file ``path``, read only after its size is checked.

    With ``size``, any other length is an IntegrityError; without it, a file longer than
    ``limit`` bytes is. FileNotFoundError passes through.
    """
    with open_regular(path, what) as (fd, info):
        if size is not None and info.st_size != size:
            raise IntegrityError(
                f"{what}: {path} holds {info.st_size} bytes, not the {size} its record implies"
            )
        if size is None and info.st_size > limit:
            raise IntegrityError(
                f"{what}: {path} holds {info.st_size} bytes, more than the {limit} allowed"
            )
        with open(fd, "rb", closefd=False) as handle:
            data = handle.read(info.st_size + 1)
    if len(data) != info.st_size:
        raise IntegrityError(f"{what}: {path} changed size while it was being read")
    return data


def read_json_object(path: Path, what: str, limit: int) -> dict:
    """Parse a UTF-8 JSON object of at most ``limit`` bytes from a regular file.

    StoreError when missing; IntegrityError when not a regular file, too large, or not a JSON
    object. Messages never quote the file.
    """
    try:
        data = read_regular(path, what, limit=limit)
    except FileNotFoundError as exc:
        raise StoreError(f"{what} is not stored ({path})") from exc
    return loads_object(data, f"{what}: {path}")


def loads_object(data: bytes, what: str) -> dict:
    """Parse ``data`` as one UTF-8 JSON object; IntegrityError otherwise, never quoting it.

    Nesting deeper than the interpreter allows (a RecursionError) is refused like any other
    malformed text, so hostile input cannot escape as a traceback.
    """
    try:
        value = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError) as exc:  # UnicodeDecodeError and JSONDecodeError too
        raise IntegrityError(f"{what} is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"{what} does not hold a JSON object")
    return value


def holds_digest(path: Path, sha256: str, size: int) -> bool:
    """True when ``path`` is a regular file of ``size`` bytes whose SHA-256 is ``sha256``."""
    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            return False
        return regular_file_digest(path, str(path))[0] == sha256
    except (FileNotFoundError, IntegrityError):  # gone, or not a regular file any more
        return False


def regular_file_digest(path: Path, what: str) -> tuple[str, int]:
    """SHA-256 (lowercase hex) and length of the regular file ``path``, read in chunks."""
    with open_regular(path, what) as (fd, _), open(fd, "rb", closefd=False) as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        return digest, handle.tell()


def input_digest(path: Path) -> tuple[str, int]:
    """SHA-256 and length of an input file (a symlink to a regular file is followed)."""
    with open(open_input(path), "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest(), handle.tell()


def open_input(path: Path) -> int:
    """A read-only descriptor of an input file, which must be (or link to) a regular file.

    Opening does not block on a FIFO; anything but a regular file is a StoreError.
    """
    fd = os.open(path, INPUT_FLAGS)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise StoreError(f"{path} is not a regular file")
    return fd


def read_input(path: Path, limit: int) -> bytes:
    """The bytes of an input file (``open_input``'s rules); StoreError past ``limit`` bytes."""
    with open(open_input(path), "rb") as handle:
        too_large = os.fstat(handle.fileno()).st_size > limit
        data = b"" if too_large else handle.read(limit + 1)  # one more byte shows growth
    if too_large or len(data) > limit:
        raise StoreError(f"{path} holds more than the {limit} bytes allowed")
    return data


@contextlib.contextmanager
def create_exclusive(path: Path) -> Generator[BinaryIO, None, None]:
    """A new file at ``path`` opened for writing; FileExistsError when anything is there."""
    fd = os.open(path, CREATE_FLAGS, 0o666)
    with open(fd, "wb") as handle:
        yield handle


def sync(handle: BinaryIO) -> None:
    """Flush ``handle`` and fsync its file."""
    handle.flush()
    os.fsync(handle.fileno())


def fsync_directory(directory: Path) -> None:
    """Make renames inside ``directory`` durable (a no-op where directories cannot be synced)."""
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in UNSYNCABLE_DIRECTORY:
            raise
    finally:
        os.close(fd)


def write_atomic(path: Path, data: bytes | memoryview) -> None:
    """Write ``data`` to ``path`` durably: temporary file, fsync, rename, directory fsync.

    The temporary ``.wvc-<uuid>.tmp`` lives in ``path``'s directory and is removed on failure.
    """
    temporary = path.with_name(temporary_name())
    try:
        with create_exclusive(temporary) as handle:
            handle.write(data)
            sync(handle)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    fsync_directory(path.parent)


@contextlib.contextmanager
def exclusive_lock(path: Path, what: str) -> Generator[None, None, None]:
    """Hold the lock file ``path`` (created with O_EXCL) and remove it on exit."""
    try:
        os.close(os.open(path, CREATE_FLAGS, 0o666))
    except FileExistsError as exc:
        raise StoreError(
            f"{what} is locked: {path} exists, so another wvc process may be updating it; "
            "if none is running, remove that file"
        ) from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def remove_stale_temporaries(directory: Path, *, max_age: float = STALE_SECONDS) -> int:
    """Delete ``.wvc-<uuid>.tmp`` entries of ``directory`` untouched for ``max_age`` seconds.

    Returns how many were removed. A stale temporary directory (from an interrupted ``init``)
    is removed with its contents; links are removed, never followed.
    """
    cutoff, removed = time.time() - max_age, 0
    try:
        entries = list(os.scandir(directory))
    except (FileNotFoundError, NotADirectoryError):
        return 0
    for entry in entries:
        if TEMP_NAME.fullmatch(entry.name) is None:
            continue
        try:
            info = entry.stat(follow_symlinks=False)
            if info.st_mtime >= cutoff:
                continue
            if stat.S_ISDIR(info.st_mode):
                shutil.rmtree(entry.path)
            else:
                os.unlink(entry.path)
        except FileNotFoundError:  # removed meanwhile by another process
            continue
        removed += 1
    return removed


def require_directory(path: Path, what: str) -> None:
    """StoreError unless ``path`` is a directory that is not a symbolic link."""
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise StoreError(f"{what} {path} is missing; is this a wvc repository?") from exc
    if stat.S_ISLNK(info.st_mode):
        raise StoreError(f"refusing {what} {path}: it is a symbolic link")
    if not stat.S_ISDIR(info.st_mode):
        raise StoreError(f"{what} {path} is not a directory")


def file_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    """What changes when a file is replaced or rewritten: device, inode, size, mtime (ns)."""
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


class SourceFile:
    """An input file opened once: its ``fstat`` at open and one read-only map of its bytes.

    The path may be a symlink to a regular file. ``check_unchanged`` compares the descriptor's
    and the path's current ``stat`` with the one taken at open, so a file rewritten in place or
    replaced by a rename during the read is noticed.

    Use it as a context manager. A constructor that fails closes the descriptor before the
    error escapes. ``close`` closes the descriptor and drops the map; ``view`` then raises
    StoreError. Views taken earlier keep the mapping (and the map's own duplicate descriptor)
    alive until they are dropped, so they stay readable but should not outlive the ``with``.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        fd = open_input(path)
        try:
            self._handle = open(fd, "rb")  # noqa: SIM115 (owns fd; closed by close or __exit__)
        except BaseException:
            os.close(fd)
            raise
        try:
            self.opened = os.fstat(fd)
            self._map: np.ndarray | None = self._mapped(self.opened.st_size)
        except BaseException:
            self._handle.close()
            raise

    def _mapped(self, size: int) -> np.ndarray:
        """A read-only map of the first ``size`` bytes (an empty array for an empty file)."""
        if not size:
            return np.zeros(0, dtype=np.uint8)
        try:
            return np.memmap(self._handle, dtype=np.uint8, mode="r", shape=(size,))
        except ValueError as exc:  # the file shrank between fstat and mmap
            raise StoreError(f"{self.path} changed while it was being opened") from exc

    def __enter__(self) -> SourceFile:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def size(self) -> int:
        return self.opened.st_size

    def view(self, begin: int, end: int) -> np.ndarray:
        """Bytes ``[begin, end)`` of the file as a read-only memory-mapped uint8 view.

        StoreError once the file is closed.
        """
        if self._map is None:
            raise StoreError(f"{self.path} is closed")
        return self._map[begin:end]

    def check_unchanged(self) -> None:
        """StoreError when the file was rewritten or the path replaced since it was opened."""
        before = file_identity(self.opened)
        now = file_identity(os.fstat(self._handle.fileno()))
        at_path = file_identity(os.stat(self.path))
        if now != before or at_path != before:
            raise StoreError(f"{self.path} changed while it was being committed")

    def close(self) -> None:
        """Close the descriptor and drop the map (views taken earlier keep it alive)."""
        self._map = None
        self._handle.close()
