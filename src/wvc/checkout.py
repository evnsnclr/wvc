"""Check out a commit: rebuild every recorded file byte-for-byte (spec sections 3.3 and 3.3a).

Before anything is written, the commit's file records are validated (names included) and its
manifest is rebuilt from the header payloads and tensor ids and compared with the stored one;
its extras records are validated again, the name and type rules re-applied. The output
directory's file system must have free space for every file and extra the commit records. Each
file is then written to a temporary file ``.wvc-<uuid>.tmp`` in the output directory (its
header payload, then its tensors in data-buffer order, each decoded and verified by the store),
and so is each extra (its payload, read at its recorded size and verified against its id);
every temporary is fsynced and hashed back from disk. Each target name must be absent or a
regular file (not a symbolic link, a directory, or anything else), checked before the first
write and again after the last verification. Only when all of them verify are they renamed into
place: every existing target is first moved aside to a temporary name, then every temporary is
renamed onto its target; if any of those renames fails, the new files are removed and the
moved-aside ones put back, so a failed checkout changes no file. The moved-aside files are
deleted after the last rename, and the result records which targets replaced an existing file.
After the renames each target must be a regular file of its recorded size, distinct from the
others.

What ``force`` may replace: an existing file at a shard name always (checkout manages shards);
an existing file at an extras name only when its bytes hash to a payload the store holds and
some stored commit lists as an extra (it came from an earlier checkout of this store), or when
the caller names it in ``overwrite``. Any other is refused before anything is written, so a
hostile store's ``package.json`` extra cannot replace a project's own. The check hashes each
such file (an extra is at most ``MAX_EXTRA_BYTES``, so a larger file is refused unhashed), looks
the hash up among the stored payloads, and only then reads the commits once to collect their
extras' payload ids; a commit that cannot be read vouches for nothing.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import stat
import time
from collections.abc import Callable, Collection
from pathlib import Path

import numpy as np

from wvc.extras import MAX_EXTRA_BYTES, commit_extras
from wvc.file_layout import MAX_PREFIX_BYTES, file_digest
from wvc.fsio import (
    create_exclusive,
    fsync_directory,
    remove_stale_temporaries,
    sync,
    temporary_name,
)
from wvc.records import IntegrityError, StoreError
from wvc.store import ObjectStore


@dataclasses.dataclass(frozen=True)
class CheckedOutFile:
    """One file a checkout wrote and verified."""

    name: str
    path: Path
    nbytes: int
    sha256: str
    seconds: float
    replaced: bool = False  # an existing file of the same name was replaced


@dataclasses.dataclass(frozen=True)
class CheckoutResult:
    """The files and extras a checkout wrote, each in the order the commit records them."""

    commit_id: str
    out_dir: Path
    files: tuple[CheckedOutFile, ...]
    seconds: float
    manifest_root: str  # verified against the commit's files before anything was written
    extras: tuple[CheckedOutFile, ...]  # spec 3.3a; an extra's sha256 is its payload id

    @property
    def nbytes(self) -> int:
        """The bytes written: every file and every extra."""
        return sum(file.nbytes for file in (*self.files, *self.extras))

    @property
    def replaced(self) -> tuple[str, ...]:
        """The names of the files and extras that replaced an existing file."""
        return tuple(file.name for file in (*self.files, *self.extras) if file.replaced)


Writer = Callable[[ObjectStore, dict, Path, Path], CheckedOutFile]


def checkout_commit(
    store: ObjectStore,
    ref: str,
    out_dir: str | Path,
    *,
    force: bool = False,
    overwrite: Collection[str] = (),
) -> CheckoutResult:
    """Write the files and extras of commit ``ref`` into ``out_dir``: all verified, or none.

    ``out_dir`` is created when missing; stale temporaries (``.wvc-*.tmp`` older than an hour)
    are removed from it. An existing non-empty directory is refused unless ``force``, which
    replaces regular files of the same names (the result lists them) and leaves other files
    alone; an existing file at an extras name must have come from an earlier checkout of this
    store or be named in ``overwrite`` (StoreError before anything is written otherwise, and
    for an ``overwrite`` name the commit does not record). IntegrityError when the commit's
    files do not hold its manifest's tensors, or a
    rebuilt file or an extra does not match its record; nothing is renamed into place then.
    StoreError, before anything is written, when the output's file system has less free space
    than the files and extras need, or when a target name is a symbolic link, a directory, or
    anything but a regular file. StoreError when a rename fails; the directory is restored.
    """
    started = time.perf_counter()
    commit_id = store.resolve(ref)
    record = store.get_commit(commit_id)
    manifest = store.verified_manifest(record, f"commit {commit_id}")
    extras = commit_extras(record, f"commit {commit_id}")  # the name and type rules, again
    allowed = _check_overwrite_names(overwrite, [*record["files"], *extras], commit_id)
    out = _prepare_out_dir(store, Path(out_dir), force=force)
    existing = _existing_targets(out, [entry["name"] for entry in (*record["files"], *extras)])
    present = [entry["name"] for entry in extras if entry["name"] in existing]
    _refuse_foreign_extras(store, out, present, allowed)
    _check_free_space(out, [*record["files"], *extras])
    files, written_extras = _write_all(store, record["files"], extras, out, existing | allowed)
    _check_targets(out, [*record["files"], *extras])
    seconds = time.perf_counter() - started
    return CheckoutResult(commit_id, out, files, seconds, manifest["root"], written_extras)


def _prepare_out_dir(store: ObjectStore, out: Path, *, force: bool) -> Path:
    """Create ``out`` or check that it may be written into."""
    if out.resolve().is_relative_to(store.dir.resolve()):
        raise StoreError(f"refusing to check out into the store itself ({out})")
    if out.exists() and not out.is_dir():
        raise StoreError(f"{out} exists and is not a directory")
    if out.is_dir():
        remove_stale_temporaries(out)
        if not force and any(out.iterdir()):
            raise StoreError(f"refusing to check out into non-empty directory {out} (use --force)")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _check_overwrite_names(
    overwrite: Collection[str], entries: list[dict], commit_id: str
) -> frozenset[str]:
    """The ``overwrite`` names; StoreError for one that the commit records no file or extra of."""
    recorded = {entry["name"] for entry in entries}
    unknown = sorted(set(overwrite) - recorded)
    if unknown:
        raise StoreError(
            f"--overwrite names {', '.join(unknown)}, which commit {commit_id} records no file "
            "or extra of"
        )
    return frozenset(overwrite)


def _refuse_foreign_extras(
    store: ObjectStore, out: Path, names: list[str], allowed: frozenset[str]
) -> None:
    """StoreError when an existing file at an extras name may not be replaced.

    It may when it is in ``allowed`` or its bytes hash to a payload the store holds that some
    stored commit lists as an extra: an earlier checkout of this store wrote it.
    """
    digests = {name: _extra_digest(out / name) for name in names if name not in allowed}
    held = {name for name, digest in digests.items() if digest and store.has_payload(digest)}
    known = _extras_payload_ids(store) if held else frozenset()
    foreign = [name for name, digest in digests.items() if name not in held or digest not in known]
    if foreign:
        raise StoreError(
            f"refusing to check out: {out} already holds {', '.join(foreign)}, which no earlier "
            "checkout of this store wrote; nothing was changed. Pass --overwrite NAME "
            "(repeatable) to replace it"
        )


def _extra_digest(path: Path) -> str | None:
    """The SHA-256 of an existing file, or None when it is larger than any extra may be."""
    if os.lstat(path).st_size > MAX_EXTRA_BYTES:
        return None
    return file_digest(path)[0]


def _extras_payload_ids(store: ObjectStore) -> frozenset[str]:
    """The payload ids that some stored commit lists as an extra; unreadable commits skipped."""
    ids: set[str] = set()
    for commit_id in store._commit_ids(""):  # the store's own listing, its layout checked
        try:
            record = store.get_commit(commit_id)
            ids.update(entry["payload"] for entry in commit_extras(record, f"commit {commit_id}"))
        except (IntegrityError, StoreError):
            continue  # a commit that cannot be read vouches for nothing
    return frozenset(ids)


def _check_free_space(out: Path, entries: list[dict]) -> None:
    """StoreError when the recorded files and extras need more bytes than ``out`` has free."""
    needed, free = sum(entry["nbytes"] for entry in entries), shutil.disk_usage(out).free
    if needed > free:
        raise StoreError(
            f"refusing to check out: the files need {needed} bytes but only {free} bytes are free"
        )


def _existing_targets(out: Path, names: list[str]) -> frozenset[str]:
    """The names in ``names`` that already exist in ``out``, each a regular file.

    StoreError when one is a symbolic link, a directory, or anything else but a regular file:
    a checkout replaces only regular files.
    """
    existing = set()
    for name in names:
        path = out / name
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISREG(mode):
            existing.add(name)
            continue
        kind = (
            "a symbolic link"
            if stat.S_ISLNK(mode)
            else "a directory"
            if stat.S_ISDIR(mode)
            else "not a regular file"
        )
        raise StoreError(
            f"refusing to check out: {path} is {kind}, and a checkout replaces only regular "
            "files; nothing was changed"
        )
    return frozenset(existing)


def _write_all(
    store: ObjectStore, files: list[dict], extras: list[dict], out: Path, approved: frozenset[str]
) -> tuple[tuple[CheckedOutFile, ...], tuple[CheckedOutFile, ...]]:
    """Write and verify every file and extra as a temporary, then rename them all into place.

    ``approved`` holds the names whose existing files passed the replace check; a file that
    appeared at an extras name after that check refuses the renames.
    """
    jobs: list[tuple[Writer, dict]] = [
        *((_write_temporary, entry) for entry in files),
        *((_write_extra, entry) for entry in extras),
    ]
    names = [entry["name"] for _, entry in jobs]
    temporaries: list[Path] = []
    try:
        written = []
        for write, entry in jobs:
            temporaries.append(out / temporary_name())
            written.append(write(store, entry, temporaries[-1], out / entry["name"]))
        existing = _existing_targets(out, names)  # again, as late as possible
        _refuse_new_extras(out, [entry["name"] for entry in extras], existing, approved)
        _rename_into_place(out, list(zip(temporaries, names, strict=True)), existing)
        written = [dataclasses.replace(file, replaced=file.name in existing) for file in written]
    except BaseException:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
        raise
    return tuple(written[: len(files)]), tuple(written[len(files) :])


def _refuse_new_extras(
    out: Path, names: list[str], existing: frozenset[str], approved: frozenset[str]
) -> None:
    """StoreError when a file appeared at an extras name after the replace check."""
    new = [name for name in names if name in existing and name not in approved]
    if new:
        raise StoreError(
            f"refusing to check out: {', '.join(new)} appeared in {out} during the checkout; "
            "nothing was changed"
        )


def _rename_into_place(out: Path, moves: list[tuple[Path, str]], existing: frozenset[str]) -> None:
    """Rename each temporary onto its name in ``out``, recoverably.

    Every existing target is first moved aside to a temporary name, then every temporary is
    renamed into place. When a rename fails, the new files are removed and the moved-aside ones
    put back, in reverse order, and a StoreError says so; only after the last rename are the
    moved-aside files deleted and the directory fsynced.
    """
    aside: list[tuple[Path, Path]] = []  # (target, where its previous file now is)
    placed: list[Path] = []
    try:
        for _, name in moves:
            if name in existing:
                backup = out / temporary_name()
                os.replace(out / name, backup)
                aside.append((out / name, backup))
        for temporary, name in moves:
            os.replace(temporary, out / name)
            placed.append(out / name)
    except BaseException as exc:
        problems = _undo_renames(placed, aside)
        if not isinstance(exc, Exception):
            raise
        if problems:
            raise StoreError(
                f"checkout into {out} failed while renaming files into place ({exc}), and the "
                f"directory could not be fully restored: {'; '.join(problems)}"
            ) from exc
        raise StoreError(
            f"checkout into {out} failed while renaming files into place ({exc}); the "
            "directory was restored, nothing was changed"
        ) from exc
    for _, backup in aside:
        backup.unlink()
    fsync_directory(out)


def _undo_renames(placed: list[Path], aside: list[tuple[Path, Path]]) -> list[str]:
    """Remove the files renamed into place and put back the moved-aside ones, in reverse.

    Returns what could not be undone, each naming where the previous file is left.
    """
    problems = []
    for path in reversed(placed):
        try:
            path.unlink()
        except OSError as exc:
            problems.append(f"could not remove the new {path} ({exc.strerror})")
    for target, backup in reversed(aside):
        try:
            os.replace(backup, target)
        except OSError as exc:
            problems.append(f"the previous {target.name} is left at {backup} ({exc.strerror})")
    return problems


def _write_temporary(
    store: ObjectStore, entry: dict, temporary: Path, target: Path
) -> CheckedOutFile:
    """Rebuild one file into ``temporary``, fsync it, and check it against its record."""
    started = time.perf_counter()
    with create_exclusive(temporary) as handle:
        handle.write(store.get_payload(entry["header"], limit=MAX_PREFIX_BYTES))
        for tensor_id in entry["tensors"]:  # one decoded tensor in memory at a time
            handle.write(np.ascontiguousarray(store.read_tensor(tensor_id)).data)
        sync(handle)
    _check_written(temporary, entry["name"], entry["sha256"], entry["nbytes"])
    seconds = time.perf_counter() - started
    return CheckedOutFile(entry["name"], target, entry["nbytes"], entry["sha256"], seconds)


def _write_extra(store: ObjectStore, entry: dict, temporary: Path, target: Path) -> CheckedOutFile:
    """Write one extra into ``temporary``: its payload, read at its recorded size and checked
    against its id by the store, then fsynced and hashed back from disk."""
    started = time.perf_counter()
    data = store.get_payload(entry["payload"], size=entry["nbytes"])
    with create_exclusive(temporary) as handle:
        handle.write(data)
        sync(handle)
    _check_written(temporary, entry["name"], entry["payload"], entry["nbytes"])
    seconds = time.perf_counter() - started
    return CheckedOutFile(entry["name"], target, entry["nbytes"], entry["payload"], seconds)


def _check_written(path: Path, name: str, sha256: str, nbytes: int) -> None:
    """Hash the written file back from disk and compare it with what the commit records."""
    actual, size = file_digest(path)
    if (actual, size) != (sha256, nbytes):
        raise IntegrityError(
            f"{name}: the rebuilt file has SHA-256 {actual} and {size} bytes, "
            f"but the commit records {sha256} and {nbytes} bytes"
        )


def _check_targets(out: Path, entries: list[dict]) -> None:
    """After the renames, each target must be a distinct regular file of its recorded size."""
    identities = set()
    for entry in entries:
        path = out / entry["name"]
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_size != entry["nbytes"]:
            raise IntegrityError(
                f"{path}: after the renames it is not a regular file of {entry['nbytes']} bytes; "
                "another file of this checkout may have replaced it"
            )
        identities.add((info.st_dev, info.st_ino))
    if len(identities) != len(entries):
        raise IntegrityError(f"{out}: two files of this checkout are one file on this file system")
