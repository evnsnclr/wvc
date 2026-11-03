"""Repository integrity check (``wvc fsck``): layout, payloads, objects, manifests, commits, refs.

The layout is checked again, whatever the store instance cached: ``.wvc`` and its directories,
shard directories included, must be directories and not symbolic links, and a store that fails
is not read further. Payloads are re-hashed in chunks. Object records are validated, their
payloads must have the length the record implies, no record may claim more than the store's
tensor limit, and, unless decoding is switched off, every tensor is rebuilt through its delta
chain and checked against its id; an object that decodes gets a verification marker
(``wvc.reuse``) and one that fails loses its marker, while the ``verified`` directory itself
is not part of the layout check. Manifests have their roots recomputed and must name stored
objects. Commits have their ids recomputed, must name stored parents, and their files must hold
exactly their manifest's tensors (rebuilt from the header payloads); each of their extras (spec
3.3a) needs its payload stored at the recorded size and hashing to its id. Refs and HEAD must
parse and point at stored commits; branches named like commit ids, branch names that differ
only in case, and lock files left behind are reported.
Problems are collected, not raised, so one damaged file does not hide the others; running out
of memory while rebuilding a tensor is reported as a problem of that object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from wvc.extras import check_extra_payloads
from wvc.fsio import TEMP_NAME
from wvc.records import StoreError, expected_payload_size, is_id
from wvc.refs import HEX_NAME, LOCK_SUFFIX
from wvc.store import ObjectStore


@dataclass
class FsckReport:
    payloads: int = 0
    objects: int = 0
    tensors_decoded: int = 0
    manifests: int = 0
    commits: int = 0
    refs: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "payloads": self.payloads,
            "objects": self.objects,
            "tensors_decoded": self.tensors_decoded,
            "manifests": self.manifests,
            "commits": self.commits,
            "refs": self.refs,
            "problems": list(self.problems),
        }


def fsck(store: ObjectStore, *, decode: bool = True) -> FsckReport:
    """Check the whole store; ``decode=False`` skips rebuilding tensors (faster, weaker)."""
    report = FsckReport()
    try:
        store.check_layout()
    except StoreError as exc:
        report.problems.append(f"layout: {exc}")
        return report
    damaged = _check_payloads(store, report)
    _check_objects(store, report, decode)
    _check_manifests(store, report)
    _check_commits(store, report, damaged)
    _check_refs(store, report)
    return report


def _ids_under(directory: Path, suffix: str = "") -> list[str]:
    """The ids named by the files of a sharded store directory, sorted."""
    names = (p.name.removesuffix(suffix) for p in directory.glob(f"*/*{suffix}"))
    return sorted(name for name in names if is_id(name))


def _ids_in(directory: Path) -> list[str]:
    """The ids named by the ``<id>.json`` files of a flat store directory, sorted."""
    names = (name.removesuffix(".json") for name in os.listdir(directory))
    return sorted(name for name in names if is_id(name))


def _check_payloads(store: ObjectStore, report: FsckReport) -> set[str]:
    """Re-hash every payload; return the ids of those that failed."""
    damaged: set[str] = set()
    for pid in _ids_under(store.dir / "payloads"):
        report.payloads += 1
        try:
            store.verify_payload(pid)
        except StoreError as exc:
            report.problems.append(f"payload {pid}: {exc}")
            damaged.add(pid)
    return damaged


def _check_objects(store: ObjectStore, report: FsckReport, decode: bool) -> None:
    for tensor_id in _ids_under(store.dir / "objects", ".json"):
        report.objects += 1
        nbytes = None
        try:
            nbytes = _check_object(store, tensor_id)
            if decode:
                store.read_tensor(tensor_id)
                report.tensors_decoded += 1
                store.mark_verified(tensor_id)
        except StoreError as exc:
            report.problems.append(f"object {tensor_id}: {exc}")
            store.unmark_verified(tensor_id)
        except MemoryError:
            report.problems.append(f"object {tensor_id}: out of memory rebuilding {nbytes} bytes")


def _check_object(store: ObjectStore, tensor_id: str) -> int:
    """Check one object record and its payload's size; return the tensor's nbytes."""
    record = store.get_object(tensor_id)
    pid, expected = record["payload"], expected_payload_size(record)
    size = store.payload_size(pid)
    if size is None:
        raise StoreError(f"payload {pid} is missing")
    if size != expected:
        raise StoreError(f"payload {pid} holds {size} bytes; the record implies {expected}")
    if "base" in record and not store.has_object(record["base"]):
        raise StoreError(f"base {record['base']} is missing")
    return record["nbytes"]


def _check_manifests(store: ObjectStore, report: FsckReport) -> None:
    for root in _ids_in(store.dir / "manifests"):
        report.manifests += 1
        try:
            manifest = store.get_manifest(root)
        except StoreError as exc:
            report.problems.append(f"manifest {root}: {exc}")
            continue
        missing = [n for n, e in manifest["tensors"].items() if not store.has_object(e["sha256"])]
        if missing:
            report.problems.append(f"manifest {root}: {len(missing)} tensors have no object")


def _check_commits(store: ObjectStore, report: FsckReport, damaged: set[str]) -> None:
    for commit_id in _ids_in(store.dir / "commits"):
        report.commits += 1
        try:
            record = store.get_commit(commit_id)
            for parent in record["parents"]:
                if not store.has_commit(parent):
                    raise StoreError(f"parent {parent} is missing")
            store.verified_manifest(record, "record")
            _check_extras(store, record, damaged)
        except StoreError as exc:
            report.problems.append(f"commit {commit_id}: {exc}")


def _check_extras(store: ObjectStore, record: dict, damaged: set[str]) -> None:
    """Each extra's payload must be stored at its recorded size and hash to its id."""
    extras = record.get("extras", [])
    check_extra_payloads(extras, store.payload_size, "record")
    for entry in extras:
        if entry["payload"] in damaged:
            raise StoreError(
                f"record: extra {entry['name']}: payload {entry['payload']} does not hash to its id"
            )


def _check_refs(store: ObjectStore, report: FsckReport) -> None:
    try:
        store.current_branch()
    except StoreError as exc:
        report.problems.append(f"HEAD: {exc}")
    names = sorted(os.listdir(store.dir / "refs" / "heads"))
    folded = [name.casefold() for name in names]
    for name in names:
        if TEMP_NAME.fullmatch(name):
            continue
        if name.endswith(LOCK_SUFFIX):
            report.problems.append(f"ref {name}: a lock file; remove it if no wvc process runs")
            continue
        report.refs += 1
        try:
            _check_ref(store, name, folded.count(name.casefold()) > 1)
        except StoreError as exc:
            report.problems.append(f"ref {name}: {exc}")


def _check_ref(store: ObjectStore, name: str, case_twin: bool) -> None:
    target = store.read_ref(name)
    if target is None or not store.has_commit(target):
        raise StoreError(f"points at {target}, which is not stored")
    if HEX_NAME.fullmatch(name):
        raise StoreError("the branch is named like a commit id, which it can shadow")
    if case_twin:
        raise StoreError("another branch has the same name up to case")
