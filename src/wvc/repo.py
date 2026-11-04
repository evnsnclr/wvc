"""Repository operations over the object store: init, commit, log, and show (spec section 3).

``commit_with_meta`` (or ``commit_checkpoint``, its keyword form) stores a safetensors
checkpoint as one commit: every tensor once by its SHA-256 (a changed tensor as a delta against
the parent's tensor of the same name, dtype, and shape when that is smaller, or as a
``delta-lowrank`` when its adapter factors are given), the verbatim header of every file, the
manifest, the commit record, and finally the branch ref.
Each file is opened once and its tensors are read through one memory map, so a checkpoint never
has to fit in memory and every stored byte is a byte of the file whose SHA-256 is recorded.
A checkpoint directory's non-tensor files (config, tokenizer, chat template, licence, card) are
stored as extras, one payload each (``wvc.extras``, spec 3.3a); a single file carries its
parent's extras forward. ``show_commit`` checks a commit's files against its manifest and
compares manifests; no tensor is decoded. Checkout lives in ``wvc.checkout``.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import cast

import numpy as np

from wvc.canonical import canonical_bytes
from wvc.extras import ExtraPlan, commit_extras, plan_extras, read_extra
from wvc.file_layout import check_file_names, data_order, file_digest, parse_header_payload
from wvc.fsio import SourceFile
from wvc.hashing import Hasher, build_manifest, digest_bytes
from wvc.ids import is_count, short
from wvc.lowrank import LowRankFactors
from wvc.records import (
    COMMIT_FORMAT,
    FULL_FORM,
    OPERATION_KINDS,
    DeltaDepthError,
    IntegrityError,
    StoreError,
)
from wvc.safetensors_io import ITEMSIZES, Checkpoint, SafetensorsFile, TensorInfo, load_checkpoint
from wvc.sources import check_source
from wvc.store import ObjectStore, find_repo

AUTHOR_ENV = "WVC_AUTHOR"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"  # spec 3.4: UTC, whole seconds
REUSED = "reused"  # the outcome of a tensor whose SHA-256 the store already held


@dataclass(frozen=True)
class CommitMeta:
    """What a commit records about itself besides its content (spec sections 3.4 and 5.1)."""

    message: str
    author: str
    operation_kind: str
    operation_extra: Mapping[str, object] | None = None  # operation keys besides ``kind``
    source: dict | None = None  # the record's ``source`` field (``wvc.sources``)


@dataclass(frozen=True)
class CommitResult:
    """What ``commit_checkpoint`` stored, and how."""

    commit_id: str
    manifest_root: str
    branch: str
    parent_id: str | None
    tensors: int  # tensors in the checkpoint
    reused: int  # already stored, found by SHA-256
    stored_full: int  # new zstd-planes objects
    stored_delta: dict[str, int]  # new delta objects by form, sorted by form
    payload_bytes: int  # payload bytes this commit wrote to disk: tensors, headers, extras
    seconds: float
    extras: int  # extras recorded (spec 3.3a): the directory's non-tensor files
    extras_bytes: int  # their total size
    skipped: tuple[str, ...]  # directory entries not recorded, each "<name> (<reason>)"
    extras_carried: bool = False  # the extras are the parent's, carried for a single file
    parent_extras: int = 0  # extras the first parent records


@dataclass(frozen=True)
class CommitTarget:
    """Where a new commit goes; see ``commit_target``."""

    branch: str  # the branch the commit is written to
    parent_id: str | None  # the commit's parent; None for a commit without parents
    previous: str | None  # the branch's commit before, which it must still be when written
    switch_head: bool  # True when HEAD moves to ``branch`` after the commit


@dataclass(frozen=True)
class ManifestComparison:
    """How one manifest differs from another, by tensor name (sorted)."""

    unchanged: int
    changed: tuple[str, ...]  # same name; a different sha256, dtype, or shape
    added: tuple[str, ...]
    removed: tuple[str, ...]

    @property
    def counts(self) -> dict[str, int]:
        """Counts in the order ``wvc diff`` reports them."""
        return {
            "changed": len(self.changed),
            "unchanged": self.unchanged,
            "added": len(self.added),
            "removed": len(self.removed),
        }

    def to_dict(self) -> dict:
        names = {"changed": self.changed, "added": self.added, "removed": self.removed}
        return {"counts": self.counts, **{key: list(value) for key, value in names.items()}}


@dataclass(frozen=True)
class ShowResult:
    """A commit record, its tensor count, and its comparison with its first parent."""

    commit_id: str
    record: dict
    tensor_count: int
    parent_id: str | None  # the first parent
    comparison: ManifestComparison | None  # None for a commit without parents

    def to_dict(self) -> dict:
        return {
            "id": self.commit_id,
            "commit": self.record,
            "tensor_count": self.tensor_count,
            "first_parent": self.parent_id,
            "comparison": None if self.comparison is None else self.comparison.to_dict(),
        }


@dataclass(frozen=True)
class _ExtrasPlan:
    """Where a commit's extras come from: its directory, or its parent for a single file."""

    plans: tuple[ExtraPlan, ...]  # files of the checkpoint directory to store
    carried: tuple[dict, ...]  # the parent's extras records, carried forward unchanged
    skipped: tuple[str, ...]  # directory entries not recorded, each "<name> (<reason>)"
    parent_count: int  # extras the parent records


@dataclass(frozen=True)
class _FilePlan:
    """One source file, its name and layout validated before anything is stored."""

    file: SafetensorsFile
    size: int
    tensors: tuple[TensorInfo, ...]  # in data-buffer order


@dataclass(frozen=True)
class _StoredTensor:
    name: str
    tensor_id: str
    outcome: str  # REUSED, or the form of the object this commit stored


@dataclass(frozen=True)
class _StoredFile:
    plan: _FilePlan
    header: bytes  # the size field and JSON header, verbatim, as read for the commit
    tensors: tuple[_StoredTensor, ...]  # in data-buffer order
    sha256: str

    def record(self) -> dict:
        """The file layout record of spec section 3.3."""
        return {
            "name": self.plan.file.path.name,
            "header": hashlib.sha256(self.header).hexdigest(),
            "tensors": [tensor.tensor_id for tensor in self.tensors],
            "sha256": self.sha256,
            "nbytes": self.plan.size,
        }


def init_repository(directory: str | Path) -> Path:
    """Create ``.wvc`` in ``directory`` (made if missing); return the repository directory.

    StoreError when ``directory`` already holds ``.wvc``.
    """
    store = ObjectStore(Path(directory))
    store.init_layout()
    return store.repo.resolve()


def open_repository(repo: str | Path | None = None) -> ObjectStore:
    """Open the store of ``repo``, which must hold ``.wvc``, and validate its config.

    Without ``repo``, the nearest repository at or above the current directory is opened; one
    that belongs to another user is refused there (see ``find_repo``).
    """
    if repo is None:
        store = ObjectStore(find_repo(Path.cwd()))
    else:
        store = ObjectStore(Path(repo))
        if not store.exists():
            raise StoreError(f"{repo}: not a wvc repository (no .wvc directory)")
    store.config()
    return store


def default_author() -> str:
    """The ``WVC_AUTHOR`` environment variable; StoreError when it is unset or blank.

    wvc never records the OS user name on its own: commits are meant to be pushed, and their
    author field is published with them.
    """
    name = os.environ.get(AUTHOR_ENV, "")
    if not name.strip():
        raise StoreError(f"no commit author: pass --author NAME or set {AUTHOR_ENV}")
    return name


def commit_target(
    store: ObjectStore, parent_ref: str | None = None, branch: str | None = None
) -> CommitTarget:
    """Where a commit may go, checked before anything is stored (spec 3.6).

    The parent is ``parent_ref`` resolved, or HEAD. Without ``branch`` the commit goes on the
    current branch, so the parent must be that branch's head (StoreError otherwise: moving the
    branch would drop the commits after the parent). With ``branch``, that branch must not
    exist yet or must be an ancestor of the parent (a fast-forward), and HEAD then switches to
    it. Branch names follow ``ObjectStore.check_branch_name``.
    """
    current, head = store.current_branch(), store.head()
    parent_id = head if parent_ref is None else store.resolve(parent_ref)
    if branch is None:
        if parent_id != head:
            raise StoreError(
                f"the parent {short(parent_id)} is not the head of branch {current!r} "
                f"({short(head)}), and a commit without --branch goes on top of that head; "
                "use --branch NAME to commit onto another branch"
            )
        store.check_branch_name(current)
        return CommitTarget(current, parent_id, head, switch_head=False)
    store.check_branch_name(branch)
    previous = store.read_ref(branch)
    if previous is not None and not _is_ancestor(store, previous, parent_id):
        raise StoreError(
            f"branch {branch!r} is at {short(previous)}, which is not an ancestor of the "
            f"parent {short(parent_id)}: the commit would not fast-forward it"
        )
    return CommitTarget(branch, parent_id, previous, switch_head=branch != current)


def commit_checkpoint(
    store: ObjectStore,
    checkpoint_path: str | Path,
    *,
    message: str,
    operation_kind: str,
    author: str,
    parent_ref: str | None = None,
    branch: str | None = None,
    operation_extra: Mapping[str, object] | None = None,
    source: dict | None = None,
    lowrank: Mapping[str, LowRankFactors] | None = None,
    extras: bool = True,
) -> CommitResult:
    """``commit_with_meta`` with the fields of its ``CommitMeta`` given as keywords."""
    meta = CommitMeta(message, author, operation_kind, operation_extra, source)
    return commit_with_meta(
        store,
        checkpoint_path,
        meta,
        parent_ref=parent_ref,
        branch=branch,
        lowrank=lowrank,
        extras=extras,
    )


def commit_with_meta(
    store: ObjectStore,
    checkpoint_path: str | Path,
    meta: CommitMeta,
    *,
    parent_ref: str | None = None,
    branch: str | None = None,
    lowrank: Mapping[str, LowRankFactors] | None = None,
    extras: bool = True,
) -> CommitResult:
    """Store a checkpoint as a commit and point a branch at it.

    ``parent_ref`` and ``branch`` choose the parent and the branch (``commit_target``); the
    branch moves only if nothing moved it meanwhile. Refused as "nothing changed" when the
    manifest root, the file layout records (names, header payloads, file SHA-256s), and the
    extras all equal the parent's. ``meta`` is checked first (``check_commit_meta``): its
    ``operation_extra`` adds keys besides ``kind`` to the operation, and its ``source``
    (``wvc.sources``) becomes the record's ``source`` field (spec 5.1). A tensor named in
    ``lowrank`` becomes a ``delta-lowrank`` on the parent's tensor of the same
    name, dtype, and shape, or what ``put_tensor`` stores when that chain would be too deep.
    With ``extras``, a directory's non-tensor files are recorded too (``wvc.extras``), and a
    single file carries its parent's extras records forward unchanged (``_plan_extras``);
    without, none are recorded. Names, layouts, and extras are checked before anything is
    stored; then each file is stored through one open (``_store_file``), each extra as one
    payload, and the rest of the commit.
    """
    started = time.perf_counter()
    meta = check_commit_meta(meta)
    target = commit_target(store, parent_ref, branch)
    parent = None if target.parent_id is None else store.get_commit(target.parent_id)
    bases = {} if parent is None else store.get_manifest(parent["manifest"])["tensors"]
    ckpt = load_checkpoint(checkpoint_path)
    factors = _check_lowrank_names(lowrank, ckpt)
    plans = _plan_files(ckpt, str(checkpoint_path))
    extras_plan = _plan_extras(store, checkpoint_path, parent, target.parent_id, enabled=extras)
    written = store.payload_bytes_written
    files = [_store_file(store, plan, bases, factors) for plan in plans]
    stored_extras = [
        *(dict(entry) for entry in extras_plan.carried),
        *(_store_extra(store, plan) for plan in extras_plan.plans),
    ]
    manifest = build_manifest(ckpt, {t.name: t.tensor_id for f in files for t in f.tensors})
    record = _commit_record(
        manifest_root=manifest["root"],
        parent_id=target.parent_id,
        meta=meta,
        files=[file.record() for file in files],
        extras=stored_extras,
    )
    _refuse_unchanged(checkpoint_path, record, parent, target.parent_id)
    commit_id = _store_commit(store, target, files, manifest, record)
    payload_bytes = store.payload_bytes_written - written
    return _commit_result(
        commit_id, record, target.branch, files, payload_bytes, started, extras_plan
    )


def commit_manifest(store: ObjectStore, commit_id: str) -> dict:
    """The manifest a stored commit names."""
    return store.get_manifest(store.get_commit(commit_id)["manifest"])


def tensor_count(manifest: Mapping) -> int:
    """A manifest's ``tensor_count``, checked against its tensors (IntegrityError if wrong)."""
    count, tensors = manifest.get("tensor_count"), manifest["tensors"]
    if not is_count(count) or count != len(tensors):
        raise IntegrityError(
            f"manifest {manifest['root']}: tensor_count does not match its {len(tensors)} tensors"
        )
    return count


def log_entries(
    store: ObjectStore, start_ref: str = "HEAD", limit: int | None = None
) -> Iterator[tuple[str, dict]]:
    """Commit ids and records from ``start_ref`` back along first parents, newest first.

    ``start_ref`` is resolved before this returns; ``limit`` caps the number of commits.
    """
    if limit is not None and not is_count(limit):
        raise ValueError(f"limit must be a non-negative integer or None, got {limit!r}")
    return islice(_first_parents(store, store.resolve(start_ref)), limit)


def compare_manifests(old: Mapping, new: Mapping) -> ManifestComparison:
    """Classify tensor names of ``new`` against ``old`` from the manifests alone."""
    before, after = old["tensors"], new["tensors"]
    common = sorted(before.keys() & after.keys())
    changed = tuple(name for name in common if _identity(before[name]) != _identity(after[name]))
    return ManifestComparison(
        unchanged=len(common) - len(changed),
        changed=changed,
        added=tuple(sorted(after.keys() - before.keys())),
        removed=tuple(sorted(before.keys() - after.keys())),
    )


def show_commit(store: ObjectStore, ref: str) -> ShowResult:
    """A commit's record and, when it has parents, its comparison with the first parent.

    The commit's files are first checked to hold exactly its manifest's tensors
    (``ObjectStore.verified_manifest``); IntegrityError otherwise.
    """
    commit_id = store.resolve(ref)
    record = store.get_commit(commit_id)
    manifest = store.verified_manifest(record, f"commit {commit_id}")
    parent_id = record["parents"][0] if record["parents"] else None
    comparison = (
        None
        if parent_id is None
        else compare_manifests(commit_manifest(store, parent_id), manifest)
    )
    return ShowResult(commit_id, record, tensor_count(manifest), parent_id, comparison)


def _is_ancestor(store: ObjectStore, ancestor: str, descendant: str | None) -> bool:
    """True when ``ancestor`` is ``descendant`` or reachable from it through parents."""
    seen: set[str] = set()
    stack = [] if descendant is None else [descendant]
    while stack:
        commit_id = stack.pop()
        if commit_id == ancestor:
            return True
        if commit_id not in seen:
            seen.add(commit_id)
            stack.extend(store.get_commit(commit_id)["parents"])
    return False


def check_commit_meta(meta: CommitMeta) -> CommitMeta:
    """``meta`` checked, with ``operation_extra`` as a dict and ``source`` validated.

    StoreError as ``check_commit_text`` and ``wvc.sources.check_source`` raise it, or for an
    ``operation_extra`` that sets ``kind`` or that canonical JSON refuses.
    """
    check_commit_text(message=meta.message, operation_kind=meta.operation_kind, author=meta.author)
    extra = _check_operation_extra(meta.operation_extra)
    source = None if meta.source is None else check_source(meta.source)
    return replace(meta, operation_extra=extra, source=source)


def check_commit_text(*, message: object, operation_kind: object, author: object) -> None:
    """Refuse an unknown operation kind and an empty or non-UTF-8 message or author."""
    if not isinstance(operation_kind, str) or operation_kind not in OPERATION_KINDS:
        raise StoreError(
            f"unknown operation kind {operation_kind!r}; expected one of "
            f"{', '.join(sorted(OPERATION_KINDS))}"
        )
    for what, text in (("message", message), ("author", author)):
        if not isinstance(text, str) or not text.strip():
            raise StoreError(f"the commit {what} is empty")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise StoreError(f"the commit {what} is not valid UTF-8 text") from exc


def _check_operation_extra(extra: Mapping[str, object] | None) -> dict:
    """Extra operation keys as a dict; StoreError for ``kind`` or what canonical JSON refuses."""
    keys = {} if extra is None else dict(extra)
    if "kind" in keys:
        raise StoreError("operation_extra must not set 'kind'; pass operation_kind instead")
    try:
        canonical_bytes(keys)
    except ValueError as exc:
        raise StoreError(f"operation_extra: {exc}") from exc
    return keys


def _check_lowrank_names(
    lowrank: Mapping[str, LowRankFactors] | None, ckpt: Checkpoint
) -> dict[str, LowRankFactors]:
    """Low-rank factors by tensor name; StoreError when one names no tensor of ``ckpt``."""
    factors = {} if lowrank is None else dict(lowrank)
    missing = sorted(name for name in factors if name not in ckpt.tensors)
    if missing:
        raise StoreError(f"low-rank factors name tensors the checkpoint lacks: {missing}")
    return factors


def _plan_extras(
    store: ObjectStore,
    path: str | Path,
    parent: dict | None,
    parent_id: str | None,
    *,
    enabled: bool,
) -> _ExtrasPlan:
    """The extras a commit of ``path`` records (spec 3.3a); nothing is stored yet.

    A directory's extras are planned from the directory. A single file has none of its own, so
    it carries the parent's extras records forward unchanged; StoreError when one of their
    payloads is not stored at its recorded size. With extras off, none are recorded.
    """
    parent_extras = [] if parent is None else commit_extras(parent, f"commit {parent_id}")
    if not enabled:
        return _ExtrasPlan((), (), (), len(parent_extras))
    if Path(path).is_dir():
        plans, skipped = plan_extras(path)
        return _ExtrasPlan(tuple(plans), (), tuple(skipped), len(parent_extras))
    for entry in parent_extras:
        if store.payload_size(entry["payload"]) != entry["nbytes"]:
            raise StoreError(
                f"cannot carry the extra {entry['name']} of parent {parent_id} forward: its "
                f"payload {entry['payload']} is not stored at {entry['nbytes']} bytes"
            )
    return _ExtrasPlan((), tuple(parent_extras), (), len(parent_extras))


def _store_extra(store: ObjectStore, plan: ExtraPlan) -> dict:
    """Store one extra as a payload (its id is the file's SHA-256); return its record."""
    payload = store.put_payload(read_extra(plan))
    return {"name": plan.name, "payload": payload, "nbytes": plan.nbytes}


def _plan_files(ckpt: Checkpoint, where: str) -> list[_FilePlan]:
    """Validate every file's name and layout; nothing is stored or read beyond the headers."""
    check_file_names([file.path.name for file in ckpt.files], where)
    plans = []
    for file in cast("tuple[SafetensorsFile, ...]", ckpt.files):  # load_checkpoint's files
        size = os.stat(file.path).st_size
        plans.append(_FilePlan(file, size, data_order(file, size)))
    return plans


def _store_file(
    store: ObjectStore,
    plan: _FilePlan,
    bases: Mapping[str, Mapping],
    lowrank: Mapping[str, LowRankFactors],
) -> _StoredFile:
    """Store one file's tensors through a single open of the file, then check the file.

    The header and every tensor come from one memory map, and the running SHA-256 of the file
    is fed the header and the same tensor views that are stored, in data order (they tile the
    file). StoreError when the file changed meanwhile: its size, mtime, or inode differ from
    when it was opened, or a final read of it does not hash to the running value.
    """
    path, start = plan.file.path, plan.file.data_start
    with SourceFile(path) as source:
        if source.size != plan.size:
            raise StoreError(f"{path} changed size while it was being committed")
        header = bytes(source.view(0, start))
        if parse_header_payload(header, plan.size, str(path)) != plan.tensors:
            raise StoreError(f"{path}: the header changed after it was parsed")
        running = hashlib.sha256(header)
        tensors = tuple(
            _store_tensor(
                store,
                source.view(start + info.begin, start + info.end),
                info,
                bases.get(info.name),
                lowrank.get(info.name),
                running,
            )
            for info in plan.tensors
        )
        source.check_unchanged()
    sha256 = running.hexdigest()
    if file_digest(path) != (sha256, plan.size):
        raise StoreError(
            f"{path} changed while it was being committed: it no longer hashes to "
            "the bytes that were stored"
        )
    return _StoredFile(plan, header, tensors, sha256)


def _store_tensor(
    store: ObjectStore,
    raw: np.ndarray,
    info: TensorInfo,
    base: Mapping | None,
    factors: LowRankFactors | None,
    running: Hasher,
) -> _StoredTensor:
    """Store one tensor unless its SHA-256 is stored already; ``base`` is the parent's entry.

    ``raw`` is read once to feed both its own SHA-256 (the tensor id) and the file's running
    one; the store gets the id and does not hash the bytes again. A stored object is reused only
    when ``check_reusable`` accepts it (it decodes to the id, or a marker says it did); otherwise
    the tensor is stored in full over it. With ``factors``, the tensor is stored as a low-rank
    delta on the parent's tensor.
    """
    tensor_id = digest_bytes(raw, also=running)  # one read feeds both hashes
    itemsize = ITEMSIZES[info.dtype]
    if store.has_object(tensor_id):
        if store.check_reusable(tensor_id, raw.size) is not None:
            return _StoredTensor(info.name, tensor_id, REUSED)
        record = store.replace_tensor(raw, itemsize, tensor_id=tensor_id)
        return _StoredTensor(info.name, tensor_id, record["form"])
    base_id = _base_id(info, base)
    if factors is None:
        record = store.put_tensor(raw, itemsize, base_id=base_id, tensor_id=tensor_id)
    else:
        record = _put_lowrank(store, raw, info, base_id, factors, tensor_id)
    return _StoredTensor(info.name, tensor_id, record["form"])


def _base_id(info: TensorInfo, base: Mapping | None) -> str | None:
    """The parent's tensor id when its dtype and shape match ``info``, else None."""
    if base is None or (base["dtype"], tuple(base["shape"])) != (info.dtype, info.shape):
        return None
    return base["sha256"]


def _put_lowrank(
    store: ObjectStore,
    raw: np.ndarray,
    info: TensorInfo,
    base_id: str | None,
    factors: LowRankFactors,
    tensor_id: str,
) -> dict:
    """Store ``raw`` as a low-rank delta on ``base_id``, or in full when the chain is too deep."""
    if base_id is None:
        raise StoreError(
            f"tensor {info.name!r}: a low-rank merge needs the parent's tensor of the same name, "
            "dtype, and shape as its base"
        )
    itemsize = ITEMSIZES[info.dtype]
    try:
        return store.put_lowrank(
            raw,
            itemsize,
            base_id=base_id,
            base_dtype=info.dtype,
            shape=info.shape,
            factors=factors,
            tensor_id=tensor_id,
        )
    except DeltaDepthError:
        return store.put_tensor(raw, itemsize, base_id=base_id, tensor_id=tensor_id)


def _refuse_unchanged(
    path: str | Path, record: dict, parent: dict | None, parent_id: str | None
) -> None:
    """StoreError when ``record`` would repeat its parent: same root, file layout, and extras."""
    if (
        parent is None
        or record["manifest"] != parent["manifest"]
        or _layout(record["files"]) != _layout(parent["files"])
        or record.get("extras", []) != parent.get("extras", [])
    ):
        return
    raise StoreError(
        f"nothing changed: {path} has the manifest root, the file layout (names, headers, and "
        f"file SHA-256s), and the extras of parent {parent_id}"
    )


def _layout(files: Sequence[object]) -> list[tuple[object, ...] | None]:
    """What the nothing-changed rule compares of file records: name, header, and sha256."""
    keys = ("name", "header", "sha256")
    return [
        tuple(entry.get(key) for key in keys) if isinstance(entry, dict) else None
        for entry in files
    ]


def _store_commit(
    store: ObjectStore,
    target: CommitTarget,
    files: Sequence[_StoredFile],
    manifest: dict,
    record: dict,
) -> str:
    """Store the header payloads, the manifest, and the record; then move the branch."""
    for file in files:
        store.put_payload(file.header)
    store.put_manifest(manifest)
    commit_id = store.put_commit(record)
    store.write_ref(target.branch, commit_id, expected=target.previous)
    if target.switch_head:
        store.set_head(target.branch)
    return commit_id


def _commit_record(
    *,
    manifest_root: str,
    parent_id: str | None,
    meta: CommitMeta,
    files: list[dict],
    extras: list[dict] | None = None,
) -> dict:
    """The commit record of spec section 3.4, stamped with the current UTC time.

    ``meta.operation_extra`` holds keys merged into ``operation`` after ``kind``. Non-empty
    ``extras`` (spec 3.3a) and a ``meta.source`` (spec 5.1) become top-level fields; without
    them the record has none, as older commits.
    """
    return {
        "format": COMMIT_FORMAT,
        "manifest": manifest_root,
        "parents": [] if parent_id is None else [parent_id],
        "operation": {"kind": meta.operation_kind, **(meta.operation_extra or {})},
        "message": meta.message,
        "author": meta.author,
        "time": datetime.now(UTC).strftime(TIME_FORMAT),
        "files": files,
        **({"extras": extras} if extras else {}),
        **({} if meta.source is None else {"source": meta.source}),
    }


def _commit_result(
    commit_id: str,
    record: dict,
    branch: str,
    files: list[_StoredFile],
    payload_bytes: int,
    started: float,
    extras_plan: _ExtrasPlan,
) -> CommitResult:
    """Count what the commit stored: tensors reused, stored full, and stored by delta form."""
    tensors = [tensor for file in files for tensor in file.tensors]
    outcomes = Counter(tensor.outcome for tensor in tensors)
    deltas = {
        form: count for form, count in sorted(outcomes.items()) if form not in (REUSED, FULL_FORM)
    }
    extras = record.get("extras", [])
    return CommitResult(
        commit_id=commit_id,
        manifest_root=record["manifest"],
        branch=branch,
        parent_id=record["parents"][0] if record["parents"] else None,
        tensors=len(tensors),
        reused=outcomes[REUSED],
        stored_full=outcomes[FULL_FORM],
        stored_delta=deltas,
        payload_bytes=payload_bytes,
        seconds=time.perf_counter() - started,
        extras=len(extras),
        extras_bytes=sum(entry["nbytes"] for entry in extras),
        skipped=extras_plan.skipped,
        extras_carried=bool(extras_plan.carried),
        parent_extras=extras_plan.parent_count,
    )


def _first_parents(store: ObjectStore, commit_id: str) -> Iterator[tuple[str, dict]]:
    current: str | None = commit_id
    while current is not None:
        record = store.get_commit(current)
        yield current, record
        current = record["parents"][0] if record["parents"] else None


def _identity(entry: Mapping) -> tuple[str, tuple[int, ...], str]:
    """What makes two manifest entries of one name equal: dtype, shape, and sha256."""
    return entry["dtype"], tuple(entry["shape"]), entry["sha256"]
