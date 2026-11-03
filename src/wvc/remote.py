"""Remotes, push, pull, and clone (spec section 5.2).

A remote is another wvc store reached through a transport (``wvc.transport``): a directory
path, whose files are laid out as under a repository's ``.wvc`` (so another repository's
``.wvc`` directory works as a remote), or ``hf://OWNER/NAME[@REVISION]``, a Hub model
repository holding them under ``wvc/``. Remotes are recorded in ``.wvc/config.json`` under
``remotes`` as ``{NAME: {"url": URL}}``.

``push`` sends the commits reachable from a branch head through all parents that the remote
lacks, parents first; for each, the payloads (file headers, extras, then those of the tensors'
delta chains), the object records (every base before its deltas), the manifest, and the
commit record, in that order. The remote ref is set last, and only when it is absent or an
ancestor of the head (a fast-forward). ``pull`` mirrors this into the local store through its
validating API: every received payload is hashed and compared with its name, and every record
validated, before anything referring to it is written; every object is decoded and must
reproduce its tensor id before its record is written; a delta's depth (its chain length) is
counted in the receiving store. Verification markers (``wvc.reuse``) are never sent or
received: push and pull name every file they move. Every remote file is read under a bound
known in advance: an object or extra payload at the exact size its record implies, a header
payload, record, or ref under the store's own limits. ``clone`` is init, ``remote add
origin``, and pull.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np

from wvc.extras import commit_extras
from wvc.file_layout import MAX_PREFIX_BYTES, validate_file_records
from wvc.fsio import loads_object
from wvc.records import (
    IntegrityError,
    StoreError,
    check_manifest,
    commit_record_id,
    expected_payload_size,
    is_id,
    validate_commit,
    validate_object,
)
from wvc.refs import REF_LIMIT
from wvc.repo import init_repository
from wvc.sources import parse_hub_ref
from wvc.store import (
    DEFAULT_BRANCH,
    MAX_CHAIN_DEPTH,
    OBJECT_LIMIT,
    RECORD_LIMIT,
    ObjectStore,
    manifest_json,
    record_json,
)
from wvc.transport import DirTransport, HfTransport, Transport

DEFAULT_REMOTE = "origin"
HF_SCHEME = "hf://"
URL_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
REMOTE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
MAX_BATCH_BYTES = 1 << 30  # push hands write_many about this many bytes at most per call

CommitList = list[tuple[str, dict]]  # (commit id, record), parents before children


@dataclass(frozen=True)
class TransferResult:
    """What a push or pull moved, and where the receiving side's branch points after it."""

    remote: str
    branch: str
    previous: str | None  # the receiving side's branch before; None when it did not exist
    head: str  # the receiving side's branch after
    commits: int  # commit records transferred
    objects: int  # object records transferred
    payloads: int  # payloads transferred: file headers, extras, and tensor payloads
    nbytes: int  # bytes of the files written on the receiving side (refs excluded)
    seconds: float

    @property
    def updated(self) -> bool:
        """True when the receiving side's branch moved."""
        return self.previous != self.head


@dataclass(frozen=True)
class PushResult(TransferResult):
    """What ``push`` sent; ``previous`` and ``head`` describe the remote's branch."""


@dataclass(frozen=True)
class PullResult(TransferResult):
    """What ``pull`` received; ``previous`` and ``head`` describe the local branch."""


@dataclass(frozen=True)
class _Moved:
    """What the transfer of one or more commits moved."""

    objects: int = 0
    payloads: int = 0
    nbytes: int = 0

    @staticmethod
    def total(parts: Iterable[_Moved]) -> _Moved:
        items = list(parts)
        return _Moved(
            objects=sum(part.objects for part in items),
            payloads=sum(part.payloads for part in items),
            nbytes=sum(part.nbytes for part in items),
        )


# Remotes in config.json


def add_remote(store: ObjectStore, name: str, url: str) -> str:
    """Record remote ``name`` in ``config.json`` and return the URL as stored.

    A directory path is stored absolute; an ``hf://`` URL is checked and stored as given.
    StoreError when the name is invalid or taken, or the URL is not a supported form.
    """
    if not isinstance(name, str) or REMOTE_NAME.fullmatch(name) is None:
        raise StoreError(
            f"invalid remote name {name!r}: use up to 100 letters, digits, '.', '_', or '-', "
            "starting with a letter or digit"
        )
    stored = normalize_url(url)
    config = store.config()
    entries = _remote_entries(config)
    if name in entries:
        raise StoreError(f"remote {name!r} already exists ({entries[name]['url']})")
    store.write_config({**config, "remotes": {**entries, name: {"url": stored}}})
    return stored


def list_remotes(store: ObjectStore) -> dict[str, str]:
    """Remote names and URLs from ``config.json``, sorted by name."""
    entries = _remote_entries(store.config())
    return {name: entries[name]["url"] for name in sorted(entries)}


def remote_url(store: ObjectStore, name: str) -> str:
    """The URL of remote ``name``; StoreError when there is no such remote."""
    remotes = list_remotes(store)
    if name not in remotes:
        raise StoreError(f"no remote named {name!r}; add one with 'wvc remote add {name} URL'")
    return remotes[name]


def normalize_url(url: str) -> str:
    """Check a remote URL; return it as stored: ``hf://`` as given, a directory absolute."""
    if isinstance(url, str) and url.startswith(HF_SCHEME):
        parse_hub_ref(url.removeprefix(HF_SCHEME), f"invalid remote URL {url!r}")
        return url
    return str(_directory(url).resolve())


def transport_for(url: str) -> Transport:
    """The transport of a remote URL: a directory path, or ``hf://REPO_ID[@REVISION]``."""
    if isinstance(url, str) and url.startswith(HF_SCHEME):
        repo, revision = parse_hub_ref(url.removeprefix(HF_SCHEME), f"invalid remote URL {url!r}")
        return HfTransport(repo, revision)
    return DirTransport(_directory(url))


def _directory(url: object) -> Path:
    if not isinstance(url, str) or not url.strip():
        raise StoreError(f"a remote URL must be non-empty text, got {url!r}")
    if URL_SCHEME.match(url):
        raise StoreError(
            f"unsupported remote URL {url!r}: use a directory path or hf://OWNER/NAME[@REVISION]"
        )
    return Path(url).expanduser()


def _remote_entries(config: dict) -> dict[str, dict]:
    """The ``remotes`` table of a config, checked (empty when absent)."""
    entries = config.get("remotes", {})
    valid = isinstance(entries, dict) and all(
        isinstance(name, str) and isinstance(entry, dict) and isinstance(entry.get("url"), str)
        for name, entry in entries.items()
    )
    if not valid:
        raise StoreError('config.json: remotes must map names to {"url": URL}')
    return entries


# Push, pull, and clone


def push(
    store: ObjectStore, remote_name: str = DEFAULT_REMOTE, branch: str | None = None
) -> PushResult:
    """Send ``branch`` (default: the current branch) to a remote and fast-forward its ref.

    StoreError, before anything is sent, when the branch has no commits or the remote's
    branch is not an ancestor of the local head.
    """
    started = time.perf_counter()
    remote = transport_for(remote_url(store, remote_name))
    branch = store.current_branch() if branch is None else branch
    head = store.read_ref(branch)  # validates the branch name
    if head is None:
        raise StoreError(f"branch {branch!r} has no commits to push")
    history, _ = _walk(head, store.get_commit, _never)
    previous = _remote_ref(remote, branch)
    if previous is not None and previous not in {commit_id for commit_id, _ in history}:
        raise StoreError(
            f"refusing to push {branch!r}: {remote_name} has it at {previous[:12]}, which is "
            f"not an ancestor of {head[:12]} (the histories diverge or the remote is ahead)"
        )
    missing = [(cid, record) for cid, record in history if not remote.exists(_commit_path(cid))]
    moved = _Moved.total(_push_commit(store, remote, cid, record) for cid, record in missing)
    if previous != head:
        remote.write(_ref_path(branch), f"{head}\n".encode())
    return PushResult(
        remote=remote_name,
        branch=branch,
        previous=previous,
        head=head,
        commits=len(missing),
        objects=moved.objects,
        payloads=moved.payloads,
        nbytes=moved.nbytes,
        seconds=time.perf_counter() - started,
    )


def pull(
    store: ObjectStore, remote_name: str = DEFAULT_REMOTE, branch: str | None = None
) -> PullResult:
    """Fetch ``branch`` (default: the current branch) from a remote; fast-forward it locally.

    When the local branch already contains the remote head, nothing changes. StoreError,
    before anything is stored, when the histories diverge; IntegrityError when received
    bytes do not match their names, before anything referring to them is stored.
    """
    remote = transport_for(remote_url(store, remote_name))
    return _pull_from(store, remote, remote_name, branch)


def clone(url: str, directory: str | Path) -> PullResult:
    """Create a repository in ``directory``, add ``url`` as ``origin``, and pull ``main``.

    The URL, its transport, and the remote's ``main`` are checked before the repository is
    created. A clone that fails later keeps what it stored; ``wvc pull`` there resumes.
    """
    remote = transport_for(url)
    if _remote_ref(remote, DEFAULT_BRANCH) is None:
        raise StoreError(f"{url} has no branch {DEFAULT_BRANCH!r} to clone")
    store = ObjectStore(init_repository(directory))
    add_remote(store, DEFAULT_REMOTE, url)
    return _pull_from(store, remote, DEFAULT_REMOTE, DEFAULT_BRANCH)


def _pull_from(
    store: ObjectStore, remote: Transport, remote_name: str, branch: str | None
) -> PullResult:
    started = time.perf_counter()
    branch = store.current_branch() if branch is None else branch
    previous = store.read_ref(branch)  # validates the branch name
    head = _remote_ref(remote, branch)
    if head is None:
        raise StoreError(f"remote {remote_name} has no branch {branch!r}")
    missing, known = _walk(head, lambda cid: _fetch_commit(remote, cid), store.has_commit)
    target = _pull_target(store, previous, head, known)
    moved = _Moved.total(_pull_commit(store, remote, cid, record) for cid, record in missing)
    if target != previous:
        store.write_ref(branch, target, expected=previous)
    return PullResult(
        remote=remote_name,
        branch=branch,
        previous=previous,
        head=target,
        commits=len(missing),
        objects=moved.objects,
        payloads=moved.payloads,
        nbytes=moved.nbytes,
        seconds=time.perf_counter() - started,
    )


def _pull_target(store: ObjectStore, previous: str | None, head: str, known: set[str]) -> str:
    """The local branch after pulling ``head``: head on a fast-forward, else ``previous``
    when it already contains head; StoreError when the histories diverge.

    ``known`` holds the local commits where the walk from ``head`` stopped, so the
    ancestors of ``head`` are the commits walked plus the local ancestors of ``known``.
    """
    if previous is None or previous in _ancestry(store, known):
        return head
    if head in known and head in _ancestry(store, [previous]):
        return previous
    raise StoreError(
        f"refusing to pull: the local branch is at {previous[:12]}, which is not an ancestor "
        f"of the remote's {head[:12]} (the histories diverge)"
    )


# One commit


def _push_commit(store: ObjectStore, remote: Transport, commit_id: str, record: dict) -> _Moved:
    """Send what one commit needs that the remote lacks: payloads, objects, manifest, record."""
    files = validate_file_records(record["files"], f"commit {commit_id}")
    extras = commit_extras(record, f"commit {commit_id}")
    manifest = store.get_manifest(record["manifest"])
    objects = _missing_objects(
        _tensor_ids(manifest, files),
        lambda tensor_id: remote.exists(_object_path(tensor_id)),
        store.get_object,
    )
    payloads = _missing_payloads(
        files, extras, objects, lambda pid: remote.exists(_payload_path(pid))
    )
    root = manifest["root"]
    manifest_pairs = (
        []
        if remote.exists(_manifest_path(root))
        else [(_manifest_path(root), manifest_json(manifest))]
    )
    records = [
        *((_object_path(obj["id"]), record_json(obj)) for obj in objects),
        *manifest_pairs,
        (_commit_path(commit_id), record_json(record)),
    ]
    sizes = _payload_sizes(objects, extras)
    headers = {entry["header"] for entry in files}
    payload_pairs = (
        (_payload_path(pid), _local_payload(store, pid, sizes.get(pid), pid in headers))
        for pid in payloads
    )
    nbytes = _send(remote, chain(payload_pairs, records))
    return _Moved(objects=len(objects), payloads=len(payloads), nbytes=nbytes)


def _pull_commit(store: ObjectStore, remote: Transport, commit_id: str, record: dict) -> _Moved:
    """Store one remote commit: payloads, objects (bases first), the manifest, the record.

    The commit record was checked against its id when it was fetched; object records are
    validated when fetched and again by ``put_object``, with depths counted locally, and
    decoded before they are written (``_put_objects``).
    """
    files = validate_file_records(record["files"], f"remote commit {commit_id}")
    extras = commit_extras(record, f"remote commit {commit_id}")
    root = record["manifest"]
    new_manifest = not store.has_manifest(root)
    manifest = _fetch_manifest(remote, root) if new_manifest else store.get_manifest(root)
    objects = _missing_objects(
        _tensor_ids(manifest, files),
        store.has_object,
        lambda tensor_id: _fetch_object(remote, tensor_id),
    )
    payloads = _missing_payloads(files, extras, objects, store.has_payload)
    sizes = _payload_sizes(objects, extras)
    payload_bytes = sum(_pull_payload(store, remote, pid, sizes.get(pid)) for pid in payloads)
    placed = _put_objects(store, objects)
    store.put_manifest(manifest)
    store.put_commit(record)
    record_bytes = sum(len(record_json(item)) for item in (*placed, record))
    manifest_bytes = len(manifest_json(manifest)) if new_manifest else 0
    nbytes = payload_bytes + record_bytes + manifest_bytes
    return _Moved(objects=len(objects), payloads=len(payloads), nbytes=nbytes)


def _payload_sizes(objects: list[dict], extras: list[dict]) -> dict[str, int]:
    """The exact length of each object and extra payload, as its record implies."""
    return {
        **{obj["payload"]: expected_payload_size(obj) for obj in objects},
        **{entry["payload"]: entry["nbytes"] for entry in extras},
    }


def _local_payload(store: ObjectStore, pid: str, size: int | None, header: bool) -> bytes:
    """A local payload to send: an object's at the size its record implies, a header capped."""
    if header and size is None:
        return store.get_payload(pid, limit=MAX_PREFIX_BYTES)
    return store.get_payload(pid, size=size)


def _put_objects(store: ObjectStore, objects: list[dict]) -> list[dict]:
    """Store received object records, bases first, each decoded to its id before it is written.

    Every record gets its depth counted in this store (depth is the length of a delta chain, so
    a base stored as a delta on the remote may be stored in full here); a delta's base must
    already be stored. Each record is then decoded from the payloads already received, and
    IntegrityError stops the pull before a record that does not reproduce its tensor is
    written. A written record is marked as verified. The decoded bytes of the previous record
    are kept only while the next record is a delta on it, so a chain is decoded once.
    """
    placed: list[dict] = []
    previous: tuple[str, np.ndarray] | None = None
    for record in objects:
        if "base" in record:
            record = {**record, "depth": store.get_object(record["base"])["depth"] + 1}
        base = previous[1] if previous and previous[0] == record.get("base") else None
        previous = None  # released before the decode when this record does not use it
        data = store.check_decodes(record, base)
        store.put_object(record, verified=True)
        placed.append(record)
        previous = (record["id"], data)
    return placed


def _pull_payload(store: ObjectStore, remote: Transport, pid: str, size: int | None) -> int:
    """Fetch a payload, check that its bytes hash to its name, store it; return its size.

    ``size`` is what an object or extra record implies; without one the payload is a file
    header, read under MAX_PREFIX_BYTES.
    """
    data = remote.read(_payload_path(pid), limit=MAX_PREFIX_BYTES, size=size)
    actual = hashlib.sha256(data).hexdigest()
    if actual != pid:
        raise IntegrityError(f"remote payload {pid}: the received bytes hash to {actual}")
    store.put_payload(data)
    return len(data)


def _send(remote: Transport, pairs: Iterable[tuple[str, bytes]]) -> int:
    """Write pairs in order, through ``write_many`` calls of about MAX_BATCH_BYTES at most.

    Returns the bytes sent. Only the current batch is held in memory.
    """
    total, batch_bytes = 0, 0
    batch: list[tuple[str, bytes]] = []
    for relpath, data in pairs:
        batch.append((relpath, data))
        batch_bytes += len(data)
        total += len(data)
        if batch_bytes >= MAX_BATCH_BYTES:
            remote.write_many(batch)
            batch, batch_bytes = [], 0
    if batch:
        remote.write_many(batch)
    return total


# Commit graphs and delta chains


def _walk(
    head: str, fetch: Callable[[str], dict], known: Callable[[str], bool]
) -> tuple[CommitList, set[str]]:
    """Commits reachable from ``head`` through all parents that ``known`` does not report,
    parents first, with their records; and the known commits where the walk stopped."""
    order: CommitList = []
    records: dict[str, dict] = {}
    stopped: set[str] = set()
    stack: list[tuple[str, bool]] = [(head, False)]
    while stack:
        commit_id, expanded = stack.pop()
        if expanded:
            order.append((commit_id, records[commit_id]))
        elif commit_id in records or commit_id in stopped:
            continue
        elif known(commit_id):
            stopped.add(commit_id)
        else:
            records[commit_id] = fetch(commit_id)
            stack.append((commit_id, True))
            stack.extend((parent, False) for parent in reversed(records[commit_id]["parents"]))
    return order, stopped


def _ancestry(store: ObjectStore, starts: Iterable[str]) -> set[str]:
    """Local commits reachable from ``starts`` through all parents, the starts included."""
    seen: set[str] = set()
    stack = list(starts)
    while stack:
        commit_id = stack.pop()
        if commit_id not in seen:
            seen.add(commit_id)
            stack.extend(store.get_commit(commit_id)["parents"])
    return seen


def _never(_: str) -> bool:
    return False


def _tensor_ids(manifest: dict, files: list[dict]) -> list[str]:
    """Every tensor id a commit needs: its manifest's and its file records', sorted."""
    named = {entry["sha256"] for entry in manifest["tensors"].values()}
    return sorted(named.union(*(entry["tensors"] for entry in files)))


def _missing_objects(
    tensor_ids: Iterable[str], have: Callable[[str], bool], fetch: Callable[[str], dict]
) -> list[dict]:
    """Object records the receiving side lacks for these tensors, each base before its deltas.

    Each delta chain is followed down to the first object ``have`` reports, or to its full
    object; every record comes from ``fetch``.
    """
    ordered: list[dict] = []
    planned: set[str] = set()
    for tensor_id in tensor_ids:
        found = _chain(tensor_id, lambda oid: oid in planned or have(oid), fetch)
        ordered.extend(reversed(found))
        planned.update(record["id"] for record in found)
    return ordered


def _chain(tensor_id: str, have: Callable[[str], bool], fetch: Callable[[str], dict]) -> list[dict]:
    """Records from ``tensor_id`` down its bases until one ``have`` reports, deltas first.

    IntegrityError when an id comes back or the chain is deeper than any store follows.
    """
    records: list[dict] = []
    seen: set[str] = set()
    current: str | None = tensor_id
    while current is not None and not have(current):
        if current in seen:
            raise IntegrityError(f"object {tensor_id}: the delta chain revisits {current}")
        if len(records) > MAX_CHAIN_DEPTH:
            raise IntegrityError(
                f"object {tensor_id}: the delta chain is over {MAX_CHAIN_DEPTH} deep"
            )
        seen.add(current)
        records.append(fetch(current))
        current = records[-1].get("base")
    return records


def _missing_payloads(
    files: list[dict], extras: list[dict], objects: list[dict], have: Callable[[str], bool]
) -> list[str]:
    """File header, extras, then object payloads, each once, that the receiver lacks."""
    wanted = [
        *(entry["header"] for entry in files),
        *(entry["payload"] for entry in extras),
        *(obj["payload"] for obj in objects),
    ]
    return [pid for pid in dict.fromkeys(wanted) if not have(pid)]


# Remote files


def _remote_ref(remote: Transport, branch: str) -> str | None:
    """The commit the remote's branch names, or None when the remote lacks the branch."""
    relpath = _ref_path(branch)
    if not remote.exists(relpath):
        return None
    text = remote.read(relpath, limit=REF_LIMIT).decode("utf-8", errors="replace")
    if not (text.endswith("\n") and is_id(text[:-1])):
        raise StoreError(
            f"remote {relpath}: malformed ref (not 64 lowercase hex characters and a newline)"
        )
    return text[:-1]


def _fetch_json(remote: Transport, relpath: str, limit: int) -> dict:
    """A remote file of at most ``limit`` bytes holding one JSON object; IntegrityError
    when it does not, deep nesting included."""
    return loads_object(remote.read(relpath, limit=limit), f"remote {relpath}")


def _fetch_commit(remote: Transport, commit_id: str) -> dict:
    """A remote commit record, checked to hash to ``commit_id`` and validated."""
    record = _fetch_json(remote, _commit_path(commit_id), RECORD_LIMIT)
    try:
        actual = commit_record_id(record)
    except StoreError as exc:
        raise IntegrityError(f"remote commit {commit_id}: {exc}") from exc
    if actual != commit_id:
        raise IntegrityError(f"remote commit {commit_id}: the record hashes to {actual}")
    validate_commit(record, f"remote commit {commit_id}")
    return record


def _fetch_object(remote: Transport, tensor_id: str) -> dict:
    """A remote object record, validated and checked to describe ``tensor_id``."""
    record = _fetch_json(remote, _object_path(tensor_id), OBJECT_LIMIT)
    validate_object(record)
    if record["id"] != tensor_id:
        raise IntegrityError(f"remote object {tensor_id}: the record describes {record['id']}")
    return record


def _fetch_manifest(remote: Transport, root: str) -> dict:
    """A remote manifest, checked to state ``root`` and to hash to it."""
    manifest = _fetch_json(remote, _manifest_path(root), RECORD_LIMIT)
    if manifest.get("root") != root:
        raise IntegrityError(f"remote manifest {root}: the file states another root")
    check_manifest(manifest, f"remote manifest {root}")
    return manifest


def _payload_path(pid: str) -> str:
    return f"payloads/{pid[:2]}/{pid}"


def _object_path(tensor_id: str) -> str:
    return f"objects/{tensor_id[:2]}/{tensor_id}.json"


def _manifest_path(root: str) -> str:
    return f"manifests/{root}.json"


def _commit_path(commit_id: str) -> str:
    return f"commits/{commit_id}.json"


def _ref_path(branch: str) -> str:
    return f"refs/heads/{branch}"
