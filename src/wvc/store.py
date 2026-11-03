"""Local object store of spec section 3.1: payloads, tensor objects, manifests, and commits.

A repository is any directory holding ``.wvc/``. Payloads are bytes named by their SHA-256. An
object record says how the tensor whose raw bytes hash to its id is stored: in full
(``zstd-planes``) or as a delta against a base tensor (spec section 3.2); a ``delta-lowrank``
holds adapter factors and is rebuilt from its base by ``lowrank-merge/0``. Manifests and commit
records are JSON. Branches and HEAD are handled by ``wvc.refs``, the layout constants, limits,
and config by ``wvc.store_config``, and decoding by form and the delta-chain checks, which need
no I/O, by ``wvc.store_objects``; this module re-exports what other modules import from it.

Stored content is rewritten only to repair it: a payload or commit record already stored is
checked against its id and replaced when wrong, and an object record or manifest already
stored must equal the new one (IntegrityError otherwise). Every write goes to a temporary file
in the destination directory that is fsynced and renamed into place, and the directory is then
fsynced; temporaries older than an hour are removed from each directory written to. No record
may refer to something the store does not hold.

Nothing on disk is trusted: ``.wvc`` and its directories must not be symlinks (checked once per
instance, and again by ``check_layout``), and files are opened without following links or
blocking, must be regular files, and are refused before reading when larger than their caps.
Reads verify payload SHA-256s, the tensor SHA-256 at every level of a delta chain (whose depths
must fall by one per level and stay within the smaller of ``max_delta_depth`` and 64), manifest
roots, commit ids, and that a commit's files hold exactly its manifest's tensors; reuse decodes
what it reuses unless a local marker says it was checked (``wvc.reuse``). No object
record may claim a tensor (or a low-rank factor) larger than the smaller of ``max_tensor_bytes``
(default 16 GiB) and 1 TiB, nor a file record a file larger than 1 TiB: such records are refused
when stored and when read, before anything of their size is allocated.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from wvc.codec import CodecError, choose_encoding, encode_lowrank
from wvc.extras import check_extra_payloads
from wvc.file_layout import MAX_PREFIX_BYTES, manifest_tensors, validate_file_records
from wvc.fsio import (
    fsync_directory,
    holds_digest,
    read_json_object,
    read_regular,
    regular_file_digest,
    remove_stale_temporaries,
    require_directory,
    temporary_name,
    write_atomic,
)
from wvc.hashing import digest_bytes
from wvc.ids import RawBytes, is_id
from wvc.lowrank import PROCEDURE, LowRankError, LowRankFactors
from wvc.records import (
    FULL_FORM,
    LOWRANK_FORM,
    OBJECT_FORMAT,
    DeltaDepthError,
    IntegrityError,
    StoreError,
    check_id,
    check_manifest,
    check_tensor_bytes,
    commit_record_id,
    expected_payload_size,
    validate_commit,
    validate_object,
)
from wvc.refs import ANY, Refs, Unset
from wvc.reuse import ReuseMixin
from wvc.store_config import (
    CONFIG_LIMIT,
    DEFAULT_BRANCH,
    LAYOUT_DIRS,
    MAX_CHAIN_DEPTH,
    OBJECT_LIMIT,
    PAYLOAD_LIMIT,
    RECORD_LIMIT,
    REPO_DIR,
    SHARD,
    SHARDED_DIRS,
    TENSOR_BYTES_CEILING,
    check_config,
    default_config,
    depth_limit,
    find_repo,
    manifest_json,
    record_json,
    tensor_limit,
)
from wvc.store_objects import (
    decode_object,
    lowrank_fields,
    merge_lowrank_record,
    object_chain,
    payload_view,
    tensor_bytes,
)

__all__ = [  # what other modules import from here; the config names live in wvc.store_config
    "DEFAULT_BRANCH",
    "MAX_CHAIN_DEPTH",
    "OBJECT_LIMIT",
    "RECORD_LIMIT",
    "TENSOR_BYTES_CEILING",
    "DeltaDepthError",
    "IntegrityError",
    "ObjectStore",
    "StoreError",
    "find_repo",
    "manifest_json",
    "record_json",
]


class ObjectStore(ReuseMixin):
    """The ``.wvc`` store of one repository directory (the parent of ``.wvc``).

    An instance caches the validated config and which directories passed the symlink checks;
    ``payload_bytes_written`` counts the payload bytes this instance actually wrote.
    """

    def __init__(self, repo: str | Path) -> None:
        self.repo = Path(repo)
        self.dir = self.repo / REPO_DIR
        self.payload_bytes_written = 0
        self._config: dict | None = None
        self._layout_ok = False
        self._checked: set[Path] = set()  # shard directories found not to be links
        self._cleaned: set[Path] = set()  # directories rid of stale temporaries

    def init_layout(self) -> None:
        """Create ``.wvc`` with the layout of spec 3.1, the default config, and HEAD on main.

        The layout is built in a temporary sibling directory renamed to ``.wvc`` last, so an
        interrupted init leaves no half-made repository; stale ones are removed first.
        """
        self.repo.mkdir(parents=True, exist_ok=True)
        remove_stale_temporaries(self.repo)
        if os.path.lexists(self.dir):
            raise StoreError(f"{self.dir} already exists")
        building = self.repo / temporary_name()
        try:
            building.mkdir()
            for sub in LAYOUT_DIRS:
                (building / sub).mkdir()
            write_atomic(building / "config.json", record_json(default_config()))
            write_atomic(building / "HEAD", f"ref: refs/heads/{DEFAULT_BRANCH}\n".encode())
            if os.path.lexists(self.dir):
                raise StoreError(f"{self.dir} already exists")
            os.rename(building, self.dir)
        except BaseException:
            shutil.rmtree(building, ignore_errors=True)
            raise
        fsync_directory(self.repo)

    def exists(self) -> bool:
        """True when the repository directory holds ``.wvc``."""
        return self.dir.is_dir()

    def check_layout(self, *, shards: bool = True) -> None:
        """StoreError unless ``.wvc`` and its directories are directories, not symlinks.

        With ``shards`` (as ``wvc fsck`` runs it), every shard directory of ``objects`` and
        ``payloads`` is checked too; otherwise each is checked the first time it is used.
        """
        require_directory(self.dir, "the repository store")
        for sub in LAYOUT_DIRS:
            require_directory(self.dir / sub, "store directory")
        self._layout_ok = True
        if shards:
            for kind in SHARDED_DIRS:
                names = sorted(os.listdir(self.dir / kind))
                for directory in (
                    self.dir / kind / name for name in names if SHARD.fullmatch(name)
                ):
                    require_directory(directory, "store directory")
                    self._checked.add(directory)

    def config(self) -> dict:
        """The validated ``.wvc/config.json`` (read once per instance); other keys are kept."""
        if self._config is None:
            path = self._layout_path("config.json")
            config = read_json_object(path, "repository config", CONFIG_LIMIT)
            check_config(config, path)
            self._config = config
        return dict(self._config)

    def write_config(self, config: dict) -> None:
        """Validate a config as ``config()`` does, then replace ``config.json`` atomically."""
        path = self._layout_path("config.json")
        check_config(config, path)
        self._write(path, record_json(config))
        self._config = dict(config)

    # Branches and HEAD (wvc.refs)

    @property
    def refs(self) -> Refs:
        """The ref operations over this store's layout and commits."""
        return Refs(self._layout_path, self._write, self.has_commit, self._commit_ids)

    def branches(self) -> list[str]:
        """The branch names under ``refs/heads``, sorted (``Refs.branches``)."""
        return self.refs.branches()

    def read_ref(self, name: str) -> str | None:
        """The commit id branch ``name`` points to, or None (``Refs.read_ref``)."""
        return self.refs.read_ref(name)

    def write_ref(self, name: str, commit_id: str, *, expected: str | Unset | None = ANY) -> None:
        """Point branch ``name`` at a stored commit, compare-and-swap (``Refs.write_ref``)."""
        self.refs.write_ref(name, commit_id, expected=expected)

    def check_branch_name(self, name: str) -> None:
        """StoreError unless ``name`` may be a branch (``Refs.check_branch_name``)."""
        self.refs.check_branch_name(name)

    def current_branch(self) -> str:
        """The branch HEAD names (``Refs.current_branch``)."""
        return self.refs.current_branch()

    def set_head(self, branch: str) -> None:
        """Make ``branch`` the current branch (``Refs.set_head``)."""
        self.refs.set_head(branch)

    def head(self) -> str | None:
        """The commit HEAD's branch points to, or None before its first commit (``Refs.head``)."""
        return self.refs.head()

    def resolve(self, ref: str) -> str:
        """Resolve HEAD, a commit id, a branch, or a unique prefix (``Refs.resolve``)."""
        return self.refs.resolve(ref)

    # Payloads

    def put_payload(self, data: RawBytes) -> str:
        """Store bytes under their SHA-256 and return it.

        A payload already stored is hashed again and replaced when its bytes are wrong
        (content addressing makes the repair unobservable), and every verification marker is
        then removed. Bytes written are counted in ``payload_bytes_written``.
        """
        view = payload_view(data)
        pid = digest_bytes(view)
        path = self._sharded("payloads", pid)
        if not holds_digest(path, pid, view.size):
            if os.path.lexists(path):  # a damaged payload is replaced: markers may be stale
                self.clear_verified()
            self._write(path, view.data)
            self.payload_bytes_written += view.size
        return pid

    def get_payload(
        self, pid: str, *, size: int | None = None, limit: int = PAYLOAD_LIMIT
    ) -> bytes:
        """Read a payload, verifying its SHA-256 (IntegrityError on mismatch).

        With ``size`` (what its object record implies), any other length is refused before
        reading; without it, a payload longer than ``limit`` bytes is.
        """
        path = self._sharded("payloads", check_id(pid, "payload id"))
        try:
            data = read_regular(path, f"payload {pid}", limit=limit, size=size)
        except FileNotFoundError as exc:
            raise StoreError(f"payload {pid} is not stored") from exc
        actual = hashlib.sha256(data).hexdigest()
        if actual != pid:
            raise IntegrityError(f"payload {pid}: stored bytes hash to {actual}")
        return data

    def verify_payload(self, pid: str) -> int:
        """Hash a stored payload in chunks (any size); return its length, IntegrityError if bad."""
        path = self._sharded("payloads", check_id(pid, "payload id"))
        try:
            actual, nbytes = regular_file_digest(path, f"payload {pid}")
        except FileNotFoundError as exc:
            raise StoreError(f"payload {pid} is not stored") from exc
        if actual != pid:
            raise IntegrityError(f"payload {pid}: stored bytes hash to {actual}")
        return nbytes

    def has_payload(self, pid: str) -> bool:
        """True when a payload with this SHA-256 is stored (its bytes are not read)."""
        return os.path.lexists(self._sharded("payloads", check_id(pid, "payload id")))

    def payload_size(self, pid: str) -> int | None:
        """A stored payload's length (not read), or None when it is missing.

        IntegrityError when the payload is not a regular file.
        """
        try:
            info = os.lstat(self._sharded("payloads", check_id(pid, "payload id")))
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(info.st_mode):
            raise IntegrityError(f"payload {pid} is not a regular file")
        return info.st_size

    # Tensor objects

    def has_object(self, tensor_id: str) -> bool:
        """True when an object record for this tensor id is stored."""
        return os.path.lexists(self._sharded("objects", check_id(tensor_id, "tensor id"), ".json"))

    def get_object(self, tensor_id: str) -> dict:
        """Read and validate the object record of a tensor id."""
        path = self._sharded("objects", check_id(tensor_id, "tensor id"), ".json")
        where = f"object {tensor_id}"
        record = read_json_object(path, where, OBJECT_LIMIT)
        validate_object(record, where)
        if record["id"] != tensor_id:
            raise IntegrityError(f"{where}: the record describes another tensor id")
        check_tensor_bytes(record, self._tensor_limit(), where)
        return record

    def put_object(self, record: dict, *, verified: bool = False, replace: bool = False) -> None:
        """Validate an object record and store it.

        An object already stored must have exactly this record (IntegrityError otherwise), unless
        ``replace`` overwrites it. Its payload, and for a delta its base object, must already be
        stored; a delta has its base's nbytes and its base's depth plus one, within the chain
        depth limit. No record may claim more than the tensor limit (``max_tensor_bytes``, at
        most 1 TiB). With ``verified`` (the caller encoded the record from the tensor's bytes,
        or decoded it to its id), a written record gets a verification marker.
        """
        validate_object(record)
        check_tensor_bytes(record, self._tensor_limit(), f"object {record['id']}")
        path = self._sharded("objects", record["id"], ".json")
        if os.path.lexists(path) and not replace:
            if self.get_object(record["id"]) != record:
                raise IntegrityError(f"object {record['id']}: another record is stored for it")
            return
        if not self.has_payload(record["payload"]):
            raise StoreError(f"object {record['id']}: payload {record['payload']} is not stored")
        if "base" in record:
            self._check_delta(record)
        self._write(path, record_json(record))
        if verified:
            self.mark_verified(record["id"])

    def put_tensor(
        self,
        raw: np.ndarray,
        itemsize: int,
        *,
        base_id: str | None = None,
        tensor_id: str | None = None,
        replace: bool = False,
    ) -> dict:
        """Store one tensor's raw bytes and return its object record.

        A tensor already stored is returned unchanged, unless ``replace`` overwrites its record
        (``replace_tensor``, which passes no base). With ``base_id``, the tensor is stored
        as a delta against that stored tensor when the delta payload is strictly smaller than
        the full one and the chain stays within the config's ``max_delta_depth``. With
        ``tensor_id``, the caller vouches that it is the SHA-256 of ``raw`` and the bytes are
        not hashed again; a wrong id gives an object that every read refuses. StoreError, before
        anything is stored, when the tensor is larger than the tensor limit.
        """
        raw = tensor_bytes(raw, itemsize)
        tensor_id = digest_bytes(raw) if tensor_id is None else check_id(tensor_id, "tensor id")
        self._check_tensor_size(raw.size, tensor_id)
        if self.has_object(tensor_id) and not replace:
            return self.get_object(tensor_id)
        base_raw, depth = self._delta_base(base_id, raw.size)
        encoded = choose_encoding(raw, itemsize, base_raw)
        delta = encoded.form != FULL_FORM
        record = {
            "format": OBJECT_FORMAT,
            "id": tensor_id,
            "nbytes": raw.size,
            "itemsize": itemsize,
            "form": encoded.form,
            "payload": self.put_payload(encoded.payload),
            "depth": depth if delta else 0,
            **encoded.fields,
            **({"base": base_id} if delta else {}),
        }
        self.put_object(record, verified=True, replace=replace)
        return record

    def put_lowrank(
        self,
        raw: np.ndarray,
        itemsize: int,
        *,
        base_id: str,
        base_dtype: str,
        shape: Sequence[int],
        factors: LowRankFactors,
        tensor_id: str | None = None,
    ) -> dict:
        """Store a tensor merged from a stored base and adapter factors as ``delta-lowrank``.

        A tensor already stored is returned unchanged. Otherwise ``lowrank-merge/0`` is
        recomputed from the stored base and the factors exactly as a read decodes the record,
        and must reproduce the tensor id: StoreError otherwise, since the caller's merge did
        not follow the procedure. ``tensor_id``, when given, is vouched for by the caller as
        the SHA-256 of ``raw``, which is then not hashed again. DeltaDepthError (a StoreError)
        when the delta would exceed the config's ``max_delta_depth``; the caller may then store
        the tensor with ``put_tensor``.
        """
        raw = tensor_bytes(raw, itemsize)
        tensor_id = digest_bytes(raw) if tensor_id is None else check_id(tensor_id, "tensor id")
        self._check_tensor_size(raw.size, tensor_id)
        if self.has_object(tensor_id):
            return self.get_object(tensor_id)
        depth = self._lowrank_depth(check_id(base_id, "base id"), raw.size)
        a_bytes, b_bytes = payload_view(factors.a_raw), payload_view(factors.b_raw)
        payload = encode_lowrank(a_bytes, b_bytes)
        record = {
            "format": OBJECT_FORMAT,
            "id": tensor_id,
            "nbytes": raw.size,
            "itemsize": itemsize,
            "form": LOWRANK_FORM,
            "payload": hashlib.sha256(payload).hexdigest(),
            "depth": depth,
            "base": base_id,
            "a_size": a_bytes.size,
            **lowrank_fields(base_dtype, shape, factors),
        }
        validate_object(record)
        self._check_lowrank_merge(record, payload)
        self.put_payload(payload)
        self.put_object(record, verified=True)
        return record

    def read_tensor(self, tensor_id: str) -> np.ndarray:
        """Rebuild a tensor's raw bytes (contiguous uint8), following its chain of bases.

        Every level of the chain is checked against its id; IntegrityError on a mismatch, or
        when the chain is malformed (``wvc.store_objects.object_chain``).
        """
        chain = object_chain(tensor_id, self.get_object, self._depth_limit)
        data = self._decode(chain[-1], None)
        for record in reversed(chain[:-1]):
            data = self._decode(record, data)
        return data

    def _depth_limit(self) -> int:
        """The deepest delta chain this store writes or follows."""
        return depth_limit(self.config())

    def _tensor_limit(self) -> int:
        """The largest tensor, in bytes, this store writes or reads."""
        return tensor_limit(self.config())

    def _check_tensor_size(self, nbytes: int, tensor_id: str) -> None:
        limit = self._tensor_limit()
        if nbytes > limit:
            raise StoreError(
                f"tensor {tensor_id}: larger than the tensor limit of {limit} bytes "
                "(max_tensor_bytes in .wvc/config.json)"
            )

    def _check_delta(self, record: dict) -> None:
        """A new delta needs its stored base's nbytes and depth plus one, within the limit."""
        base, limit = self.get_object(record["base"]), self._depth_limit()
        if (record["nbytes"], record["depth"]) != (base["nbytes"], base["depth"] + 1):
            raise StoreError(
                f"object {record['id']}: a delta needs its base's nbytes ({base['nbytes']}) "
                f"and its base's depth plus one ({base['depth'] + 1})"
            )
        if record["depth"] > limit:
            raise DeltaDepthError(
                f"object {record['id']}: depth {record['depth']} is over the limit {limit}"
            )

    def _delta_base(self, base_id: str | None, nbytes: int) -> tuple[np.ndarray | None, int]:
        """The base bytes and the depth of a delta on them, or ``(None, 0)`` for no delta."""
        if base_id is None:
            return None, 0
        base = self.get_object(base_id)
        depth = base["depth"] + 1
        if base["nbytes"] != nbytes or depth > self._depth_limit():
            return None, 0
        return self.read_tensor(base_id), depth

    def _lowrank_depth(self, base_id: str, nbytes: int) -> int:
        """The depth of a low-rank delta on a stored base of ``nbytes`` bytes.

        DeltaDepthError when the depth would exceed the config's ``max_delta_depth``.
        """
        base = self.get_object(base_id)
        if base["nbytes"] != nbytes:
            raise StoreError(
                f"base {base_id} holds {base['nbytes']} bytes but the merged tensor {nbytes}"
            )
        depth, limit = base["depth"] + 1, self._depth_limit()
        if depth > limit:
            raise DeltaDepthError(
                f"a delta on {base_id} would have depth {depth}, over max_delta_depth {limit}"
            )
        return depth

    def _check_lowrank_merge(self, record: dict, payload: bytes) -> None:
        """Decode a new delta-lowrank record as a read would; it must give the tensor's id."""
        try:
            merged = merge_lowrank_record(record, self.read_tensor(record["base"]), payload)
        except (CodecError, LowRankError) as exc:
            raise StoreError(f"object {record['id']}: {exc}") from exc
        if digest_bytes(merged) != record["id"]:
            raise StoreError(
                f"object {record['id']}: {PROCEDURE} on base {record['base']} does not reproduce "
                "the tensor; the merge did not follow the procedure"
            )

    def _decode(self, record: dict, base: np.ndarray | None) -> np.ndarray:
        """Decode one object and check the result against its id.

        ``base`` holds the decoded base of a delta and is None for the full object that ends
        a chain (only deltas have a ``base`` field).
        """
        payload = self.get_payload(record["payload"], size=expected_payload_size(record))
        try:
            data = decode_object(record, base, payload)
        except (CodecError, LowRankError) as exc:
            raise IntegrityError(f"object {record['id']}: {exc}") from exc
        if data.size != record["nbytes"] or digest_bytes(data) != record["id"]:
            raise IntegrityError(f"object {record['id']}: decoded bytes do not match the id")
        return data

    # Manifests and commits

    def put_manifest(self, manifest: dict) -> str:
        """Store a manifest exactly as ``wvc hash`` prints it and return its root.

        A manifest already stored under the root must list the same tensors (IntegrityError
        otherwise: two checkpoints whose roots collide).
        """
        root = check_manifest(manifest, "manifest")
        path = self._layout_path("manifests", f"{root}.json")
        if os.path.lexists(path):
            if self.get_manifest(root)["tensors"] != manifest["tensors"]:
                raise IntegrityError(
                    f"manifest {root}: a manifest with other tensors is stored under this root"
                )
            return root
        self._write(path, manifest_json(manifest))
        return root

    def has_manifest(self, root: str) -> bool:
        """True when a manifest with this root is stored (it is not read)."""
        return os.path.lexists(self._manifest_path(root))

    def get_manifest(self, root: str) -> dict:
        """Read a manifest, checking its format and that its root matches its file name."""
        path = self._manifest_path(root)
        manifest = read_json_object(path, f"manifest {root}", RECORD_LIMIT)
        if manifest.get("root") != root:
            raise IntegrityError(f"manifest {root}: the file states another root")
        check_manifest(manifest, f"manifest {root}")
        return manifest

    def verified_manifest(self, record: dict, where: str) -> dict:
        """The stored manifest a commit record names, checked against the record's files.

        The file records must be valid (names included), and the manifest entries rebuilt
        from their header payloads and tensor ids (``wvc.file_layout.manifest_tensors``) must
        equal the stored manifest's tensors (IntegrityError otherwise). No file may claim more
        than 1 TiB. No tensor is decoded.
        """
        files = validate_file_records(record["files"], where)
        if any(entry["nbytes"] > TENSOR_BYTES_CEILING for entry in files):
            raise StoreError(
                f"{where}: a file record claims more than the ceiling of {TENSOR_BYTES_CEILING} "
                "bytes"
            )
        manifest = self.get_manifest(record["manifest"])
        rebuilt = manifest_tensors(files, self._header_payload, self.get_object, where)
        if rebuilt != manifest["tensors"]:
            raise IntegrityError(
                f"{where}: the files do not hold the tensors of manifest {record['manifest']}"
            )
        return manifest

    def put_commit(self, record: dict) -> str:
        """Store a commit record and return its id, the SHA-256 of its canonical form.

        The record must not contain ``id``; its manifest, parents, header payloads, extras
        payloads (at their recorded sizes; spec 3.3a), and tensor objects must already be stored,
        and its files must hold exactly its manifest's tensors (``verified_manifest``). A damaged
        record file already stored under the id is replaced.
        """
        validate_commit(record, "commit record")
        new_id = commit_record_id(record)
        if not self.has_manifest(record["manifest"]):
            raise StoreError(f"commit record: manifest {record['manifest']} is not stored")
        for parent in record["parents"]:
            if not self.has_commit(parent):
                raise StoreError(f"commit record: parent {parent} is not stored")
        self.verified_manifest(record, "commit record")
        check_extra_payloads(record.get("extras", []), self.payload_size, "commit record")
        if not self._commit_intact(new_id):
            self._write(self._commit_path(new_id), record_json(record))
        return new_id

    def has_commit(self, commit_id: str) -> bool:
        """True when a commit record with this id is stored (it is not read)."""
        return os.path.lexists(self._commit_path(commit_id))

    def get_commit(self, commit_id: str) -> dict:
        """Read a commit record, verifying that its canonical form hashes to ``commit_id``."""
        record = read_json_object(self._commit_path(commit_id), f"commit {commit_id}", RECORD_LIMIT)
        try:
            actual = commit_record_id(record)
        except StoreError as exc:
            raise IntegrityError(f"commit {commit_id}: {exc}") from exc
        if actual != commit_id:
            raise IntegrityError(f"commit {commit_id}: the stored record hashes to {actual}")
        validate_commit(record, f"commit {commit_id}")
        return record

    def _commit_intact(self, commit_id: str) -> bool:
        """True when a readable record that hashes to ``commit_id`` is stored under it."""
        if not self.has_commit(commit_id):
            return False
        try:
            self.get_commit(commit_id)
        except IntegrityError:
            return False
        return True

    def _header_payload(self, pid: str) -> bytes:
        return self.get_payload(pid, limit=MAX_PREFIX_BYTES)

    # Paths and writes

    def _layout_path(self, *parts: str) -> Path:
        """A path inside ``.wvc``, once its layout passed the symlink checks."""
        if not self._layout_ok:
            self.check_layout(shards=False)
        return self.dir.joinpath(*parts)

    def _shard_dir(self, kind: str, prefix: str) -> Path:
        """The shard directory ``kind/prefix``, checked not to be a link once it exists."""
        directory = self._layout_path(kind, prefix)
        if directory not in self._checked and os.path.lexists(directory):
            require_directory(directory, "store directory")
            self._checked.add(directory)
        return directory

    def _sharded(self, kind: str, oid: str, suffix: str = "") -> Path:
        return self._shard_dir(kind, oid[:2]) / f"{oid}{suffix}"

    def _commit_path(self, commit_id: str) -> Path:
        return self._layout_path("commits", f"{check_id(commit_id, 'commit id')}.json")

    def _manifest_path(self, root: str) -> Path:
        return self._layout_path("manifests", f"{check_id(root, 'manifest root')}.json")

    def _commit_ids(self, prefix: str) -> list[str]:
        """Stored commit ids that start with ``prefix``, sorted."""
        names = os.listdir(self._layout_path("commits"))
        ids = (name.removesuffix(".json") for name in names if name.endswith(".json"))
        return sorted(oid for oid in ids if oid.startswith(prefix) and is_id(oid))

    def _write(self, path: Path, data: bytes | memoryview) -> None:
        """``write_atomic`` into a store directory, made (a shard) and checked on first use."""
        directory = path.parent
        if directory not in self._cleaned:
            directory.mkdir(exist_ok=True)  # a shard directory; the others exist since init
            require_directory(directory, "store directory")
            remove_stale_temporaries(directory)
            self._cleaned.add(directory)
        write_atomic(path, data)
