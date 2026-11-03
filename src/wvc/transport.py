"""Transports for remotes (spec section 5.2): reading and writing another store's files.

A transport addresses a remote store's files by the store's own relative paths
(``payloads/aa/pid``, ``objects/aa/id.json``, ``manifests/root.json``, ``commits/id.json``,
``refs/heads/name``); a path that is absolute or has an empty, ``.``, or ``..`` part is
refused. ``read`` is bounded by what the caller knows in advance: an exact ``size`` (from a
record), or a ``limit``; a file of any other length is refused before its bytes are read.
``DirTransport`` keeps the files under a local directory, read through ``fsio.read_regular``
(regular files only, a final symlink never followed) and written durably through
``fsio.write_atomic``. ``HfTransport`` keeps them under ``wvc/`` in a Hugging Face Hub model
repository through the optional ``huggingface_hub`` package (installed as ``wvc[hf]``); it
gets its client from the module-level factory ``make_hub_client``, which tests replace with a
fake.
"""

from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TypeAlias

from wvc.fsio import open_input, read_regular, write_atomic
from wvc.quote import shown
from wvc.records import IntegrityError, StoreError
from wvc.sources import DEFAULT_REVISION, is_hub_repo_id, is_hub_revision

HUB_PACKAGE = "huggingface_hub"
HUB_EXTRA = "wvc[hf]"
HUB_DIR = "wvc"  # the top-level directory of a Hub repository that holds the store's files
HUB_REPO_TYPE = "model"

# Aliases, because a class with a ``list`` method shadows the builtin in its annotations.
RelPaths: TypeAlias = list[str]  # store-relative paths, sorted
FilePairs: TypeAlias = list[tuple[str, bytes]]  # (relpath, data), in write order
RangeReader: TypeAlias = Callable[[str, str, str, int, int], bytes]  # repo, file, rev, start, len
PARTIAL_CONTENT = 206  # the status a satisfied range request returns
RANGE_TIMEOUT_SECONDS = 30.0  # to connect, and for each read of a range response


class Transport(ABC):
    """The files of a remote store, addressed by the store's relative paths."""

    @abstractmethod
    def exists(self, relpath: str) -> bool:
        """True when the remote holds the file ``relpath``."""

    @abstractmethod
    def read(self, relpath: str, *, limit: int, size: int | None = None) -> bytes:
        """The bytes of ``relpath``; StoreError when the remote lacks it.

        With ``size``, a file of any other length is an IntegrityError; without it, a file
        longer than ``limit`` bytes is. Either is refused before the file is read whole.
        """

    @abstractmethod
    def write(self, relpath: str, data: bytes) -> None:
        """Create or replace ``relpath`` with ``data``."""

    def write_many(self, pairs: FilePairs) -> None:
        """Write ``(relpath, data)`` pairs in order: one ``write`` each unless overridden."""
        for relpath, data in pairs:
            self.write(relpath, data)

    @abstractmethod
    def list(self, prefix: str = "") -> RelPaths:
        """The sorted relative paths of the remote's files that start with ``prefix``."""


class DirTransport(Transport):
    """A remote store in a local directory; it and its subdirectories are made on write."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"DirTransport({str(self.root)!r})"

    def exists(self, relpath: str) -> bool:
        return self._path(relpath).is_file()

    def read(self, relpath: str, *, limit: int, size: int | None = None) -> bytes:
        """A regular file's bytes; a symlink, FIFO, or device is an IntegrityError."""
        path = self._path(relpath)
        try:
            return read_regular(path, "remote file", limit=limit, size=size)
        except FileNotFoundError as exc:
            raise StoreError(f"remote directory {self.root} lacks {relpath}") from exc

    def write(self, relpath: str, data: bytes) -> None:
        """Write ``data`` durably: a ``.wvc-<uuid>.tmp`` temporary, fsync, rename, fsync."""
        path = self._path(relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, data)

    def list(self, prefix: str = "") -> RelPaths:
        """Files under ``root`` whose path starts with ``prefix``; hidden paths are skipped."""
        directory = self.root / _check_prefix(prefix).rpartition("/")[0]
        if not directory.is_dir():
            return []
        found = (path.relative_to(self.root).as_posix() for path in directory.rglob("*"))
        return sorted(
            relpath
            for relpath in found
            if relpath.startswith(prefix)
            and not any(part.startswith(".") for part in relpath.split("/"))
            and (self.root / relpath).is_file()
        )

    def _path(self, relpath: str) -> Path:
        return self.root / check_relpath(relpath)


@dataclass(frozen=True)
class HubClient:
    """What ``HfTransport`` uses from ``huggingface_hub``; tests substitute a fake."""

    api: Any  # an HfApi: list_repo_files, hf_hub_download, create_commit, model_info
    operation_add: Callable[..., Any]  # CommitOperationAdd(path_in_repo=, path_or_fileobj=)
    errors: tuple[type[Exception], ...]  # client exceptions reported as StoreError
    read_range: RangeReader  # exactly that byte range of a repository file; b"" for none


def load_hub() -> ModuleType:
    """Import ``huggingface_hub``; StoreError naming ``wvc[hf]`` when it is not installed."""
    try:
        return importlib.import_module(HUB_PACKAGE)
    except ImportError as exc:
        raise StoreError(
            f"hf:// remotes need the optional package {HUB_PACKAGE}; install {HUB_EXTRA}, "
            f"for example: pip install '{HUB_EXTRA}'"
        ) from exc


def make_hub_client() -> HubClient:
    """A default ``HfApi`` (token from the environment or a saved login) and its types.

    ``huggingface_hub.errors.HTTPError`` is the base of the client's HTTP and network errors.
    """
    hub = load_hub()
    errors = importlib.import_module(f"{HUB_PACKAGE}.errors")
    return HubClient(
        api=hub.HfApi(),
        operation_add=hub.CommitOperationAdd,
        errors=(errors.HTTPError,),
        read_range=_range_reader(hub),
    )


def _range_reader(hub: ModuleType) -> RangeReader:
    """Read a byte range of a repository file with the client's own session and headers.

    A zero-length range makes no request: ``bytes=N-(N-1)`` is not a valid range, and a server
    may answer it with the whole file.
    """
    utils = importlib.import_module(f"{HUB_PACKAGE}.utils")

    def read_range(repo_id: str, filename: str, revision: str, start: int, length: int) -> bytes:
        if length == 0:
            return b""
        url = hub.hf_hub_url(repo_id, filename, revision=revision, repo_type=HUB_REPO_TYPE)
        what = f"{filename} in {repo_id}@{revision}"
        return fetch_range(utils.get_session(), url, utils.build_hf_headers(), start, length, what)

    return read_range


def fetch_range(
    session: Any, url: str, headers: dict[str, str], start: int, length: int, what: str
) -> bytes:
    """Bytes ``start`` to ``start + length - 1`` of ``url``, exactly, or a StoreError.

    ``session`` is an httpx-style client (``huggingface_hub.utils.get_session()``); the
    response is streamed with a timeout of ``RANGE_TIMEOUT_SECONDS``. A status other than 206
    (a 200 means the server ignored the range), a declared ``Content-Length`` other than
    ``length``, or an encoded body is refused before the body is read; the body is read as the
    transport delivers it and refused at the first read that brings it past ``length`` bytes.
    The Hub's own error formatting is not used, because it reads the whole error body.
    """
    if start < 0 or length < 0:
        raise StoreError(f"range request for {what}: invalid range at {start} of {length} bytes")
    if length == 0:
        return b""
    request_headers = {
        **headers,
        "range": f"bytes={start}-{start + length - 1}",
        "accept-encoding": "identity",
    }
    expected = f"expected {PARTIAL_CONTENT} with {length} bytes"
    with session.stream(
        "GET", url, headers=request_headers, timeout=RANGE_TIMEOUT_SECONDS
    ) as response:
        _check_range_response(response, length, f"range request for {what}", expected)
        data = bytearray()
        for chunk in response.iter_raw():  # no re-chunking buffer: one transport read each
            data += chunk
            if len(data) > length:
                raise StoreError(
                    f"range request for {what}: more than {length} bytes arrived; {expected}"
                )
    if len(data) != length:
        raise StoreError(f"range request for {what}: {len(data)} bytes arrived; {expected}")
    return bytes(data)


def _check_range_response(response: Any, length: int, what: str, expected: str) -> None:
    """StoreError for a status, ``Content-Length``, or encoding that a range cannot carry."""
    status = response.status_code
    if status != PARTIAL_CONTENT:
        ignored = " (the server ignored the range)" if status == 200 else ""
        raise StoreError(f"{what}: status {status}{ignored}; {expected}")
    declared = response.headers.get("content-length")
    if declared is not None and declared.strip() != str(length):
        raise StoreError(f"{what}: Content-Length {shown(declared)}; {expected}")
    encoding = response.headers.get("content-encoding", "identity")
    if encoding.strip().lower() != "identity":
        raise StoreError(f"{what}: Content-Encoding {shown(encoding)}; {expected}")


class HfTransport(Transport):
    """A remote store under ``wvc/`` in a Hub model repository, at one revision.

    ``exists`` and ``list`` use one ``list_repo_files`` listing per transport instance, kept
    current with this instance's own writes; ``read`` downloads through ``hf_hub_download``
    and checks the cached file's size before reading it;
    ``write_many`` sends one ``create_commit`` of ``CommitOperationAdd`` operations whose
    bytes are held in memory, and ``write`` is ``write_many`` of one pair.
    """

    def __init__(self, repo_id: str, revision: str = DEFAULT_REVISION) -> None:
        if not (is_hub_repo_id(repo_id) and is_hub_revision(revision)):
            raise StoreError(
                f"invalid Hub repository {repo_id!r} at revision {revision!r}: expected "
                "OWNER/NAME and a branch, tag, or commit"
            )
        self.repo_id = repo_id
        self.revision = revision
        self._client = make_hub_client()
        self._files: frozenset[str] | None = None

    def __repr__(self) -> str:
        return f"HfTransport({self.repo_id!r}, {self.revision!r})"

    @property
    def url(self) -> str:
        return f"hf://{self.repo_id}@{self.revision}"

    def exists(self, relpath: str) -> bool:
        return _hub_path(relpath) in self._listing()

    def read(self, relpath: str, *, limit: int, size: int | None = None) -> bytes:
        path = _hub_path(relpath)
        if path not in self._listing():
            raise StoreError(f"{self.url} lacks {path}")
        local = self._call(
            self._client.api.hf_hub_download,
            repo_id=self.repo_id,
            filename=path,
            revision=self.revision,
            repo_type=HUB_REPO_TYPE,
        )
        what = f"{self.url} file {path}"
        with open(open_input(Path(local)), "rb") as handle:  # Hub cache files are symlinks
            length = os.fstat(handle.fileno()).st_size
            _check_length(length, what, limit=limit, size=size)
            data = handle.read(length + 1)  # one byte more reveals a file that grew
        if len(data) != length:
            raise IntegrityError(f"{what} changed size while it was being read")
        return data

    def write(self, relpath: str, data: bytes) -> None:
        self.write_many([(relpath, data)])

    def write_many(self, pairs: FilePairs) -> None:
        """Send every pair as one Hub commit; nothing is sent for no pairs."""
        paths = [_hub_path(relpath) for relpath, _ in pairs]
        if not paths:
            return
        operations = [
            self._client.operation_add(path_in_repo=path, path_or_fileobj=bytes(data))
            for path, (_, data) in zip(paths, pairs, strict=True)
        ]
        message = f"wvc: write {paths[0]}" if len(paths) == 1 else f"wvc: write {len(paths)} files"
        self._call(
            self._client.api.create_commit,
            repo_id=self.repo_id,
            operations=operations,
            commit_message=message,
            revision=self.revision,
            repo_type=HUB_REPO_TYPE,
        )
        if self._files is not None:
            self._files = self._files | frozenset(paths)

    def list(self, prefix: str = "") -> RelPaths:
        start = f"{HUB_DIR}/{_check_prefix(prefix)}"
        return sorted(
            path.removeprefix(f"{HUB_DIR}/") for path in self._listing() if path.startswith(start)
        )

    def _listing(self) -> frozenset[str]:
        """Every file path of the repository at the revision, listed once per instance."""
        if self._files is None:
            files = self._call(
                self._client.api.list_repo_files,
                repo_id=self.repo_id,
                revision=self.revision,
                repo_type=HUB_REPO_TYPE,
            )
            self._files = frozenset(files)
        return self._files

    def _call(self, method: Callable[..., Any], **kwargs: Any) -> Any:
        """Call the client, reporting its errors as StoreError naming this remote."""
        try:
            return method(**kwargs)
        except self._client.errors as exc:
            raise StoreError(f"{self.url}: {exc}") from exc


def _check_length(length: int, what: str, *, limit: int, size: int | None) -> None:
    """IntegrityError when ``length`` is not ``size`` or, without a size, exceeds ``limit``."""
    if size is not None and length != size:
        raise IntegrityError(f"{what} holds {length} bytes, not the {size} its record implies")
    if size is None and length > limit:
        raise IntegrityError(f"{what} holds {length} bytes, more than the {limit} allowed")


def check_relpath(relpath: object) -> str:
    """Return ``relpath`` if it is a relative '/'-separated path without '', '.', or '..'."""
    if not isinstance(relpath, str) or not relpath:
        raise StoreError(f"a remote path must be non-empty text, got {relpath!r}")
    if relpath.startswith("/") or "\\" in relpath or "\x00" in relpath:
        raise StoreError(f"refusing remote path {relpath!r}: it must be relative")
    if any(part in ("", ".", "..") for part in relpath.split("/")):
        raise StoreError(f"refusing remote path {relpath!r}: empty, '.', or '..' part")
    return relpath


def _check_prefix(prefix: object) -> str:
    """A listing prefix: empty, or a relative path that may end with '/'."""
    if prefix == "":
        return ""
    if not isinstance(prefix, str):
        raise StoreError(f"a listing prefix must be text, got {prefix!r}")
    check_relpath(prefix.removesuffix("/"))
    return prefix


def _hub_path(relpath: str) -> str:
    return f"{HUB_DIR}/{check_relpath(relpath)}"
