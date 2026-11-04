"""Checkpoints on the Hugging Face Hub, named ``hf:OWNER/NAME[@REVISION][:FILE.gguf]``.

``wvc hash``, ``wvc diff``, and ``wvc verify`` take a Hub model repository in place of a
path. The repository is described once (``model_info`` with ``files_metadata``): the
revision is pinned to its commit sha, and every top-level ``*.safetensors`` file is listed
with its size and LFS SHA-256; a listing that names one outside the portable file-name rule
of ``wvc.file_layout`` is refused before anything is fetched. Files are fetched through
``hf_hub_download`` into the Hub cache, never into the working directory, and a fetched file
whose size differs from the listing is refused. A diff of two Hub repositories skips every
shard whose name, size, and SHA-256 agree on both sides: byte-identical files hold identical
tensors, so those are reported unchanged from the shard's header, which is read with two range
requests (the size field, then the header) and parsed by the same reader as a local file,
bounded by the listed size. Everything the Hub returns is treated as hostile. The client comes from
``wvc.transport.make_hub_client``, which tests replace with a fake; the optional
dependency is ``wvc[hf]``.

A repository that ships GGUF files usually holds several quantizations of one model, so a
GGUF checkpoint is named by its file: ``hf:OWNER/NAME@REVISION:path/to/file.gguf``. The file
and, for a split checkpoint (``wvc.gguf_split``), every part are taken from the same listing
with their sizes and LFS SHA-256, fetched the same way, and each fetched file's SHA-256 is
checked against the listing before it is parsed. A bare ``hf:`` argument on a repository with
GGUF files and no safetensors is refused with the GGUF checkpoints it lists
(``GgufChoiceError``), so that the user can name one.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wvc import transport
from wvc.diff import DEFAULT_RANK_CAP, TensorDiff, diff_checkpoints, unchanged_diff
from wvc.errors import WvcError
from wvc.file_layout import check_file_names
from wvc.fsio import input_digest
from wvc.gguf_io import GgufError, read_gguf
from wvc.gguf_split import check_parts, check_single, split_names
from wvc.hashing import build_manifest, tensor_digests
from wvc.quote import shown
from wvc.records import StoreError
from wvc.safetensors_io import (
    SIZE_FIELD_BYTES,
    Checkpoint,
    SafetensorsError,
    TensorInfo,
    header_size_from,
    parse_header,
    read_header,
)
from wvc.sources import parse_hub_ref
from wvc.transport import HUB_REPO_TYPE, HubClient

HUB_PREFIX = "hf:"
REMOTE_PREFIX = "hf://"
SHARD_SUFFIX = ".safetensors"
GGUF_SUFFIX = ".gguf"
HUB_FORM = "hf:OWNER/NAME[@REVISION][:FILE.gguf]"
# One part of a GGUF path in a repository: the characters gguf-hash lines and file systems take.
GGUF_PATH_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._+-]{0,199}")
MAX_GGUF_PATH_PARTS = 8
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


class HubError(WvcError):
    """A Hub checkpoint could not be named, described, or fetched."""


@dataclass(frozen=True)
class GgufChoice:
    """One GGUF checkpoint a repository lists: a file, or the first part of a split one."""

    name: str
    nbytes: int  # of every listed part
    parts: int  # the part count its name gives (1 for a plain file)
    listed: int  # how many of those parts the listing holds


class GgufChoiceError(HubError):
    """A bare ``hf:`` argument names a repository of GGUF files: the user must pick one."""

    def __init__(self, url: str, choices: list[GgufChoice]) -> None:
        self.url = url
        self.choices = choices
        super().__init__(
            f"{url}: no top-level *.safetensors files, but {len(choices)} GGUF "
            f"checkpoint{'' if len(choices) == 1 else 's'}; name one as {url}:FILE.gguf"
        )


@dataclass(frozen=True)
class HubRef:
    """A parsed Hub argument; ``file`` is the GGUF file it names, None for the repository."""

    repo_id: str
    revision: str
    file: str | None = None


@dataclass(frozen=True)
class HubFile:
    name: str
    nbytes: int
    sha256: str | None  # the LFS object id; None when the Hub stores the file as a plain blob

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "nbytes": self.nbytes, "sha256": self.sha256}


@dataclass(frozen=True)
class HubRepo:
    repo_id: str
    revision: str  # as given
    sha: str  # the commit the revision resolved to when the repository was described
    # the top-level safetensors files, sorted by name; for a GGUF selection, its parts in order
    files: tuple[HubFile, ...]
    selected: str | None = None  # the GGUF file the argument named, None for safetensors

    @property
    def url(self) -> str:
        where = f"{HUB_PREFIX}{self.repo_id}@{self.revision}"
        return where if self.selected is None else f"{where}:{self.selected}"

    def to_dict(self) -> dict[str, Any]:
        about: dict[str, Any] = {"repo": self.repo_id, "revision": self.revision, "sha": self.sha}
        if self.selected is not None:
            about["file"] = self.selected
            about["files"] = [file.to_dict() for file in self.files]
        return about

    def named(self, names: Iterable[str]) -> list[HubFile]:
        wanted = set(names)
        return [file for file in self.files if file.name in wanted]


@dataclass(frozen=True)
class PairPlan:
    """Which shards of two repositories a diff can skip, and which it must fetch."""

    skipped: tuple[HubFile, ...]  # same name, size, and SHA-256 on both sides
    fetch_a: tuple[str, ...]
    fetch_b: tuple[str, ...]


@dataclass(frozen=True)
class PairDiff:
    diffs: list[TensorDiff]  # sorted by name
    plan: PairPlan
    root_a: str | None  # None when a shard was skipped: its tensors were never hashed
    root_b: str | None
    inspected: object = None  # what diff_hub_pair's ``inspect`` returned


def is_hub_checkpoint(arg: object) -> bool:
    return isinstance(arg, str) and arg.startswith(HUB_PREFIX)


def parse_hub_checkpoint(arg: str) -> tuple[str, str]:
    """``hf:OWNER/NAME[@REVISION]`` as (repo id, revision), the revision ``main`` by default."""
    ref = parse_hub_argument(arg)
    if ref.file is not None:
        raise HubError(f"{arg!r} names one file; expected hf:OWNER/NAME[@REVISION] here")
    return ref.repo_id, ref.revision


def parse_hub_argument(arg: str) -> HubRef:
    """``hf:OWNER/NAME[@REVISION][:FILE.gguf]``; the revision is ``main`` by default.

    The file follows the first ``:`` after the prefix (neither a repository id nor a revision
    can hold one). It must be a relative ``/``-separated path ending in ``.gguf`` whose every
    part is a portable name (``GGUF_PATH_PART``).
    """
    if arg.startswith(REMOTE_PREFIX):
        raise HubError(
            f"{arg!r} names a wvc remote (a store under wvc/ in a repository); "
            f"a Hub checkpoint is {HUB_FORM}"
        )
    if not is_hub_checkpoint(arg):
        raise HubError(f"expected {HUB_FORM}, got {arg!r}")
    head, separator, file = arg.removeprefix(HUB_PREFIX).partition(":")
    try:
        repo_id, revision = parse_hub_ref(head, f"invalid Hub checkpoint {arg!r}")
    except StoreError as exc:
        raise HubError(str(exc)) from exc
    if not separator:
        return HubRef(repo_id, revision)
    check_gguf_path(file, arg)
    return HubRef(repo_id, revision, file)


def check_gguf_path(path: str, where: str) -> None:
    """HubError unless ``path`` names a GGUF file by a relative path of portable parts."""
    if not path.endswith(GGUF_SUFFIX):
        raise HubError(
            f"{shown(where)}: the file after ':' must be a .gguf file; a repository of "
            "safetensors shards is named without one"
        )
    parts = path.split("/")
    if len(parts) > MAX_GGUF_PATH_PARTS or any(
        GGUF_PATH_PART.fullmatch(part) is None or part in (".", "..") for part in parts
    ):
        raise HubError(
            f"{shown(where)}: a GGUF file must be a relative path of at most "
            f"{MAX_GGUF_PATH_PARTS} parts, each of letters, digits, '.', '_', '+', and '-' "
            "(not starting with '.', '+', or '-')"
        )


class Hub:
    """One Hub client and the calls the checkpoint commands make; errors name the checkpoint."""

    def __init__(self, client: HubClient | None = None) -> None:
        self._client = transport.make_hub_client() if client is None else client

    def _listing(self, url: str, repo_id: str, revision: str) -> tuple[str, tuple[Any, ...]]:
        """The commit sha the revision resolves to and the repository's listed files."""
        info = self._call(
            url,
            self._client.api.model_info,
            repo_id=repo_id,
            revision=revision,
            files_metadata=True,
        )
        sha = getattr(info, "sha", None)
        if not isinstance(sha, str) or COMMIT_SHA.fullmatch(sha) is None:
            raise HubError(
                f"{url}: the Hub returned no commit sha for the revision, got {shown(sha)}"
            )
        return sha, tuple(getattr(info, "siblings", None) or ())

    def describe(self, repo_id: str, revision: str) -> HubRepo:
        """Pin the revision to its commit and list the top-level safetensors files.

        A repository without them that lists GGUF files raises ``GgufChoiceError``.
        """
        url = f"{HUB_PREFIX}{repo_id}@{revision}"
        sha, siblings = self._listing(url, repo_id, revision)
        files = sorted(
            (
                _hub_file(url, sibling)
                for sibling in siblings
                if _is_shard(getattr(sibling, "rfilename", None))
            ),
            key=lambda file: file.name,
        )
        if not files:
            choices = gguf_choices(url, siblings)
            if choices:
                raise GgufChoiceError(url, choices)
            raise HubError(f"{url}: no top-level *.safetensors files")
        try:  # before anything is fetched, under the rule a commit applies to its file names
            check_file_names([file.name for file in files], url)
        except StoreError as exc:
            raise HubError(str(exc)) from exc
        return HubRepo(repo_id, revision, sha, tuple(files))

    def describe_gguf(self, ref: HubRef) -> HubRepo:
        """Pin the revision and list the GGUF file ``ref`` names, or every part of its split.

        Every part must be in the listing, with a size and (unless the Hub stores it as a
        plain blob) an LFS SHA-256; nothing is fetched here.
        """
        if ref.file is None:
            raise HubError(f"hf:{ref.repo_id}@{ref.revision} names no GGUF file")
        url = f"{HUB_PREFIX}{ref.repo_id}@{ref.revision}:{ref.file}"
        sha, siblings = self._listing(url, ref.repo_id, ref.revision)
        listed = {
            sibling.rfilename: sibling
            for sibling in siblings
            if isinstance(getattr(sibling, "rfilename", None), str)
        }
        if ref.file not in listed:
            raise HubError(f"{url}: the repository lists no file {ref.file} at {sha}")
        try:
            names = split_names(ref.file) or [ref.file]
        except GgufError as exc:
            raise HubError(f"{url}: {exc}") from exc
        missing = [name for name in names if name not in listed]
        if missing:
            raise HubError(
                f"{url}: {len(missing)} of the {len(names)} parts of this split checkpoint "
                f"are not in the repository, the first {missing[0]}"
            )
        files = tuple(_hub_file(url, listed[name]) for name in names)
        return HubRepo(ref.repo_id, ref.revision, sha, files, selected=ref.file)

    def fetch(self, repo: HubRepo, file: HubFile) -> Path:
        """Download one file of the pinned commit into the Hub cache; its local path.

        For a GGUF selection the file's SHA-256 is checked against the listing's LFS SHA-256.
        """
        local = Path(
            self._call(
                repo.url,
                self._client.api.hf_hub_download,
                repo_id=repo.repo_id,
                filename=file.name,
                revision=repo.sha,
                repo_type=HUB_REPO_TYPE,
            )
        )
        size = local.stat().st_size
        if size != file.nbytes:
            raise HubError(
                f"{repo.url}: {file.name} downloaded as {size:,} bytes, "
                f"the Hub listed {file.nbytes:,}"
            )
        if repo.selected is not None and file.sha256 is not None:
            digest, _ = input_digest(local)
            if digest != file.sha256:
                raise HubError(
                    f"{repo.url}: {file.name} downloaded with SHA-256 {digest}, "
                    f"the Hub listed {file.sha256}"
                )
        return local

    def header(self, repo: HubRepo, file: HubFile) -> dict[str, TensorInfo]:
        """The tensors of one shard from its header alone: two range requests, one parser."""
        path = Path(f"{repo.url}/{file.name}")  # names the file in the parser's messages
        header_size = header_size_from(path, self._read(repo, file, 0, SIZE_FIELD_BYTES))
        if SIZE_FIELD_BYTES + header_size > file.nbytes:
            raise SafetensorsError(
                f"{path}: header of {header_size} bytes exceeds the {file.nbytes}-byte file"
            )
        header_bytes = self._read(repo, file, SIZE_FIELD_BYTES, header_size)
        _, tensors = parse_header(path, header_bytes, file.nbytes - SIZE_FIELD_BYTES - header_size)
        return tensors

    def _read(self, repo: HubRepo, file: HubFile, start: int, length: int) -> bytes:
        if length == 0:  # an empty header: no request, since ``bytes=8-7`` is not a range
            return b""
        data = self._call(
            repo.url, self._client.read_range, repo.repo_id, file.name, repo.sha, start, length
        )
        if not isinstance(data, bytes) or len(data) != length:
            raise HubError(
                f"{repo.url}: a range request for {file.name} returned "
                f"{len(data) if isinstance(data, bytes) else type(data).__name__}, "
                f"expected {length} bytes"
            )
        return data

    def _call(self, url: str, method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call the client, reporting its errors as HubError naming the checkpoint."""
        try:
            return method(*args, **kwargs)
        except self._client.errors as exc:
            raise HubError(f"{url}: {exc}") from exc


def fetch_all(hub: Hub, repo: HubRepo, files: Iterable[HubFile]) -> Checkpoint:
    """Fetch the given shards of a repository and open them as one checkpoint."""
    return Checkpoint([read_header(hub.fetch(repo, file)) for file in files])


def load_hub_checkpoint(hub: Hub, repo: HubRepo) -> Checkpoint:
    """Fetch every shard of a repository and open them as one checkpoint."""
    return fetch_all(hub, repo, repo.files)


def load_hub_gguf(hub: Hub, repo: HubRepo) -> Checkpoint:
    """Fetch a GGUF selection's parts, check them as one checkpoint, and open it.

    A split checkpoint's parts are read in part order, so its tensors are too.
    """
    if repo.selected is None:
        raise HubError(f"{repo.url} names no GGUF file")
    parts = [read_gguf(hub.fetch(repo, file)) for file in repo.files]
    if split_names(repo.selected) is None:
        check_single(parts[0])
    else:
        check_parts(parts)
    return Checkpoint(parts)


def gguf_choices(url: str, siblings: Iterable[Any]) -> list[GgufChoice]:
    """The GGUF checkpoints a listing holds, a split one once under its first part's name.

    Names outside the GGUF path rule are left out: they could not be named in an argument.
    """
    plain: list[GgufChoice] = []
    splits: dict[str, list[HubFile]] = {}
    counts: dict[str, int] = {}
    for sibling in siblings:
        name = getattr(sibling, "rfilename", None)
        if not isinstance(name, str) or not _is_gguf_path(name):
            continue
        file = _hub_file(url, sibling)
        try:
            names = split_names(name)
        except GgufError:  # a part number no writer makes: listed as a file of its own
            names = None
        if names is None:
            plain.append(GgufChoice(name, file.nbytes, 1, 1))
        else:
            splits.setdefault(names[0], []).append(file)
            counts[names[0]] = len(names)
    grouped = [
        GgufChoice(first, sum(file.nbytes for file in files), counts[first], len(files))
        for first, files in splits.items()
    ]
    return sorted([*plain, *grouped], key=lambda choice: choice.name)


def _is_gguf_path(name: str) -> bool:
    try:
        check_gguf_path(name, name)
    except HubError:
        return False
    return True


def plan_pair(a: HubRepo, b: HubRepo) -> PairPlan:
    """Skip the shards whose name, size, and LFS SHA-256 agree; fetch the rest on each side."""
    by_name_b = {file.name: file for file in b.files}
    skipped, fetch_a = [], []
    for file in a.files:
        other = by_name_b.get(file.name)
        if (
            other is not None
            and file.sha256 is not None
            and (file.sha256, file.nbytes) == (other.sha256, other.nbytes)
        ):
            skipped.append(file)
        else:
            fetch_a.append(file.name)
    skipped_names = {file.name for file in skipped}
    fetch_b = [file.name for file in b.files if file.name not in skipped_names]
    return PairPlan(tuple(skipped), tuple(fetch_a), tuple(fetch_b))


def diff_hub_pair(
    hub: Hub,
    a: HubRepo,
    b: HubRepo,
    *,
    rank: bool = False,
    rank_cap: int = DEFAULT_RANK_CAP,
    inspect: Callable[[Checkpoint, Checkpoint, list[TensorDiff]], object] | None = None,
) -> PairDiff:
    """Diff two Hub repositories, fetching only the shards that differ on the Hub.

    The skipped shards' headers are read and checked first, before any download. Their
    tensors are ``unchanged`` by construction (the two files are byte-identical). A tensor
    name that appears in a skipped shard and in any other shard on either side is refused,
    as a local reader refuses a duplicate. ``inspect``, when given, runs while the fetched
    shards are open, on them and on the whole sorted diff; its result is ``inspected``.
    """
    plan = plan_pair(a, b)
    unchanged: dict[str, TensorDiff] = {}
    for file in plan.skipped:
        for name, info in hub.header(a, file).items():
            if name in unchanged:
                raise SafetensorsError(
                    f"tensor {shown(name)} appears in both {file.name} and another shard"
                )
            unchanged[name] = unchanged_diff(name, info)
    with (
        fetch_all(hub, a, a.named(plan.fetch_a)) as ckpt_a,
        fetch_all(hub, b, b.named(plan.fetch_b)) as ckpt_b,
    ):
        if overlap := sorted((set(ckpt_a.tensors) | set(ckpt_b.tensors)) & set(unchanged)):
            raise SafetensorsError(
                f"tensor {shown(overlap[0])} appears in both a skipped shard and a fetched one"
            )
        digests_a, digests_b = tensor_digests(ckpt_a), tensor_digests(ckpt_b)
        diffs = diff_checkpoints(ckpt_a, ckpt_b, digests_a, digests_b, rank=rank, rank_cap=rank_cap)
        diffs = sorted([*diffs, *unchanged.values()], key=lambda item: item.name)
        inspected = None if inspect is None else inspect(ckpt_a, ckpt_b, diffs)
    if plan.skipped:
        return PairDiff(diffs, plan, None, None, inspected)
    root_a = build_manifest(ckpt_a, digests_a)["root"]  # from the digests: no file is read
    root_b = build_manifest(ckpt_b, digests_b)["root"]
    return PairDiff(diffs, plan, root_a, root_b, inspected)


def _is_shard(name: object) -> bool:
    """A top-level, non-hidden, printable ``*.safetensors`` name, as a directory listing gives."""
    return (
        isinstance(name, str)
        and name.endswith(SHARD_SUFFIX)
        and "/" not in name
        and not name.startswith(".")
        and name.isprintable()
    )


def _hub_file(url: str, sibling: Any) -> HubFile:
    name = sibling.rfilename
    size = getattr(sibling, "size", None)
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise HubError(f"{url}: the Hub listed {shown(name)} with no valid size, got {shown(size)}")
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return HubFile(name, size, None)
    sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if not isinstance(sha256, str) or SHA256_HEX.fullmatch(sha256) is None:
        raise HubError(f"{url}: the Hub listed {shown(name)} with a malformed LFS sha256")
    return HubFile(name, size, sha256)
