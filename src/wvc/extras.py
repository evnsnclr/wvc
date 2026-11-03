"""Extras: the non-tensor files of a checkpoint directory (spec section 3.3a).

A checkpoint directory usually holds files a loader needs besides the weights: the model
config, the tokenizer files, a chat template, the shard index, a licence, a model card. A
commit records each top-level entry that is (or links to) a regular file, is neither a
``.safetensors`` shard nor a dot file, has an allowed name, and holds at most
``MAX_EXTRA_BYTES``, as one payload whose id is the file's SHA-256; the commit record lists them
as ``"extras": [{"name", "payload", "nbytes"}, ...]`` sorted by name, at most
``MAX_EXTRAS_TOTAL`` bytes in all. Every other entry is skipped with a one-line reason.

A symbolic link is followed only when its fully resolved target lies inside the resolved
checkpoint directory, or inside the blob store of a Hugging Face cache snapshot
(``<repo>/snapshots/<sha>/config.json -> ../../blobs/<hash>``): the ``blobs`` directory beside
the ``snapshots`` directory that holds the checkpoint. Any other link is skipped as a "symlink
outside the checkpoint", so a directory unpacked from an archive cannot make a commit store, or
a push publish, a file such as ``~/.ssh/id_ed25519`` behind a link named ``tokenizer.json``.

An allowed name follows the portable rule of ``wvc.file_layout`` without its ``.safetensors``
suffix and either ends in an allowed suffix (case-sensitive) or starts with a licence prefix and
holds no further dot: ``LICENSE`` and ``LICENSE-MIT`` qualify, ``LICENSE.py`` does not, so no
extra is a code file type (such as ``.py``) or a shard. The rules apply when a commit plans its
extras, when a record is validated (``wvc.records.validate_commit``, so ``put_commit`` and
every read), and again at checkout, so a received repository can never make a checkout write a
code file type or an unlisted type. The rules judge file types, not contents, so some risk
remains: a ``.jinja`` chat template is a program in a template language that a loader runs,
and an ``auto_map`` entry in a config such as ``tokenizer_config.json`` can point a loader at
remote code when the user opts in to it. Extras are stored and written verbatim, never
inspected. Extras never enter the manifest: identity stays tensors-only.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from wvc.errors import IntegrityError, StoreError
from wvc.file_layout import name_key, portable_problem
from wvc.fsio import file_identity, open_input
from wvc.ids import is_count, is_id

ALLOWED_SUFFIXES = (".json", ".txt", ".jinja", ".model", ".tiktoken", ".md", ".yaml", ".yml")
ALLOWED_PREFIXES = ("LICENSE", "LICENCE", "NOTICE")
MAX_EXTRA_BYTES = 64 << 20  # one extra
MAX_EXTRAS_TOTAL = 256 << 20  # all the extras of one commit
EXTRA_KEYS = frozenset({"name", "payload", "nbytes"})
SHARD_SUFFIX = ".safetensors"
EXTRA_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
NAME_RULE = (
    "an extra's name must be letters, digits, '.', '_', or '-' and start with a letter or digit"
)
TYPE_RULE = (
    f"an extra's name must end in {', '.join(ALLOWED_SUFFIXES)}, or start with "
    f"{', '.join(ALLOWED_PREFIXES)} and hold no further dot"
)
# Why plan_extras skips an entry of a checkpoint directory.
DOT_FILE = "dot file"
DIRECTORY = "a directory"
NOT_REGULAR = "not a regular file"
UNSUPPORTED_NAME = "unsupported name"
TOO_LARGE = "too large"
SYMLINK_OUTSIDE = "symlink outside the checkpoint"
# A Hugging Face cache entry: <repo>/snapshots/<sha>/<name> links to <repo>/blobs/<hash>.
HUB_SNAPSHOTS = "snapshots"
HUB_BLOBS = "blobs"


@dataclass(frozen=True)
class ExtraPlan:
    """A file of a checkpoint directory to record as an extra, as listed before storing."""

    name: str
    path: Path  # may be a symlink to a regular file inside the checkpoint or its Hub blob store
    nbytes: int


def is_extra_name(name: object) -> bool:
    """True when ``name`` may name an extra (``extra_name_problem`` finds nothing wrong)."""
    return extra_name_problem(name) is None


def extra_name_problem(name: object) -> str | None:
    """Why ``name`` cannot name an extra, or None when it can (see the module docstring)."""
    if not isinstance(name, str) or EXTRA_NAME.fullmatch(name) is None:
        return NAME_RULE
    problem = portable_problem(name)
    if problem is not None:
        return problem
    if name.endswith(ALLOWED_SUFFIXES):
        return None
    if name.startswith(ALLOWED_PREFIXES) and "." not in name:
        return None
    return TYPE_RULE


def plan_extras(directory: str | Path) -> tuple[list[ExtraPlan], list[str]]:
    """The extras of a checkpoint directory, sorted by name, and the entries it skips.

    Nothing is read. Shards (``*.safetensors`` that are not dot files) belong to the file
    records and appear in neither list. A symlink is followed only into the directory itself or
    a Hub snapshot's blob store (see the module docstring). Each skipped entry is
    ``"<name> (<reason>)"``: a dot file, a directory, not a regular file, a symlink outside the
    checkpoint, an unsupported name, or too large. StoreError when two extras, or an extra and a
    shard, would be one file on a case-insensitive disk, or when the extras hold more than
    ``MAX_EXTRAS_TOTAL`` bytes.
    """
    directory = Path(directory)
    plans: list[ExtraPlan] = []
    skipped: list[str] = []
    shards: list[str] = []
    roots = _link_roots(directory)
    for name in sorted(os.listdir(directory)):
        if name.endswith(SHARD_SUFFIX) and not name.startswith("."):
            shards.append(name)
            continue
        outcome = _classify(directory, name, roots)
        if isinstance(outcome, ExtraPlan):
            plans.append(outcome)
        else:
            skipped.append(f"{name} ({outcome})")
    check_extra_names([plan.name for plan in plans], str(directory), shards)
    total = sum(plan.nbytes for plan in plans)
    if total > MAX_EXTRAS_TOTAL:
        raise StoreError(
            f"{directory}: its extras hold {total:,} bytes, more than the {MAX_EXTRAS_TOTAL:,} "
            "one commit may record; use --no-extras to record none"
        )
    return plans, skipped


def read_extra(plan: ExtraPlan) -> bytes:
    """The bytes of a planned extra, read through one descriptor (a symlink is followed).

    The file must be a regular file of the planned size before a byte is read, and unchanged
    after the read: the same device, inode, size, and mtime at the descriptor and at the path.
    StoreError otherwise.
    """
    with open(open_input(plan.path), "rb") as handle:
        opened = os.fstat(handle.fileno())
        if opened.st_size != plan.nbytes:
            raise StoreError(f"{plan.path} changed size since the commit listed it")
        data = handle.read(plan.nbytes + 1)
        after = os.fstat(handle.fileno())
    identity = file_identity(opened)
    if (
        len(data) != plan.nbytes
        or file_identity(after) != identity
        or file_identity(os.stat(plan.path)) != identity
    ):
        raise StoreError(f"{plan.path} changed while it was being committed")
    return data


def validate_extra_records(
    extras: object, where: str, shard_names: Iterable[object] = ()
) -> list[dict]:
    """Check a commit's extras records and return them; StoreError when any is malformed.

    Each record has exactly ``name``, ``payload`` (a 64-hex id), and ``nbytes`` (a count of at
    most ``MAX_EXTRA_BYTES``); the names are allowed (``extra_name_problem``), sorted, and
    distinct from each other and from ``shard_names`` after NFC and case folding; the sizes add
    up to at most ``MAX_EXTRAS_TOTAL``. A bad record is named by its position, never quoted.
    """
    if not isinstance(extras, list):
        raise StoreError(f"{where}: extras must be a list")
    for index, entry in enumerate(extras):
        _check_extra_record(entry, f"{where}: extras[{index}]")
    names = [entry["name"] for entry in extras]
    if names != sorted(names):
        raise StoreError(f"{where}: extras must be sorted by name")
    check_extra_names(names, where, shard_names)
    if sum(entry["nbytes"] for entry in extras) > MAX_EXTRAS_TOTAL:
        raise StoreError(f"{where}: the extras hold more than {MAX_EXTRAS_TOTAL} bytes")
    return extras


def commit_extras(record: Mapping, where: str) -> list[dict]:
    """A commit record's extras, validated; empty for a record without them (older commits)."""
    files = record.get("files")
    entries = files if isinstance(files, list) else []
    names = [entry.get("name") for entry in entries if isinstance(entry, dict)]
    return validate_extra_records(record.get("extras", []), where, names)


def check_extra_names(names: Sequence[str], where: str, shard_names: Iterable[object] = ()) -> None:
    """StoreError when two extras, or an extra and a shard, are one name after NFC and casefold."""
    seen = {name_key(name): name for name in shard_names if isinstance(name, str)}
    for name in names:
        key = name_key(name)
        if key in seen:
            how = "share a name" if seen[key] == name else "differ only in case or normalisation"
            raise StoreError(f"{where}: two recorded files {how}: {seen[key]!r} and {name!r}")
        seen[key] = name


def check_extra_payloads(
    extras: Sequence[Mapping], payload_size: Callable[[str], int | None], where: str
) -> None:
    """StoreError unless every extra's payload is stored at its recorded size (none is read)."""
    for entry in extras:
        what = f"{where}: extra {entry['name']}: payload {entry['payload']}"
        size = payload_size(entry["payload"])
        if size is None:
            raise StoreError(f"{what} is not stored")
        if size != entry["nbytes"]:
            raise IntegrityError(f"{what} holds {size} bytes, not the {entry['nbytes']} recorded")


def _link_roots(directory: Path) -> tuple[Path, ...]:
    """The directories an extra's symlink may resolve into: the checkpoint directory, resolved,
    and for a Hugging Face cache snapshot its repository's ``blobs`` directory.

    ``blobs`` is joined to the resolved path, not resolved itself, so a ``blobs`` that is a
    link to somewhere else admits nothing.
    """
    resolved = directory.resolve()
    if resolved.parent.name == HUB_SNAPSHOTS:
        return (resolved, resolved.parent.parent / HUB_BLOBS)
    return (resolved,)


def _classify(directory: Path, name: str, link_roots: Sequence[Path]) -> ExtraPlan | str:
    """One directory entry as a planned extra, or the reason it is skipped."""
    if name.startswith("."):
        return DOT_FILE
    path = directory / name
    try:
        info = os.stat(path)  # follows a symlink; where it points is checked below
        linked = stat.S_ISLNK(os.lstat(path).st_mode)
        target = path.resolve(strict=True) if linked else None
    except (OSError, RuntimeError):  # a dangling or looping link, or an entry removed meanwhile
        return NOT_REGULAR
    if stat.S_ISDIR(info.st_mode):
        return DIRECTORY
    if not stat.S_ISREG(info.st_mode):
        return NOT_REGULAR
    if target is not None and not any(target.is_relative_to(root) for root in link_roots):
        return SYMLINK_OUTSIDE
    if not is_extra_name(name):
        return UNSUPPORTED_NAME
    if info.st_size > MAX_EXTRA_BYTES:
        return TOO_LARGE
    return ExtraPlan(name, path, info.st_size)


def _check_extra_record(entry: object, where: str) -> None:
    if not isinstance(entry, dict) or set(entry) != EXTRA_KEYS:
        raise StoreError(f"{where}: an extras record needs exactly the keys {sorted(EXTRA_KEYS)}")
    problem = extra_name_problem(entry["name"])
    if problem is not None:
        raise StoreError(f"{where}: not an allowed extra name: {problem}")
    if not is_id(entry["payload"]):
        raise StoreError(f"{where}: payload must be a 64-hex id")
    if not is_count(entry["nbytes"]) or entry["nbytes"] > MAX_EXTRA_BYTES:
        raise StoreError(f"{where}: nbytes must be a byte count of at most {MAX_EXTRA_BYTES}")
