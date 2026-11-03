"""The object store's layout constants, size limits, config, and stored JSON forms.

Split out of ``wvc.store``, which re-exports every public name here. ``config.json`` holds the
repository format, ``max_delta_depth``, and ``max_tensor_bytes`` (absent in repositories made
before it existed, which get the default); the limits that no config can raise are
``MAX_CHAIN_DEPTH`` and ``TENSOR_BYTES_CEILING``. ``find_repo`` locates a repository from a
directory inside it and refuses one that belongs to another user.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from wvc.errors import StoreError
from wvc.ids import is_count

REPO_DIR = ".wvc"
REPO_FORMAT = "wvc-repo/0"
DEFAULT_MAX_DELTA_DEPTH = 8
MAX_CHAIN_DEPTH = 64  # no delta chain is followed further, whatever config.json says
DEFAULT_MAX_TENSOR_BYTES = 1 << 34  # 16 GiB
TENSOR_BYTES_CEILING = 1 << 40  # no tensor or file may claim more, whatever config.json says
DEFAULT_BRANCH = "main"
LAYOUT_DIRS = ("refs", "refs/heads", "commits", "manifests", "objects", "payloads")
SHARDED_DIRS = ("objects", "payloads")
SHARD = re.compile(r"[0-9a-f]{2}")  # a shard directory: the first two hex characters of ids
CONFIG_LIMIT = OBJECT_LIMIT = 1 << 20  # bytes: config.json and object records
RECORD_LIMIT = 256 << 20  # bytes: commit records and manifests
PAYLOAD_LIMIT = 4 << 30  # bytes: a payload read without a record that implies its size


def find_repo(start: str | Path) -> Path:
    """Return the nearest directory at or above ``start`` that holds ``.wvc``.

    A ``.wvc`` found this way that belongs to another user is refused, as git refuses a
    repository it does not own; naming the repository explicitly (``--repo``) skips this.
    """
    start = Path(start).resolve()
    for directory in (start, *start.parents):
        if (directory / REPO_DIR).is_dir():
            _check_owner(directory / REPO_DIR)
            return directory
    raise StoreError(f"not inside a wvc repository: no {REPO_DIR} at or above {start}")


def default_config() -> dict:
    """The ``config.json`` of a new repository."""
    return {
        "format": REPO_FORMAT,
        "max_delta_depth": DEFAULT_MAX_DELTA_DEPTH,
        "max_tensor_bytes": DEFAULT_MAX_TENSOR_BYTES,
    }


def check_config(config: object, path: Path) -> None:
    """StoreError unless ``config`` is an object with this format and a max_delta_depth.

    ``max_tensor_bytes`` may be absent (repositories made before it existed get the default)
    and must otherwise be a positive integer.
    """
    if (
        not isinstance(config, dict)
        or config.get("format") != REPO_FORMAT
        or not is_count(config.get("max_delta_depth"))
    ):
        raise StoreError(
            f"{path}: expected format {REPO_FORMAT!r} and a non-negative integer max_delta_depth"
        )
    limit = config.get("max_tensor_bytes", DEFAULT_MAX_TENSOR_BYTES)
    if not is_count(limit) or limit == 0:
        raise StoreError(f"{path}: max_tensor_bytes must be a positive integer")


def depth_limit(config: dict) -> int:
    """The deepest delta chain a store with this validated config writes or follows."""
    return min(config["max_delta_depth"], MAX_CHAIN_DEPTH)


def tensor_limit(config: dict) -> int:
    """The largest tensor, in bytes, a store with this validated config writes or reads."""
    return min(config.get("max_tensor_bytes", DEFAULT_MAX_TENSOR_BYTES), TENSOR_BYTES_CEILING)


def record_json(obj: object) -> bytes:
    """Object and commit records and the config as stored: readable UTF-8 JSON, keys sorted.

    Ids never depend on this form (spec 3.5); remotes receive records in it too.
    """
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def manifest_json(manifest: dict) -> bytes:
    """A manifest as the store keeps it: exactly what ``wvc hash`` prints for it."""
    return (json.dumps(manifest, indent=2) + "\n").encode("utf-8")


def _check_owner(path: Path) -> None:
    getuid = getattr(os, "getuid", None)
    if getuid is None:  # Windows has no uids
        return
    owner = os.lstat(path).st_uid
    if owner != getuid():
        raise StoreError(
            f"refusing to use {path}: it belongs to another user (uid {owner}); name the "
            "repository with --repo to use it anyway"
        )
