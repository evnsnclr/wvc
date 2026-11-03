"""Commit sources (spec section 5.1): where a committed checkpoint came from.

``wvc commit --source SPEC`` records a source in the commit record's ``source`` field.
``hf:REPO_ID[@REVISION]`` names a Hugging Face Hub model repository (revision ``main`` when
omitted); any other text is kept as a path. Sources are hashed with the commit record, so every
value is a string (canonical JSON forbids floats). The Hub repo-id rules here are shared with
``hf://`` remote URLs.
"""

from __future__ import annotations

import re

from wvc.records import StoreError

HUB_SOURCE_PREFIX = "hf:"
DEFAULT_REVISION = "main"
SOURCE_KEYS = {"hf": frozenset({"kind", "repo", "revision"}), "path": frozenset({"kind", "value"})}
_HUB_NAME = r"[A-Za-z0-9_](?:[A-Za-z0-9_.-]{0,94}[A-Za-z0-9_])?"  # 1 to 96 characters
HUB_REPO_ID = re.compile(rf"{_HUB_NAME}/{_HUB_NAME}")
HUB_REVISION = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+/-]{0,254}")  # a branch, tag, or commit


def is_hub_repo_id(value: object) -> bool:
    """True for an ``owner/name`` Hub repo id; '--' and '..' are refused as the Hub does."""
    return (
        isinstance(value, str)
        and HUB_REPO_ID.fullmatch(value) is not None
        and "--" not in value
        and ".." not in value
    )


def is_hub_revision(value: object) -> bool:
    """True for a branch, tag, or commit name without whitespace, '..', or a leading '/'."""
    return (
        isinstance(value, str) and HUB_REVISION.fullmatch(value) is not None and ".." not in value
    )


def parse_hub_ref(text: str, what: str) -> tuple[str, str]:
    """Split ``REPO_ID[@REVISION]`` into the repo id and the revision (``main`` when omitted).

    StoreError naming ``what`` unless REPO_ID looks like ``owner/name`` and REVISION is valid.
    """
    repo, separator, revision = text.partition("@")
    revision = revision if separator else DEFAULT_REVISION
    if not (is_hub_repo_id(repo) and is_hub_revision(revision)):
        raise StoreError(f"{what}: expected OWNER/NAME[@REVISION], got {text!r}")
    return repo, revision


def parse_source(spec: str) -> dict:
    """Parse a ``--source`` SPEC into the commit record's ``source`` field.

    ``hf:REPO_ID[@REVISION]`` gives ``{"kind": "hf", "repo": ..., "revision": ...}``; any other
    non-empty text gives ``{"kind": "path", "value": spec}``.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise StoreError(f"a source must be non-empty text, got {spec!r}")
    if spec.startswith(HUB_SOURCE_PREFIX):
        rest = spec.removeprefix(HUB_SOURCE_PREFIX)
        repo, revision = parse_hub_ref(rest, f"invalid hf source {spec!r}")
        return check_source({"kind": "hf", "repo": repo, "revision": revision})
    return check_source({"kind": "path", "value": spec})


def check_source(source: object) -> dict:
    """Return a copy of a valid ``source`` field; StoreError when it is malformed.

    The kinds are ``hf`` (exactly ``kind``, ``repo``, ``revision``) and ``path`` (exactly
    ``kind`` and ``value``); every value is non-empty UTF-8 text.
    """
    if not isinstance(source, dict):
        raise StoreError(f"a source must be a JSON object, got {type(source).__name__}")
    kind = source.get("kind")
    if not isinstance(kind, str) or kind not in SOURCE_KEYS:
        raise StoreError(f"source kind must be one of {sorted(SOURCE_KEYS)}, got {kind!r}")
    if set(source) != SOURCE_KEYS[kind]:
        found = sorted(map(str, source))
        raise StoreError(
            f"a {kind!r} source needs the keys {sorted(SOURCE_KEYS[kind])}, got {found}"
        )
    for key, value in source.items():
        _check_text(value, f"source {key}")
    if kind == "hf" and not (
        is_hub_repo_id(source["repo"]) and is_hub_revision(source["revision"])
    ):
        raise StoreError(f"source {source!r}: repo must be OWNER/NAME with a valid revision")
    return dict(source)


def _check_text(value: object, what: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise StoreError(f"{what} must be non-empty text, got {value!r}")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise StoreError(f"{what} is not valid UTF-8 text") from exc
