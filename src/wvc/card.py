"""Model-card front matter for a commit (spec section 5.1).

``model_card`` renders the YAML front matter for publishing a committed checkpoint on the
Hugging Face Hub: ``base_model`` is the Hub repo of the nearest commit on REF's first-parent
chain, starting with REF itself, whose source is ``hf``; the tags are ``wvc`` and
``wvc:op:<kind>``; comment lines after the front matter carry the manifest root and the commit
id. ``write_manifest`` writes the manifest exactly as ``wvc hash`` prints it, for publishing
as a sidecar file (conventionally ``wvc-manifest.json``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wvc.records import StoreError
from wvc.repo import log_entries
from wvc.sources import is_hub_repo_id
from wvc.store import ObjectStore, manifest_json

CARD_TAG = "wvc"


@dataclass(frozen=True)
class CardResult:
    """The model-card front matter of one commit."""

    commit_id: str
    manifest_root: str
    operation_kind: str
    base_model: str | None  # the nearest hf source on the first-parent chain, if any
    yaml: str  # the front matter and the comment lines, ending with a newline


def model_card(store: ObjectStore, ref: str) -> CardResult:
    """The model-card front matter of commit ``ref`` (StoreError when it does not resolve)."""
    commit_id = store.resolve(ref)
    record = store.get_commit(commit_id)
    root = record["manifest"]
    store.get_manifest(root)  # the card states this root, so it must be stored and valid
    kind = record["operation"]["kind"]
    base_model = nearest_hub_source(store, commit_id)
    text = render_card(
        base_model=base_model, operation_kind=kind, manifest_root=root, commit_id=commit_id
    )
    return CardResult(commit_id, root, kind, base_model, text)


def nearest_hub_source(store: ObjectStore, commit_id: str) -> str | None:
    """The ``source.repo`` of the nearest commit whose source kind is ``hf``, or None.

    The search follows first parents from ``commit_id`` itself. A matching source whose repo
    is not an ``owner/name`` Hub repo id is a StoreError rather than text put into YAML.
    """
    for current_id, record in log_entries(store, commit_id):
        source = record.get("source")
        if isinstance(source, dict) and source.get("kind") == "hf":
            repo = source.get("repo")
            if not is_hub_repo_id(repo):
                raise StoreError(f"commit {current_id}: source.repo is not OWNER/NAME")
            return repo
    return None


def render_card(
    *, base_model: str | None, operation_kind: str, manifest_root: str, commit_id: str
) -> str:
    """YAML front matter (``base_model`` only when known), then the provenance comments."""
    lines = [
        "---",
        *([] if base_model is None else [f"base_model: {base_model}"]),
        "tags:",
        f"- {CARD_TAG}",
        f"- {CARD_TAG}:op:{operation_kind}",
        "---",
        "# wvc provenance",
        f"# manifest root: {manifest_root}",
        f"# commit: {commit_id}",
    ]
    return "\n".join(lines) + "\n"


def write_manifest(store: ObjectStore, manifest_root: str, path: str | Path) -> Path:
    """Write a stored manifest to ``path`` exactly as ``wvc hash`` prints it; return the path."""
    target = Path(path)
    target.write_bytes(manifest_json(store.get_manifest(manifest_root)))
    return target
