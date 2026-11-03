"""Merge a PEFT LoRA adapter into a commit and commit the result (spec section 5.3).

Everything that can be refused is checked before the output directory is written: the adapter,
the commit text, the base commit's file records, the branch the merge goes on, and every target
against the base manifest. The base is then checked out into the output directory; each
target's byte range is overwritten in place with the tensor ``lowrank-merge/0`` computes, so
the files keep the base's headers and shard layout; and the directory is committed with the
base as its parent and operation kind ``adapter-merge``, each target stored as a
``delta-lowrank`` object against the base tensor (recomputed and verified before it is stored)
and everything else reused by hash. As for any commit, a base that is not the head of the
current branch needs ``branch`` (``--branch``), so that no branch loses commits.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from wvc.checkout import checkout_commit
from wvc.file_layout import validate_file_records
from wvc.hashing import digest_bytes
from wvc.lowrank import (
    FLOAT_DTYPES,
    PROCEDURE,
    LowRankError,
    LowRankFactors,
    check_factor_shapes,
    merge_lowrank,
)
from wvc.peft import Adapter, load_adapter
from wvc.records import IntegrityError, StoreError
from wvc.repo import (
    CommitMeta,
    CommitResult,
    check_commit_text,
    commit_target,
    commit_with_meta,
)
from wvc.safetensors_io import SafetensorsFile, TensorInfo, read_header
from wvc.store import ObjectStore

OPERATION_KIND = "adapter-merge"


def merge_adapter(
    store: ObjectStore,
    base_ref: str,
    adapter_dir: str | Path,
    out_dir: str | Path,
    *,
    message: str,
    author: str,
    branch: str | None = None,
) -> CommitResult:
    """Merge the PEFT LoRA adapter in ``adapter_dir`` into commit ``base_ref`` and commit it.

    ``out_dir`` must be missing or empty; it receives the merged checkpoint. The commit goes on
    ``branch`` when given, else on the current branch, whose head the base must then be (see
    ``wvc.repo.commit_target``). StoreError for an adapter, target, base, or branch that cannot
    be merged into, before anything is written.
    """
    adapter = load_adapter(adapter_dir)
    return merge_loaded_adapter(
        store, base_ref, adapter, out_dir, message=message, author=author, branch=branch
    )


def merge_loaded_adapter(
    store: ObjectStore,
    base_ref: str,
    adapter: Adapter,
    out_dir: str | Path,
    *,
    message: str,
    author: str,
    branch: str | None = None,
) -> CommitResult:
    """``merge_adapter`` for an adapter already read by ``load_adapter``."""
    check_commit_text(message=message, operation_kind=OPERATION_KIND, author=author)
    base_id = store.resolve(base_ref)
    commit_target(store, base_id, branch)  # refuses a branch the merge may not go on
    record = store.get_commit(base_id)
    files = validate_file_records(record["files"], f"commit {base_id}")
    entries = store.get_manifest(record["manifest"])["tensors"]
    _check_base(adapter, entries)
    checkout_commit(store, base_id, out_dir)  # refuses a non-empty directory
    located = _locate(Path(out_dir), files)
    factors = {
        target: _merge_target(adapter, target, entries[target], located)
        for target in adapter.targets
    }
    meta = CommitMeta(message, author, OPERATION_KIND, {"adapter": adapter.operation()})
    return commit_with_meta(
        store, out_dir, meta, parent_ref=base_id, branch=branch, lowrank=factors
    )


def _check_base(adapter: Adapter, entries: Mapping[str, Mapping]) -> None:
    """Refuse targets the base cannot take (file names were checked with the file records)."""
    for target, (a, b) in adapter.targets.items():
        entry = entries.get(target)
        if entry is None:
            raise StoreError(f"adapter target {target!r} is not a tensor of the base commit")
        if entry["dtype"] not in FLOAT_DTYPES:
            raise StoreError(
                f"adapter target {target!r} is {entry['dtype']} in the base; {PROCEDURE} merges "
                f"into {', '.join(FLOAT_DTYPES)} tensors only"
            )
        try:
            check_factor_shapes(entry["shape"], a.shape, b.shape)
        except LowRankError as exc:
            raise StoreError(f"adapter target {target!r}: {exc}") from exc


def _locate(out: Path, files: list[dict]) -> dict[str, tuple[SafetensorsFile, TensorInfo]]:
    """Every tensor of the checked-out files, with the file that holds it."""
    headers = [read_header(out / entry["name"]) for entry in files]
    return {name: (file, info) for file in headers for name, info in file.tensors.items()}


def _merge_target(
    adapter: Adapter,
    target: str,
    entry: Mapping,
    located: Mapping[str, tuple[SafetensorsFile, TensorInfo]],
) -> LowRankFactors:
    """Overwrite one target's byte range in its checked-out file with the merged tensor."""
    file, info = _checked_location(target, entry, located)
    factors = adapter.factors(target)
    offset = file.data_start + info.begin
    with file.path.open("r+b") as handle:
        handle.seek(offset)
        base = np.frombuffer(handle.read(info.nbytes), dtype=np.uint8)
        if base.size != info.nbytes or digest_bytes(base) != entry["sha256"]:
            raise IntegrityError(
                f"{file.path}: the bytes of {target!r} are not the base tensor {entry['sha256']}"
            )
        merged = _merged(base, info, factors)
        handle.seek(offset)
        handle.write(merged.data)
    return factors


def _checked_location(
    target: str, entry: Mapping, located: Mapping[str, tuple[SafetensorsFile, TensorInfo]]
) -> tuple[SafetensorsFile, TensorInfo]:
    """Where a target sits in the checked-out files; its dtype and shape must be the manifest's."""
    found = located.get(target)
    if found is None or (found[1].dtype, list(found[1].shape)) != (
        entry["dtype"],
        list(entry["shape"]),
    ):
        raise IntegrityError(
            f"the checked-out files do not hold {target!r} as {entry['dtype']}{entry['shape']}, "
            "as the base manifest records"
        )
    return found


def _merged(base: np.ndarray, info: TensorInfo, factors: LowRankFactors) -> np.ndarray:
    """The bytes of the merged tensor under ``lowrank-merge/0``."""
    try:
        return merge_lowrank(base, info.dtype, info.shape, factors)
    except LowRankError as exc:
        raise StoreError(f"adapter target {info.name!r}: {exc}") from exc
