"""Tensor-level diff between two checkpoints (spec section 2).

Identity first: two tensors of the same dtype and shape are ``unchanged`` only when their
stored bytes are identical, and ``changed`` otherwise, with ``changed`` counting the elements
whose bytes differ (so a sign flip of zero, a NaN payload, or an integer beyond float
precision all count). A dtype or shape difference is ``reshaped``; when the shapes agree and
both dtypes decode, the numeric statistics are still computed from the decoded values.
Tensors are compared in fixed-size chunks of their raw bytes, so a diff never holds more than
a few chunks in memory. The optional rank estimate decodes only the touched rows and columns.
Its ``rank`` and ``top1_energy`` come from LAPACK's SVD, so unlike every other statistic here
they are not bit-reproducible across machines.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace

import numpy as np

from wvc.errors import WvcError
from wvc.safetensors_io import (
    FP8_FORMATS,
    ITEMSIZES,
    NUMPY_DTYPES,
    Checkpoint,
    TensorInfo,
    decodable,
    fp8_table,
)

RANK_REL_TOL = 1e-2  # singular values below this fraction of the largest are ignored
# Skip the SVD when the touched submatrix has more elements. 2^26 covers a 9B-class projection
# matrix (4096 x 12288, E008). The block is float64 and is copied once by concatenate and once
# by LAPACK, so a block at the cap peaks near 1.1 GiB and can take about 20 s.
DEFAULT_RANK_CAP = 1 << 26
CHUNK_ELEMENTS = 1 << 22  # elements compared per chunk (32 MiB of float64 per side)
UINT_VIEWS = {size: np.dtype(f"<u{size}") for size in (1, 2, 4, 8)}
OPTIONAL_FIELDS = frozenset(
    ("compared", "name_a", "name_b", "transform_a", "transform_b", "sha256", "renamed_to")
)
RowDecoder = Callable[[int, int], np.ndarray]  # float64 values of rows [first, stop), flat


@dataclass(frozen=True)
class TensorDiff:
    name: str
    status: str  # unchanged | changed | reshaped | added | removed
    dtype_a: str | None = None
    dtype_b: str | None = None
    shape_a: tuple[int, ...] | None = None
    shape_b: tuple[int, ...] | None = None
    numel: int | None = None
    changed: int | None = None  # elements whose bytes differ (same dtype) or values differ
    nonfinite_changed: int | None = None  # changed elements with a NaN or infinite delta
    max_abs_delta: float | None = None
    delta_norm: float | None = None
    rel_delta_norm: float | None = None
    rows_touched: int | None = None
    cols_touched: int | None = None
    rank: int | None = None
    top1_energy: float | None = None
    # Present only in a quantization comparison (spec section 2.1), and omitted when None:
    compared: str | None = None  # "values": compared by decoded values, not by bytes
    name_a: str | None = None  # the tensor's name in a's file, when it is not ``name``
    name_b: str | None = None
    transform_a: str | None = None  # what was undone to bring a's values into the Hub frame
    transform_b: str | None = None
    # Omitted when None (spec section 2): the id of an added or removed tensor, and the name
    # in b of a tensor paired across a rename, whose ``name`` is then its name in a.
    sha256: str | None = None
    renamed_to: str | None = None

    def to_dict(self) -> dict:
        """A JSON-safe dict: non-finite floats become null so strict parsers accept it.

        The quantization fields, ``sha256``, and ``renamed_to`` appear only when set, so an
        entry that is neither one-sided nor renamed keeps exactly the keys it had before.
        """
        return {
            key: (None if isinstance(value, float) and not np.isfinite(value) else value)
            for key, value in asdict(self).items()
            if key not in OPTIONAL_FIELDS or value is not None
        }


@dataclass
class _Accumulator:
    """Running statistics over the chunks of one tensor pair."""

    changed: int = 0
    nonfinite: int = 0
    max_abs: float = 0.0
    any_finite: bool = False
    delta_sumsq: float = 0.0
    base_sumsq: float = 0.0


def decode_f64(raw: np.ndarray, dtype: str) -> np.ndarray:
    """Decode raw little-endian bytes to float64 (exact for every dtype up to 2^53)."""
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << np.uint32(16)
        return bits.view(np.float32).astype(np.float64)
    if dtype in FP8_FORMATS:
        return fp8_table(dtype)[np.frombuffer(raw, dtype=np.uint8)].astype(np.float64)
    return np.frombuffer(raw, dtype=np.dtype(NUMPY_DTYPES[dtype]).newbyteorder("<")).astype(
        np.float64
    )


def bytes_differ(a: np.ndarray, b: np.ndarray, itemsize: int) -> np.ndarray:
    """Per-element mask of the elements whose stored bytes differ."""
    view = UINT_VIEWS.get(itemsize)
    if view is not None:
        return a.view(view) != b.view(view)
    return (a.reshape(-1, itemsize) != b.reshape(-1, itemsize)).any(axis=1)


def _fold(acc: _Accumulator, mask: np.ndarray, a: np.ndarray | None, b: np.ndarray | None) -> None:
    """Fold one chunk's change mask and decoded values (when decodable) into ``acc``."""
    acc.changed += int(mask.sum())
    if a is None or b is None:
        return
    acc.base_sumsq += float(np.sum(np.square(a[np.isfinite(a)])))
    if not mask.any():
        return
    delta = b[mask] - a[mask]
    finite = np.isfinite(delta)
    acc.nonfinite += int((~finite).sum())
    if finite.any():
        acc.any_finite = True
        acc.max_abs = max(acc.max_abs, float(np.abs(delta[finite]).max()))
        acc.delta_sumsq += float(np.sum(np.square(delta[finite])))


def _mark_hits(mask: np.ndarray, start: int, hits: tuple | None) -> None:
    if hits is None or not mask.any():
        return
    row_hits, col_hits, ncols = hits
    index = np.flatnonzero(mask) + start
    row_hits[np.unique(index // ncols)] = True
    col_hits[np.unique(index % ncols)] = True


def _chunk_mask(
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    info_a: TensorInfo,
    info_b: TensorInfo,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """The change mask of elements ``start:stop`` plus the decoded chunks when decodable."""
    size_a, size_b = ITEMSIZES[info_a.dtype], ITEMSIZES[info_b.dtype]
    slice_a = raw_a[start * size_a : stop * size_a]
    slice_b = raw_b[start * size_b : stop * size_b]
    both = decodable(info_a.dtype) and decodable(info_b.dtype)
    a = decode_f64(slice_a, info_a.dtype) if both else None
    b = decode_f64(slice_b, info_b.dtype) if both else None
    if info_a.dtype == info_b.dtype:
        return bytes_differ(slice_a, slice_b, size_a), a, b
    if a is None or b is None:
        raise WvcError(f"cannot compare {info_a.dtype} with {info_b.dtype} by value")
    return (a != b) & ~(np.isnan(a) & np.isnan(b)), a, b


def _delta_submatrix(
    raw_a: np.ndarray, raw_b: np.ndarray, info_a: TensorInfo, info_b: TensorInfo, hits: tuple
) -> np.ndarray:
    """The change restricted to the touched rows and columns, decoded block by block."""
    row_hits, col_hits, ncols = hits
    rows, cols = np.flatnonzero(row_hits), np.flatnonzero(col_hits)
    block = max(1, CHUNK_ELEMENTS // ncols)
    pieces = []
    for first in range(0, len(row_hits), block):
        selected = rows[(rows >= first) & (rows < first + block)]
        if selected.size == 0:
            continue
        start, stop = first * ncols, min(first + block, len(row_hits)) * ncols
        _, a, b = _chunk_mask(raw_a, raw_b, info_a, info_b, start, stop)
        if a is None or b is None:
            return np.zeros((0, 0))
        local = selected - first
        pieces.append(b.reshape(-1, ncols)[local][:, cols] - a.reshape(-1, ncols)[local][:, cols])
    return np.concatenate(pieces) if pieces else np.zeros((0, 0))


def _rank_stats(sub: np.ndarray) -> tuple[int | None, float | None]:
    if sub.size == 0 or not np.isfinite(sub).all():
        return None, None
    singular = np.linalg.svd(sub, compute_uv=False)
    if singular.size == 0 or singular[0] <= 0:
        return None, None
    rank = int((singular > RANK_REL_TOL * singular[0]).sum())
    return rank, float(singular[0] ** 2 / np.sum(singular**2))


def _hits_for(info: TensorInfo) -> tuple | None:
    if len(info.shape) >= 2 and info.shape[0] > 0 and info.numel > 0:
        nrows = info.shape[0]
        return (
            np.zeros(nrows, dtype=bool),
            np.zeros(info.numel // nrows, dtype=bool),
            info.numel // nrows,
        )
    return None


def diff_tensor(
    name: str,
    info_a: TensorInfo,
    info_b: TensorInfo,
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    *,
    rank: bool = False,
    rank_cap: int = DEFAULT_RANK_CAP,
) -> TensorDiff:
    """Compare two tensors of equal shape from their raw stored bytes (spec section 2)."""
    same_layout = info_a.dtype == info_b.dtype
    summary = TensorDiff(
        name=name,
        status="changed" if same_layout else "reshaped",
        dtype_a=info_a.dtype,
        dtype_b=info_b.dtype,
        shape_a=info_a.shape,
        shape_b=info_b.shape,
        numel=info_a.numel,
    )
    if not same_layout and not (decodable(info_a.dtype) and decodable(info_b.dtype)):
        return summary
    hits = _hits_for(info_a)
    acc = _Accumulator()
    for start in range(0, info_a.numel, CHUNK_ELEMENTS):
        stop = min(start + CHUNK_ELEMENTS, info_a.numel)
        mask, a, b = _chunk_mask(raw_a, raw_b, info_a, info_b, start, stop)
        _fold(acc, mask, a, b)
        _mark_hits(mask, start, hits)
    if same_layout and acc.changed == 0:
        return replace(summary, status="unchanged", changed=0)
    fields = _finish(acc, hits, info_a, decodable(info_a.dtype) and decodable(info_b.dtype))
    if rank and hits is not None and fields["rows_touched"] * fields["cols_touched"] <= rank_cap:
        sub = _delta_submatrix(raw_a, raw_b, info_a, info_b, hits)
        fields["rank"], fields["top1_energy"] = _rank_stats(sub)
    return replace(summary, **fields)


def _finish(acc: _Accumulator, hits: tuple | None, info: TensorInfo, numeric: bool) -> dict:
    """The statistics of a changed tensor once every chunk has been folded in."""
    fields: dict = {"changed": acc.changed}
    if numeric:
        delta_norm = float(np.sqrt(acc.delta_sumsq))
        base_norm = float(np.sqrt(acc.base_sumsq))
        fields.update(
            nonfinite_changed=acc.nonfinite,
            max_abs_delta=acc.max_abs if acc.any_finite else None,
            delta_norm=delta_norm,
            rel_delta_norm=delta_norm / base_norm if base_norm > 0 else None,
        )
    if hits is None:
        fields["rows_touched"] = acc.changed if len(info.shape) == 1 else None
        return fields
    row_hits, col_hits, _ = hits
    fields.update(rows_touched=int(row_hits.sum()), cols_touched=int(col_hits.sum()))
    return fields


def value_grid(shape: tuple[int, ...]) -> tuple[int, int]:
    """The (rows, columns) a value comparison walks: the first axis by the rest; 1-D by element."""
    numel = math.prod(shape)
    if len(shape) >= 2 and shape[0] > 0:
        return shape[0], numel // shape[0]
    return numel, 1


def _value_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a != b) & ~(np.isnan(a) & np.isnan(b))


def diff_values(
    name: str,
    info_a: TensorInfo,
    info_b: TensorInfo,
    rows_a: RowDecoder,
    rows_b: RowDecoder,
    *,
    equal_is_unchanged: bool,
    rank: bool = False,
    rank_cap: int = DEFAULT_RANK_CAP,
) -> TensorDiff:
    """Compare two tensors of equal shape by their decoded values (spec section 2.1).

    ``rows_a`` and ``rows_b`` decode whole rows of ``value_grid`` to float64. When every value
    is equal (NaN equal to NaN) the pair is ``unchanged`` if ``equal_is_unchanged`` (the
    dtypes differ, or a conversion step was undone) and ``changed`` with no element changed
    otherwise: a pair of one dtype with nothing undone reaches here only when its bytes
    differ. Statistics are those of ``diff_tensor``.
    """
    nrows, ncols = value_grid(info_a.shape)
    hits = _hits_for(info_a)
    acc = _Accumulator()
    step = max(1, CHUNK_ELEMENTS // max(ncols, 1))
    for first in range(0, nrows, step):
        stop = min(first + step, nrows)
        a, b = rows_a(first, stop), rows_b(first, stop)
        mask = _value_mask(a, b)
        _fold(acc, mask, a, b)
        _mark_hits(mask, first * ncols, hits)
    summary = TensorDiff(
        name=name,
        status="changed",
        dtype_a=info_a.dtype,
        dtype_b=info_b.dtype,
        shape_a=info_a.shape,
        shape_b=info_b.shape,
        numel=info_a.numel,
        compared="values",
    )
    if acc.changed == 0 and equal_is_unchanged:
        return replace(summary, status="unchanged", changed=0)
    fields = _finish(acc, hits, info_a, numeric=True)
    if rank and hits is not None and fields["rows_touched"] * fields["cols_touched"] <= rank_cap:
        sub = _value_submatrix(rows_a, rows_b, hits)
        fields["rank"], fields["top1_energy"] = _rank_stats(sub)
    return replace(summary, **fields)


def _value_submatrix(rows_a: RowDecoder, rows_b: RowDecoder, hits: tuple) -> np.ndarray:
    """The change restricted to the touched rows and columns, decoded block by block."""
    row_hits, col_hits, ncols = hits
    rows, cols = np.flatnonzero(row_hits), np.flatnonzero(col_hits)
    block = max(1, CHUNK_ELEMENTS // ncols)
    pieces = []
    for first in range(0, len(row_hits), block):
        selected = rows[(rows >= first) & (rows < first + block)]
        if selected.size == 0:
            continue
        stop = min(first + block, len(row_hits))
        a = rows_a(first, stop).reshape(-1, ncols)
        b = rows_b(first, stop).reshape(-1, ncols)
        local = selected - first
        pieces.append(b[local][:, cols] - a[local][:, cols])
    return np.concatenate(pieces) if pieces else np.zeros((0, 0))


def unchanged_diff(name: str, info: TensorInfo) -> TensorDiff:
    """The record of a tensor whose stored bytes are identical in both checkpoints."""
    return TensorDiff(
        name=name,
        status="unchanged",
        dtype_a=info.dtype,
        dtype_b=info.dtype,
        shape_a=info.shape,
        shape_b=info.shape,
        numel=info.numel,
        changed=0,
    )


def _one_sided(
    name: str, info_a: TensorInfo | None, info_b: TensorInfo | None, sha256: str | None = None
) -> TensorDiff:
    """Describe a tensor that is present in only one of the two checkpoints, with its id."""
    if info_a is None and info_b is not None:
        return TensorDiff(
            name,
            "added",
            dtype_b=info_b.dtype,
            shape_b=info_b.shape,
            numel=info_b.numel,
            sha256=sha256,
        )
    if info_b is None and info_a is not None:
        return TensorDiff(
            name,
            "removed",
            dtype_a=info_a.dtype,
            shape_a=info_a.shape,
            numel=info_a.numel,
            sha256=sha256,
        )
    raise ValueError(f"tensor {name!r} is missing from both checkpoints")


def _layout(info: TensorInfo) -> tuple[str, tuple[int, ...]]:
    return info.dtype, info.shape


def _unique_pairs(candidates: Mapping[str, set[str]]) -> list[tuple[str, str]]:
    """The (removed, added) pairs in which each of the two is the other's only candidate."""
    reverse: dict[str, set[str]] = {}
    for removed, found in candidates.items():
        for added in found:
            reverse.setdefault(added, set()).add(removed)
    pairs = []
    for removed, found in candidates.items():
        if len(found) == 1:
            (added,) = found
            if len(reverse[added]) == 1:
                pairs.append((removed, added))
    return sorted(pairs)


def pair_by_id(
    removed: Mapping[str, TensorInfo],
    added: Mapping[str, TensorInfo],
    digests_a: Mapping[str, str],
    digests_b: Mapping[str, str],
) -> list[tuple[str, str]]:
    """Removed and added tensors of one id, dtype, and shape: one tensor under two names.

    A pair is made only when its key (id, dtype, shape) holds exactly one removed and one
    added tensor; copies of one tensor under several names leave every one of them unpaired.
    """
    by_key: dict[tuple, set[str]] = {}
    for name, info in added.items():
        by_key.setdefault((digests_b[name], *_layout(info)), set()).add(name)
    candidates = {
        name: by_key.get((digests_a[name], *_layout(info)), set()) for name, info in removed.items()
    }
    return _unique_pairs(candidates)


def _proper_suffixes(name: str) -> list[str]:
    """``b.c`` and ``c`` for ``a.b.c``: the name with one or more leading components dropped."""
    parts = name.split(".")
    return [".".join(parts[index:]) for index in range(1, len(parts))]


def pair_by_suffix(
    removed: Mapping[str, TensorInfo], added: Mapping[str, TensorInfo]
) -> list[tuple[str, str]]:
    """Removed and added tensors of one dtype and shape whose names differ by a leading prefix.

    One name must be the other with one or more dotted components prepended, in either
    direction (``model.norm.weight`` and ``language_model.model.norm.weight``). A pair is
    made only when each of the two is the other's only candidate.
    """
    longer: dict[str, set[str]] = {}
    for name in added:
        for suffix in _proper_suffixes(name):
            longer.setdefault(suffix, set()).add(name)
    candidates = {}
    for name, info in removed.items():
        found = longer.get(name, set()) | {s for s in _proper_suffixes(name) if s in added}
        candidates[name] = {other for other in found if _layout(added[other]) == _layout(info)}
    return _unique_pairs(candidates)


def pair_renames(
    removed: Mapping[str, TensorInfo],
    added: Mapping[str, TensorInfo],
    digests_a: Mapping[str, str],
    digests_b: Mapping[str, str],
) -> dict[str, str]:
    """Name in a to name in b of the tensors paired across a rename: by id, then by suffix."""
    by_id = dict(pair_by_id(removed, added, digests_a, digests_b))
    taken = set(by_id.values())
    rest_a = {name: info for name, info in removed.items() if name not in by_id}
    rest_b = {name: info for name, info in added.items() if name not in taken}
    return {**by_id, **dict(pair_by_suffix(rest_a, rest_b))}


def diff_checkpoints(
    ckpt_a: Checkpoint,
    ckpt_b: Checkpoint,
    digests_a: Mapping[str, str],
    digests_b: Mapping[str, str],
    *,
    rank: bool = False,
    rank_cap: int = DEFAULT_RANK_CAP,
) -> list[TensorDiff]:
    """Diff two checkpoints tensor by tensor, decoding only tensors whose digests differ.

    A tensor present under one name only is first paired across a rename (``pair_renames``,
    spec section 2): a pair is reported under its name in a, with ``renamed_to`` its name in
    b, and compared as a same-name pair would be. What stays unpaired is ``added`` or
    ``removed`` and carries its id in ``sha256``.

    The result is sorted by name. The tensors whose bytes must be read are compared grouped by
    the pair of files that hold them, in offset order within a file, so each checkpoint keeps
    only one file in use at a time whatever its shard count.
    """
    tensors_a, tensors_b = ckpt_a.tensors, ckpt_b.tensors
    removed = {name: info for name, info in tensors_a.items() if name not in tensors_b}
    added = {name: info for name, info in tensors_b.items() if name not in tensors_a}
    renamed = pair_renames(removed, added, digests_a, digests_b)
    paired_b = set(renamed.values())
    results: dict[str, TensorDiff] = {}
    to_read: list[tuple[str, str]] = []
    for name in sorted(set(tensors_a) | set(tensors_b)):
        if name in paired_b:
            continue
        other = renamed.get(name, name)
        info_a, info_b = tensors_a.get(name), tensors_b.get(other)
        if info_a is None or info_b is None:
            digest = digests_a[name] if info_a is not None else digests_b[name]
            results[name] = _one_sided(name, info_a, info_b, digest)
        elif info_a.shape != info_b.shape:
            results[name] = TensorDiff(
                name=name,
                status="reshaped",
                dtype_a=info_a.dtype,
                dtype_b=info_b.dtype,
                shape_a=info_a.shape,
                shape_b=info_b.shape,
                numel=info_a.numel,
            )
        elif info_a.dtype == info_b.dtype and digests_a[name] == digests_b[other]:
            results[name] = unchanged_diff(name, info_a)
        else:
            to_read.append((name, other))
    for name, other in sorted(to_read, key=_file_pair_order(ckpt_a, ckpt_b)):
        results[name] = diff_tensor(
            name,
            tensors_a[name],
            tensors_b[other],
            ckpt_a.raw(name),
            ckpt_b.raw(other),
            rank=rank,
            rank_cap=rank_cap,
        )
    for name, other in renamed.items():
        results[name] = replace(results[name], renamed_to=other)
    return [results[name] for name in sorted(results)]


def _file_pair_order(ckpt_a: Checkpoint, ckpt_b: Checkpoint) -> Callable[[tuple[str, str]], tuple]:
    """A sort key grouping (name in a, name in b) by (file in a, file in b), then offset in a."""
    index_a = {file.path: index for index, file in enumerate(ckpt_a.files)}
    index_b = {file.path: index for index, file in enumerate(ckpt_b.files)}

    def key(pair: tuple[str, str]) -> tuple[int, int, int, str]:
        name, other = pair
        file_a, file_b = ckpt_a.file_of(name), ckpt_b.file_of(other)
        return index_a[file_a.path], index_b[file_b.path], ckpt_a.tensors[name].begin, name

    return key


def summarize(diffs: list[TensorDiff]) -> dict:
    """Counts by status, the tensors paired across a rename, and elements changed of compared."""
    counts = {status: 0 for status in ("unchanged", "changed", "added", "removed", "reshaped")}
    for item in diffs:
        counts[item.status] += 1
    compared = [d for d in diffs if d.status in ("unchanged", "changed")]
    total = sum(d.numel or 0 for d in compared)
    changed = sum(d.changed or 0 for d in compared)
    return {
        "tensors": len(diffs),
        "by_status": counts,
        "renamed": sum(1 for d in diffs if d.renamed_to is not None),
        "elements_compared": total,
        "elements_changed": changed,
        "fraction_changed": (changed / total) if total else None,
    }
