"""Diffs with a GGUF side: a quantization comparison (spec section 2.1).

When one checkpoint is a GGUF file and the other a safetensors checkpoint, the GGUF names are
mapped to Hub names by ``general.architecture`` (``wvc.gguf_map``), the conversion's
permutation and shift are undone, and each pair is compared:

- same dtype, nothing to undo, identical bytes: ``unchanged``, by identity, as in section 2;
- same plain dtype, nothing to undo, different bytes: by bytes, as in section 2;
- anything else (a quantized tensor against floats, F32 against BF16, a permuted projection):
  by values, every value decoded or dequantised exactly to float64 (``diff.diff_values``),
  with ``compared: "values"`` in the tensor's record.

Two GGUF files are compared under their own names, with no mapping. The record says what was
done: a top-level ``comparison`` object, a ``gguf`` object on each GGUF side, and per tensor
the file's own name and what was undone. A type wvc cannot dequantise is refused by name.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from collections.abc import Mapping as MappingType
from dataclasses import dataclass, replace

import numpy as np

from wvc.diff import (
    DEFAULT_RANK_CAP,
    RowDecoder,
    TensorDiff,
    decode_f64,
    diff_tensor,
    diff_values,
    unchanged_diff,
    value_grid,
)
from wvc.gguf_io import BLOCK_TYPES, GgufFile, gguf_files
from wvc.gguf_map import Located, Mapping, MappingError, locate, mapping_for, rotary_rows
from wvc.gguf_quants import DequantError, can_dequantize, dequantize, supported_text
from wvc.safetensors_io import ITEMSIZES, Checkpoint, TensorInfo

COMPARISON_KIND = "quantization"
COMPARISON_DTYPE = "float64"


@dataclass(frozen=True)
class Side:
    """One checkpoint as the comparison sees it: tensors by comparison name."""

    ckpt: Checkpoint
    gguf: GgufFile | None
    mapping: Mapping | None
    located: dict[str, Located]  # comparison name -> where it is and what to undo

    def info(self, name: str) -> TensorInfo:
        return self.ckpt.tensors[self.located[name].gguf_name]

    def transform(self, name: str) -> str | None:
        return self.located[name].transform

    def file_name(self, name: str) -> str | None:
        """The tensor's name in its file, when the comparison calls it something else."""
        stored = self.located[name].gguf_name
        return None if stored == name else stored


def _side(ckpt: Checkpoint, map_names: bool) -> Side:
    files = gguf_files(ckpt)
    gguf = files[0] if files else None
    mapping = mapping_for(gguf) if gguf is not None and map_names else None
    located: dict[str, Located] = {}
    for stored in ckpt.tensors:
        entry = (
            locate(gguf, mapping, stored)
            if gguf is not None and mapping is not None
            else Located(stored, stored)
        )
        if entry.name in located:
            raise MappingError(
                f"tensors {located[entry.name].gguf_name!r} and {stored!r} both map to "
                f"{entry.name!r}"
            )
        located[entry.name] = entry
    return Side(ckpt=ckpt, gguf=gguf, mapping=mapping, located=located)


def is_quantization_pair(ckpt_a: Checkpoint, ckpt_b: Checkpoint) -> bool:
    """True when either checkpoint is a GGUF file."""
    return bool(gguf_files(ckpt_a) or gguf_files(ckpt_b))


def _decode(raw: np.ndarray, dtype: str) -> np.ndarray:
    if dtype in BLOCK_TYPES:
        block, size = BLOCK_TYPES[dtype]
        return dequantize(raw, dtype, block, size).astype(np.float64)
    return decode_f64(raw, dtype)


def _rows(side: Side, name: str) -> RowDecoder:
    """Decode whole rows of ``value_grid`` in the Hub frame, undoing permutation and shift."""
    located = side.located[name]
    info = side.info(name)
    nrows, ncols = value_grid(info.shape)
    block, size = BLOCK_TYPES.get(info.dtype, (1, info.nbytes // max(info.numel, 1)))
    order = None if located.heads is None else rotary_rows(nrows, located.heads, name)
    if order is not None and ncols % block:
        raise DequantError(
            f"tensor {located.gguf_name!r} of type {info.dtype} cannot be unpermuted: its "
            "rows are not whole blocks"
        )

    def span(first: int, stop: int) -> slice:
        """The bytes of elements [first, stop), which must begin and end on block bounds."""
        if first % block or (stop % block and stop != info.numel):
            raise DequantError(f"elements [{first}, {stop}) of {name!r} split a {info.dtype} block")
        return slice(first // block * size, -(-stop // block) * size)

    def rows(first: int, stop: int) -> np.ndarray:
        raw = side.ckpt.raw(located.gguf_name)
        if order is None:
            chunk = raw[span(first * ncols, stop * ncols)]
        else:
            row_bytes = ncols // block * size
            chunk = raw.reshape(nrows, row_bytes)[order[first:stop]].reshape(-1)
        values = _decode(chunk, info.dtype)
        return values - located.shift if located.shift else values

    return rows


def _with_sources(item: TensorDiff, a: Side, b: Side) -> TensorDiff:
    name = item.name
    fields = {}
    if name in a.located:
        fields.update(name_a=a.file_name(name), transform_a=a.transform(name))
    if name in b.located:
        fields.update(name_b=b.file_name(name), transform_b=b.transform(name))
    return replace(item, **fields)


def _require_decodable(name: str, info_a: TensorInfo, info_b: TensorInfo) -> None:
    for info in (info_a, info_b):
        if not can_dequantize(info.dtype):
            raise DequantError(
                f"tensor {name!r} has type {info.dtype}, which wvc cannot decode or dequantise "
                f"(it reads {supported_text()} and the safetensors dtypes)"
            )


def _pair(
    name: str, a: Side, b: Side, digests: tuple[MappingType[str, str], MappingType[str, str]]
) -> TensorDiff | None:
    """The record of a pair that needs no reading, or None when its values must be read."""
    if name not in b.located:
        info = a.info(name)
        return TensorDiff(name, "removed", dtype_a=info.dtype, shape_a=info.shape, numel=info.numel)
    if name not in a.located:
        info = b.info(name)
        return TensorDiff(name, "added", dtype_b=info.dtype, shape_b=info.shape, numel=info.numel)
    info_a, info_b = a.info(name), b.info(name)
    if info_a.shape != info_b.shape:
        return TensorDiff(
            name, "reshaped", info_a.dtype, info_b.dtype, info_a.shape, info_b.shape, info_a.numel
        )
    plain = a.transform(name) is None and b.transform(name) is None
    same_bytes = digests[0][a.located[name].gguf_name] == digests[1][b.located[name].gguf_name]
    if plain and info_a.dtype == info_b.dtype and same_bytes:
        return unchanged_diff(name, info_a)
    return None


def compare_quantized(
    ckpt_a: Checkpoint,
    ckpt_b: Checkpoint,
    digests_a: MappingType[str, str],
    digests_b: MappingType[str, str],
    *,
    rank: bool = False,
    rank_cap: int = DEFAULT_RANK_CAP,
) -> tuple[list[TensorDiff], dict]:
    """Diff two checkpoints of which at least one is a GGUF file; return the record parts.

    Returns the per-tensor diffs sorted by comparison name and the ``comparison`` object.
    """
    one_gguf = bool(gguf_files(ckpt_a)) != bool(gguf_files(ckpt_b))
    a, b = _side(ckpt_a, one_gguf), _side(ckpt_b, one_gguf)
    names = sorted(set(a.located) | set(b.located))
    results: dict[str, TensorDiff] = {}
    to_read = []
    for name in names:
        done = _pair(name, a, b, (digests_a, digests_b))
        if done is None:
            _require_decodable(name, a.info(name), b.info(name))
            to_read.append(name)
        else:
            results[name] = done
    for name in sorted(to_read, key=_reading_order(a, b)):
        results[name] = _compare(name, a, b, rank=rank, rank_cap=rank_cap)
    diffs = [_with_sources(results[name], a, b) for name in names]
    return diffs, _comparison(a, b)


def _compare(name: str, a: Side, b: Side, *, rank: bool, rank_cap: int) -> TensorDiff:
    info_a, info_b = a.info(name), b.info(name)
    by_bytes = (
        a.transform(name) is None
        and b.transform(name) is None
        and info_a.dtype == info_b.dtype
        and info_a.dtype in ITEMSIZES
    )
    if by_bytes:
        raw_a, raw_b = a.ckpt.raw(a.located[name].gguf_name), b.ckpt.raw(b.located[name].gguf_name)
        return diff_tensor(name, info_a, info_b, raw_a, raw_b, rank=rank, rank_cap=rank_cap)
    undone = a.transform(name) is not None or b.transform(name) is not None
    return diff_values(
        name,
        info_a,
        info_b,
        _rows(a, name),
        _rows(b, name),
        equal_is_unchanged=undone or info_a.dtype != info_b.dtype,
        rank=rank,
        rank_cap=rank_cap,
    )


def _reading_order(a: Side, b: Side) -> Callable[[str], tuple[int, int, int, str]]:
    index_a = {file.path: i for i, file in enumerate(a.ckpt.files)}
    index_b = {file.path: i for i, file in enumerate(b.ckpt.files)}

    def key(name: str) -> tuple[int, int, int, str]:
        stored_a, stored_b = a.located[name].gguf_name, b.located[name].gguf_name
        return (
            index_a[a.ckpt.file_of(stored_a).path],
            index_b[b.ckpt.file_of(stored_b).path],
            a.ckpt.tensors[stored_a].begin,
            name,
        )

    return key


def gguf_about(files: Sequence[GgufFile]) -> dict:
    """The ``gguf`` object of a GGUF side: container facts and the tensor types it holds.

    For a split checkpoint the facts are its first part's, the types count every part's
    tensors, and ``parts`` gives the number of parts (the key is absent for one file).
    """
    first = files[0]
    types = Counter(info.dtype for file in files for info in file.tensors.values())
    about = {
        "version": first.version,
        "architecture": first.architecture,
        "alignment": first.alignment,
        "types": dict(sorted(types.items())),
    }
    return about if len(files) == 1 else {**about, "parts": len(files)}


def _comparison(a: Side, b: Side) -> dict:
    mapping = a.mapping or b.mapping
    return {
        "kind": COMPARISON_KIND,
        "gguf": [label for label, side in (("a", a), ("b", b)) if side.gguf is not None],
        "mapping": None if mapping is None else mapping.name,
        "values": COMPARISON_DTYPE,
    }
