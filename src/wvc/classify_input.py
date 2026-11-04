"""The classifier's input: a diff JSON record, validated, and the view the rules read.

A record comes from a file someone else may have written (a census receipt), so every value
is checked before use and an error names the key, never the content. The view sorts the
record's tensors into the groups the rules of docs/classification.md speak of.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import wvc.store
from wvc.errors import WvcError
from wvc.features import MANTISSA_BITS, widens
from wvc.fsio import read_input
from wvc.quote import shown
from wvc.tensor_names import (
    has_quant_suffix,
    is_embedding,
    is_output_side,
    layer_of,
    stack_of,
)

STATUSES = ("unchanged", "changed", "added", "removed", "reshaped")
INT_FIELDS = ("numel", "changed", "nonfinite_changed", "rows_touched", "cols_touched", "rank")
FLOAT_FIELDS = ("max_abs_delta", "delta_norm", "rel_delta_norm", "top1_energy")
FEATURE_INTS = ("signal_rank", "energy_rank")
FEATURE_FLOATS = ("signal_top1", "noise_edge", "snr", "direction_alignment")
MAX_DIMS = 64
MAX_SIGMA = 64
SNR_MIN = 4.0  # v2: features count only when the top singular value is 4x the noise edge


class ClassifyError(WvcError):
    """A diff record the classifier cannot read."""


def _fail(where: str, rule: str) -> NoReturn:
    raise ClassifyError(f"{where}: {rule}")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _check_shape(where: str, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, (list, tuple)) or len(value) > MAX_DIMS:
        _fail(where, f"must be null or a list of at most {MAX_DIMS} non-negative integers")
    if not all(map(_is_int, value)):
        _fail(where, f"must be null or a list of at most {MAX_DIMS} non-negative integers")


def _check_optional(where: str, value: object, kind: str) -> None:
    if value is None:
        return
    if kind == "int" and not _is_int(value):
        _fail(where, "must be null or a non-negative integer")
    if kind == "number" and not _is_number(value):
        _fail(where, "must be null or a finite number")
    if kind == "str" and not isinstance(value, str):
        _fail(where, "must be null or a string")


def _check_features(where: str, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        _fail(where, "must be null or an object")
    for key in FEATURE_INTS:
        _check_optional(f"{where}.{key}", value.get(key), "int")
        if value.get(key) is None:
            _fail(f"{where}.{key}", "is required")
    for key in FEATURE_FLOATS:
        _check_optional(f"{where}.{key}", value.get(key), "number")
    sigma = value.get("sigma", [])
    if not isinstance(sigma, list) or len(sigma) > MAX_SIGMA or not all(map(_is_number, sigma)):
        _fail(f"{where}.sigma", f"must be a list of at most {MAX_SIGMA} finite numbers")


def _check_quantization(where: str, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        _fail(where, "must be null or an object with a string 'type'")


def _check_tensor(where: str, entry: object) -> None:
    if not isinstance(entry, dict):
        _fail(where, "must be an object")
    if not isinstance(entry.get("name"), str):
        _fail(f"{where}.name", "must be a string")
    if entry.get("status") not in STATUSES:
        _fail(f"{where}.status", f"must be one of {', '.join(STATUSES)}")
    for key in ("dtype_a", "dtype_b", "renamed_to", "sha256"):
        _check_optional(f"{where}.{key}", entry.get(key), "str")
    for key in ("shape_a", "shape_b"):
        _check_shape(f"{where}.{key}", entry.get(key))
    for key in INT_FIELDS:
        _check_optional(f"{where}.{key}", entry.get(key), "int")
    for key in FLOAT_FIELDS:
        _check_optional(f"{where}.{key}", entry.get(key), "number")
    _check_features(f"{where}.features", entry.get("features"))
    _check_quantization(f"{where}.quantization", entry.get("quantization"))


def _check_summary(where: str, summary: object) -> int:
    if not isinstance(summary, dict) or not _is_int(summary.get("tensors")):
        _fail(f"{where}.summary", "must be an object with a non-negative integer 'tensors'")
    by_status = summary.get("by_status")
    if not isinstance(by_status, dict) or not all(_is_int(by_status.get(s)) for s in STATUSES):
        _fail(f"{where}.summary.by_status", f"must give a count for each of {', '.join(STATUSES)}")
    if sum(by_status[s] for s in STATUSES) != summary["tensors"]:
        _fail(f"{where}.summary.by_status", "counts must add up to summary.tensors")
    return summary["tensors"]


def validate_record(data: object, source: str = "diff record") -> dict:
    """Check the structure the classifier reads; return the record unchanged."""
    if not isinstance(data, dict):
        _fail(source, "must be a JSON object")
    total = _check_summary(source, data.get("summary"))
    tensors = data.get("tensors")
    if not isinstance(tensors, list) or len(tensors) > total:
        _fail(f"{source}.tensors", "must be a list of at most summary.tensors entries")
    seen: set[str] = set()
    counts = dict.fromkeys(STATUSES, 0)
    for index, entry in enumerate(tensors):
        _check_tensor(f"{source}.tensors[{index}]", entry)
        if entry["name"] in seen:
            _fail(f"{source}.tensors[{index}].name", f"repeats {shown(entry['name'])}")
        seen.add(entry["name"])
        counts[entry["status"]] += 1
    by_status = data["summary"]["by_status"]
    if any(counts[s] > by_status[s] for s in STATUSES):
        _fail(f"{source}.tensors", "lists more tensors of a status than summary.by_status counts")
    if any(counts[s] != by_status[s] for s in STATUSES if s != "unchanged"):
        _fail(f"{source}.tensors", "must list every tensor that is not unchanged")
    pair = data.get("features")
    if pair is not None:
        if not isinstance(pair, dict):
            _fail(f"{source}.features", "must be null or an object")
        _check_optional(f"{source}.features.hidden_size", pair.get("hidden_size"), "int")
        _check_optional(f"{source}.features.directions", pair.get("directions"), "int")
        _check_optional(
            f"{source}.features.direction_coherence", pair.get("direction_coherence"), "number"
        )
    return data


def load_record(path: str | Path) -> dict:
    """Read and validate a diff JSON record (at most the store's record size limit)."""
    path = Path(path)
    raw = read_input(path, wvc.store.RECORD_LIMIT)
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise ClassifyError(f"{path}: diff record is not valid JSON: {exc}") from exc
    return validate_record(data, source=str(path))


@dataclass(frozen=True)
class Tensor:
    """One tensor of the record, with what its name and statistics say."""

    name: str
    status: str
    layer: int | None
    module: str
    shape: tuple[int, ...] | None
    dtype_a: str | None
    dtype_b: str | None
    entry: dict
    stack: str | None = None

    @property
    def ndim(self) -> int | None:
        return None if self.shape is None else len(self.shape)

    @property
    def features(self) -> dict | None:
        """The measured features, or None when the change is not clear of rounding noise.

        A change wholly below the noise edge has none (signal_rank 0). Rules v2 also set aside
        features whose top singular value is under ``SNR_MIN`` times the edge: there the count
        of components above the edge measures the noise's structure, not the change's (E010,
        F06). The rules then use the record's statistics.
        """
        found = self.entry.get("features")
        if found is None or found.get("signal_rank", 0) <= 0:
            return None
        snr = found.get("snr")
        return None if snr is not None and snr < SNR_MIN else found

    @property
    def output_side(self) -> bool:
        return self.layer is not None and is_output_side(self.module)

    @property
    def embedding(self) -> bool:
        return is_embedding(self.name)


def _cast_kind(entry: dict) -> str | None:
    """How a same-shape dtype change behaves; None when there is no such change.

    ``lossless``: no value changed. ``widening``: the new dtype holds every value of the old
    one (``features.widens``) and values changed, so they changed beyond the cast.
    ``reencoded`` (v2): the new dtype has at least the old one's mantissa bits but another
    range (BF16 to F16), and the change exceeds twice its unit roundoff, so it is a change
    stored in another encoding. ``narrowing``: anything else, judged by R2. Rules v2 read the
    dtypes whatever the status, since a quantization comparison (spec 2.1) reports a
    dequantised tensor as ``changed`` or ``unchanged``.
    """
    dtype_a, dtype_b = entry.get("dtype_a"), entry.get("dtype_b")
    if dtype_a is None or dtype_b is None or dtype_a == dtype_b:
        return None
    if entry.get("shape_a") != entry.get("shape_b"):
        return None
    if entry["status"] == "unchanged" or entry.get("changed") == 0:
        return "lossless"
    if widens(dtype_a, dtype_b):
        return "widening"
    bits_a, bits_b = MANTISSA_BITS.get(dtype_a), MANTISSA_BITS.get(dtype_b)
    rel = entry.get("rel_delta_norm")
    finer = bits_a is not None and bits_b is not None and bits_b >= bits_a
    if finer and isinstance(rel, (int, float)) and rel > 2.0 * 2.0**-bits_b:
        return "reencoded"
    return "narrowing"


def _gguf_sides(record: dict) -> tuple[str, ...]:
    """The sides a quantization comparison (spec 2.1) read from GGUF files."""
    comparison = record.get("comparison")
    if not isinstance(comparison, dict) or not isinstance(comparison.get("gguf"), list):
        return ()
    return tuple(side for side in comparison["gguf"] if side in ("a", "b"))


def _with_quantization(entry: dict, gguf: tuple[str, ...]) -> dict:
    """The entry with the M8 quantization type made explicit, when a GGUF side quantized it."""
    if entry.get("quantization") is not None:
        return entry
    for side in gguf:
        dtype = entry.get(f"dtype_{side}")
        if isinstance(dtype, str) and dtype not in MANTISSA_BITS and dtype not in ("I32", "U8"):
            return {**entry, "quantization": {"type": dtype}}
    return entry


@dataclass
class View:
    """The record's tensors sorted into the groups the rules use."""

    total: int
    full: bool  # the record lists unchanged tensors too (--all)
    layers: int  # the main stack's layer count (a lower bound when not full)
    stack: str | None = None  # the main layer stack
    moved: list[Tensor] = field(default_factory=list)  # changed values at equal or wider dtype
    narrowing: list[Tensor] = field(default_factory=list)
    lossless: list[Tensor] = field(default_factory=list)
    resized: list[Tensor] = field(default_factory=list)
    added: list[Tensor] = field(default_factory=list)
    removed: list[Tensor] = field(default_factory=list)
    marked: list[Tensor] = field(default_factory=list)  # quantized by a GGUF side (M8)
    pair: dict = field(default_factory=dict)
    module_layers: dict[str, set[int]] = field(default_factory=dict)  # full records only
    main_tensors: int = 0  # tensors of the main stack on both sides (full records only)
    main_matrices: int = 0  # of which 2-D
    renamed: int = 0  # tensors paired across a rename (spec 2), placed by their status

    @property
    def both_sides(self) -> int:
        return self.total - len(self.added) - len(self.removed)

    @property
    def has_features(self) -> bool:
        return bool(self.pair) or any(t.features is not None for t in self.moved)


def _tensor(entry: dict, main: str | None) -> Tensor:
    layer, module = layer_of(entry["name"])
    stack = stack_of(entry["name"])
    if stack != main:  # another stack (a vision tower): grouped with the global tensors
        layer, module = None, entry["name"]
    shape = entry.get("shape_a") or entry.get("shape_b")
    return Tensor(
        entry["name"],
        entry["status"],
        layer,
        module,
        None if shape is None else tuple(shape),
        entry.get("dtype_a"),
        entry.get("dtype_b"),
        entry,
        stack,
    )


def _main_stack(entries: list[dict]) -> str | None:
    """The stack holding the most changed tensors, else the most tensors (v2)."""
    changed = Counter(  # added tensors do not count: an adapter shipped beside a merge
        s for e in entries if e["status"] in ("changed", "reshaped") and (s := stack_of(e["name"]))
    )
    every = Counter(s for e in entries if (s := stack_of(e["name"])) is not None)
    counts = changed or every
    return max(counts, key=lambda stack: (counts[stack], every[stack])) if counts else None


def build_view(record: dict) -> View:
    """Group a validated record's tensors (rules v2)."""
    gguf = _gguf_sides(record)
    entries = [_with_quantization(entry, gguf) for entry in record["tensors"]]
    main = _main_stack(entries)
    tensors = [_tensor(entry, main) for entry in entries]
    total = record["summary"]["tensors"]
    indices = [t.layer for t in tensors if t.layer is not None]
    full = len(tensors) == total
    layers = len(set(indices)) if full else (max(indices) + 1 if indices else 0)
    view = View(
        total=total, full=full, layers=layers, stack=main, pair=record.get("features") or {}
    )
    for t in tensors:
        _place(view, t)
        view.renamed += t.entry.get("renamed_to") is not None
        if full and t.layer is not None and t.status not in ("added", "removed"):
            view.module_layers.setdefault(t.module, set()).add(t.layer)
            view.main_tensors += 1
            view.main_matrices += t.ndim == 2
    return view


def _place(view: View, t: Tensor) -> None:
    kind = _cast_kind(t.entry)
    if t.status == "added":
        view.added.append(t)
    elif t.status == "removed":
        view.removed.append(t)
    elif t.status == "reshaped" and t.entry.get("shape_a") != t.entry.get("shape_b"):
        view.resized.append(t)
    elif kind == "lossless":
        view.lossless.append(t)
    elif kind is not None and t.entry.get("quantization") is not None:
        view.marked.append(t)  # a dequantised GGUF tensor: compared as a cast, never as moved
    elif kind in ("widening", "reencoded") or (kind is None and t.status != "unchanged"):
        view.moved.append(t)
    elif kind == "narrowing":
        view.narrowing.append(t)


def quant_named(tensors: list[Tensor]) -> int:
    """How many of ``tensors`` carry a quantization format's companion name."""
    return sum(1 for t in tensors if has_quant_suffix(t.name))
