"""Change classification from a diff record: which kind of change, on what evidence, by which rule.

The rules, their thresholds, and why they are set where they are, are in
docs/classification.md (rules version ``v2``; what changed from ``v1`` after E010 is listed
there). Each rule checks structural conditions in a
fixed order; the first failed condition is what an ``unknown`` result reports. Rules are tried
in the order R1, R2, R3, R4, R5, R6; then the mixed patterns; then ``unknown``.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from wvc.classify_input import Tensor, View, build_view, quant_named, validate_record
from wvc.features import MANTISSA_BITS

FORMAT = "wvc-classification/1"
RULES_VERSION = "v2"
KINDS = (
    "reupload",
    "quantization",
    "direction-removal",
    "direct-edit",
    "merged-lora",
    "dense-finetune",
    "mixed",
    "unknown",
)
CAST_SHARE = 0.9  # R2: casts among the tensors both sides hold
QUANT_NAME_SHARE = 0.5  # R2b: quantization companion names among B's tensors
LAYER_SHARE_SPARSE = 0.25  # R3, R5: touched layers, at least this share of the layer count
LAYER_SHARE_DENSE = 0.75  # R6
MODULE_CONSISTENCY = 0.8  # R5: touched layers whose changed modules are the common set
LOW_RANK_SHARE = 0.8  # R5: changed projections that pass the low-rank test
LORA_RANK_LIMIT = 64  # low rank (R5) is at most min(64, side/4); high (R6) above min(64, side/2)
EDIT_RANK_LIMIT = 4  # R4
EDIT_MAX_TENSORS = 8  # R4
EDIT_TENSOR_SHARE = 0.05  # R4
LOCAL_SHARE = 0.25  # localized: rows or columns touched, at most this share
RANK1_TOP1_RECORD = 0.75  # rank-1-like from the record's top-1 energy
RANK1_TOP1_FEATURES = 0.75  # rank-1-like from the signal's top-1 energy (v2; 0.9 in v1)
EDIT_TOP1 = 0.9  # R4: low rank from the record's top-1 energy
COHERENCE_SHARED = 0.2  # R3 (v2; 0.5 in v1): the output-side directions share one direction
MIXED_REMAINDER = 0.1  # M2: the rest of the change, as a share of the changed tensors
MAX_LOSSLESS_EXTRA = 2  # R1b: added or removed global tensors a reupload may differ by
GGML_BITS = {"Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5, "Q5_0": 5.5, "Q5_1": 6.0, "Q4_K": 4.5}
GGML_BITS.update({"Q4_0": 4.5, "Q4_1": 5.0, "Q3_K": 3.4375, "Q2_K": 2.625})


@dataclass(frozen=True)
class Classification:
    kind: str
    tier: str  # strong | weak
    rule: str
    reason: str
    evidence: dict = field(default_factory=dict)
    features: bool = False

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "rules": RULES_VERSION,
            "kind": self.kind,
            "tier": self.tier,
            "rule": self.rule,
            "reason": self.reason,
            "features": self.features,
            "evidence": self.evidence,
        }


@dataclass
class _Attempt:
    """The conditions one rule checked, in order, until the first that failed."""

    rule: str
    met: int = 0
    failed: str | None = None

    def need(self, condition: bool, description: str) -> bool:
        if self.failed is None:
            if condition:
                self.met += 1
            else:
                self.failed = description
        return self.failed is None


# --- per-tensor tests -------------------------------------------------------------------------


def _int(t: Tensor, key: str) -> int | None:
    value = t.entry.get(key)
    return value if isinstance(value, int) else None


def _num(t: Tensor, key: str) -> float | None:
    value = t.entry.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def rank1_like(t: Tensor) -> bool:
    if t.features is not None:
        return t.features["signal_top1"] >= RANK1_TOP1_FEATURES
    top1, rank = _num(t, "top1_energy"), _int(t, "rank")
    return rank == 1 or (top1 is not None and top1 >= RANK1_TOP1_RECORD)


def rank_of(t: Tensor) -> int | None:
    """The best rank estimate: the signal rank when measured, else the record's."""
    return t.features["signal_rank"] if t.features is not None else _int(t, "rank")


def _small_side(t: Tensor) -> int:
    return min(t.shape) if t.shape else 0


def low_rank_measured(t: Tensor) -> bool:
    """R5: the signal rank is at most min(64, a quarter of the smaller side)."""
    found = t.features
    limit = min(LORA_RANK_LIMIT, _small_side(t) / 4)
    return found is not None and found["signal_rank"] <= limit


def thin_full_rank(t: Tensor) -> bool:
    """v2: in a matrix whose smaller side is at most 64 (a 16 x 1024 gate), a LoRA of the
    ranks R5 accepts is already full rank, so the low-rank test cannot tell; R5 leaves such
    matrices out of its share."""
    return _small_side(t) <= LORA_RANK_LIMIT


def high_rank(t: Tensor) -> bool | None:
    """R6: the best rank estimate exceeds min(64, half the smaller side); None if unknown."""
    rank = rank_of(t)
    return None if rank is None else rank > min(LORA_RANK_LIMIT, _small_side(t) / 2)


def localized(t: Tensor) -> bool:
    if t.shape is None or t.ndim == 0:
        return False
    if t.ndim == 1:
        changed = _int(t, "changed")
        return changed is not None and changed <= LOCAL_SHARE * t.shape[0]
    rows, cols = _int(t, "rows_touched"), _int(t, "cols_touched")
    nrows = t.shape[0]
    ncols = 1
    for dim in t.shape[1:]:
        ncols *= dim
    return (rows is not None and rows <= LOCAL_SHARE * nrows) or (
        cols is not None and cols <= LOCAL_SHARE * ncols
    )


def edit_low_rank(t: Tensor) -> bool:
    rank, top1 = rank_of(t), _num(t, "top1_energy")
    if rank is not None and rank <= EDIT_RANK_LIMIT:
        return True
    return top1 is not None and top1 >= EDIT_TOP1


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _mode(values: Sequence[int]) -> int | None:
    return Counter(values).most_common(1)[0][0] if values else None


# --- evidence ---------------------------------------------------------------------------------


def _layers_touched(tensors: Sequence[Tensor]) -> list[int]:
    return sorted({t.layer for t in tensors if t.layer is not None})


def _modules(tensors: Sequence[Tensor]) -> dict[str, int]:
    return dict(sorted(Counter(t.module for t in tensors if t.layer is not None).items()))


def _module_coverage(view: View, tensors: Sequence[Tensor]) -> tuple[float, list[str]]:
    """v2: the smallest share, over the changed modules, of the layers holding a module that
    change it; a hybrid stack (linear and full attention layers) holds different modules in
    different layers. Falls back to v1's common-set consistency without ``--all``."""
    if not view.full:
        return _module_consistency(tensors)
    changed: dict[str, set[int]] = {}
    for t in tensors:
        if t.layer is not None:
            changed.setdefault(t.module, set()).add(t.layer)
    if not changed:
        return 0.0, []
    shares = [len(layers) / len(view.module_layers.get(m, layers)) for m, layers in changed.items()]
    return min(shares), sorted(changed)


def _module_consistency(tensors: Sequence[Tensor]) -> tuple[float, list[str]]:
    by_layer: dict[int, set[str]] = {}
    for t in tensors:
        if t.layer is not None:
            by_layer.setdefault(t.layer, set()).add(t.module)
    if not by_layer:
        return 0.0, []
    common, count = Counter(frozenset(s) for s in by_layer.values()).most_common(1)[0]
    return count / len(by_layer), sorted(common)


def _cast_pairs(tensors: Sequence[Tensor]) -> dict[str, int]:
    return dict(sorted(Counter(f"{t.dtype_a}->{t.dtype_b}" for t in tensors).items()))


def _rank_summary(tensors: Sequence[Tensor]) -> dict:
    matrices = [t for t in tensors if t.ndim == 2]
    record = [r for t in matrices if (r := _int(t, "rank")) is not None]
    top1 = [v for t in matrices if (v := _num(t, "top1_energy")) is not None]
    measured = [t.features for t in matrices if t.features is not None]
    summary: dict = {
        "matrices": len(matrices),
        "record_rank_median": _median(record),
        "record_rank_max": max(record) if record else None,
        "record_top1_median": _median(top1),
        "measured": len(measured),
    }
    if measured:
        ranks = [f["signal_rank"] for f in measured]
        summary.update(
            signal_rank_mode=_mode(ranks),
            signal_rank_median=_median(ranks),
            signal_rank_max=max(ranks),
            signal_top1_median=_median([f["signal_top1"] for f in measured]),
        )
    return summary


def common_evidence(view: View) -> dict:
    vectors = [t for t in view.moved if t.ndim == 1]
    evidence = {
        "tensors": view.total,
        "changed": len(view.moved),
        "cast": len(view.narrowing) + len(view.marked),
        "cast_lossless": len(view.lossless),
        "resized": [t.name for t in view.resized],
        "added": len(view.added),
        "removed": len(view.removed),
        "layers": view.layers,
        "layers_known": view.full,
        "stack": view.stack,
        "layers_touched": _layers_touched(view.moved),
        "modules_changed": _modules(view.moved),
        "vectors_changed": len(vectors),
        "embeddings_changed": sorted(t.name for t in view.moved if t.embedding),
        "rank": _rank_summary(view.moved),
    }
    if view.narrowing or view.marked:
        evidence["casts"] = _cast_pairs([*view.narrowing, *view.marked])
    if view.pair:
        evidence["direction_coherence"] = view.pair.get("direction_coherence")
        evidence["directions"] = view.pair.get("directions")
        evidence["hidden_size"] = view.pair.get("hidden_size")
    return evidence


def _result(view: View, kind: str, strong: bool, rule: str, reason: str, **extra) -> Classification:
    tier = "strong" if strong and not view.resized else "weak"
    if view.resized:
        reason += f"; {len(view.resized)} tensor(s) resized"
    return Classification(
        kind, tier, rule, reason, {**common_evidence(view), **extra}, features=view.has_features
    )


# --- the rules --------------------------------------------------------------------------------


def _unit_roundoff(dtype: str | None) -> float | None:
    bits = MANTISSA_BITS.get(dtype or "")
    return None if bits is None else 2.0**-bits


def _marked_bound(t: Tensor) -> float | None:
    quant = t.entry["quantization"]
    bits = quant.get("bits_per_weight") or GGML_BITS.get(quant["type"])
    if isinstance(bits, (int, float)) and not isinstance(bits, bool) and bits > 1:
        return 2.0 * 2.0 ** -(float(bits) - 1)
    return None


def _within_rounding(t: Tensor, marked: bool) -> tuple[bool, float | None, float | None]:
    """Whether a cast's deviation is what rounding to the narrower type costs (rel <= 2u)."""
    rel = _num(t, "rel_delta_norm")
    bound = _marked_bound(t) if marked else None
    if bound is None:
        u = _unit_roundoff(t.dtype_b)
        bound = None if u is None else 2.0 * u
    if bound is None:
        return False, rel, None
    exact = rel is None and _int(t, "changed") == 0
    return exact or (rel is not None and rel <= bound), rel, bound


def rule_reupload(view: View, attempt: _Attempt) -> Classification | None:
    extra = len(view.added) + len(view.removed)
    if not (
        attempt.need(not view.moved, "no tensor may change value")
        and attempt.need(not view.narrowing and not view.marked, "no tensor may be quantized")
        and attempt.need(not view.resized, "no tensor may be resized")
        and attempt.need(
            extra <= MAX_LOSSLESS_EXTRA
            and all(t.layer is None for t in [*view.added, *view.removed]),
            f"at most {MAX_LOSSLESS_EXTRA} global tensors may be added or removed",
        )
    ):
        return None
    if not extra and not view.lossless:
        return _result(view, "reupload", True, "R1", f"all {view.total} tensors are byte-identical")
    parts = []
    if view.lossless:
        parts.append(f"{len(view.lossless)} stored in another dtype with identical values")
    if extra:
        parts.append(f"{extra} global tensor(s) added or removed")
    reason = "every value is identical; " + " and ".join(parts)
    return _result(view, "reupload", False, "R1b", reason)


def _quant_named_share(view: View) -> float:
    b_count = view.total - len(view.removed)
    return quant_named(view.added) / b_count if b_count else 0.0


def _cast_share(view: View, casts: Sequence[Tensor]) -> float:
    """v2: casts among the main stack's matrices (a quantization leaves norms in floats);
    v1's share over every tensor both sides hold when the record is not full."""
    if view.full and view.main_matrices:
        cast_matrices = sum(1 for t in casts if t.layer is not None and t.ndim == 2)
        return cast_matrices / view.main_matrices
    return len(casts) / view.both_sides if view.both_sides else 0.0


def rule_quantization(view: View, attempt: _Attempt) -> Classification | None:
    casts = [*view.narrowing, *view.marked]
    every_cast = [*casts, *view.lossless]
    share = _cast_share(view, every_cast)
    if _quant_named_share(view) >= QUANT_NAME_SHARE:
        attempt.need(True, "")
        reason = (
            f"{quant_named(view.added)} of {view.total - len(view.removed)} tensors in B carry "
            "quantization companion names (a packed format the diff cannot compare by value)"
        )
        return _result(view, "quantization", False, "R2b", reason)
    fp8 = sum(1 for t in casts if (t.dtype_b or "").startswith("F8_"))
    if casts and fp8 >= 0.5 * len(casts) and quant_named(view.added) > 0:
        attempt.need(True, "")
        reason = f"{fp8} tensors cast to FP8 with added scale tensors (scaled FP8)"
        return _result(view, "quantization", False, "R2c", reason, cast_share=share)
    if not (
        attempt.need(bool(casts), "some cast must change values")
        and attempt.need(share >= CAST_SHARE, f"at least {CAST_SHARE:.0%} of matrices cast")
    ):
        return None
    checks = [(t, *_within_rounding(t, t in view.marked)) for t in casts]
    worst = max(checks, key=lambda c: (c[2] or 0.0) / (c[3] or 1.0))
    evidence = {"cast_share": share, "max_rel_delta": worst[2], "tolerance": worst[3]}
    pairs = ", ".join(_cast_pairs(every_cast))
    head = f"{len(every_cast)} tensors cast ({pairs})"
    vectors = [t for t in view.moved if t.ndim == 1]
    if view.moved:
        reason = (
            f"{head}, but {len(view.moved)} tensor(s) changed beyond a cast, {len(vectors)} of "
            "them norm or bias vectors (a quantization leaves them as they are)"
        )
        return _result(view, "mixed", True, "M3", reason, **evidence)
    if all(ok for _, ok, _, _ in checks):
        reason = (
            f"{head}, {len(casts)} with changed values, all within rounding (largest "
            f"relative change {worst[2] or 0.0:.3g} <= {worst[3]:.3g})"
        )
        return _result(view, "quantization", True, "R2", reason, **evidence)
    reason = (
        f"{head}, but {worst[0].name} deviates by {worst[2] or 0.0:.3g}, more than rounding "
        f"allows ({worst[3] or 0.0:.3g})"
    )
    return _result(view, "mixed", True, "M1", reason, **evidence)


def _output_pattern(view: View) -> tuple[list[Tensor], list[Tensor]]:
    """The rank-1-like output-side changes, and every other changed tensor."""
    pattern = [t for t in view.moved if t.output_side and t.ndim == 2 and rank1_like(t)]
    rest = [t for t in view.moved if t not in pattern and not (t.embedding and t.ndim == 2)]
    return pattern, rest


def _rank1_pattern(view: View) -> tuple[list[Tensor], list[Tensor]]:
    """v2: every rank-1-like matrix change, in any stack (a multi-token prediction head is
    edited too), and everything else changed, embeddings aside; some tools also project the
    direction out of the input side."""
    pattern = [t for t in view.moved if t.ndim == 2 and rank1_like(t)]
    rest = [t for t in view.moved if t not in pattern and not (t.embedding and t.ndim == 2)]
    return pattern, rest


def rule_direction_removal(view: View, attempt: _Attempt) -> Classification | None:
    pattern, rest = _rank1_pattern(view)
    output = [t for t in pattern if t.output_side]
    layers = _layers_touched(output)
    needed = max(3, LAYER_SHARE_SPARSE * view.layers)
    coherence = view.pair.get("direction_coherence") if view.pair else None
    if not (
        attempt.need(bool(output), "rank-1 output-side projections must change")
        and attempt.need(not rest, "every other change must be a rank-1 projection or embedding")
        and attempt.need(len(layers) >= needed, f"at least {needed:g} layers must be touched")
        and attempt.need(
            coherence is None or coherence >= COHERENCE_SHARED,
            f"the layers' directions must be shared (coherence >= {COHERENCE_SHARED})",
        )
    ):
        return None
    modules = ", ".join(_modules(pattern))
    reason = f"rank-1 changes in {modules} across {len(layers)} of {view.layers} layers"
    if coherence is None:
        reason += "; direction sharing not measured (run with --features)"
    else:
        reason += f", along one shared direction (coherence {coherence:.2f})"
    strong = coherence is not None
    evidence = {"projections": sorted(_modules(pattern)), "layers_list": layers}
    return _result(view, "direction-removal", strong, "R3", reason, **evidence)


def rule_direct_edit(view: View, attempt: _Attempt) -> Classification | None:
    moved = view.moved
    limit = min(EDIT_MAX_TENSORS, max(1, int(EDIT_TENSOR_SHARE * view.total)))
    local = [t for t in moved if localized(t)]
    low = [t for t in moved if edit_low_rank(t)]
    if not (
        attempt.need(bool(moved), "some tensor must change")
        and attempt.need(len(moved) <= limit, f"at most {limit} tensors may change")
        and attempt.need(
            not (view.added or view.removed or view.narrowing or view.marked),
            "nothing may be added, removed, or quantized",
        )
        and attempt.need(
            all(t in local or t in low for t in moved),
            "every change must be localized or low rank",
        )
    ):
        return None
    strong = all(t in local and t in low for t in moved)
    shapes = "; ".join(_describe_edit(t) for t in moved)
    reason = f"{len(moved)} of {view.total} tensors changed: {shapes}"
    return _result(view, "direct-edit", strong, "R4", reason)


def _describe_edit(t: Tensor) -> str:
    rows, cols, rank = _int(t, "rows_touched"), _int(t, "cols_touched"), rank_of(t)
    where = f"{rows} rows" if cols is None else f"{rows} rows x {cols} cols"
    return f"{t.name} ({where}, rank {'-' if rank is None else rank})"


def rule_merged_lora(view: View, attempt: _Attempt) -> Classification | None:
    inner = [t for t in view.moved if t.layer is not None and t.ndim == 2]
    others = [t for t in view.moved if t not in inner]
    layers = _layers_touched(inner)
    needed = max(2, LAYER_SHARE_SPARSE * view.layers)
    consistency, common = _module_coverage(view, inner)
    ranks = [f["signal_rank"] for t in inner if (f := t.features) is not None]
    measured = [t for t in inner if t.features is not None and not thin_full_rank(t)]
    low = [t for t in measured if low_rank_measured(t)]
    informative = [t for t in inner if not thin_full_rank(t)]
    has_measure = bool(informative) and len(measured) >= LOW_RANK_SHARE * len(informative)
    if not (
        attempt.need(bool(inner), "projections inside layers must change")
        and attempt.need(
            all(t.embedding and t.ndim == 2 for t in others) and len(others) <= 2,
            "only projections inside layers and at most two embeddings may change",
        )
        and attempt.need(len(layers) >= needed, f"at least {needed:g} layers must be touched")
        and attempt.need(
            consistency >= MODULE_CONSISTENCY,
            f"each changed module must change in at least {MODULE_CONSISTENCY:.0%} of the layers "
            "holding it",
        )
        and attempt.need(
            not has_measure or len(low) >= LOW_RANK_SHARE * len(measured),
            f"at least {LOW_RANK_SHARE:.0%} of measured projections must have signal rank "
            f"at most min({LORA_RANK_LIMIT}, a quarter of the smaller side)",
        )
    ):
        return None
    record_ranks = [r for t in inner if (r := _int(t, "rank")) is not None]
    r = _mode(ranks) if has_measure else _mode(record_ranks)
    evidence = {
        "r": r,
        "r_source": "features" if has_measure else "record (unreliable below the rounding floor)",
        "projections": common,
        "layers_list": layers,
        "module_consistency": consistency,
    }
    shown_r = "unmeasured" if not has_measure else str(r)
    reason = (
        f"changes in the same {len(common)} projection(s) across {len(layers)} of {view.layers} "
        f"layers with norms unchanged; rank r = {shown_r}"
    )
    if not has_measure:
        reason += " (run with --features to measure it)"
    return _result(view, "merged-lora", has_measure, "R5", reason, **evidence)


def rule_dense(view: View, attempt: _Attempt) -> Classification | None:
    layers = _layers_touched(view.moved)
    vector_layers = {t.layer for t in view.moved if t.ndim == 1 and t.layer is not None}
    inner = [t for t in view.moved if t.layer is not None and t.ndim == 2]
    median_rank = _median([r for t in inner if (r := rank_of(t)) is not None])
    main_moved = sum(1 for t in view.moved if t.layer is not None)
    if view.full and view.main_tensors:  # v2: the main stack's tensors, not a vision tower's
        share = main_moved / view.main_tensors
    else:
        share = len(view.moved) / view.both_sides if view.both_sides else 0.0
    vectors = len(vector_layers) >= 0.5 * len(layers) if layers else False
    ranked = [h for t in inner if (h := high_rank(t)) is not None]
    high = bool(ranked) and sum(ranked) >= 0.5 * len(ranked)
    needed = LAYER_SHARE_DENSE * view.layers
    if not (
        attempt.need(
            bool(layers) and len(layers) >= needed,
            f"changes in at least {LAYER_SHARE_DENSE:.0%} of the {view.layers} layers",
        )
        and attempt.need(share >= 0.5, "at least half of the tensors must change")
        and attempt.need(
            vectors or high,
            "norms must change, or half the projections must exceed rank "
            f"min({LORA_RANK_LIMIT}, half the smaller side)",
        )
    ):
        return None
    reason = (
        f"{len(view.moved)} of {view.both_sides} tensors changed across {len(layers)} of "
        f"{view.layers} layers; norms changed in {len(vector_layers)} layers; median rank "
        f"{'-' if median_rank is None else f'{median_rank:g}'}"
    )
    evidence = {"changed_share": share, "vector_layers": len(vector_layers)}
    return _result(view, "dense-finetune", vectors and high, "R6", reason, **evidence)


def rule_mixed(view: View, attempt: _Attempt) -> Classification | None:
    pattern, rest = _output_pattern(view)
    layers = _layers_touched(pattern)
    needed = max(3, LAYER_SHARE_SPARSE * view.layers)
    if not (
        attempt.need(len(layers) >= needed, "a rank-1 output-side pattern across layers")
        and attempt.need(
            len(rest) > MIXED_REMAINDER * len(view.moved), "a remainder beyond the pattern"
        )
    ):
        return None
    reason = (
        f"rank-1 output-side changes across {len(layers)} layers (the direction-removal "
        f"pattern), plus {len(rest)} other changed tensor(s)"
    )
    return _result(view, "mixed", False, "M2", reason, layers_list=layers)


RULES: tuple[tuple[str, Callable[[View, _Attempt], Classification | None]], ...] = (
    ("R1", rule_reupload),
    ("R2", rule_quantization),
    ("R3", rule_direction_removal),
    ("R4", rule_direct_edit),
    ("R5", rule_merged_lora),
    ("R6", rule_dense),
    ("M2", rule_mixed),
)


def classify_view(view: View) -> Classification:
    attempts = []
    for name, rule in RULES:
        attempt = _Attempt(name)
        found = rule(view, attempt)
        if found is not None:
            return found
        attempts.append(attempt)
    closest = max(attempts, key=lambda a: a.met)
    reason = f"no rule fired; the closest, {closest.rule}, failed: {closest.failed}"
    near = {a.rule: a.failed for a in attempts}
    return Classification(
        "unknown",
        "weak",
        "U",
        reason,
        {**common_evidence(view), "failed": near},
        features=view.has_features,
    )


def classify_record(record: object, source: str = "diff record") -> Classification:
    """Classify a diff JSON record (validated here; see classify_input)."""
    return classify_view(build_view(validate_record(record, source)))


def classification_lines(result: Classification) -> list[str]:
    """``kind (tier): reason [rule]``, then the evidence as indented ``key: value`` lines."""
    lines = [f"{result.kind} ({result.tier}): {result.reason} [{result.rule}]"]
    lines.extend(f"  {key}: {value}" for key, value in result.evidence.items())
    return lines
