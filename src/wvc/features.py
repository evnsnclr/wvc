"""The second pass of change classification: spectra measured against the rounding floor.

A merged change is stored as ``round(W + delta) - W``, so the stored delta carries a rounding
error of up to half a unit in the last place of every element. In bfloat16 that error fills
the spectrum and the diff record's rank reads as full for most LoRA merges
(docs/classification.md). This pass decodes each changed matrix again, bounds the rounding
noise's largest singular value from the spacing of the stored values (Bandeira and van Handel,
2016: at most about the largest row norm plus the largest column norm of the entries' standard
deviations), and counts the singular values above it. It also keeps, for each matrix, the unit
singular vector on the hidden-size side, so the classifier can test whether the layers share
one direction.

Singular values come from the eigenvalues of the Gram matrix of the smaller side, which is
accumulated block by block, so a matrix above the cap (an embedding) never sits in memory
whole. Like the record's rank, these numbers come from LAPACK and are not bit-reproducible
across machines.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from wvc.diff import CHUNK_ELEMENTS, DEFAULT_RANK_CAP, TensorDiff, decode_f64
from wvc.safetensors_io import ITEMSIZES, Checkpoint, decodable
from wvc.tensor_names import is_embedding, is_output_side, layer_of

FEATURES_REL_TOL = 1e-2  # the record's tolerance, kept as a floor (structured rounding error)
ENERGY_SHARE = 0.95  # energy_rank: the fewest singular values holding this share of the signal
SIGMA_KEPT = 16  # largest singular values written out per tensor
MAX_GRAM_SIDE = 8192  # a Gram matrix above this side (512 MiB of float64) is not built
MANTISSA_BITS = {  # precision p, counting the implicit bit; u = 2^-p is the unit roundoff
    "F64": 53,
    "F32": 24,
    "F16": 11,
    "BF16": 8,
    "F8_E4M3": 4,
    "F8_E4M3FNUZ": 4,
    "F8_E5M2": 3,
    "F8_E5M2FNUZ": 3,
    "F8_E8M0": 1,
}
FP8 = ("F8_E4M3", "F8_E4M3FNUZ", "F8_E5M2", "F8_E5M2FNUZ", "F8_E8M0")
WIDER = {  # the dtypes that hold every value of a dtype: a cast to one changes no value
    "F32": ("F64",),
    "F16": ("F32", "F64"),
    "BF16": ("F32", "F64"),
    **{fp8: ("F16", "BF16", "F32", "F64") for fp8 in FP8 if fp8 != "F8_E8M0"},
    "F8_E8M0": ("BF16", "F32", "F64"),  # powers of two from 2^-127: beyond F16's range
}


def widens(dtype_a: str | None, dtype_b: str | None) -> bool:
    """Whether every value of ``dtype_a`` is a value of ``dtype_b`` (F16 and BF16: neither)."""
    return dtype_b in WIDER.get(dtype_a or "", ())


@dataclass(frozen=True)
class TensorFeatures:
    signal_rank: int  # 0 when the whole change lies below the noise edge
    energy_rank: int
    signal_top1: float | None  # None when signal_rank is 0
    noise_edge: float
    snr: float | None
    sigma: tuple[float, ...]
    direction: np.ndarray | None  # unit vector on the hidden-size side; not written out
    direction_alignment: float | None = None

    def to_dict(self) -> dict:
        return {
            "signal_rank": self.signal_rank,
            "energy_rank": self.energy_rank,
            "signal_top1": self.signal_top1,
            "noise_edge": self.noise_edge,
            "snr": self.snr,
            "sigma": list(self.sigma),
            "direction_alignment": self.direction_alignment,
        }


def spacing(values: np.ndarray, dtype: str) -> np.ndarray:
    """The gap to the next representable magnitude of ``dtype`` at each value (zeros if unknown)."""
    magnitude = np.abs(values)
    if dtype == "F64":
        return np.spacing(magnitude)
    if dtype == "F32":
        return np.spacing(magnitude.astype(np.float32)).astype(np.float64)
    if dtype == "F16":
        with np.errstate(over="ignore"):
            return np.spacing(magnitude.astype(np.float16)).astype(np.float64)
    if dtype == "BF16":  # float32's spacing scaled by the 16 mantissa bits bfloat16 drops
        return np.spacing(magnitude.astype(np.float32)).astype(np.float64) * 65536.0
    return np.zeros_like(magnitude)


def noise_dtype_of(dtype_a: str, dtype_b: str) -> str:
    """The dtype whose rounding bounds the noise: B's, unless B re-encodes A's values in a
    dtype with other range (BF16 to F16), where a value may have been rounded in A's dtype
    first, so the coarser of the two (v2)."""
    if dtype_a == dtype_b or widens(dtype_a, dtype_b):
        return dtype_b
    bits_a, bits_b = MANTISSA_BITS.get(dtype_a, 0), MANTISSA_BITS.get(dtype_b, 0)
    return dtype_a if bits_a < bits_b else dtype_b


def hidden_size(diffs: Iterable[TensorDiff]) -> int | None:
    """The dimension most 2-D tensors share: the residual stream's width in a transformer."""
    counts: Counter[int] = Counter()
    for item in diffs:
        shape = item.shape_a or item.shape_b
        if shape is not None and len(shape) == 2:
            counts.update(set(shape))
    return counts.most_common(1)[0][0] if counts else None


def wants_features(item: TensorDiff) -> bool:
    """A changed matrix whose values decode, at equal dtype, widened, or re-encoded in a dtype
    with at least as many mantissa bits (BF16 to F16); never a quantization."""
    if item.shape_a is None or len(item.shape_a) != 2 or item.shape_a != item.shape_b:
        return False
    if item.status == "changed":
        return item.dtype_a is not None and decodable(item.dtype_a)
    if item.status != "reshaped" or not item.changed:
        return False
    bits_a, bits_b = MANTISSA_BITS.get(item.dtype_a or ""), MANTISSA_BITS.get(item.dtype_b or "")
    finer = bits_a is not None and bits_b is not None and bits_b >= bits_a  # BF16 to F16 (v2)
    return widens(item.dtype_a, item.dtype_b) or finer


@dataclass
class _Gram:
    gram: np.ndarray
    row_var: np.ndarray
    col_var: np.ndarray
    delta: np.ndarray | None  # the whole delta when it fits under the cap


def _accumulate(raw_a: np.ndarray, raw_b: np.ndarray, item: TensorDiff, cap: int) -> _Gram | None:
    """Decode the delta in row blocks; keep the Gram matrix of the smaller side."""
    if item.shape_a is None or item.dtype_a is None or item.dtype_b is None:
        return None
    rows, cols = item.shape_a
    rows_small = rows <= cols
    side = rows if rows_small else cols
    keep = rows * cols <= cap
    if side > MAX_GRAM_SIDE or (rows_small and not keep):
        return None
    size_a, size_b = ITEMSIZES[item.dtype_a], ITEMSIZES[item.dtype_b]
    noise_dtype = noise_dtype_of(item.dtype_a, item.dtype_b)
    gram = np.zeros((side, side))
    row_var, col_var = np.zeros(rows), np.zeros(cols)
    blocks: list[np.ndarray] = []
    step = max(1, CHUNK_ELEMENTS // max(cols, 1))
    for first in range(0, rows, step):
        last = min(first + step, rows)
        start, stop = first * cols, last * cols
        a = decode_f64(raw_a[start * size_a : stop * size_a], item.dtype_a).reshape(-1, cols)
        b = decode_f64(raw_b[start * size_b : stop * size_b], item.dtype_b).reshape(-1, cols)
        delta = b - a
        if not np.isfinite(delta).all():
            return None
        noise = spacing(b, noise_dtype) ** 2 / 12.0
        row_var[first:last] = noise.sum(axis=1)
        col_var += noise.sum(axis=0)
        if keep:
            blocks.append(delta)
        if not rows_small:
            gram += delta.T @ delta
    whole = np.concatenate(blocks) if keep else None
    if rows_small and whole is not None:
        gram = whole @ whole.T
    return _Gram(gram, row_var, col_var, whole)


def _direction(
    acc: _Gram, vectors: np.ndarray, top: float, shape: tuple[int, ...], hidden: int | None
) -> np.ndarray | None:
    """The top singular vector on the hidden-size side, or None when no side has that size."""
    rows, cols = shape
    small = vectors[:, 0]
    rows_small = rows <= cols
    if hidden is None:
        return None
    if (rows if rows_small else cols) == hidden:
        return small
    if acc.delta is None or top <= 0:
        return None
    other = acc.delta.T @ small if rows_small else acc.delta @ small
    if other.shape[0] != hidden:
        return None
    norm = float(np.linalg.norm(other))
    return other / norm if norm > 0 else None


def tensor_features(
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    item: TensorDiff,
    hidden: int | None,
    cap: int = DEFAULT_RANK_CAP,
) -> TensorFeatures | None:
    """The features of one changed matrix, or None when it cannot be measured."""
    acc = _accumulate(raw_a, raw_b, item, cap)
    if acc is None:
        return None
    values, vectors = np.linalg.eigh(acc.gram)
    order = np.argsort(values)[::-1]
    sigma = np.sqrt(np.clip(values[order], 0.0, None))
    vectors = vectors[:, order]
    if sigma.size == 0 or sigma[0] <= 0:
        return None
    edge = float(np.sqrt(acc.row_var.max()) + np.sqrt(acc.col_var.max()))
    signal = sigma[sigma > max(edge, FEATURES_REL_TOL * sigma[0])]
    if signal.size == 0:  # no singular value above the rounding floor: nothing to measure
        return TensorFeatures(0, 0, None, edge, float(sigma[0] / edge), _kept(sigma), None)
    energy = np.square(signal)
    total = float(energy.sum())
    energy_rank = int(np.searchsorted(np.cumsum(energy), ENERGY_SHARE * total) + 1)
    return TensorFeatures(
        signal_rank=int(signal.size),
        energy_rank=min(energy_rank, int(signal.size)),
        signal_top1=float(energy[0] / total),
        noise_edge=edge,
        snr=float(sigma[0] / edge) if edge > 0 else None,
        sigma=_kept(sigma),
        direction=_direction(acc, vectors, float(sigma[0]), item.shape_a or (0, 0), hidden),
    )


def _kept(sigma: np.ndarray) -> tuple[float, ...]:
    return tuple(float(s) for s in sigma[:SIGMA_KEPT])


def coherence(features: dict[str, TensorFeatures]) -> tuple[float | None, int, dict[str, float]]:
    """How much the output-side directions agree: top eigenvalue of the mean of u uᵀ.

    1 when every tensor's direction is the same up to sign; about 1/n for unrelated ones.
    Returns the coherence (None for fewer than two directions), their count, and each
    tensor's |cos| to the consensus direction.
    """
    names = [
        name
        for name, item in sorted(features.items())
        if item.direction is not None and (is_embedding(name) or is_output_side(layer_of(name)[1]))
    ]
    if len(names) < 2:
        return None, len(names), {}
    stacked = np.stack([features[name].direction for name in names])
    _, singular, rows = np.linalg.svd(stacked, full_matrices=False)
    consensus = rows[0]
    alignment = np.abs(stacked @ consensus)
    value = float(singular[0] ** 2 / len(names))
    return value, len(names), {name: float(a) for name, a in zip(names, alignment, strict=True)}


def compute_features(
    ckpt_a: Checkpoint, ckpt_b: Checkpoint, diffs: list[TensorDiff], cap: int = DEFAULT_RANK_CAP
) -> tuple[dict[str, dict], dict]:
    """Per-tensor features of every changed matrix the two open checkpoints hold, and the pair's.

    ``diffs`` may list tensors the checkpoints do not hold (skipped Hub shards); those are
    unchanged and need no features.
    """
    hidden = hidden_size(diffs)
    measured: dict[str, TensorFeatures] = {}
    for item in diffs:
        if not wants_features(item) or item.name not in ckpt_a.tensors:
            continue
        found = tensor_features(ckpt_a.raw(item.name), ckpt_b.raw(item.name), item, hidden, cap)
        if found is not None:
            measured[item.name] = found
    value, count, alignment = coherence(measured)
    per_tensor = {
        name: {**item.to_dict(), "direction_alignment": alignment.get(name)}
        for name, item in measured.items()
    }
    pair = {"hidden_size": hidden, "direction_coherence": value, "directions": count}
    return per_tensor, pair
