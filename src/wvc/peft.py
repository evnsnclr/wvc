"""Read PEFT LoRA adapters for ``wvc merge-adapter`` (spec section 5.3).

An adapter directory holds ``adapter_config.json`` and ``adapter_model.safetensors``. Only plain
LoRA is read: ``peft_type`` must be ``LORA``; ``use_dora`` or ``fan_in_fan_out`` set to true, a
non-empty ``rank_pattern``, ``alpha_pattern``, or ``modules_to_save``, and a set
``layers_pattern`` or ``layers_to_transform`` are refused. The scale is ``lora_alpha / r``, or
``lora_alpha / sqrt(r)`` with ``use_rslora``, the float PEFT computes. Every tensor of the file
must be one half of a pair ``base_model.model.<path>.lora_A[.<name>].weight`` and
``base_model.model.<path>.lora_B[.<name>].weight``: 2-D F32, F16, or BF16 factors A [r, in] and
B [out, r], which update the base tensor ``<path>.weight``. Refusals raise StoreError.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wvc.fsio import loads_object, read_input
from wvc.hashing import build_manifest
from wvc.lowrank import FLOAT_DTYPES, LowRankFactors
from wvc.records import StoreError
from wvc.safetensors_io import Checkpoint, TensorInfo, load_checkpoint

CONFIG_NAME = "adapter_config.json"
CONFIG_LIMIT = 1 << 20  # bytes of adapter_config.json
WEIGHTS_NAME = "adapter_model.safetensors"
PEFT_TYPE = "LORA"
LORA_KEY = re.compile(
    r"base_model\.model\.(?P<path>.+)\.lora_(?P<half>[AB])(?:\.(?P<name>[^.]+))?\.weight"
)
REFUSED_IF_TRUE = ("use_dora", "fan_in_fan_out")
REFUSED_IF_NON_EMPTY = ("rank_pattern", "alpha_pattern", "modules_to_save")
REFUSED_IF_SET = ("layers_pattern", "layers_to_transform")


@dataclass(frozen=True)
class Adapter:
    """A LoRA adapter with each pair of factors mapped to the base tensor it updates."""

    r: int
    lora_alpha: int | float
    scale: float  # lora_alpha / r, or lora_alpha / sqrt(r) with use_rslora
    target_modules: tuple[str, ...] | str | None  # names (sorted) or a regex, as configured
    targets: dict[str, tuple[TensorInfo, TensorInfo]]  # base tensor name -> (A, B), sorted
    checkpoint: Checkpoint  # the adapter file
    tensors_root: str  # the manifest root of the adapter file

    def factors(self, target: str) -> LowRankFactors:
        """A copy of one target's factors exactly as stored, with the adapter's scale."""
        a, b = self.targets[target]
        return LowRankFactors(
            a_raw=np.array(self.checkpoint.raw(a.name)),
            a_dtype=a.dtype,
            a_shape=a.shape,
            b_raw=np.array(self.checkpoint.raw(b.name)),
            b_dtype=b.dtype,
            b_shape=b.shape,
            scale=self.scale,
        )

    def operation(self) -> dict:
        """The ``adapter`` entry of an ``adapter-merge`` commit's operation (spec section 5.3).

        Floats become their decimal text, since hashed records hold no floats (spec 3.5).
        """
        alpha, modules = self.lora_alpha, self.target_modules
        return {
            "peft_type": PEFT_TYPE,
            "r": self.r,
            "lora_alpha": alpha if isinstance(alpha, int) else repr(alpha),
            "scale": repr(self.scale),
            "target_modules": list(modules) if isinstance(modules, tuple) else modules,
            "tensors_root": self.tensors_root,
        }


def load_adapter(adapter_dir: str | Path) -> Adapter:
    """Read and check a PEFT LoRA adapter directory; StoreError for anything wvc cannot merge."""
    directory = Path(adapter_dir)
    config = _read_config(directory / CONFIG_NAME)
    r, lora_alpha, scale = _lora_settings(config)
    weights = directory / WEIGHTS_NAME
    if not weights.is_file():
        raise StoreError(
            f"{weights} is missing or not a regular file; a PEFT adapter directory holds "
            f"{WEIGHTS_NAME}"
        )
    ckpt = load_checkpoint(weights)
    return Adapter(
        r=r,
        lora_alpha=lora_alpha,
        scale=scale,
        target_modules=_target_modules(config),
        targets=_pair_targets(ckpt, r),
        checkpoint=ckpt,
        tensors_root=build_manifest(ckpt)["root"],
    )


def _read_config(path: Path) -> dict:
    """The adapter config as a JSON object; StoreError when missing, not a regular file of at
    most ``CONFIG_LIMIT`` bytes, or malformed."""
    try:
        data = read_input(path, CONFIG_LIMIT)
    except FileNotFoundError as exc:
        raise StoreError(
            f"{path} is missing; a PEFT adapter directory holds {CONFIG_NAME}"
        ) from exc
    return loads_object(data, str(path))


def _lora_settings(config: dict) -> tuple[int, int | float, float]:
    """``r``, ``lora_alpha``, and the scale, after refusing what wvc cannot merge."""
    _refuse_unsupported(config)
    r, alpha = config.get("r"), config.get("lora_alpha")
    if isinstance(r, bool) or not isinstance(r, int) or r < 1:
        raise StoreError(f"{CONFIG_NAME}: r must be a positive integer, got {r!r}")
    if isinstance(alpha, bool) or not isinstance(alpha, int | float) or not _is_finite(alpha):
        raise StoreError(f"{CONFIG_NAME}: lora_alpha must be a finite number, got {alpha!r}")
    scale = alpha / math.sqrt(r) if _flag(config, "use_rslora") else alpha / r
    return r, alpha, scale


def _refuse_unsupported(config: dict) -> None:
    """StoreError for a PEFT type or option whose merge is not ``W + scale * B @ A``."""
    if config.get("peft_type") != PEFT_TYPE:
        raise StoreError(
            f"{CONFIG_NAME}: peft_type {config.get('peft_type')!r} is not supported; "
            f"wvc merges {PEFT_TYPE} adapters only"
        )
    refused = [key for key in REFUSED_IF_TRUE if _flag(config, key)]
    refused += [key for key in REFUSED_IF_NON_EMPTY if config.get(key)]
    refused += [key for key in REFUSED_IF_SET if config.get(key) not in (None, "", [])]
    if refused:
        key = refused[0]
        raise StoreError(
            f"{CONFIG_NAME}: {key} is {config[key]!r}; wvc merges only plain LoRA adapters "
            "(spec section 5.3)"
        )


def _flag(config: dict, key: str) -> bool:
    """A boolean option; absent or null counts as false, anything but a boolean is refused."""
    value = config.get(key)
    if value is not None and not isinstance(value, bool):
        raise StoreError(f"{CONFIG_NAME}: {key} must be true or false, got {value!r}")
    return bool(value)


def _is_finite(value: int | float) -> bool:
    """Whether a number is finite; an int beyond the float range counts as infinite."""
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _target_modules(config: dict) -> tuple[str, ...] | str | None:
    """``target_modules`` as a sorted tuple of names, a pattern string, or None."""
    modules = config.get("target_modules")
    if modules is None or isinstance(modules, str):
        return modules
    if isinstance(modules, list) and all(isinstance(module, str) for module in modules):
        return tuple(sorted(modules))
    raise StoreError(
        f"{CONFIG_NAME}: target_modules must be a list of names or a pattern, got {modules!r}"
    )


def _pair_targets(ckpt: Checkpoint, r: int) -> dict[str, tuple[TensorInfo, TensorInfo]]:
    """Map each A and B pair to its base tensor name; every tensor must belong to a pair."""
    halves: dict[tuple[str, str], dict[str, TensorInfo]] = {}
    for name, info in ckpt.tensors.items():
        match = LORA_KEY.fullmatch(name)
        if match is None:
            raise StoreError(
                f"{WEIGHTS_NAME}: {name!r} is not a LoRA factor named "
                "base_model.model.<path>.lora_A[.<name>].weight or lora_B; wvc merges adapter "
                "files holding only LoRA A and B weights"
            )
        halves.setdefault((match["path"], match["name"] or ""), {})[match["half"]] = info
    targets: dict[str, tuple[TensorInfo, TensorInfo]] = {}
    for (path, _), pair in sorted(halves.items()):
        target = f"{path}.weight"
        if target in targets:
            raise StoreError(f"{WEIGHTS_NAME}: two adapters update {target}")
        targets[target] = _check_pair(path, pair, r)
    if not targets:
        raise StoreError(f"{WEIGHTS_NAME} holds no LoRA weights")
    return dict(sorted(targets.items()))


def _check_pair(path: str, pair: dict[str, TensorInfo], r: int) -> tuple[TensorInfo, TensorInfo]:
    """Both factors of one module, 2-D F32, F16, or BF16, shaped [r, in] and [out, r]."""
    for half, other in (("A", "B"), ("B", "A")):
        if half not in pair:
            raise StoreError(f"{WEIGHTS_NAME}: {path} has lora_{other} but no lora_{half}")
    a, b = pair["A"], pair["B"]
    for info in (a, b):
        if info.dtype not in FLOAT_DTYPES or len(info.shape) != 2:
            raise StoreError(
                f"{WEIGHTS_NAME}: {info.name} is {info.dtype}{list(info.shape)}; LoRA factors "
                f"must be 2-D {', '.join(FLOAT_DTYPES)}"
            )
    if a.shape[0] != r or b.shape[1] != r:
        raise StoreError(
            f"{WEIGHTS_NAME}: {path} has lora_A {list(a.shape)} and lora_B {list(b.shape)}, "
            f"which are not of rank r = {r}"
        )
    return a, b
