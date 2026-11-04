"""What a tensor's name says about its place in a transformer (docs/classification.md).

Names are matched by convention only: a layer index after ``layers.``, ``layer.``, ``h.``,
``blocks.``, or ``block.``; the module is the rest of the name. Nothing here is trusted beyond
grouping: a name that fits no convention is a global tensor.
"""

from __future__ import annotations

import re

LAYER_PATTERN = re.compile(r"^((?:.*?\.)?(?:layers|layer|h|blocks|block|blk))\.(\d{1,6})\.(.+)$")
OUTPUT_SIDE = (
    "o_proj",
    "down_proj",
    "out_proj",
    "dense_4h_to_h",
    "c_proj",
    "fc2",
    "w2",
    "wo",
    "attention.dense",
    "attn.dense",
    "attn_output",  # GGUF names, when a quantization comparison maps none
    "ffn_down",
)
EMBEDDING_MARKERS = ("embed", "wte", "wpe", "lm_head", "token_embd")
QUANT_SUFFIXES = (
    "qweight",
    "qzeros",
    "scales",
    "g_idx",
    "weight_scale",
    "weight_scale_inv",
    "input_scale",
    "absmax",
    "quant_map",
    "quant_state",
    "SCB",
    "weight_packed",
    "weight_shape",
    "biases",
)


def layer_of(name: str) -> tuple[int | None, str]:
    """The layer index and module of a tensor name; ``(None, name)`` for a global tensor."""
    match = LAYER_PATTERN.match(name)
    if match is None:
        return None, name
    return int(match.group(2)), match.group(3)


def stack_of(name: str) -> str | None:
    """The layer stack a tensor belongs to (``model.language_model.layers``), or None.

    A multimodal checkpoint holds several stacks (a vision tower's ``blocks``, a multi-token
    prediction head's ``layers``); rules v2 count layers in the main stack only.
    """
    match = LAYER_PATTERN.match(name)
    return None if match is None else match.group(1)


def _parts(module: str) -> list[str]:
    parts = module.split(".")
    return parts[:-1] if parts and parts[-1] in ("weight", "bias") else parts


def is_output_side(module: str) -> bool:
    """A projection whose rows are the residual stream (attention output, MLP down)."""
    parts = _parts(module)
    dotted = ".".join(parts)
    return any(
        (marker in parts) if "." not in marker else dotted.endswith(marker)
        for marker in OUTPUT_SIDE
    )


def is_embedding(name: str) -> bool:
    """A global token or position embedding, or the output head."""
    layer, _ = layer_of(name)
    return layer is None and (
        any(marker in name for marker in EMBEDDING_MARKERS) or name == "output.weight"
    )


QUANT_MARKERS = ("quant_state", "bitsandbytes")  # bitsandbytes' per-tensor state names


def has_quant_suffix(name: str) -> bool:
    """A quantization format's companion tensor: scales, zero points, packed state."""
    last = name.rsplit(".", 1)[-1]
    if last in QUANT_SUFFIXES or any(marker in name for marker in QUANT_MARKERS):
        return True
    return any(last.endswith("_" + suffix) for suffix in (*QUANT_SUFFIXES, "scale", "quant_map"))
