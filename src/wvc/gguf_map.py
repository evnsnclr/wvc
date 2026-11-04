"""How a GGUF file's tensors correspond to the Hugging Face checkpoint it was converted from.

llama.cpp's ``convert_hf_to_gguf.py`` renames every tensor (``model.layers.17.mlp.down_proj
.weight`` becomes ``blk.17.ffn_down.weight``), reverses the dimension order (which leaves the
row-major bytes alone), and for some architectures changes values on the way: the llama
family's query and key projections are permuted so that each head's rotary pairs sit next to
each other, and Gemma 3's RMS norm weights get 1.0 added. The tables here undo exactly those
steps, per ``general.architecture``; an architecture without a table is refused by name.

A mapping is data: ``GLOBAL`` and ``BLOCK`` name pairs, the block suffixes whose rows are
permuted (with the head count that permutation uses), and the suffixes whose values are
shifted. ``locate`` turns one GGUF name into its Hub name and the steps that undo the
conversion; a GGUF tensor the table does not name keeps its name, so it is reported as added
or removed rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from wvc.errors import WvcError
from wvc.gguf_io import GgufFile

BLOCK_NAME = re.compile(r"blk\.(0|[1-9][0-9]*)\.(.+)")
HUB_BLOCK = "model.layers.{}.{}"
QUERY_HEADS, KEY_HEADS = "head_count", "head_count_kv"

GLOBAL = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
LLAMA_BLOCK = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
QK_NORM = {
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}
GEMMA3_BLOCK = {
    **{key: value for key, value in LLAMA_BLOCK.items() if not key.startswith("ffn_norm")},
    **QK_NORM,
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
}
PHI3_BLOCK = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_qkv.weight": "self_attn.qkv_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_up.weight": "mlp.gate_up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


@dataclass(frozen=True)
class Mapping:
    """One architecture's correspondence between GGUF and Hub tensors."""

    name: str
    block: dict[str, str]
    # block suffix -> the metadata key suffix of the head count its rows were permuted with
    permuted: dict[str, str] = field(default_factory=dict)
    shift_suffix: str | None = None  # names ending so had ``shift`` added at conversion
    shift: float = 0.0


LLAMA_PERMUTED = {
    "attn_q.weight": QUERY_HEADS,
    "attn_q.bias": QUERY_HEADS,
    "attn_k.weight": KEY_HEADS,
    "attn_k.bias": KEY_HEADS,
}
MAPPINGS = {
    "llama": Mapping("llama", LLAMA_BLOCK, permuted=LLAMA_PERMUTED),
    "qwen2": Mapping("qwen2", LLAMA_BLOCK),
    "qwen3": Mapping("qwen3", {**LLAMA_BLOCK, **QK_NORM}),
    "gemma3": Mapping("gemma3", GEMMA3_BLOCK, shift_suffix="norm.weight", shift=1.0),
    "phi3": Mapping("phi3", PHI3_BLOCK),
}


class MappingError(WvcError):
    """A GGUF file's tensors cannot be matched with a Hub checkpoint's."""


@dataclass(frozen=True)
class Located:
    """A GGUF tensor as seen from the Hub checkpoint: its name there and how to undo the rest.

    ``heads`` is the head count whose rotary permutation must be undone along the first axis
    (None for no permutation); ``shift`` is subtracted from every value.
    """

    name: str
    gguf_name: str
    heads: int | None = None
    shift: float = 0.0

    @property
    def transform(self) -> str | None:
        """The steps that undo the conversion, as the diff record states them."""
        steps = []
        if self.heads is not None:
            steps.append(f"unpermute-rotary(heads={self.heads})")
        if self.shift:
            steps.append(f"subtract({self.shift:g})")
        return ", ".join(steps) or None


def mapping_for(file: GgufFile) -> Mapping:
    architecture = file.architecture
    if architecture is None:
        raise MappingError(f"{file.path}: no general.architecture, so its names cannot be mapped")
    mapping = MAPPINGS.get(architecture)
    if mapping is None:
        raise MappingError(
            f"{file.path}: architecture {architecture!r} has no name mapping in wvc; it maps "
            f"{', '.join(sorted(MAPPINGS))}"
        )
    return mapping


def _head_count(file: GgufFile, architecture: str, suffix: str) -> int:
    key = f"{architecture}.attention.{suffix}"
    value = file.metadata.get(key)
    if value is None and suffix == KEY_HEADS:  # no separate key heads: as many as queries
        return _head_count(file, architecture, QUERY_HEADS)
    if type(value) is not int or value <= 0:
        raise MappingError(f"{file.path}: {key} must be a positive integer to undo the permutation")
    return value


def locate(file: GgufFile, mapping: Mapping, gguf_name: str) -> Located:
    """The Hub name of one GGUF tensor and the steps that undo its conversion."""
    if gguf_name in GLOBAL:
        located = Located(GLOBAL[gguf_name], gguf_name)
    else:
        match = BLOCK_NAME.fullmatch(gguf_name)
        suffix = match.group(2) if match else None
        if match is None or suffix not in mapping.block:
            return Located(gguf_name, gguf_name)  # unknown to the table: kept as it is
        located = Located(HUB_BLOCK.format(match.group(1), mapping.block[suffix]), gguf_name)
        if suffix in mapping.permuted:
            heads = _head_count(file, mapping.name, mapping.permuted[suffix])
            located = Located(located.name, gguf_name, heads=heads)
    if mapping.shift_suffix is not None and gguf_name.endswith(mapping.shift_suffix):
        located = Located(located.name, gguf_name, heads=located.heads, shift=mapping.shift)
    return located


def rotary_rows(rows: int, heads: int, where: str) -> np.ndarray:
    """For each Hub row, the GGUF row that holds it after llama.cpp's q/k permutation.

    The conversion reshapes the rows to ``(heads, 2, rows // heads // 2)`` and swaps the last
    two axes; so Hub row ``h * d + t * d / 2 + j`` (head ``h``, half ``t``, index ``j``) sits
    at GGUF row ``h * d + 2 * j + t``, with ``d = rows // heads``.
    """
    if rows % (2 * heads):
        raise MappingError(f"{where}: {rows} rows do not split into {heads} heads of pairs")
    half = rows // heads // 2
    head, rest = np.divmod(np.arange(rows, dtype=np.int64), 2 * half)
    which, index = np.divmod(rest, half)
    return head * 2 * half + 2 * index + which
