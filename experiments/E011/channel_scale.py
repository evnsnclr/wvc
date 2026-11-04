"""Unplanned check (E011): is the Qwen GGUF's change a per-channel scale folded across norms?

    uv run python experiments/E011/channel_scale.py BASE.safetensors FILE.gguf

The hypothesis tested: for each layer and each input channel j, the GGUF's RMS-norm weight is
the base's divided by some s_j and the columns j of the projections the norm feeds (q, k, v
after input_layernorm; gate and up after post_attention_layernorm) are the base's times s_j,
which leaves the function unchanged. With s_j = base_norm_j / gguf_norm_j, the ratio of
column norms ||gguf[:, j]|| / ||base[:, j]|| should equal s_j up to quantization error. Prints,
per layer and per projection, the median and the 99th percentile over channels of
|log(column ratio / s_j)|; values near 0 support the hypothesis, values near |log s_j| refute
it. Also prints the spread of s_j itself. Reads values through wvc (dequantised exactly).
"""

from __future__ import annotations

import sys

import numpy as np

from wvc.gguf_diff import _rows, _side
from wvc.gguf_io import load_gguf
from wvc.safetensors_io import load_checkpoint

GROUPS = {
    "input_layernorm": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "post_attention_layernorm": ("mlp.gate_proj", "mlp.up_proj"),
}


def values(side, name: str) -> np.ndarray:
    info = side.info(name)
    rows = info.shape[0]
    return _rows(side, name)(0, rows).reshape(info.shape)


def main(base_path: str, gguf_path: str) -> int:
    with load_checkpoint(base_path) as base_ckpt, load_gguf(gguf_path) as gguf_ckpt:
        base, gguf = _side(base_ckpt, False), _side(gguf_ckpt, True)
        names = [n for n in base.located if n.startswith("model.layers.")]
        layers = sorted({int(n.split(".")[2]) for n in names})
        for layer in layers:
            prefix = f"model.layers.{layer}."
            parts = []
            for norm, projections in GROUPS.items():
                n_base = values(base, prefix + norm + ".weight")
                n_gguf = values(gguf, prefix + norm + ".weight")
                scale = n_base / n_gguf
                spread = np.percentile(np.abs(np.log(scale)), [50, 99])
                parts.append(f"{norm} |log s| p50 {spread[0]:.3f} p99 {spread[1]:.3f}")
                for projection in projections:
                    w_base = values(base, prefix + projection + ".weight")
                    w_gguf = values(gguf, prefix + projection + ".weight")
                    ratio = np.linalg.norm(w_gguf, axis=0) / np.linalg.norm(w_base, axis=0)
                    misfit = np.abs(np.log(ratio / scale))
                    p50, p99 = np.percentile(misfit, [50, 99])
                    parts.append(f"{projection.split('.')[1]} misfit p50 {p50:.4f} p99 {p99:.4f}")
            print(f"layer {layer}: " + "; ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
