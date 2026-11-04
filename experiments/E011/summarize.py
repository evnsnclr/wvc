"""Summarise an E011 diff record by GGUF type and by projection (E011).

    uv run python experiments/E011/summarize.py DIFF.json

Prints, per ggml type: tensors by status, the range of relative Frobenius error and of the
fraction of elements changed over changed tensors, and the largest top-1 energy; then the
changed F32 tensors by name pattern; then, for a llama file, each layer's q, k, and v relative
error (P6).
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict


def main(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for t in record["tensors"]:
        by_type[t.get("dtype_b") or "-"].append(t)
    print(f"comparison: {record['comparison']}")
    print(f"summary: {record['summary']}")
    for dtype, items in sorted(by_type.items()):
        status = defaultdict(int)
        for t in items:
            status[t["status"]] += 1
        changed = [t for t in items if t["status"] == "changed" and t["numel"]]
        line = f"{dtype}: {dict(status)}"
        if changed:
            rel = [t["rel_delta_norm"] for t in changed if t["rel_delta_norm"] is not None]
            frac = [t["changed"] / t["numel"] for t in changed]
            top1 = [t["top1_energy"] for t in changed if t["top1_energy"] is not None]
            line += f"; rel |d|/|a| {min(rel):.4g} to {max(rel):.4g}"
            line += f"; changed fraction {min(frac):.4f} to {max(frac):.4f}"
            if top1:
                line += f"; top-1 energy max {max(top1):.4f} over {len(top1)} with a rank"
        print(line)
    patterns = defaultdict(list)
    for t in record["tensors"]:
        if t.get("dtype_b") == "F32" and t["status"] == "changed":
            patterns[re.sub(r"\.\d+\.", ".N.", t["name"])].append(t)
    for pattern, items in sorted(patterns.items()):
        worst = max(t["max_abs_delta"] or 0 for t in items)
        count = sum(t["changed"] for t in items)
        print(f"changed F32 {pattern}: {len(items)} tensors, {count} elements, max |d| {worst:.3g}")
    rel = {t["name"]: t for t in record["tensors"]}
    rows = []
    for name, t in rel.items():
        m = re.fullmatch(r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight", name)
        if m and t.get("transform_b"):
            layer = m.group(1)
            k = rel[f"model.layers.{layer}.self_attn.k_proj.weight"]
            v = rel[f"model.layers.{layer}.self_attn.v_proj.weight"]
            rows.append((int(layer), t, k, v))
    for layer, q, k, v in sorted(rows, key=lambda row: row[0]):
        errors = ", ".join(
            f"{label} {t['dtype_b']} {t['rel_delta_norm']:.4g}"
            for label, t in (("q", q), ("k", k), ("v", v))
        )
        print(f"layer {layer}: {errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
