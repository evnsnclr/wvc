"""Draw astra-{dark,light}.png: the LLM-written edit of Qwen3.5-0.8B on its own, for the post.

    uv run --with matplotlib python docs/assets/make_astra.py

The two heat maps are the top panels of deltas-{theme}.png, which make_deltas.py drew from the
weights after checking them against the receipt; they are cut out pixel for pixel and enlarged
1.5 times by repeating pixels, so nothing is redrawn or smoothed. The labels are read from
E001's diff record and checked against the authors' manifest, as the receipt page does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from canvas import Canvas
from make_deltas import H as DELTAS_H
from make_deltas import W as DELTAS_W
from make_deltas import ramp
from make_figures import HERE, THEMES, receipt

W, H = 1600, 900
SCALE = 1.5
# make_deltas.draw: two panels side by side under the first headline, 40 px apart.
PANEL_W = round((DELTAS_W - 52 - 52 - 40) / 2)
PANEL_H = round(PANEL_W * 1024 / 3584)
PANEL_X, PANEL_Y = (52, 52 + PANEL_W + 40), 92


def checked() -> tuple[dict, list[dict]]:
    diff = receipt("E001/diff.json")
    authors = {t["name"]: t for t in receipt("E001/authors-edit-manifest.json")["tensors"]}
    changed = [t for t in diff["tensors"] if t["status"] == "changed"]
    if [t["name"] for t in changed] != sorted(authors):
        raise SystemExit("the changed tensors differ from the authors' manifest")
    for t in changed:
        a = authors[t["name"]]
        mine = (t["changed"], t["rows_touched"], t["cols_touched"], t["shape_a"])
        theirs = (a["changed_entries"], a["changed_rows"], a["changed_columns"], a["shape"])
        if mine != theirs or t["shape_a"] != [1024, 3584] or t["rank"] != 1:
            raise SystemExit(f"{t['name']}: {mine} against the authors' {theirs}")
    return diff["summary"], changed


def panels(theme: str) -> list[np.ndarray]:
    from PIL import Image

    source = Image.open(HERE / f"deltas-{theme}.png").convert("RGB")
    if source.size != (DELTAS_W, DELTAS_H):
        raise SystemExit(f"deltas-{theme}.png is {source.size}; run make_deltas.py")
    size = (round(PANEL_W * SCALE), round(PANEL_H * SCALE))
    out = []
    for x in PANEL_X:
        crop = source.crop((x, PANEL_Y, x + PANEL_W, PANEL_Y + PANEL_H))
        out.append(np.asarray(crop.resize(size, Image.Resampling.NEAREST)))
    return out


def draw(theme: str, summary: dict, changed: list[dict]) -> Path:
    f = Canvas(THEMES[theme], W, H)
    n, total = summary["by_status"]["changed"], summary["tensors"]
    f.text(52, 66, f"An LLM agent edited {n} of the {total} tensors in Qwen3.5-0.8B", 36, "text",
           weight="semibold")  # fmt: skip
    f.text(52, 104, "wvc diff of the edit against its base, |edited − base| for every weight. "
           "The counts equal the authors' own manifest.", 22, "muted")  # fmt: skip
    pw, ph = round(PANEL_W * SCALE), round(PANEL_H * SCALE)
    x0 = W - 52 - pw
    for i, (rgb, t) in enumerate(zip(panels(theme), changed, strict=True)):
        y = 140 + i * (ph + 30)
        f.image(rgb, x0, y, pw, ph)
        layer = t["name"].split(".layers.")[1].split(".")[0]
        rows, cols = t["shape_a"]
        f.text(52, y + 40, f"Layer {layer}", 32, "text", weight="semibold")
        f.text(52, y + 78, "mlp.down_proj.weight", 21, "accent", mono=True)
        lines = [
            f"{rows:,} × {cols:,} · {t['dtype_a']}",
            f"{t['changed']:,} weights changed",
            f"{t['cols_touched']} of {cols:,} columns",
            f"rank {t['rank']}",
        ]
        for k, s in enumerate(lines):
            f.text(52, y + 124 + 38 * k, s, 23, "body")
    ly = H - 44
    f.image(ramp(np.linspace(0, 1, 300)[None, :].repeat(14, 0), f.c, gamma=1), x0, ly - 14, 300, 14)
    f.text(x0 - 16, ly - 1, "|edited − base|", 20, "muted", ha="right")
    f.text(x0, ly + 30, "unchanged", 19, "muted")
    f.text(x0 + 300, ly + 30, "largest", 19, "muted", ha="right")
    rest = summary["by_status"]["unchanged"]
    f.text(W - 52, ly + 2, f"The other {rest} tensors are byte-identical to the base.", 23,
           "body", ha="right")  # fmt: skip
    path = HERE / f"astra-{theme}.png"
    f.save(path)
    return path


def main() -> None:
    summary, changed = checked()
    for theme in THEMES:
        path = draw(theme, summary, changed)
        print(f"wrote docs/assets/{path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
