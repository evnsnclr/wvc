"""Draw gguf-{dark,light}.png: two GGUF quantizations of one model, tensor by tensor.

    uv run --with matplotlib python docs/assets/make_gguf.py

Reads two committed diff records against the same base, Qwen2.5-0.5B-Instruct at 7ae5576:
Qwen's own Q4_K_M (experiments/E011/qwen-q4_k_m.diff.json) and Felladrin's Q5_K_M split in three
(experiments/E012/qwen25-q5_k_m-split.diff.json). Before drawing, the counts and ranges shown are
checked against the receipt pages' tables. Nothing is fetched.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from canvas import Canvas
from make_figures import EXP, HERE, THEMES, receipt

W, H = 1600, 820
BASE_REV = "7ae557604adf67be50417f59c2c2f167def9a775"
SIDES = [
    ("Qwen/Qwen2.5-0.5B-Instruct-GGUF, Q4_K_M", "E011/qwen-q4_k_m.diff.json", "amber"),
    ("Felladrin/gguf-Q5_K_M-Qwen2.5-0.5B-Instruct", "E012/qwen25-q5_k_m-split.diff.json", "blue"),
]
GROUPS = [
    ("layer norms", re.compile(r"layers\.\d+\.\w*layernorm\.weight$")),
    ("q, k, v", re.compile(r"self_attn\.[qkv]_proj\.weight$")),
    ("o_proj", re.compile(r"o_proj\.weight$")),
    ("gate_proj", re.compile(r"gate_proj\.weight$")),
    ("up_proj", re.compile(r"up_proj\.weight$")),
    ("down_proj", re.compile(r"down_proj\.weight$")),
    ("embedding", re.compile(r"embed_tokens\.weight$")),
]
LO, HI = -3.0, 1.0  # log10 of |d|/|a| at the bottom and top of the plot
X0, X1, Y0, Y1, ZERO = 190, 1548, 196, 618, 672  # plot box and the "unchanged" lane


def values(record: dict) -> dict[str, list[tuple[int, float]]]:
    """(layer, relative change; 0 when unchanged) for every tensor of each group."""
    out: dict[str, list[tuple[int, float]]] = {g: [] for g, _ in GROUPS}
    for t in record["tensors"]:
        for group, pattern in GROUPS:
            if pattern.search(t["name"]):
                layer = re.search(r"layers\.(\d+)\.", t["name"])
                rel = 0.0 if t["status"] == "unchanged" else t["rel_delta_norm"]
                out[group].append((int(layer.group(1)) if layer else 0, rel))
    return out


def span(rows: list[tuple[int, float]]) -> tuple[float, float]:
    xs = [v for _, v in rows]
    return min(xs), max(xs)


def check() -> list[dict[str, list[tuple[int, float]]]]:
    official, felladrin = (receipt(path) for _, path, _ in SIDES)
    if felladrin["a"]["hf"]["revision"] != BASE_REV or BASE_REV not in official["a"]["path"]:
        raise SystemExit("the two records are not against the same base revision")
    e011 = (EXP / "E011-gguf-reading.md").read_text(encoding="utf-8")
    e012 = (EXP / "E012-gguf-uploads.md").read_text(encoding="utf-8")
    off, fel = values(official), values(felladrin)
    norms_changed = sum(v > 0 for _, v in off["layer norms"])
    biggest = max(t["max_abs_delta"] or 0 for t in official["tensors"] if "layernorm" in t["name"])
    if (norms_changed, len(off["layer norms"]), round(biggest, 2)) != (48, 48, 6.86):
        raise SystemExit("official norms differ from E011 (48 of 48 changed, up to 6.86)")
    if "changed (48 of 48)" not in e011 or "max abs delta 6.86" not in e011:
        raise SystemExit("E011's norm row has changed")
    expected = {"q, k, v": "0.219 to 0.594", "o_proj": "0.0546 to 0.100",
                "gate_proj": "0.125 to 0.797", "up_proj": "0.138 to 6.15",
                "down_proj": "0.206 to 0.785"}  # fmt: skip
    for group, text in expected.items():
        got = span(off[group])
        want = [float(x) for x in text.split(" to ")]
        if not all(math.isclose(a, b, rel_tol=5e-3) for a, b in zip(got, want, strict=True)):
            raise SystemExit(f"official {group}: {got}, E011 says {text}")
        if text not in e011:
            raise SystemExit(f"E011 no longer says {text} for {group}")
    f32 = [t for t in felladrin["tensors"] if t["dtype_b"] == "F32"]
    if len(f32) != 121 or any(t["status"] != "unchanged" for t in f32):
        raise SystemExit("Felladrin: not every one of 121 F32 tensors unchanged")
    if "121 (every F32)" not in e012 or felladrin["summary"]["by_status"]["changed"] != 169:
        raise SystemExit("E012's row for Felladrin has changed")
    rels = [t["rel_delta_norm"] for t in felladrin["tensors"] if t["status"] == "changed"]
    if round(min(rels), 4) < 0.0055 or round(max(rels), 4) > 0.0446:
        raise SystemExit(f"Felladrin relative change {min(rels)} to {max(rels)} outside E012's")
    return [off, fel]


def ypos(v: float) -> float:
    return Y1 - (Y1 - Y0) * (math.log10(v) - LO) / (HI - LO)


def axes(f: Canvas) -> None:
    for e in range(int(LO), int(HI) + 1):
        y = ypos(10.0**e)
        f.line([X0, X1], [y, y], "grid", 1.4)
        label = {-3: "0.001", -2: "0.01", -1: "0.1", 0: "1", 1: "10"}[e]
        f.text(X0 - 18, y + 8, label, 20, "muted", ha="right")
    f.line([X0, X1], [ZERO, ZERO], "grid", 1.4, dash=(4, 4))
    f.text(X0 - 18, ZERO + 8, "unchanged", 20, "muted", ha="right")
    f.text(52, Y0 - 30, "relative change of each tensor, |d| / |base|", 21, "body")


def draw(theme: str, sides: list[dict]) -> Path:
    f = Canvas(THEMES[theme], W, H)
    f.text(52, 74, "Same model, two GGUF quantizations: one changes every layer norm", 36, "text",
           weight="semibold")  # fmt: skip
    f.text(52, 116, "Qwen2.5-0.5B-Instruct against two Q4/Q5 GGUF uploads. One dot per tensor, "
           "values dequantised exactly and compared with the base.", 22, "muted")  # fmt: skip
    axes(f)
    width = (X1 - X0) / len(GROUPS)
    for g, (group, _) in enumerate(GROUPS):
        cx = X0 + width * (g + 0.5)
        f.text(cx, ZERO + 56, group, 22, "body", ha="center")
        for s, (rows, (_, _, colour)) in enumerate(zip(sides, SIDES, strict=True)):
            lane = cx + (-1 if s == 0 else 1) * width * 0.2
            for i, (layer, v) in enumerate(rows[group]):
                jitter = ((layer * 7 + i * 3) % 13 - 6) / 6 * width * 0.12
                y = ypos(v) if v > 0 else ZERO
                f.dot(lane + jitter, y, 6.2, colour, alpha=0.8)
    ly = H - 44
    for k, (label, _, colour) in enumerate(SIDES):
        x = 52 + k * 760
        f.dot(x + 10, ly - 8, 9, colour)
        f.text(x + 30, ly, label, 22, "body")
    norms = sides[0]["layer norms"]
    changed = f"{sum(v > 0 for _, v in norms)} of {len(norms)} layer norms changed"
    f.text(X0 + 14, ypos(max(v for _, v in norms)) - 22, changed, 20, "accent")
    f.text(X0 + 14, ZERO - 18, "every F32 tensor unchanged", 20, "blue")
    path = HERE / f"gguf-{theme}.png"
    f.save(path)
    return path


def main() -> None:
    sides = check()
    for theme in THEMES:
        path = draw(theme, sides)
        print(f"wrote docs/assets/{path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
