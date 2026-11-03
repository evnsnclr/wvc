"""Draw deltas-{dark,light}.png: every changed tensor of E001 and E008, from the weights.

    uv run --with matplotlib python docs/assets/make_deltas.py

Needs the experiments' inputs in the Hugging Face cache: one 1.7 GB file per side for E001 and
shard 2 of each side for E008 (5.3 GB each); it names what is missing and stops otherwise. The
tensors are read with a minimal header parser of its own, not wvc's, and each one's changed
element, row, and column counts must equal its receipt before anything is drawn.
"""

import json
import struct
from pathlib import Path

import numpy as np
from make_figures import HERE, THEMES, receipt

HUB = Path.home() / ".cache" / "huggingface" / "hub"
PAIRS = [
    ("E001", "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17",
     "RikiyaT/Qwen3.5-0.8B-weight-space-programming@d1671a178705bcab7644806d23c535517f35723b",
     "model.safetensors-00001-of-00001.safetensors", "E001/diff.json"),
    ("E008", "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
     "bowmanslayer/Qwen3.5-9B-Evil-BF16@b66fc0a575c13af75c5d438111559b04ed015cd4",
     "model.safetensors-00002-of-00004.safetensors", "E008/diff-rankcap.json"),
]  # fmt: skip
W, H, DPI = 1600, 940, 200


def cached(spec: str, shard: str) -> Path:
    repo, rev = spec.split("@")
    return HUB / f"models--{repo.replace('/', '--')}" / "snapshots" / rev / shard


def bf16_bits(path: Path, name: str) -> np.ndarray:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        info = json.loads(f.read(n))[name]
    if info["dtype"] != "BF16":
        raise SystemExit(f"{name}: expected BF16, found {info['dtype']}")
    start, end = info["data_offsets"]
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n + start, shape=(end - start,))
    return np.frombuffer(raw, dtype="<u2").reshape(info["shape"])


def changed_tensors() -> list[tuple[str, dict, np.ndarray]]:
    """(experiment, diff record, |edited - base|) for every changed tensor, checked."""
    found = []
    for exp, a, b, shard, record in PAIRS:
        paths = [cached(a, shard), cached(b, shard)]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise SystemExit("not in the Hugging Face cache:\n  " + "\n  ".join(missing))
        for t in (t for t in receipt(record)["tensors"] if t["status"] == "changed"):
            bits = [bf16_bits(p, t["name"]) for p in paths]
            differs = bits[0] != bits[1]
            got = (int(differs.sum()), int(differs.any(1).sum()), int(differs.any(0).sum()))
            if got != (t["changed"], t["rows_touched"], t["cols_touched"]):
                raise SystemExit(f"{exp} {t['name']}: {got} differs from the receipt")
            a32, b32 = ((x.astype(np.uint32) << 16).view(np.float32) for x in bits)
            found.append((exp, t, np.abs(b32 - a32)))
    return found


def pool(d: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Mean over a rows x cols grid of near-equal blocks."""
    r = np.linspace(0, d.shape[0], rows + 1).astype(int)
    q = np.linspace(0, d.shape[1], cols + 1).astype(int)
    s = np.add.reduceat(np.add.reduceat(d.astype(np.float64), r[:-1], 0), q[:-1], 1)
    return s / np.outer(np.diff(r), np.diff(q))


def ramp(p: np.ndarray, c: dict, gamma: float = 0.55) -> np.ndarray:
    """Background (unchanged) through the accent to the peak colour, scaled per panel."""
    rgb = [
        np.array([int(c[k][i : i + 2], 16) / 255 for i in (1, 3, 5)]) for k in ("bg", "mid", "peak")
    ]
    t = (np.clip(p / np.percentile(p[p > 0], 99.5), 0, 1) ** gamma)[..., None]
    low = rgb[0] + np.clip(2 * t, 0, 1) * (rgb[1] - rgb[0])
    return np.where(t <= 0.5, low, rgb[1] + np.clip(2 * t - 1, 0, 1) * (rgb[2] - rgb[1]))


class Figure:
    """Pixel coordinates from the top-left of a W x H canvas, on top of matplotlib."""

    def __init__(self, c: dict) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        fonts = Path.home() / "Library" / "Fonts"
        for f in ("Inter-Regular.otf", "Inter-SemiBold.otf", "JetBrainsMonoNLNerdFont-Regular.ttf"):
            if (fonts / f).exists():
                font_manager.fontManager.addfont(str(fonts / f))
        have = {f.name for f in font_manager.fontManager.ttflist}
        self.sans = "Inter" if "Inter" in have else "DejaVu Sans"
        self.mono = (
            "JetBrainsMonoNL Nerd Font"
            if "JetBrainsMonoNL Nerd Font" in have
            else "DejaVu Sans Mono"
        )
        self.c, self.plt = c, plt
        self.fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=c["bg"])

    def text(self, x, y, s, size, colour, mono=False, **kw) -> None:
        family = self.mono if mono else self.sans
        self.fig.text(x / W, 1 - y / H, s, fontsize=size * 72 / DPI, color=self.c[colour],
                      family=family, va="baseline", **kw)  # fmt: skip

    def image(self, rgb: np.ndarray, x: float, y: float, w: float, h: float, frame=True) -> None:
        ax = self.fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
        ax.imshow(rgb, interpolation="nearest", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(frame)
            spine.set_color(self.c["line"])
            spine.set_linewidth(0.8)

    def heat(self, d: np.ndarray, x: float, y: float, w: int) -> int:
        """The whole matrix at its true aspect ratio, |delta| averaged over each pixel's block."""
        h = round(w * d.shape[0] / d.shape[1])
        self.image(ramp(pool(d, h, w), self.c), x, y, w, h)
        return h

    def save(self, path: Path) -> None:
        from PIL import Image

        self.fig.savefig(path, dpi=DPI, facecolor=self.c["bg"])
        self.plt.close(self.fig)
        image = Image.open(path).convert("RGB").quantize(256, dither=Image.Dither.NONE)
        image.save(path, optimize=True)


def headline(f: Figure, y: float, exp: str, pair: tuple, summary: dict) -> None:
    a, b = (s.split("@")[0] for s in pair[1:3])
    n = summary["by_status"]["changed"]
    f.text(52, y, exp, 30, "text", weight="semibold")
    f.text(146, y, f"{a}  →  {b}", 25, "muted")
    f.text(W - 52, y, f"{n} of {summary['tensors']} tensors changed", 25, "body", ha="right")


def draw(theme: str, found: list[tuple[str, dict, np.ndarray]]) -> Path:
    f = Figure(THEMES[theme])
    headline(f, 62, "E001", PAIRS[0], receipt("E001/diff.json")["summary"])
    headline(f, 470, "E008", PAIRS[1], receipt("E008/diff-rankcap.json")["summary"])
    pw = round((W - 52 - 52 - 40) / 2)
    for i, (_, t, d) in enumerate(x for x in found if x[0] == "E001"):
        x = 52 + i * (pw + 40)
        h = f.heat(d, x, 92, pw)
        cols = f"{t['cols_touched']} of {t['shape_a'][1]:,} columns"
        f.text(x, 92 + h + 38, t["name"].removeprefix("model.language_model."), 23, "accent", True)
        f.text(
            x,
            92 + h + 70,
            f"{t['changed']:,} weights changed · {cols} · rank {t['rank']}",
            22,
            "body",
        )
    (_, t, d) = next(x for x in found if x[0] == "E008")
    h, cx = f.heat(d, 52, 500, 1060), 52 + 1060 + 44
    rows, cols = t["shape_a"]
    touched = (t["rows_touched"], t["cols_touched"]) == (rows, cols)
    lines = [
        f"BF16  [{rows}, {cols}]",
        f"{t['changed']:,} of {t['numel']:,}",
        f"weights changed ({100 * t['changed'] / t['numel']:.0f}%)",
        "every row and column touched" if touched else f"{t['rows_touched']:,} rows touched",
        f"rank {t['rank']} (top-1 energy {t['top1_energy']:.4f})",
    ]
    f.text(cx, 532, t["name"].removeprefix("model.language_model."), 20, "accent", True)
    for k, s in enumerate(lines):
        f.text(cx, 574 + 34 * k, s, 22, "body")
    ly = 500 + h - 34
    f.image(
        ramp(np.linspace(0, 1, 256)[None, :].repeat(2, 0), f.c, gamma=1), cx, ly, 300, 14, False
    )
    f.text(cx, ly - 14, "|edited − base|, per panel", 20, "muted")
    f.text(cx, ly + 44, "unchanged", 19, "muted")
    f.text(cx + 300, ly + 44, "largest", 19, "muted", ha="right")
    path = HERE / f"deltas-{theme}.png"
    f.save(path)
    return path


def main() -> None:
    found = changed_tensors()
    for theme in THEMES:
        path = draw(theme, found)
        print(f"wrote docs/assets/{path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
