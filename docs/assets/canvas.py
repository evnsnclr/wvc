"""A pixel canvas on matplotlib for the PNG figures drawn from receipts (census, GGUF).

Coordinates are pixels from the top-left of a w x h canvas; colours are keys of
``make_figures.THEMES``. Fonts are Inter and JetBrains Mono when installed, as in
``make_deltas.py``, and DejaVu otherwise.
"""

from __future__ import annotations

from pathlib import Path

DPI = 200


class Canvas:
    def __init__(self, c: dict, w: int, h: int) -> None:
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
        mono = "JetBrainsMonoNL Nerd Font"
        self.mono = mono if mono in have else "DejaVu Sans Mono"
        self.c, self.w, self.h, self.plt = c, w, h, plt
        self.fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI, facecolor=c["bg"])
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.axis("off")

    def text(self, x: float, y: float, s: str, size: float, colour: str, mono: bool = False,
             **kw: object) -> None:  # fmt: skip
        family = self.mono if mono else self.sans
        self.ax.text(x, y, s, fontsize=size * 72 / DPI, color=self.c[colour], family=family,
                     va="baseline", **kw)  # fmt: skip

    def rect(self, x: float, y: float, w: float, h: float, fill: str | None,
             edge: str | None = None, width: float = 0.0, radius: float = 0.0) -> None:  # fmt: skip
        from matplotlib.patches import FancyBboxPatch

        style = f"round,pad=0,rounding_size={radius}" if radius else "square,pad=0"
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle=style,
                facecolor=self.c[fill] if fill else "none",
                edgecolor=self.c[edge] if edge else "none",
                linewidth=width * 72 / DPI,
            )
        )  # fmt: skip

    def line(self, xs: list[float], ys: list[float], colour: str, width: float = 1.0,
             dash: tuple[float, float] | None = None) -> None:  # fmt: skip
        style = {"linestyle": (0, dash)} if dash else {}
        self.ax.plot(xs, ys, color=self.c[colour], linewidth=width * 72 / DPI, **style)

    def dot(self, x: float, y: float, r: float, fill: str, alpha: float = 1.0,
            edge: str | None = None) -> None:  # fmt: skip
        from matplotlib.patches import Circle

        self.ax.add_patch(
            Circle((x, y), r, facecolor=self.c[fill], alpha=alpha,
                   edgecolor=self.c[edge] if edge else "none", linewidth=0.6)
        )  # fmt: skip

    def save(self, path: Path) -> None:
        from PIL import Image

        self.fig.savefig(path, dpi=DPI, facecolor=self.c["bg"])
        self.plt.close(self.fig)
        image = Image.open(path).convert("RGB")
        if image.size != (self.w, self.h):
            raise SystemExit(f"{path.name}: {image.size}, expected {(self.w, self.h)}")
        image.quantize(256, dither=Image.Dither.NONE).save(path, optimize=True)

    def image(self, rgb: object, x: float, y: float, w: float, h: float) -> None:
        """An RGB array drawn into the box, pixel for pixel when its size equals the box."""
        self.ax.imshow(rgb, extent=(x, x + w, y + h, y), interpolation="nearest", zorder=0)
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(self.h, 0)
