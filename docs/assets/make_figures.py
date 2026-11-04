"""Draw the README's SVG figures from the experiment receipts.

    uv run python docs/assets/make_figures.py

Writes hero.svg and demo.svg (E008), how-{dark,light}.svg (E001, E003), and
next to this file. Every number drawn is read from a file under
experiments/; nothing here touches weights (make_deltas.py does, for the delta figure).

GitHub serves README images without scripts or external fonts, so the SVGs use system fonts and
CSS animation only. An animated element's static style is its final state, which is also what
readers who ask for reduced motion see.
"""

import json
import math
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiments"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"
THEMES = {
    "dark": {
        "bg": "#0d1117", "panel": "#161b22", "line": "#30363d", "grid": "#21262d",
        "text": "#f0f6fc", "body": "#c9d1d9", "muted": "#9198a1", "faint": "#6e7681",
        "blue": "#58a6ff", "green": "#3fb950", "amber": "#f0883e", "accent": "#ffa657",
        "mid": "#db6d28", "peak": "#ffe1b8",
    },
    "light": {
        "bg": "#ffffff", "panel": "#f6f8fa", "line": "#d0d7de", "grid": "#eaeef2",
        "text": "#1f2328", "body": "#424a53", "muted": "#59636e", "faint": "#818b98",
        "blue": "#0969da", "green": "#1a7f37", "amber": "#e16f24", "accent": "#bc4c00",
        "mid": "#fb8f44", "peak": "#762c00",
    },
}  # fmt: skip


def receipt(path: str) -> dict:
    return json.loads((EXP / path).read_text(encoding="utf-8"))


def tag(name: str, body: str = "", **attrs: object) -> str:
    """One SVG element. Attribute names use _ for -; a trailing _ escapes a keyword."""
    parts = [name]
    for key, value in attrs.items():
        if isinstance(value, float):
            value = f"{value:.2f}".rstrip("0").rstrip(".")
        parts.append(f'{key.rstrip("_").replace("_", "-")}="{escape(str(value))}"')
    return f"<{' '.join(parts)}>{body}</{name}>" if body else f"<{' '.join(parts)}/>"


def text(x: float, y: float, s: str, size: float, fill: str, font: str = SANS, **kw) -> str:
    return tag("text", escape(s), x=x, y=y, fill=fill, font_size=size, font_family=font, **kw)


def texts(rows: list[tuple], c: dict) -> list[str]:
    """Rows of (x, y, string, size, colour key[, font[, extra attributes]])."""
    out = []
    for x, y, s, size, colour, *rest in rows:
        font = rest[0] if rest else SANS
        extra = rest[1] if len(rest) > 1 else {}
        out.append(text(x, y, s, size, c[colour], font, **extra))
    return out


def document(size: tuple[int, int], title: str, desc: str, parts: list[str]) -> str:
    w, h = size
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc">'
    )
    meta = [f'<title id="title">{escape(title)}</title>', f'<desc id="desc">{escape(desc)}</desc>']
    return "\n".join([head, *meta, *parts, "</svg>", ""])


class Timeline:
    """Looping CSS animations on one shared clock."""

    def __init__(self, seconds: float) -> None:
        self.seconds, self.rules = seconds, []

    def add(self, name: str, stops: list[tuple[float, str]], rest: str) -> str:
        frames = " ".join(f"{100 * t / self.seconds:.2f}% {{ {css} }}" for t, css in stops)
        self.rules.append(f"@keyframes {name} {{ {frames} }}")
        self.rules.append(
            f".{name} {{ {rest} animation: {name} {self.seconds}s linear infinite; }}"
        )
        return name

    def show(self, name: str, start: float, end: float, opacity: float = 1, fade: float = 0.25):
        on = f"opacity: {opacity}"
        stops = [(0, "opacity: 0"), (start, "opacity: 0"), (start + fade, on), (end, on)]
        return self.add(name, [*stops, (end + 0.5, "opacity: 0")], f"{on};")

    def slide(self, name: str, start: float, end: float, dx: float) -> str:
        """Move right by dx between start and end, visible only while moving."""
        a, b = "transform: translateX(0px)", f"transform: translateX({dx:.1f}px)"
        stops = [(0, f"opacity: 0; {a}"), (start, f"opacity: 0; {a}"), (start + 0.1, "opacity: 1")]
        stops += [(end - 0.1, "opacity: 1"), (end, f"opacity: 0; {b}"), (self.seconds, b)]
        return self.add(name, stops, "opacity: 0;")

    def style(self) -> str:
        reduce = "@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }"
        return "<style>\n" + "\n".join([*self.rules, reduce]) + "\n</style>"


# Hero: E008's Hub diff, shard by shard ------------------------------------------------------

X0, X1, YA, YB, SH = 860.0, 1530.0, 96.0, 176.0, 40.0  # strips: left, right, tops, height
HOLD = 9.2  # the hero holds its final frame until here, then fades and loops


def hero_text(c: dict, diff: dict) -> list[str]:
    command = f"$ wvc diff --rank hf:{diff['a']['hf']['repo']} hf:{diff['b']['hf']['repo']}"
    big = {"font_weight": 700, "letter_spacing": -8}
    rows = [
        (72, 236, "wvc", 196, "text", SANS, big),
        (80, 306, "Version control for", 44, "body"),
        (80, 360, "open model weights.", 44, "body"),
        (80, 478, "hash · diff · verify · commit · checkout", 21, "faint", MONO),
        (X0, 58, command, 15, "faint", MONO),
        (X0 - 16, YA + 27, "base", 20, "muted", MONO, {"text_anchor": "end"}),
        (X0 - 16, YB + 27, "upload", 20, "muted", MONO, {"text_anchor": "end"}),
    ]
    return texts(rows, c)


def hero_shards(c: dict, tl: Timeline, shards: list[dict], fetched: int, n: int):
    """Both checkpoints' shards, widths in proportion to bytes; also returns the fetched box."""
    gap = 10.0
    scale = (X1 - X0 - gap * (len(shards) - 1)) / sum(f["nbytes"] for f in shards)
    out, x, picked = [], X0, (X0, 0.0)
    for i, f in enumerate(shards):
        w = f["nbytes"] * scale
        t = 0.6 + 2.4 * (x + w / 2 - X0) / (X1 - X0)  # when the beam reaches this shard
        box = {"x": x, "width": w, "height": SH, "rx": 6}
        for y in (YA, YB):
            out.append(tag("rect", y=y, fill=c["panel"], stroke=c["line"], stroke_width=1.5, **box))
        glyph, colour = ("≠", c["blue"]) if i == fetched else ("=", c["green"])
        mark = {"class_": tl.show(f"mark{i}", t, HOLD), "text_anchor": "middle"}
        out.append(text(x + w / 2, YA + SH + 29, glyph, 30, colour, **mark))
        if i != fetched:
            dim = tl.show(f"dim{i}", t, HOLD, opacity=0.62)
            out += [tag("rect", class_=dim, y=y, fill=c["bg"], **box) for y in (YA, YB)]
        else:
            grow = [(0, "transform: scaleX(0)"), (t, "transform: scaleX(0)")]
            grow += [(t + 1.1, "transform: scaleX(1)"), (HOLD, "opacity: 1")]
            grow += [(HOLD + 0.5, "opacity: 0")]
            fill = tl.add("fetch", grow, "transform-box: fill-box; transform-origin: left center;")
            blue = {"class_": fill, "fill": c["blue"], "fill_opacity": 0.32, **box}
            out += [tag("rect", y=y, **blue) for y in (YA, YB)]
            ticks = "".join(f"M{x + w * k / n:.2f} {YB + 6}v{SH - 12}" for k in range(1, n))
            out.append(tag("path", d=ticks, stroke=c["bg"], stroke_opacity=0.5))
            picked = (x, w)
        x += w + gap
    beam = {"class_": tl.slide("beam", 0.6, 3.0, X1 - X0 + 20), "fill": "url(#beam)"}
    out.append(tag("rect", x=X0 - 30, y=YA - 10, width=30, height=YB + SH - YA + 20, **beam))
    return out, picked


def hero_zoom(c: dict, tl: Timeline, picked: tuple, names: list[str], changed: dict) -> list[str]:
    """The fetched shard's tensors in file order (all one size here), then the changed one."""
    sx, sw = picked
    n, hit = len(names), names.index(changed["name"])
    row, rh, pitch = 292.0, 56.0, (X1 - X0) / n
    cell = {"y": row, "width": pitch - 3.2, "height": rh, "rx": 2.5}
    cells = "".join(tag("rect", x=X0 + k * pitch + 1.6, fill=c["grid"], **cell) for k in range(n))
    corners = f"{sx:.1f},{YB + SH} {sx + sw:.1f},{YB + SH} {X1},{row - 8} {X0},{row - 8}"
    blue = {"fill": c["blue"], "fill_opacity": 0.07, "stroke": c["blue"], "stroke_opacity": 0.35}
    t_hit = 3.5 + 1.5 * (hit + 0.5) / n  # the scan reaches the changed tensor
    cx, bx, amber = X0 + (hit + 0.5) * pitch, sx + sw * (hit + 0.5) / n, c["amber"]
    glow = {"fill": c["accent"], "opacity": 0.35, "filter": "url(#blur)", "rx": 4}
    lit = [
        tag("rect", x=X0 + hit * pitch - 2, y=row - 6, width=pitch + 4, height=rh + 12, **glow),
        tag("rect", x=X0 + hit * pitch + 1.6, fill=amber, **cell),
        tag("rect", x=bx - 2.5, y=YB + 2, width=5, height=SH - 4, rx=1.5, fill=amber),
        tag("path", d=f"M{cx:.2f} {row + rh + 8}l-8 12h16z", fill=amber),
    ]
    detail = (
        f"rank {changed['rank']} · {changed['changed']:,} of {changed['numel']:,} weights changed"
    )
    note = text(X0, 404, changed["name"], 21, c["accent"], MONO)
    note += text(X0, 434, detail, 19, c["muted"], MONO)
    scan = {"class_": tl.slide("scan", 3.4, 5.1, X1 - X0), "fill": c["green"], "fill_opacity": 0.35}
    return [
        tag("g", tag("polygon", points=corners, **blue) + cells, class_=tl.show("zoom", 3.0, HOLD)),
        tag("rect", x=X0 - pitch, y=row - 4, width=pitch, height=rh + 8, rx=3, **scan),
        tag("g", "".join(lit), class_=tl.show("hit", t_hit, HOLD)),
        tag("g", note, class_=tl.show("note", t_hit + 0.4, HOLD)),
    ]


def hero() -> str:
    c, tl = THEMES["dark"], Timeline(10.0)
    diff, rec = receipt("E008/diff-rankcap.json"), receipt("E008/receipt.json")
    got = rec["shards"]["a"]["downloaded_shard"]
    shards = diff["a"]["files"]["skipped"] + [{"name": got["name"], "nbytes": got["bytes"]}]
    shards.sort(key=lambda f: f["name"])
    names = got["tensor_names_in_shard"]
    (changed,) = (t for t in diff["tensors"] if t["status"] == "changed")
    fetched = [f["name"] for f in shards].index(got["name"])
    strips, picked = hero_shards(c, tl, shards, fetched, len(names))
    skipped = len(diff["a"]["files"]["skipped"])
    tail = f"{rec['summary']['tensors']} tensors compared · {skipped} of {len(shards)} shards "
    tail += "never downloaded"
    defs = (
        '<defs><pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">'
        '<path d="M40 0H0V40" fill="none" stroke="#151b23"/></pattern>'
        f'<linearGradient id="beam" x1="0" x2="1"><stop offset="0" stop-color="{c["blue"]}" '
        f'stop-opacity="0"/><stop offset="0.7" stop-color="{c["blue"]}" stop-opacity="0.55"/>'
        '<stop offset="1" stop-color="#cae8ff" stop-opacity="0.9"/></linearGradient>'
        '<filter id="blur" x="-1" y="-1" width="3" height="3"><feGaussianBlur stdDeviation="6"/>'
        "</filter></defs>"
    )
    parts = [tag("rect", width=1600, height=500, rx=24, fill=f) for f in (c["bg"], "url(#grid)")]
    parts += hero_text(c, diff) + strips + hero_zoom(c, tl, picked, names, changed)
    parts.append(text(X0, 478, tail, 19, c["faint"], MONO, class_=tl.show("tail", 5.2, HOLD)))
    desc = (
        f"An animation of a real diff: two Qwen3.5-9B checkpoints on the Hugging Face Hub, "
        f"{len(shards)} shards each. {skipped} shards have the same SHA-256 on both sides and are "
        f"skipped; the one that differs is fetched; of its {len(names)} tensors one, "
        f"{changed['name']}, changed, with rank {changed['rank']}."
    )
    title = "wvc: version control for open model weights"
    return document((1600, 500), title, desc, [defs, tl.style(), *parts])


# Terminal demo: the E008 command typed, then its recorded output -------------------------------

FS, ADV = 16.5, 0.6 * 16.5  # font size and nominal monospace advance (cover and cursors only)


def term(x: float, y: float, s: str, fill: str, **kw) -> str:
    body = escape(s).replace(" ", " ")  # non-breaking spaces keep the table's columns
    return tag("text", body, x=x, y=y, fill=fill, font_size=FS, font_family=MONO, **kw)


def window(c: dict, w: int, h: int) -> list[str]:
    lights = ("#ff5f57", "#febc2e", "#28c840")
    bar = f"M1 52V15a14 14 0 0 1 14-14h{w - 30}a14 14 0 0 1 14 14v37z"
    return [
        tag("rect", x=1, y=1, width=w - 2, height=h - 2, rx=14, fill=c["bg"], stroke=c["line"]),
        tag("path", d=bar, fill=c["panel"]),
        tag("path", d=f"M1 52H{w - 1}", stroke=c["line"]),
        *(tag("circle", cx=32 + 26 * i, cy=27, r=8, fill=f) for i, f in enumerate(lights)),
        text(w / 2, 33, "wvc — zsh", 17, c["faint"], text_anchor="middle"),
    ]


def typing(c: dict, tl: Timeline, x: float, y: float, command: str) -> list[str]:
    """The command appears one character at a time: a cover slides off it, a cursor leads."""
    start, end = 0.6, 3.4
    move = f"transform: translateX({len(command) * ADV:.1f}px)"
    steps = f"animation-timing-function: steps({len(command)}, end)"
    # The cover goes once the command is typed, so a font wider than ADV never hides its end.
    cover = [
        (0, "opacity: 1; transform: translateX(0px)"),
        (start, steps),
        (end, f"opacity: 1; {move}"),
    ]
    cover += [(end + 0.05, f"opacity: 0; {move}"), (tl.seconds, f"opacity: 0; {move}")]
    caret = [(0, "opacity: 1; transform: translateX(0px)"), (start, steps)]
    caret += [(end, f"opacity: 1; {move}"), (end + 0.3, "opacity: 1"), (end + 0.31, "opacity: 0")]
    caret += [(tl.seconds, f"opacity: 0; {move}")]
    box = {"x": x, "y": y - FS + 1, "width": ADV, "height": FS + 4}
    return [
        term(x, y, command, c["text"]),
        tag("rect", class_=tl.add("cover", cover, f"opacity: 0; {move};"), x=x - 2, y=y - FS - 2,
            width=1600 - x - 2, height=FS + 10, fill=c["bg"]),
        tag("rect", class_=tl.add("caret", caret, "opacity: 0;"), fill=c["body"], **box),
    ]  # fmt: skip


def demo() -> str:
    lines = (EXP / "E008" / "diff-default-cap.txt").read_text(encoding="utf-8").splitlines()
    diff = receipt("E008/diff-rankcap.json")
    command = f"wvc diff --rank hf:{diff['a']['hf']['repo']} hf:{diff['b']['hf']['repo']}"
    c, tl = THEMES["dark"], Timeline(14.0)
    w, pad, lh, top = 1600, 40, 31, 96
    h, cmd_x = top + lh * (len(lines) + 1) + 26, pad + 2 * ADV
    out = [*window(c, w, h), term(pad, top, "$", c["green"]), *typing(c, tl, cmd_x, top, command)]
    at = [4.6, 4.95, 5.4, 5.55, 5.7, 5.85, 6.0]  # when each output line appears
    for i, (line, t) in enumerate(zip(lines, at, strict=True)):
        y = top + lh * (i + 1)
        if line.startswith("changed  "):
            amber = {"fill": c["amber"], "fill_opacity": 0.16, "stroke": c["amber"]}
            mark = {"class_": tl.show("mark", t + 0.6, 13.2), "rx": 6, **amber}
            out.append(
                tag(
                    "rect", x=pad - 10, y=y - FS - 4, width=w - 2 * pad + 20, height=FS + 13, **mark
                )
            )
        if line:
            out.append(term(pad, y, line, c["body"], class_=tl.show(f"l{i}", t, 13.2, fade=0.05)))
    y, t = top + lh * (len(lines) + 1), at[-1] + 0.9
    blink = [(0, "opacity: 0"), (t - 0.01, "opacity: 0")]
    blink += [(t + 0.5 * k + d, f"opacity: {1 - k % 2}") for k in range(12) for d in (0, 0.49)]
    blink += [(t + 6, "opacity: 0")]
    box = {"x": cmd_x, "y": y - FS + 1, "width": ADV, "height": FS + 4, "fill": c["body"]}
    out += [
        term(pad, y, "$", c["green"], class_=tl.show("prompt", t, 13.2, fade=0.05)),
        tag("rect", class_=tl.add("blink", blink, "opacity: 1;"), **box),
    ]
    desc = f"A terminal runs: {command}. It prints: " + " / ".join(x for x in lines if x)
    title = "wvc diff on two Hugging Face Hub repositories"
    return document((w, h), title, desc, [tl.style(), *out])


# How it works: hash, diff, commit and checkout (E001, E003) ----------------------------------


def arrow(x: float, y0: float, y1: float, colour: str) -> str:
    d = f"M{x} {y0}V{y1 - 9}M{x - 6} {y1 - 12}l6 10 6-10"
    return tag("path", d=d, stroke=colour, stroke_width=2, fill="none", stroke_linecap="round")


def short(name: str) -> str:
    return name.removeprefix("model.language_model.")


def how_hash(c: dict, base: dict, names: list[str]) -> list[str]:
    x, sx, out = 32, 32, []
    for k, w in enumerate([58, 64, 120, 52, 120, 42]):  # a header, then tensors of varied size
        fill = c["grid"] if k == 0 else c["bg"]
        out.append(
            tag("rect", x=sx, y=150, width=w - 3, height=40, rx=5, fill=fill, stroke=c["line"])
        )
        sx += w
    root = {"fill": c["amber"], "fill_opacity": 0.1, "stroke": c["amber"], "stroke_opacity": 0.6}
    out += [
        tag("rect", x=x, y=254, width=456, height=150, rx=8, fill=c["bg"], stroke=c["line"]),
        tag("rect", x=x, y=460, width=456, height=44, rx=22, **root),
        arrow(x + 120, 198, 246, c["faint"]),
        arrow(x + 120, 412, 452, c["faint"]),
    ]
    rows = [
        (x, 136, "model.safetensors", 16.5, "muted", MONO),
        (x + 27, 175, "hdr", 14, "faint", MONO, {"text_anchor": "middle"}),
        (x + 138, 228, "SHA-256 of each tensor's bytes", 17, "muted"),
        (x + 138, 437, "SHA-256 of the sorted lines", 17, "muted"),
        (x + 22, 489, f"root  {base['root'][:28]}…", 18, "accent", MONO),
        (x, 552, "Sharding, header order and file names don't", 18, "muted"),
        (x, 578, "change the root. Bytes, dtype and shape do.", 18, "muted"),
    ]
    for k, name in enumerate(names):
        line = f"{short(name):<32}  {base['tensors'][name]['sha256'][:8]}…"
        rows.append((x + 16, 290 + 30 * k, line, 16.5, "body", MONO, {"style": "white-space: pre"}))
    return out + texts(rows, c)


def how_diff(c: dict, base: dict, edit: dict, names: list[str], t15: dict) -> list[str]:
    x, out = 572, []
    rows = [
        (x, 136, "a  base", 15.5, "faint", MONO),
        (x + 180, 136, "b  edited", 15.5, "faint", MONO),
    ]
    for k, name in enumerate(names):
        y, a, b = (
            172 + 58 * k,
            base["tensors"][name]["sha256"][:8],
            edit["tensors"][name]["sha256"][:8],
        )
        same = a == b
        if not same:
            box = {"fill": c["amber"], "fill_opacity": 0.1, "rx": 8}
            out.append(tag("rect", x=x - 12, y=y - 22, width=300, height=58, **box))
        tone = "body" if same else "accent"
        middle = {"text_anchor": "middle"}
        rows += [
            (x, y, short(name), 14.5, "faint", MONO),
            (x, y + 26, a, 18, tone, MONO),
            (
                x + 135,
                y + 27,
                "=" if same else "≠",
                22,
                "green" if same else "accent",
                SANS,
                middle,
            ),
            (x + 180, y + 26, b, 18, tone, MONO),
        ]
    mx, my, mw, mh = x + 318, 214, 138, 92  # a schematic of b - a; the delta figure has the data
    stripes = [0.02, 0.05, 0.09, 0.11, 0.16, 0.21, 0.27, 0.34, 0.46, 0.58, 0.77, 0.84]
    stripe = {"y": my + 4, "width": 2.2, "height": mh - 8, "fill": c["amber"]}
    pointer = f"M{x + 290} 244H{mx - 10}M{mx - 18} 238l8 6-8 6"
    out += [
        tag("path", d=pointer, stroke=c["accent"], stroke_width=2, fill="none"),
        tag("rect", x=mx, y=my, width=mw, height=mh, rx=4, fill=c["bg"], stroke=c["line"]),
        *(tag("rect", x=mx + 4 + f * (mw - 8), **stripe) for f in stripes),
    ]
    cols = f"{t15['cols_touched']} of {t15['shape_a'][1]:,} columns, {t15['rows_touched']:,} rows"
    rows += [
        (mx + mw / 2, my + mh + 24, "b − a", 15, "muted", MONO, {"text_anchor": "middle"}),
        (x, 448, f"{t15['changed']:,} weights changed", 19, "body"),
        (x, 476, cols, 19, "body"),
        (x, 504, f"rank {t15['rank']}, max |Δ| {t15['max_abs_delta']:.3f}", 19, "body"),
        (x, 552, "Equal ids are never decoded. Hub shards with", 18, "muted"),
        (x, 578, "the same SHA-256 are never downloaded.", 18, "muted"),
    ]
    return out + texts(rows, c)


def how_store(c: dict, base: dict, e3: dict) -> list[str]:
    x, first, second = 1112, e3["commits"]["first"], e3["commits"]["second"]
    stored = e3["store_bytes_after_first_commit"] / 1e9
    delta_kb = 513_248 / 1e3  # payload bytes of the second commit (E003, Outcome table)
    n = base["tensor_count"]
    out = [
        tag("path", d=f"M{x + 10} 158V266", stroke=c["line"], stroke_width=3),
        tag("circle", cx=x + 10, cy=150, r=9, fill=c["faint"]),
        tag("circle", cx=x + 10, cy=266, r=9, fill=c["amber"]),
        arrow(x + 10, 284, 340, c["faint"]),
        tag("rect", x=x, y=346, width=456, height=118, rx=8, fill=c["bg"], stroke=c["line"]),
    ]
    rows = [
        (x + 34, 148, first[:12], 17, "body", MONO),
        (x + 172, 148, "pretrain-checkpoint", 17, "muted"),
        (x + 34, 176, f"{n} tensors · {stored:.3f} GB stored", 18, "muted"),
        (x + 34, 264, second[:12], 17, "body", MONO),
        (x + 172, 264, "direct-edit", 17, "muted"),
        (x + 34, 292, f"{n - 2} reused by id · 2 deltas, {delta_kb:.0f} KB", 18, "accent"),
        (x + 18, 378, "$ wvc checkout HEAD --out model/", 16.5, "body", MONO),
        (x + 18, 410, "model.safetensors  1,746,943,032 bytes", 16.5, "muted", MONO),
        (x + 18, 442, "sha256 fd1e42ac…  = the Hub file  ✓", 16.5, "green", MONO),
        (x, 552, "Every file is checked before it is renamed into", 18, "muted"),
        (x, 578, "place. Both revisions came back byte-identical.", 18, "muted"),
    ]
    return out + texts(rows, c)


def how(theme: str) -> str:
    c = THEMES[theme]
    base, edit = receipt("E001/manifest-base.json"), receipt("E001/manifest-edited.json")
    t15, e3 = receipt("E001/diff.json")["tensors"][0], receipt("E003/receipt.json")
    parts = ("input_layernorm", "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj")
    names = [f"model.language_model.layers.15.{p}.weight" for p in parts]
    heads = [
        ("wvc hash", "Every tensor gets an id"),
        ("wvc diff", "Decode only what differs"),
        ("wvc commit · checkout", "Store only what changed"),
    ]
    panel = {"y": 1, "width": 518, "height": 598, "rx": 16, "fill": c["panel"], "stroke": c["line"]}
    out = []
    for i, (kicker, title) in enumerate(heads):
        out.append(tag("rect", x=540 * i + 1, stroke_width=1.5, **panel))
        out.append(text(540 * i + 32, 54, kicker, 18, c["accent"], MONO))
        out.append(text(540 * i + 32, 90, title, 27, c["text"], font_weight=600))
    out += how_hash(c, base, names) + how_diff(c, base, edit, names, t15) + how_store(c, base, e3)
    desc = (
        "How wvc works, with values from the receipts. wvc hash: each tensor's bytes "
        "get a SHA-256, and the sorted lines get a root. wvc diff: equal ids are skipped; of four "
        f"layer-15 tensors only mlp.down_proj differs, {t15['changed']:,} weights in "
        f"{t15['cols_touched']} columns, rank {t15['rank']}. wvc commit and checkout: the edited "
        "revision reuses 486 tensors by id and stores 2 deltas of 513 KB; checkout reproduces the "
        "Hub file byte for byte."
    )
    return document((1600, 600), "How wvc works", desc, out)


# Storage: E002's four revision pairs -------------------------------------------------------

PAIRS = [
    ("P3", "Qwen3.5-0.8B → weight-space edit", "sparse rank-1 edit (the LLM-written edit)"),
    ("P4", "Qwen2.5-0.5B + rank-8 merge", "synthetic adapter merge"),
    ("P1", "pythia-160m, step 142k → 143k", "consecutive pretraining checkpoints"),
    ("P2", "Qwen2.5-0.5B → Instruct", "dense post-training fine-tune"),
]
LEGEND = [
    ("ring", "compress each tensor (zstd)"),
    ("blue", "byte-chunk dedup, best case"),
    ("amber", "wvc delta"),
]


def percent(r: float) -> str:
    v = 100 * r
    return f"{v:.1f}%" if v >= 10 else f"{v:.2f}%" if v >= 0.1 else f"{v:.3f}%"


def dot(kind: str, x: float, y: float, c: dict) -> str:
    if kind == "ring":
        return tag("circle", cx=x, cy=y, r=10, fill=c["panel"], stroke=c["muted"], stroke_width=3)
    return tag("circle", cx=x, cy=y, r=6.5 if kind == "blue" else 10, fill=c[kind])


def xpos(r: float) -> float:
    """Log axis from 0.01% (x 600) to 100% (x 1460)."""
    return 600 + 860 * (math.log10(r) + 4) / 4


def main() -> None:
    figures = {"hero.svg": hero(), "demo.svg": demo()}
    for theme in THEMES:
        figures[f"how-{theme}.svg"] = how(theme)
    for name, svg in figures.items():
        (HERE / name).write_text(svg, encoding="utf-8", newline="\n")
        print(f"wrote docs/assets/{name} ({len(svg.encode()):,} bytes)")


if __name__ == "__main__":
    main()
