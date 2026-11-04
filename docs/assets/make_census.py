"""Draw census-{dark,light}.png: the 93 Hub uploads of the census, one square each, by category.

    uv run --with matplotlib python docs/assets/make_census.py

Reads the census receipts (experiments/E009/census.csv, then rediff-2026-11-04/census.csv, and
the receipts beside each) with the summarizer's own loading and category rule, then checks the
counts against the committed summary and the receipt page before drawing. Nothing is fetched.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

from canvas import Canvas
from make_figures import EXP, HERE, THEMES

sys.path.insert(0, str(EXP / "E009"))
import summarize

W, H = 1600, 820
CSVS = [EXP / "E009" / "census.csv", EXP / "E009" / "rediff-2026-11-04" / "census.csv"]
# Found after rename pairing (E009 addendum, "What this does to P9"): tagged finetune, yet its
# language model is the base's byte for byte. The other eleven follow the frozen P9 rule.
POST_HOC = "autotrust/JEV-9B"
MERGED_LORAS = [f"JackKozmo29/{n}-jev-style-1.5b" for n in ("codeguard", "depguard", "mailguard",
                "sevclassifier")]  # fmt: skip
TAGS = {"finetune": "blue", "quantized": "amber", "merge": "faint"}
ORDER = [
    "byte-identical",
    "low rank, few projections",
    "renamed, unchanged",
    "renamed, changed",
    "dtype rewrite",
    "partial change",
    "renamed layout",
]


def load() -> list[dict[str, str]]:
    """The diffed pairs as the committed summary sees them (later CSVs supersede earlier)."""
    rows = []
    for i, path in enumerate(CSVS):
        with path.open(encoding="utf-8", newline="") as handle:
            extra = {"_dir": str(path.parent)} if i else {}
            rows += [row | extra for row in csv.DictReader(handle)]
    state = summarize.with_receipts(CSVS[0].parent, summarize.latest(rows))
    return [row for row in state if row["outcome"] == "diffed"]


def contradicted(row: dict[str, str]) -> bool:
    """The frozen P9 rule of summarize.predictions, plus the one pair found after pairing."""
    cat, tag = summarize.category(row), row["relation_tag"]
    frozen = (tag in ("finetune", "quantized") and cat == "byte-identical") or (
        tag == "quantized" and summarize.num(row, "reshaped") == summarize.num(row, "added") == 0
    )
    return frozen or row["upload"] == POST_HOC


def summary_tables() -> tuple[dict[str, dict[str, int]], set[str]]:
    """The by-tag category table and the P9 uploads, as committed in experiments/E009/summary.md."""
    text = (EXP / "E009" / "summary.md").read_text(encoding="utf-8")
    block = text.split("## Categories by the Hub's relation tag")[1].split("\n## ")[0]
    lines = [ln for ln in block.splitlines() if ln.startswith("|")]
    header = [h.strip() for h in lines[0].strip("|").split("|")]
    table = {}
    for ln in lines[2:]:
        cells = [x.strip() for x in ln.strip("|").split("|")]
        table[cells[0]] = dict(zip(header[2:], map(int, cells[2:]), strict=True))
    p9 = text.split("- P9 ")[1].split("\n")[0]
    return table, set(re.findall(r"`([^`]+/[^`]+)` \(", p9))


def check(diffed: list[dict[str, str]]) -> dict[str, list[str]]:
    """Every count drawn, against the summary and the receipt page; the sub-label of each row."""
    cats = Counter(summarize.category(r) for r in diffed)
    table, p9 = summary_tables()
    for tag, counts in table.items():
        got = Counter(summarize.category(r) for r in diffed if r["relation_tag"] == tag)
        if any(got[k] != v for k, v in counts.items()):
            raise SystemExit(f"{tag}: {dict(got)} differs from summary.md {counts}")
    page = (EXP / "E009-hub-census.md").read_text(encoding="utf-8")
    bases = {r["base"] for r in diffed}
    if (len(diffed), len(bases)) != (93, 11) or "93 pairs diffed on 11 bases" not in page:
        raise SystemExit(f"{len(diffed)} pairs on {len(bases)} bases, not the page's 93 on 11")
    if cats["every tensor changed"] or set(cats) - set(ORDER):
        raise SystemExit(f"categories outside the figure: {dict(cats)}")
    frozen = {r["upload"] for r in diffed if contradicted(r)} - {POST_HOC}
    if frozen != p9 or len(p9) != 11 or "a twelfth" not in page:
        raise SystemExit(f"contradicted tags {sorted(frozen)} differ from summary.md P9")
    by = {r["upload"]: r for r in diffed}
    jev = by[POST_HOC]
    if (summarize.category(jev), jev["relation_tag"], jev.get("renamed")) != (
        "renamed, unchanged", "finetune", "426"
    ):  # fmt: skip
        raise SystemExit(f"{POST_HOC} no longer reads as the base's weights renamed")
    return sub_labels(diffed, by)


def sub_labels(diffed: list[dict[str, str]], by: dict[str, dict]) -> dict[str, list[str]]:
    def rows(cat: str) -> list[dict[str, str]]:
        return [r for r in diffed if summarize.category(r) == cat]

    mirrors = rows("byte-identical")
    if not all(r["upload"].startswith("unsloth/") and r["relation_tag"] == "finetune"
               for r in mirrors):  # fmt: skip
        raise SystemExit("a byte-identical upload is not an unsloth mirror tagged finetune")
    low = rows("low rank, few projections")
    if not all("heretic" in r["upload"] and summarize.num(r, "rank_le_64") ==
               summarize.num(r, "rank_measured") for r in low):  # fmt: skip
        raise SystemExit("the low-rank row is not the two heretic uploads")
    kept = Counter(r["relation_tag"] for r in rows("renamed, unchanged"))
    lora = {u for u in MERGED_LORAS if summarize.category(by[u]) == "dtype rewrite"}
    partial = rows("partial change")
    # The page's P2 row: at least half the elements changed, with full-rank deltas.
    dense = [r for r in partial if float(r["fraction_changed"] or 0) >= 0.5
             and summarize.num(r, "rank_le_64") < summarize.num(r, "rank_measured")]  # fmt: skip
    fp8 = [r for r in partial if r["relation_tag"] == "quantized" and "FP8" in r["upload"]
           and summarize.num(r, "changed") == 0]  # fmt: skip
    if (len(dense), len(fp8), len(lora)) != (13, 5, 4):
        raise SystemExit(f"{len(dense)} dense (page: 13), {len(fp8)} FP8, {len(lora)} F32 merges")
    unpaired = Counter(r["relation_tag"] for r in rows("renamed layout"))
    return {
        "byte-identical": ["Byte-identical to the base", f"all {len(mirrors)} unsloth mirrors, "
                           "every one tagged finetune"],
        "low rank, few projections": ["A few projections, low rank",
                                      "two named as abliterations: o_proj, down_proj"],
        "renamed, unchanged": ["Renamed, values kept", f"{kept['finetune']} repackagings, "
                               f"{kept['quantized']} MLX quantizations"],
        "renamed, changed": ["Renamed, values changed",
                             "multimodal wrappers, prefix-dropping fine-tunes"],
        "dtype rewrite": ["Saved in another dtype", f"{len(lora)} are LoRA merges saved in F32"],
        "partial change": ["Some tensors changed", f"{len(dense)} dense fine-tunes at full rank, "
                           f"{len(fp8)} FP8 casts"],
        "renamed layout": ["Renamed, still unpaired", f"{unpaired['quantized']} of "
                           f"{sum(unpaired.values())} tagged quantized: packed weights"],
    }  # fmt: skip


def draw(theme: str, diffed: list[dict[str, str]], labels: dict[str, list[str]]) -> Path:
    f = Canvas(THEMES[theme], W, H)
    n_bases = len({r["base"] for r in diffed})
    f.text(52, 74, f"{len(diffed)} popular uploads, diffed against the base they declare", 36,
           "text", weight="semibold")  # fmt: skip
    per_base = max(Counter(r["base"] for r in diffed).values())
    f.text(52, 116, f"The most-downloaded derivatives of {n_bases} open-weight bases on the "
           f"Hugging Face Hub, up to {per_base} per base, 2026-11-04. One square per upload.", 22,
           "muted")  # fmt: skip
    size, gap, per_line, x0, y = 30, 8, 24, 640, 168
    for cat in ORDER:
        group = [r for r in diffed if summarize.category(r) == cat]
        group.sort(key=lambda r: (not contradicted(r), list(TAGS).index(r["relation_tag"])))
        title, sub = labels[cat]
        f.text(52, y + 24, title, 25, "text")
        f.text(52, y + 52, sub, 20, "muted")
        f.text(600, y + 26, str(len(group)), 30, "body", ha="right", weight="semibold")
        for i, r in enumerate(group):
            x = x0 + (i % per_line) * (size + gap)
            top = y + (i // per_line) * (size + gap)
            f.rect(x, top, size, size, TAGS[r["relation_tag"]], radius=4)
            if contradicted(r):
                f.rect(x - 4.5, top - 4.5, size + 9, size + 9, None, "text", 3.2, radius=6)
        lines = (len(group) - 1) // per_line + 1
        y += max(70, lines * (size + gap) + 34)
    ringed = Counter(summarize.category(r) for r in diffed if contradicted(r))
    if ringed != Counter({"byte-identical": 9, "low rank, few projections": 1,
                          "partial change": 1, "renamed, unchanged": 1}):  # fmt: skip
        raise SystemExit(f"ringed uploads by category changed: {dict(ringed)}")
    f.text(
        52,
        H - 104,
        "Ringed: 9 mirrors tagged finetune; a heretic upload and a BF16 fine-tune "
        "tagged quantized; the base's own weights, renamed, tagged finetune",
        19,
        "muted",
    )
    ly = H - 52
    x = 52
    for tag, colour in TAGS.items():
        n = sum(r["relation_tag"] == tag for r in diffed)
        f.rect(x, ly - 20, 22, 22, colour, radius=3)
        f.text(x + 34, ly, f"tagged {tag} ({n})", 21, "body")
        x += 300
    n = sum(contradicted(r) for r in diffed)
    f.rect(x + 2, ly - 18, 18, 18, "panel", radius=3)
    f.rect(x - 2.5, ly - 22.5, 27, 27, None, "text", 3.2, radius=5)
    f.text(x + 34, ly, f"the Hub's tag contradicted by the bytes ({n})", 21, "body")
    path = HERE / f"census-{theme}.png"
    f.save(path)
    return path


def main() -> None:
    diffed = load()
    labels = check(diffed)
    for theme in THEMES:
        path = draw(theme, diffed, labels)
        print(f"wrote docs/assets/{path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
