"""The E009 summary table, generated from ``census.csv`` and the selection walks.

    uv run python experiments/E009/summarize.py experiments/E009/census.csv \
        > experiments/E009/summary.md
    uv run python experiments/E009/summarize.py --rediff experiments/E009/census.csv
    uv run python experiments/E009/summarize.py experiments/E009/census.csv \
        experiments/E009/rediff-2026-11-04/census.csv > experiments/E009/summary.md

Further CSVs are read after the first, so their rows supersede earlier rows of the same pair;
each row's receipt is read from ``pairs/`` beside the CSV it came from, and the selections
from beside the first. With one CSV the output is what it was.

``--rediff`` lists instead the diffed pairs whose receipts have tensors both added and removed
and no tensor ids, with both pinned commits: the pairs to diff again with a wvc that pairs
renamed tensors. A receipt that has ids (``summary.renamed``) splits "renamed layout" into
"renamed, unchanged" and "renamed, changed"; those columns and the renamed section appear
only then, so the summary of receipts without ids is what it was.

The latest row for each (base, upload) is its state. Categories follow the definitions frozen
in ``experiments/E009-hub-census.md``; a diffed pair falls in the first that applies.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

LOW_RANK_MAX = 64
SMALL_K = 10  # the small group takes ten uploads per base (census.BASES)
PREDICTED = {
    "P1": "5% to 20%",
    "P2": "35% to 65%",
    "P3": "5% to 20%",
    "P4": "10% to 30%, mostly Qwen3.5",
    "P5": "at least 80%",
    "P6": "at least 30%",
    "P7": "at least one crash at 40% credence; refusals 5% to 15%",
    "P8": "at least 0.9",
    "P9": "at least 2",
}
CATEGORIES = (
    "byte-identical",
    "every tensor changed",
    "low rank, few projections",
    "renamed layout",
    "dtype rewrite",
    "partial change",
)
# Receipts written by a wvc that pairs renamed tensors (spec section 2, ``summary.renamed``)
# split "renamed layout" into these two; it then holds only what stayed unpaired.
RENAMED_SPLIT = ("renamed, unchanged", "renamed, changed")
RENAMED_ANY = ("renamed layout", *RENAMED_SPLIT)


def latest(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    state: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        state[(row["base"], row["upload"])] = row
    return list(state.values())


def num(row: dict[str, str], key: str) -> int:
    value = row.get(key)
    return int(value) if value not in ("", None) else 0


def receipt_path(out: Path, row: dict[str, str]) -> Path:
    """The row's receipt, beside the CSV the row came from (``_dir``) or else under ``out``."""
    return Path(row.get("_dir") or out) / "pairs" / f"{row['upload'].replace('/', '__')}.json"


def renamed_counts(diff: dict) -> dict[str, str] | None:
    """Renamed pairs in a receipt's diff record, or None for a record without ids (old wvc).

    The census diffs without ``--all``, so renamed tensors that are unchanged are counted from
    the summary, not listed.
    """
    total = diff["summary"].get("renamed")
    if total is None:
        return None
    changed = sum(
        1 for t in diff["tensors"] if t.get("renamed_to") is not None and t["status"] != "unchanged"
    )
    return {
        "renamed": str(total),
        "renamed_changed": str(changed),
        "renamed_unchanged": str(total - changed),
    }


def with_receipts(out: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """The rows, each diffed one with its receipt's renamed counts when the receipt has them."""
    enriched = []
    for row in rows:
        path = receipt_path(out, row)
        counts = None
        if row["outcome"] == "diffed" and path.is_file():
            counts = renamed_counts(json.loads(path.read_text(encoding="utf-8"))["diff"])
        enriched.append(row if counts is None else {**row, **counts})
    return enriched


def categories(rows: list[dict[str, str]]) -> tuple[str, ...]:
    """CATEGORIES, with the renamed split after "renamed layout" once a row falls in it."""
    if not any(category(row) in RENAMED_SPLIT for row in rows):
        return CATEGORIES
    at = CATEGORIES.index("renamed layout") + 1
    return (*CATEGORIES[:at], *RENAMED_SPLIT, *CATEGORIES[at:])


def category(row: dict[str, str]) -> str:
    tensors = num(row, "tensors")
    unchanged, changed = num(row, "unchanged"), num(row, "changed")
    added, removed, reshaped = num(row, "added"), num(row, "removed"), num(row, "reshaped")
    renamed = num(row, "renamed")  # 0 for a receipt without ids
    if unchanged == tensors and tensors and not renamed:
        return "byte-identical"
    if renamed and (renamed + added + removed >= tensors / 2 or unchanged == tensors):
        return RENAMED_SPLIT[0] if changed == reshaped == 0 else RENAMED_SPLIT[1]
    if added + removed >= tensors / 2:
        return "renamed layout"
    if reshaped >= tensors / 2:
        return "dtype rewrite"
    if unchanged == 0 and added == removed == 0:
        return "every tensor changed"
    measured, low = num(row, "rank_measured"), num(row, "rank_le_64")
    if (
        changed
        and added == removed == reshaped == 0
        and changed <= tensors / 2
        and measured
        and low == measured
    ):
        return "low rank, few projections"
    return "partial change"


def share(part: int, whole: int) -> str:
    return f"{part} of {whole} ({100 * part / whole:.0f}%)" if whole else f"{part} of 0"


def touched_share(row: dict[str, str]) -> float:
    tensors = num(row, "tensors")
    return (num(row, "changed") + num(row, "reshaped")) / tensors if tensors else 0.0


def bucket(value: float) -> str:
    if value == 0:
        return "0"
    if value <= 0.1:
        return "(0, 10%]"
    if value <= 0.5:
        return "(10%, 50%]"
    if value < 1:
        return "(50%, 100%)"
    return "100%"


def table(header: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def walks(out: Path) -> dict[str, dict]:
    return {
        json.loads(path.read_text(encoding="utf-8"))["base"]["repo"]: json.loads(
            path.read_text(encoding="utf-8")
        )
        # by name, case-sensitively: Windows paths compare without case
        for path in sorted((out / "selection").glob("*.json"), key=lambda p: p.name)
    }


def per_base(rows: list[dict[str, str]], selections: dict[str, dict]) -> list[str]:
    order = categories([row for row in rows if row["outcome"] == "diffed"])
    by_base = defaultdict(list)
    for row in rows:
        by_base[row["base"]].append(row)
    body = []
    for base, selection in selections.items():
        mine = by_base.get(base, [])
        outcomes = Counter(row["outcome"] for row in mine)
        cats = Counter(category(row) for row in mine if row["outcome"] == "diffed")
        shares = [touched_share(row) for row in mine if row["outcome"] == "diffed"]
        body.append(
            [
                base,
                len(selection["walked"]),
                outcomes["skipped"],
                outcomes["diffed"],
                outcomes["failed"],
                sum(1 for e in selection["walked"] if e["eligible"])
                - outcomes["diffed"]
                - outcomes["failed"],
                *(cats[name] for name in order),
                f"{min(shares):.2f} / {statistics.median(shares):.2f} / {max(shares):.2f}"
                if shares
                else "-",
            ]
        )
    header = [
        "base",
        "walked",
        "skipped",
        "diffed",
        "failed",
        "not run",
        *order,
        "tensors touched, min / median / max",
    ]
    return table(header, body)


def skipped_reasons(rows: list[dict[str, str]]) -> list[str]:
    counts = Counter(row["reason"].split(":")[0] for row in rows if row["outcome"] == "skipped")
    return table(["reason", "candidates"], sorted(counts.items(), key=lambda kv: -kv[1]))


def failures(rows: list[dict[str, str]]) -> list[str]:
    failed = [row for row in rows if row["outcome"] == "failed"]
    if not failed:
        return ["None."]
    return table(
        ["base", "upload", "tag", "reason"],
        [[row["base"], row["upload"], row["relation_tag"], row["reason"]] for row in failed],
    )


def same_values(diffed: list[dict[str, str]]) -> list[str]:
    """Not a frozen definition: uploads whose shared tensors are all unchanged."""
    rows = [
        row
        for row in diffed
        if num(row, "unchanged")
        and num(row, "changed") == num(row, "reshaped") == 0
        and category(row) != "renamed layout"
    ]
    body = [
        [
            row["base"],
            row["upload"],
            row["relation_tag"],
            row["unchanged"],
            row["added"],
            row["removed"],
        ]
        for row in sorted(rows, key=lambda r: (r["base"], r["upload"]))
    ]
    return table(["base", "upload", "tag", "unchanged", "added", "removed"], body)


def value_ranks(out: Path, diffed: list[dict[str, str]]) -> list[str]:
    """Not a frozen definition: ranks over every same-shape tensor whose values changed,
    whatever its dtype, read from the receipts (the CSV counts same-dtype tensors only)."""
    body = []
    for row in sorted(diffed, key=lambda r: (r["base"], r["upload"])):
        path = receipt_path(out, row)
        tensors = json.loads(path.read_text(encoding="utf-8"))["diff"]["tensors"]
        moved = [
            t
            for t in tensors
            if t["status"] in ("changed", "reshaped") and t.get("changed") and t["rank"] is not None
        ]
        ranks = sorted(t["rank"] for t in moved)
        full = any(t["rank"] >= min(t["rows_touched"], t["cols_touched"]) for t in moved)
        if not ranks or max(ranks) > LOW_RANK_MAX or full:
            continue
        kinds = Counter(t["name"].rsplit(".", 2)[-2] for t in moved)
        body.append(
            [
                row["base"],
                row["upload"],
                row["relation_tag"],
                category(row),
                len(ranks),
                f"{ranks[0]} to {ranks[-1]}",
                ", ".join(f"{name} {count}" for name, count in sorted(kinds.items())),
            ]
        )
    header = ["base", "upload", "tag", "category", "tensors", "ranks", "projections"]
    return table(header, body)


def tensor_issues(t: dict) -> list[str]:
    issues = []
    changed, numel = t.get("changed"), t.get("numel")
    if changed is not None and numel is not None and changed > numel:
        issues.append("changed exceeds numel")
    if t["status"] == "changed" and changed == 0:
        issues.append("changed with no element changed")
    if t.get("rank") is not None:
        if not 1 <= t["rank"] <= min(t["rows_touched"], t["cols_touched"]):
            issues.append("rank outside 1..min(rows, cols) touched")
        if not 0 < t["top1_energy"] <= 1 + 1e-9:
            issues.append("top-1 energy outside (0, 1]")
    for key in ("max_abs_delta", "delta_norm", "rel_delta_norm"):
        value = t.get(key)
        if isinstance(value, float) and (value != value or value < 0 or value == float("inf")):
            issues.append(f"{key} not a finite non-negative number")
    if t["status"] == "reshaped" and (t["dtype_a"], t["shape_a"]) == (t["dtype_b"], t["shape_b"]):
        issues.append("reshaped with equal dtype and shape")
    return issues


def consistency(dirs: list[Path]) -> list[str]:
    """Internal checks on every receipt: a wrong number is a wvc bug even without a crash."""
    receipts = tensors = 0
    found: Counter = Counter()
    for path in [
        p for d in dirs for p in sorted((d / "pairs").glob("*.json"), key=lambda p: p.name)
    ]:
        diff = json.loads(path.read_text(encoding="utf-8"))["diff"]
        receipts += 1
        summary = diff["summary"]
        if sum(summary["by_status"].values()) != summary["tensors"]:
            found["status counts do not sum to the tensor count"] += 1
        for t in diff["tensors"]:
            tensors += 1
            found.update(tensor_issues(t))
    lines = [f"Checked {receipts} receipts and {tensors} reported tensors."]
    lines += [f"- {issue}: {count}" for issue, count in sorted(found.items())] or [
        "- no issue found"
    ]
    return lines


def distribution(diffed: list[dict[str, str]]) -> list[str]:
    order = ["0", "(0, 10%]", "(10%, 50%]", "(50%, 100%)", "100%"]
    counts = Counter(bucket(touched_share(row)) for row in diffed)
    return table(["share of tensors changed or reshaped", "pairs"], [[b, counts[b]] for b in order])


def by_tag(diffed: list[dict[str, str]]) -> list[str]:
    order = categories(diffed)
    tags = sorted({row["relation_tag"] for row in diffed})
    body = []
    for tag in tags:
        cats = Counter(category(row) for row in diffed if row["relation_tag"] == tag)
        body.append([tag or "(none)", sum(cats.values()), *(cats[name] for name in order)])
    return table(["Hub tag", "diffed", *order], body)


def rank_rows(diffed: list[dict[str, str]]) -> list[str]:
    measured = [row for row in diffed if num(row, "rank_measured")]
    body = [
        [
            row["upload"],
            category(row),
            num(row, "changed"),
            row["rank_measured"],
            row["rank_le_64"],
            row["rank_median"],
            row["rank_max"],
        ]
        for row in sorted(measured, key=lambda r: (category(r), r["upload"]))
    ]
    header = [
        "upload",
        "category",
        "changed",
        "ranks measured",
        "rank <= 64",
        "median rank",
        "max rank",
    ]
    return table(header, body)


def predictions(
    rows: list[dict[str, str]], diffed: list[dict[str, str]], selections: dict[str, dict]
) -> list[str]:
    cats = Counter(category(row) for row in diffed)
    total = len(diffed)
    finetune = [row for row in diffed if row["relation_tag"] == "finetune"]
    quantized = [row for row in diffed if row["relation_tag"] == "quantized"]
    small = [s for s in selections.values() if s["k"] == SMALL_K]
    walked = sum(len(s["walked"]) for s in small)
    no_st = sum(1 for s in small for e in s["walked"] if e["reason"] == "no-safetensors")
    attempted = [row for row in rows if row["outcome"] in ("diffed", "failed")]
    crashes = [row for row in rows if row["reason"].startswith("crash")]
    refusals = [row for row in rows if row["reason"].startswith("refused")]
    dense = [
        float(row["fraction_changed"])
        for row in diffed
        if category(row) == "every tensor changed" and row["fraction_changed"]
    ]
    contradict = [
        row
        for row in diffed
        if (row["relation_tag"] in ("finetune", "quantized") and category(row) == "byte-identical")
        or (
            row["relation_tag"] == "quantized"
            and num(row, "reshaped") == 0
            and num(row, "added") == 0
        )
    ]
    quant_ok = sum(1 for row in quantized if category(row) in ("dtype rewrite", *RENAMED_ANY))
    renamed = [row for row in diffed if category(row) in RENAMED_ANY]
    split = ", ".join(f"{name} {cats[name]}" for name in RENAMED_SPLIT if cats[name])
    renamed_q35 = sum(1 for row in renamed if row["base"].startswith("Qwen/Qwen3.5-"))
    low = sum(1 for r in finetune if category(r) == "low rank, few projections")
    tagged = ", ".join(f"`{r['upload']}` ({r['relation_tag']}, {category(r)})" for r in contradict)
    p8 = f"{statistics.median(dense):.4f} over {len(dense)} pairs" if dense else "no such pairs"
    observed = {
        "P1": f"byte-identical: {share(cats['byte-identical'], total)}",
        "P2": f"every tensor changed: {share(cats['every tensor changed'], total)}",
        "P3": f"low rank among `finetune`: {share(low, len(finetune))}",
        "P4": f"renamed layout: {share(len(renamed), total)}, {renamed_q35} on Qwen3.5 bases"
        + (f" ({split})" if split else ""),
        "P5": f"`quantized` that are dtype rewrites or renamed: {share(quant_ok, len(quantized))}",
        "P6": f"`no-safetensors` among small-base candidates: {share(no_st, walked)}",
        "P7": f"crashes: {len(crashes)}; refusals: {share(len(refusals), len(attempted))}",
        "P8": f"median elements changed, every-tensor-changed pairs: {p8}",
        "P9": f"pairs contradicting their tag: {len(contradict)}: {tagged or 'none'}",
    }
    return [f"- {key} {text} (predicted {PREDICTED[key]})" for key, text in observed.items()]


def renamed_rows(diffed: list[dict[str, str]]) -> list[str]:
    """Pairs whose receipt pairs renamed tensors: how many, and how many of them changed."""
    rows = [row for row in diffed if num(row, "renamed")]
    body = [
        [
            row["base"],
            row["upload"],
            category(row),
            row["renamed"],
            row["renamed_unchanged"],
            row["renamed_changed"],
            row["added"],
            row["removed"],
        ]
        for row in sorted(rows, key=lambda r: (r["base"], r["upload"]))
    ]
    header = ["base", "upload", "category", "renamed", "unchanged", "changed", "added", "removed"]
    return table(header, body)


def renamed_section(diffed: list[dict[str, str]]) -> list[str]:
    """The section on renamed pairs; nothing when no receipt carries ``summary.renamed``."""
    if not any("renamed" in row for row in diffed):
        return []
    return [
        "## Renamed tensors paired by the diff",
        "",
        "From receipts written by a wvc that pairs renamed tensors by id and by name prefix "
        '(spec section 2); receipts without ids keep the frozen "renamed layout".',
        "",
        *renamed_rows(diffed),
        "",
    ]


def needs_rediff(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Diffed pairs with tensors both added and removed whose receipt carries no ids."""
    return sorted(
        (
            row
            for row in rows
            if row["outcome"] == "diffed"
            and "renamed" not in row
            and num(row, "added")
            and num(row, "removed")
        ),
        key=lambda r: (r["base"], r["upload"]),
    )


def rediff_list(rows: list[dict[str, str]]) -> list[str]:
    """What ``--rediff`` prints: the pairs to diff again, with both pinned commits."""
    chosen = needs_rediff(rows)
    body = [
        [
            row["base"],
            row["base_sha"],
            row["upload"],
            row["upload_sha"],
            row["added"],
            row["removed"],
        ]
        for row in chosen
    ]
    header = ["base", "base sha", "upload", "upload sha", "added", "removed"]
    bases = len({row["base"] for row in chosen})
    return [
        f"{len(chosen)} diffed pairs on {bases} bases have tensors both added and removed and "
        "a receipt without tensor ids; diff them again to pair their renames.",
        "",
        *table(header, body),
    ]


def pair_table(diffed: list[dict[str, str]]) -> list[str]:
    body = [
        [
            row["base"],
            row["upload"],
            row["relation_tag"],
            category(row),
            row["tensors"],
            f"{row['unchanged']}/{row['changed']}/{row['added']}/{row['removed']}/{row['reshaped']}",
            f"{float(row['fraction_changed']):.4f}" if row["fraction_changed"] else "-",
            f"{row['rank_le_64']}/{row['rank_measured']}" if num(row, "rank_measured") else "-",
            f"{row['fetched_a']}/{row['fetched_b']} ({row['shards_skipped']} skipped)",
            f"{num(row, 'bytes_new') / 1e9:.2f}",
            row["wall_seconds"],
        ]
        for row in sorted(diffed, key=lambda r: (r["base"], int(r["listing_rank"])))
    ]
    header = [
        "base",
        "upload",
        "tag",
        "category",
        "tensors",
        "unch/chg/add/rem/resh",
        "elements changed",
        "rank <= 64 / measured",
        "shards fetched a/b",
        "GB new",
        "seconds",
    ]
    return table(header, body)


def shown_path(path: Path, out: Path) -> str:
    """``path`` relative to ``out`` where it can be (a sibling too), else as given."""
    try:
        return Path(os.path.relpath(path.resolve(), out.resolve())).as_posix()
    except ValueError:  # another drive on Windows
        return path.as_posix()


def main(argv: list[str]) -> int:
    rediff = "--rediff" in argv[1:]
    csv_paths = [Path(arg) for arg in argv[1:] if arg != "--rediff"]
    out = csv_paths[0].parent
    all_rows = []
    for index, csv_path in enumerate(csv_paths):
        with csv_path.open(encoding="utf-8", newline="") as handle:
            extra = {"_dir": str(csv_path.parent)} if index else {}
            all_rows += [row | extra for row in csv.DictReader(handle)]
    rows = with_receipts(out, latest(all_rows))
    if rediff:
        sys.stdout.buffer.write(("\n".join(rediff_list(rows)) + "\n").encode("utf-8"))
        return 0
    selections = walks(out)
    diffed = [row for row in rows if row["outcome"] == "diffed"]
    total_bytes = sum(num(row, "bytes_new") for row in all_rows)
    hosts = Counter(row["host"] for row in diffed)
    walked = sum(len(s["walked"]) for s in selections.values())
    lines = [
        "# E009 summary",
        "",
        "Generated by `experiments/E009/summarize.py` from `census.csv` and `selection/`; "
        "do not edit by hand. Definitions and predictions: [E009](../E009-hub-census.md).",
        *[
            f"Rows and receipts from `{shown_path(path, out)}` supersede earlier "
            "rows of the same pair."
            for path in csv_paths[1:]
        ],
        "",
        f"- Bases: {len(selections)}; candidates walked: {walked}; "
        f"pairs diffed: {len(diffed)} ({', '.join(f'{h} {n}' for h, n in sorted(hosts.items()))}); "
        f"failed: {sum(1 for r in rows if r['outcome'] == 'failed')}.",
        f"- Bytes written into the census caches over all attempts: {total_bytes / 1e9:.1f} GB.",
        "",
        "## Per base",
        "",
        *per_base(rows, selections),
        "",
        "## Candidates passed over, by reason",
        "",
        *skipped_reasons(rows),
        "",
        "## Uploads whose shared tensors are all unchanged",
        "",
        "Added after the run, not one of the frozen definitions: byte-identical uploads plus "
        "those that only add or drop tensors (a tied `lm_head`, for example).",
        "",
        *same_values(diffed),
        "",
        "## Distribution of tensors changed",
        "",
        *distribution(diffed),
        "",
        "## Categories by the Hub's relation tag",
        "",
        *by_tag(diffed),
        "",
        "## Rank structure where `--rank` measured one",
        "",
        "Ranks at the 1e-2 relative tolerance, for changed tensors whose touched block "
        "is at most 2^26 elements.",
        "",
        *rank_rows(diffed),
        "",
        "## Uploads whose every measured value change has rank at most 64",
        "",
        "Added after the run, not one of the frozen definitions: every same-shape tensor whose "
        "values changed and whose rank was measured, whatever its dtype, from the receipts; "
        "a pair is left out when any such tensor is full rank in its touched block.",
        "",
        *value_ranks(out, diffed),
        "",
        *renamed_section(diffed),
        "## Receipt consistency",
        "",
        *consistency([out, *(path.parent for path in csv_paths[1:])]),
        "",
        "## Failures",
        "",
        *failures(rows),
        "",
        "## Predictions, computed",
        "",
        *predictions(rows, diffed, selections),
        "",
        "## Every diffed pair",
        "",
        *pair_table(diffed),
        "",
    ]
    sys.stdout.buffer.write("\n".join(lines).encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
