"""``wvc diff --format agent``: a compact Markdown view of a diff record for an LLM agent.

The view is rendered from the same record ``--json`` prints (every tensor, unchanged ones
included), so the two never disagree; it groups and truncates so that a model of several
hundred tensors fits in about 150 lines. Its layout is a stable format, ``wvc-diff-agent/0``
(spec section 2.2): the sections, their order, the table columns, and the truncation lines
change only with a new format id. ``docs/agents-reading-diffs.md`` says how to read it.

Layer grouping is by name: the first dot-separated part that is all digits is the layer
index, what precedes it the stack (``model.layers``), what follows it the projection
(``mlp.down_proj.weight``). Tensors without such a part are grouped as ``(no layer)``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from wvc.quote import ELLIPSIS, printable

AGENT_FORMAT = "wvc-diff-agent/0"
MAX_PROJECTION_ROWS = 24
MAX_LAYER_ROWS = 36
MAX_LARGEST_ROWS = 12
MAX_LISTED = 8  # added, removed, reshaped, and unchanged files listed by name
MAX_NAME_CHARS = 120  # a longer name or path is cut, marked with an ellipsis
NO_LAYER = "(no layer)"
FOOTER = (
    "A diff shows where a model changed and how, not what the change does.",
    "Shown: which tensors differ, how many elements, how large the change is relative to the "
    "base, and the rank structure of each change (`rank` at relative tolerance 1e-2, "
    "`top-1` the share of the change's energy in its largest singular direction).",
    "Not shown: behaviour, intent, quality, or safety. No model was run. A low-rank change "
    "can be a merged adapter, a steering direction, or a direct edit, and the diff alone does "
    "not say which.",
)
QUANTIZATION_FOOTER = (
    "This is a quantization comparison: every quantized tensor differs from its float base by "
    "its rounding error, which is not an edit, and a rescaling that leaves the function "
    "unchanged (E011) also shows as change."
)


@dataclass
class _Group:
    """Changed tensors sharing a projection, across layers."""

    tensors: list[dict] = field(default_factory=list)
    layers: set[int] = field(default_factory=set)


def code(text: object) -> str:
    """A name or path as inline code that cannot leave its span or its table cell.

    Names come from the files compared: control characters are escaped (``quote.printable``),
    a backtick or a pipe is written as its ``\\uNNNN`` escape, and the text is cut at
    ``MAX_NAME_CHARS`` characters.
    """
    safe = printable(text).replace("`", "\\u0060").replace("|", "\\u007c")
    if len(safe) > MAX_NAME_CHARS:
        safe = safe[:MAX_NAME_CHARS] + ELLIPSIS
    return f"`{safe}`"


def split_name(name: str) -> tuple[str, int | None, str]:
    """(stack, layer, projection): ``model.layers.17.mlp.down_proj.weight`` gives
    (``model.layers``, 17, ``mlp.down_proj.weight``); no digit part gives ("", None, name)."""
    parts = name.split(".")
    for index, part in enumerate(parts):
        if part.isdigit() and len(part) < 7:
            return ".".join(parts[:index]), int(part), ".".join(parts[index + 1 :])
    return "", None, name


def _projection_key(name: str) -> str:
    stack, layer, projection = split_name(name)
    if layer is None:
        return name
    return f"{stack}.*.{projection}" if stack else f"*.{projection}"


def ranges(numbers: Iterable[int]) -> str:
    """``0-3, 7, 9-11`` for {0, 1, 2, 3, 7, 9, 10, 11}."""
    values = sorted(set(numbers))
    if not values:
        return "-"
    spans, start, last = [], values[0], values[0]
    for value in values[1:]:
        if value != last + 1:
            spans.append((start, last))
            start = value
        last = value
    spans.append((start, last))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def _num(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}g}"


def _span(values: Sequence[float | int | None], digits: int = 3) -> str:
    """The range of the known values: ``0.1-0.3``, one value alone, or ``-``."""
    known = [value for value in values if value is not None]
    if not known:
        return "-"
    low, high = min(known), max(known)
    if isinstance(low, int) and isinstance(high, int):
        return str(low) if low == high else f"{low}-{high}"
    low_text, high_text = _num(low, digits), _num(high, digits)
    return low_text if low_text == high_text else f"{low_text} to {high_text}"


def _count(changed: int, total: int) -> str:
    share = f" ({changed / total:.2%})" if total else ""
    return f"{changed:,} of {total:,}{share}"


def _total(tensors: list[dict]) -> str:
    """Elements changed over elements compared, summed over ``tensors``."""
    return _count(sum(t["changed"] or 0 for t in tensors), sum(t["numel"] or 0 for t in tensors))


def _names(names: Sequence[str], limit: int = 4) -> str:
    """Up to ``limit`` names as code, then ``+N`` for the rest."""
    shown = ", ".join(code(name) for name in names[:limit])
    return shown + (f", +{len(names) - limit}" if len(names) > limit else "")


def _max_rel(tensors: list[dict]) -> str:
    rel = [float(t["rel_delta_norm"]) for t in tensors if t.get("rel_delta_norm") is not None]
    return _num(max(rel)) if rel else "-"


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _dtype(t: dict) -> str:
    a, b = t.get("dtype_a"), t.get("dtype_b")
    return str(a or b) if a == b or a is None or b is None else f"{a}->{b}"


def _side_line(label: str, side: dict) -> str:
    if "hf" in side:
        hf = side["hf"]
        reference = f"hf:{hf['repo']}@{hf['sha']}"
        # the file in a span of its own: a sha and a file name together pass MAX_NAME_CHARS
        file = f", file {code(hf['file'])}" if "file" in hf else ""
        parts = len(hf.get("files", ()))
        split = f", {parts} parts" if parts > 1 else ""
        where = f"{code(reference)}{file} (revision {code(hf['revision'])}{split})"
    else:
        where = code(side.get("path"))
    root = side.get("root")
    text = f"- {label}: {where}; manifest root {code(root[:16])}" if root else f"- {label}: {where}"
    if side.get("format") == "gguf":
        gguf = side["gguf"]
        types = ", ".join(f"{dtype} {count}" for dtype, count in gguf["types"].items())
        text += f"; GGUF v{gguf['version']}, architecture {code(gguf['architecture'])} ({types})"
    return text


def _pair_section(record: dict) -> list[str]:
    lines = ["## Pair", _side_line("a (base)", record["a"]), _side_line("b", record["b"])]
    comparison = record.get("comparison")
    if comparison:
        mapping = comparison["mapping"]
        lines.append(
            f"- comparison: {comparison['kind']}; GGUF side {' and '.join(comparison['gguf'])}; "
            + (f"names mapped by the {code(mapping)} table" if mapping else "names as stored")
            + f"; values dequantised to {comparison['values']} where bytes cannot be compared"
            + _file_types(comparison)
        )
    return lines


def _file_types(comparison: dict) -> str:
    """``; file type a BF16, b Q4_K_M`` from a Hub comparison's ``quantizations``, if any."""
    known = [
        f"{label} {code(value)}"
        for label, value in sorted(comparison.get("quantizations", {}).items())
        if value is not None
    ]
    return f"; file type {', '.join(known)}" if known else ""


def _headline(record: dict, tensors: list[dict], unchanged_files: dict | None) -> list[str]:
    summary = record["summary"]
    by = summary["by_status"]
    changed = [t for t in tensors if t["status"] == "changed"]
    layers_all = {(split_name(t["name"])[0], split_name(t["name"])[1]) for t in tensors}
    layers_changed = {(split_name(t["name"])[0], split_name(t["name"])[1]) for t in changed}
    layers_all.discard(("", None))
    layers_changed.discard(("", None))
    lines = [
        "## Headline",
        f"- tensors: {summary['tensors']} ({by['changed']} changed, {by['unchanged']} "
        f"unchanged, {by['added']} added, {by['removed']} removed, {by['reshaped']} reshaped)",
        f"- elements changed: {_count(summary['elements_changed'], summary['elements_compared'])}",
        f"- layers with a changed tensor: {len(layers_changed)} of {len(layers_all)}",
    ]
    if unchanged_files is not None:
        names = unchanged_files["names"]
        listed = ", ".join(code(name) for name in names[:MAX_LISTED])
        more = f", and {len(names) - MAX_LISTED} more" if len(names) > MAX_LISTED else ""
        lines.append(
            f"- {unchanged_files['label']}: {len(names)} of {unchanged_files['total']}"
            + (f" ({listed}{more})" if names else "")
        )
    return lines


def _projection_rows(changed: list[dict]) -> list[str]:
    groups: dict[str, _Group] = defaultdict(_Group)
    for t in changed:
        group = groups[_projection_key(t["name"])]
        group.tensors.append(t)
        layer = split_name(t["name"])[1]
        if layer is not None:
            group.layers.add(layer)
    lines = [
        "## Changed tensors by projection",
        "| projection | layers | dtype | elements changed | rel. norm | rank | top-1 |",
        "|---|---|---|---|---|---|---|",
    ]
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1].tensors), item[0]))
    for key, group in ordered[:MAX_PROJECTION_ROWS]:
        ts = group.tensors
        layers = ranges(group.layers) if group.layers else "-"
        cells = (
            code(key),
            f"{layers} ({len(ts)})",
            ", ".join(sorted({_dtype(t) for t in ts})),
            _total(ts),
            _span([t.get("rel_delta_norm") for t in ts]),
            _span([t.get("rank") for t in ts]),
            _span([t.get("top1_energy") for t in ts], 4),
        )
        lines.append(_row(cells))
    if len(ordered) > MAX_PROJECTION_ROWS:
        rest = ordered[MAX_PROJECTION_ROWS:]
        count = sum(len(group.tensors) for _, group in rest)
        lines.append(_row((f"and {len(rest)} more projections ({count} tensors)", *[""] * 6)))
    return lines


@dataclass
class _Run:
    """Consecutive layers of one stack whose changed projections are the same set."""

    stack: str
    first: int
    last: int
    labels: frozenset[str]
    every: frozenset[str]  # every projection in each of these layers, changed or not
    tensors: list[dict] = field(default_factory=list)


def _layer_runs(tensors: list[dict], changed: list[dict]) -> list[_Run]:
    in_layer: dict[tuple[str, int], set[str]] = defaultdict(set)
    for t in tensors:
        stack, layer, projection = split_name(t["name"])
        if layer is not None:
            in_layer[(stack, layer)].add(projection)
    by_layer: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for t in changed:
        stack, layer, _ = split_name(t["name"])
        if layer is not None:
            by_layer[(stack, layer)].append(t)
    runs: list[_Run] = []
    for (stack, layer), ts in sorted(by_layer.items()):
        labels = frozenset(split_name(t["name"])[2] for t in ts)
        every = frozenset(in_layer[(stack, layer)])
        last = runs[-1] if runs else None
        joins = last is not None and (last.stack, last.last + 1, last.labels, last.every) == (
            stack,
            layer,
            labels,
            every,
        )
        if last is not None and joins:
            last.last = layer
            last.tensors += ts
        else:
            runs.append(_Run(stack, layer, layer, labels, every, list(ts)))
    return runs


def _changed_of(labels: frozenset[str], every: frozenset[str]) -> str:
    """``all 9``, ``all 9 but `a```, or the changed names ``(of 9)``, whichever is shorter."""
    missing = sorted(every - labels)
    if not missing:
        return f"all {len(every)}"
    if len(missing) < len(labels) and len(missing) <= 4:
        return f"all {len(every)} but {_names(missing)}"
    return f"{_names(sorted(labels))} (of {len(every)})"


def _layer_rows(tensors: list[dict], changed: list[dict]) -> list[str]:
    runs = _layer_runs(tensors, changed)
    lines = [
        "## Changed tensors by layer",
        "| layers | tensors changed | projections changed | elements changed | max rel. norm |",
        "|---|---|---|---|---|",
    ]
    for run in runs[:MAX_LAYER_ROWS]:
        span = str(run.first) if run.first == run.last else f"{run.first}-{run.last}"
        key = f"{run.stack}.{span}" if run.stack else span
        cells = (
            code(key),
            str(len(run.tensors)),
            _changed_of(run.labels, run.every),
            _total(run.tensors),
            _max_rel(run.tensors),
        )
        lines.append(_row(cells))
    if len(runs) > MAX_LAYER_ROWS:
        rest = runs[MAX_LAYER_ROWS:]
        count = sum(len(run.tensors) for run in rest)
        lines.append(_row((f"and {len(rest)} more layer runs ({count} tensors)", *[""] * 4)))
    outside = [t for t in changed if split_name(t["name"])[1] is None]
    if outside:
        names = _names([t["name"] for t in outside])
        lines.append(_row((NO_LAYER, str(len(outside)), names, _total(outside), _max_rel(outside))))
    return lines


def _largest_rows(changed: list[dict]) -> list[str]:
    def size(t: dict) -> tuple:
        rel = t.get("rel_delta_norm")
        return (rel is None, -(rel or 0.0), t["name"])

    ordered = sorted(changed, key=size)
    lines = [
        f"## Largest changes (by rel. norm, top {min(len(ordered), MAX_LARGEST_ROWS)})",
        "| tensor | dtype | shape | elements changed | rows x cols touched | rel. norm "
        "| max abs | rank | top-1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in ordered[:MAX_LARGEST_ROWS]:
        shape = ", ".join(str(dim) for dim in (t.get("shape_a") or t.get("shape_b") or []))
        cells = (
            code(t["name"]),
            _dtype(t),
            f"[{shape}]",
            _count(t["changed"] or 0, t["numel"] or 0),
            f"{_num(t.get('rows_touched'), 9)} x {_num(t.get('cols_touched'), 9)}",
            _num(t.get("rel_delta_norm")),
            _num(t.get("max_abs_delta")),
            _num(t.get("rank"), 9),
            _num(t.get("top1_energy"), 4),
        )
        lines.append(_row(cells))
    if len(ordered) > MAX_LARGEST_ROWS:
        lines.append(
            f"| and {len(ordered) - MAX_LARGEST_ROWS} more changed tensors | | | | | | | | |"
        )
    return lines


def _listed(status: str, tensors: list[dict]) -> list[str]:
    chosen = [t for t in tensors if t["status"] == status]
    if not chosen:
        return [f"- {status}: none"]
    lines = [f"- {status} ({len(chosen)}):"]
    for t in chosen[:MAX_LISTED]:
        shape_a = ", ".join(str(d) for d in t.get("shape_a") or [])
        shape_b = ", ".join(str(d) for d in t.get("shape_b") or [])
        shape = f"[{shape_a}]->[{shape_b}]" if shape_a and shape_b else f"[{shape_a or shape_b}]"
        lines.append(f"  - {code(t['name'])} {_dtype(t)} {shape}")
    if len(chosen) > MAX_LISTED:
        lines.append(f"  - and {len(chosen) - MAX_LISTED} more")
    return lines


def render_agent(record: dict, unchanged_files: dict | None = None) -> str:
    """The ``wvc-diff-agent/0`` Markdown for a diff record whose ``tensors`` lists every tensor.

    ``unchanged_files`` is ``{"label", "names", "total"}``: the files (or Hub shards) whose
    tensors are all unchanged, and how they were found; None leaves the line out.
    """
    tensors = record["tensors"]
    changed = [t for t in tensors if t["status"] == "changed"]
    sections = [
        [f"# wvc diff: agent view ({AGENT_FORMAT})"],
        _pair_section(record),
        _headline(record, tensors, unchanged_files),
    ]
    if changed:
        sections += [
            _projection_rows(changed),
            _layer_rows(tensors, changed),
            _largest_rows(changed),
        ]
    else:
        sections.append(["## Changed tensors", "- none"])
    sections.append(
        ["## Added, removed, reshaped", *(line for s in ("added", "removed", "reshaped")
                                          for line in _listed(s, tensors))]
    )  # fmt: skip
    footer = [*FOOTER, QUANTIZATION_FOOTER] if record.get("comparison") else list(FOOTER)
    sections.append(["## What this does and does not show", *(f"- {line}" for line in footer)])
    return "\n\n".join("\n".join(section) for section in sections) + "\n"
