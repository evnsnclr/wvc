"""The ``wvc`` command: the argument parser, ``main``, and the checkpoint and repository commands.

``main`` turns errors into exit codes (see ``EXIT_CODES``). The checkpoint commands (hash,
diff, verify) and the repository commands of spec section 3.6 live here; bisect, fsck, PEFT,
and remote commands register themselves from their ``wvc.cli_*`` modules, and what the CLI
modules share is in ``wvc.cli_common``.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Sequence
from typing import Any, cast

from wvc import __version__
from wvc.agent_report import AGENT_FORMAT, render_agent
from wvc.checkout import CheckoutResult, checkout_commit
from wvc.classify import classification_lines
from wvc.cli_bisect import add_bisect_command
from wvc.cli_classify import (
    FEATURES_HELP,
    PairRun,
    add_classify_command,
    classify_run,
    tensor_dicts,
)
from wvc.cli_common import (
    REF_HELP,
    add_repo_option,
    format_table,
    plural,
    render_commit,
    short_id,
)
from wvc.cli_fsck import add_fsck_command
from wvc.cli_peft import add_peft_commands
from wvc.cli_remote import add_remote_commands, source_arg
from wvc.diff import DEFAULT_RANK_CAP, TensorDiff, diff_checkpoints, summarize
from wvc.errors import WvcError
from wvc.features import compute_features
from wvc.gguf_diff import compare_quantized, gguf_about, is_quantization_pair
from wvc.gguf_io import gguf_files, is_gguf, load_gguf
from wvc.hashing import build_manifest, gguf_hash_lines, tensor_digests
from wvc.hub import (
    GgufChoiceError,
    Hub,
    HubFile,
    HubRef,
    HubRepo,
    PairPlan,
    diff_hub_pair,
    is_hub_checkpoint,
    load_hub_checkpoint,
    load_hub_gguf,
    parse_hub_argument,
)
from wvc.output import configure_streams, emit_bytes, emit_json, emit_text
from wvc.quote import printable
from wvc.records import OPERATION_KINDS
from wvc.repo import (
    AUTHOR_ENV,
    CommitResult,
    ManifestComparison,
    ShowResult,
    commit_checkpoint,
    default_author,
    init_repository,
    log_entries,
    open_repository,
    show_commit,
    tensor_count,
)
from wvc.safetensors_io import Checkpoint, load_checkpoint
from wvc.store import ObjectStore, manifest_json
from wvc.verify import TensorRecord, VerifyReport, load_manifest, verify_checkpoint

CHECKPOINT_HELP = (
    "a .safetensors file, a directory of shards, a .gguf file, hf:OWNER/NAME[@REVISION], or "
    "hf:OWNER/NAME[@REVISION]:FILE.gguf"
)
MANIFEST_FORMATS = ("json", "gguf-hash")
DIFF_FORMATS = ("table", "json", "agent")
DEBUG_ENV = "WVC_DEBUG"  # when non-empty, errors propagate with their traceback
INTERRUPTED_EXIT = 130  # 128 + SIGINT, as a shell reports a command stopped by Ctrl-C
DIFF_COLUMNS = (
    "status",
    "tensor",
    "dtype",
    "shape",
    "changed",
    "rows",
    "cols",
    "max|d|",
    "|d|/|a|",
    "rank",
    "top1",
)
EXIT_CODES = f"""\
exit codes:
  0    success (bisect exits 0 whatever it finds)
  1    diff found a difference, verify a mismatch, or fsck a problem
  2    a wvc error or bad usage
  {INTERRUPTED_EXIT}  interrupted (Ctrl-C)
"""


def _fmt_shape(shape: tuple[int, ...] | None) -> str:
    return "" if shape is None else "[" + ",".join(str(d) for d in shape) + "]"


def _fmt_num(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def render_diff_table(diffs: list[TensorDiff], *, show_all: bool) -> str:
    rows = [d for d in diffs if show_all or d.status != "unchanged"]
    table = [DIFF_COLUMNS]
    for d in rows:
        dtype = (
            d.dtype_a
            if d.dtype_a == d.dtype_b or d.dtype_b is None
            else f"{d.dtype_a}->{d.dtype_b}"
        )
        shape = _fmt_shape(d.shape_a or d.shape_b)
        if d.shape_b is not None and d.shape_a is not None and d.shape_a != d.shape_b:
            shape = f"{_fmt_shape(d.shape_a)}->{_fmt_shape(d.shape_b)}"
        changed = "-" if d.changed is None else f"{d.changed:,}/{d.numel:,}"
        table.append(
            (
                d.status,
                d.name,
                dtype or "-",
                shape,
                changed,
                _fmt_num(d.rows_touched, ","),
                _fmt_num(d.cols_touched, ","),
                _fmt_num(d.max_abs_delta, ".3g"),
                _fmt_num(d.rel_delta_norm, ".3g"),
                _fmt_num(d.rank, "d"),
                _fmt_num(d.top1_energy, ".4f"),
            )
        )
    return format_table(table)


def render_summary(summary: dict) -> str:
    by = summary["by_status"]
    fraction = summary["fraction_changed"]
    lines = [
        f"tensors: {summary['tensors']} ({by['changed']} changed, {by['unchanged']} unchanged, "
        f"{by['added']} added, {by['removed']} removed, {by['reshaped']} reshaped)",
        f"elements changed: {summary['elements_changed']:,} of {summary['elements_compared']:,}"
        + (f" ({fraction:.4%})" if fraction is not None else ""),
    ]
    return "\n".join(lines)


def _fmt_record(record: TensorRecord | None, *, layout: bool) -> str:
    """A sha256 prefix, or dtype and shape when ``layout`` is set; '-' when absent."""
    if record is None:
        return "-"
    if layout:
        return f"{record.dtype}{_fmt_shape(record.shape)}"
    return short_id(record.sha256)


def render_verify_table(report: VerifyReport, *, show_all: bool) -> str:
    table: list[tuple[str, ...]] = [("status", "tensor", "expected", "actual")]
    for check in report.tensors:
        if show_all or check.status != "ok":
            layout = check.status == "reshaped"
            table.append(
                (
                    check.status,
                    check.name,
                    _fmt_record(check.expected, layout=layout),
                    _fmt_record(check.actual, layout=layout),
                )
            )
    return format_table(table)


def render_verify_summary(report: VerifyReport) -> str:
    counts = ", ".join(f"{count} {status}" for status, count in report.summary.items())
    return f"verify: {counts}; root {'ok' if report.root_match else 'MISMATCH'}"


def _open_checkpoint(arg: str) -> tuple[Checkpoint, dict[str, Any]]:
    """Open a path or a Hub checkpoint, with the record that names it in JSON output.

    A Hub argument is parsed before any client is built, and its pinned commit goes to stderr.
    ``Hub.describe`` refuses a shard name outside the portable rule before anything is
    fetched, which also keeps whitespace out of gguf-hash lines. The caller closes the
    checkpoint (``with``).
    """
    if not is_hub_checkpoint(arg):
        if is_gguf(arg):
            return load_gguf(arg), {"path": str(arg), "format": "gguf"}
        return load_checkpoint(arg), {"path": str(arg)}
    ref = parse_hub_argument(arg)
    hub = Hub()
    if ref.file is None:
        repo = _describe(hub, ref)
        print(f"pinned: {printable(repo.url)} at {repo.sha}", file=sys.stderr)
        return load_hub_checkpoint(hub, repo), {"hf": repo.to_dict()}
    repo = hub.describe_gguf(ref)
    print(f"pinned: {printable(repo.url)} at {repo.sha}", file=sys.stderr)
    ckpt = load_hub_gguf(hub, repo)
    for file in repo.files:
        digest = f"sha256 {file.sha256} checked" if file.sha256 else "no LFS sha256 listed"
        print(f"file: {printable(file.name)} ({file.nbytes:,} bytes, {digest})", file=sys.stderr)
    return ckpt, {"hf": repo.to_dict(), "format": "gguf"}


def _describe(hub: Hub, ref: HubRef) -> HubRepo:
    """``Hub.describe``; for a repository of GGUF files, list them on stderr and refuse."""
    try:
        return hub.describe(ref.repo_id, ref.revision)
    except GgufChoiceError as exc:
        print(f"GGUF checkpoints in {printable(exc.url)}:", file=sys.stderr)
        for choice in exc.choices:
            parts = "" if choice.parts == 1 else f" in {choice.parts} parts"
            if choice.listed != choice.parts:
                parts += f" ({choice.listed} listed)"
            print(f"  {printable(choice.name)}  {choice.nbytes:,} bytes{parts}", file=sys.stderr)
        raise


def cmd_hash(args: argparse.Namespace) -> int:
    if args.root_only and args.format != "json":
        args.usage_error("--root-only applies to the JSON manifest, not to --format gguf-hash")
    ckpt, _ = _open_checkpoint(args.path)
    with ckpt:
        if args.format == "gguf-hash":
            emit_text("\n".join(gguf_hash_lines(ckpt)) + "\n")
            return 0
        manifest = build_manifest(ckpt)
    if args.root_only:
        emit_text(manifest["root"] + "\n")  # LF on every platform, as the manifest
    else:
        emit_bytes(manifest_json(manifest))  # the bytes the store keeps, on every platform
    return 0


def render_hub_header(a: HubRepo, b: HubRepo, plan: PairPlan) -> list[str]:
    """Two lines naming the pinned commits and the shards a Hub-to-Hub diff skipped."""
    return [
        f"hub: a = {a.url} ({short_id(a.sha)}), b = {b.url} ({short_id(b.sha)})",
        f"shards: {len(a.files)} in a, {len(b.files)} in b; {len(plan.skipped)} byte-identical "
        f"on the Hub and skipped; fetched {len(plan.fetch_a)} from a and "
        f"{len(plan.fetch_b)} from b",
    ]


def _hub_side(
    repo: HubRepo, root: str | None, skipped: tuple[HubFile, ...], fetched: tuple[str, ...]
) -> dict[str, Any]:
    files = {"skipped": [file.to_dict() for file in skipped], "fetched": list(fetched)}
    return {"hf": repo.to_dict(), "root": root, "files": files}


def _report_diff(args: argparse.Namespace, run: PairRun) -> int:
    diffs = run.diffs
    summary = summarize(diffs)
    output = _diff_format(args)
    classification = classify_run(run) if args.classify else None
    if output != "table":
        payload: dict[str, Any] = {
            "a": run.a,
            "b": run.b,
            **({} if run.comparison is None else {"comparison": run.comparison}),
            "summary": summary,
            "tensors": tensor_dicts(run, include_unchanged=args.all or output == "agent"),
        }
        if run.features is not None:
            payload["features"] = run.features[1]
        if classification is not None:
            payload["classification"] = classification.to_dict()
        if output == "agent":
            emit_text(render_agent(payload, run.unchanged_files))
        else:
            emit_json(payload, allow_nan=False)
    else:
        for line in run.header:
            print(printable(line))
        print(render_summary(summary))
        if classification is not None:
            lines = classification_lines(classification)
            print(printable("classification: " + lines[0]))
        if any(d.status != "unchanged" for d in diffs) or args.all:
            print()
            print(render_diff_table(diffs, show_all=args.all))
    return 1 if summary["by_status"]["unchanged"] != summary["tensors"] else 0


def _diff_format(args: argparse.Namespace) -> str:
    """``table``, ``json``, or ``agent``; ``--json`` is ``--format json``."""
    if args.json and args.format not in (None, "json"):
        args.usage_error(f"--json and --format {args.format} ask for two formats")
    return "json" if args.json else (args.format or "table")


def _unchanged_files(ckpt: Checkpoint, diffs: list[TensorDiff]) -> dict:
    """The files of side b whose every tensor the diff reports unchanged."""
    status = {d.name_b or d.name: d.status for d in diffs}
    names = [
        file.path.name
        for file in ckpt.files
        if all(status.get(name) == "unchanged" for name in file.tensors)
    ]
    return {"label": "files of b whose tensors are all unchanged", "names": names,
            "total": len(ckpt.files)}  # fmt: skip


def _run_hub_pair(
    ref_a: HubRef, ref_b: HubRef, *, rank: bool, rank_cap: int, features: bool
) -> PairRun:
    """Two Hub repositories of safetensors shards: fetch only the shards that differ."""
    hub = Hub()
    a, b = _describe(hub, ref_a), _describe(hub, ref_b)

    def measure(ckpt_a: Checkpoint, ckpt_b: Checkpoint, diffs: list[TensorDiff]) -> object:
        return compute_features(ckpt_a, ckpt_b, diffs, rank_cap)

    result = diff_hub_pair(
        hub, a, b, rank=rank, rank_cap=rank_cap, inspect=measure if features else None
    )
    plan = result.plan
    measured = cast("tuple[dict[str, dict], dict] | None", result.inspected)
    return PairRun(
        result.diffs,
        _hub_side(a, result.root_a, plan.skipped, plan.fetch_a),
        _hub_side(b, result.root_b, plan.skipped, plan.fetch_b),
        render_hub_header(a, b, plan),
        measured,
        unchanged_files={
            "label": "shards byte-identical on the Hub (not fetched)",
            "names": [file.name for file in plan.skipped],
            "total": len(b.files),
        },
    )


def run_pair(arg_a: str, arg_b: str, *, rank: bool, rank_cap: int, features: bool) -> PairRun:
    """Diff two checkpoints (paths, Hub repositories, or GGUF files), with features if asked.

    Features are measured only when both sides are safetensors: a quantization comparison's
    deltas are against dequantised values, which the rounding-noise model does not describe.
    """
    if is_hub_checkpoint(arg_a) and is_hub_checkpoint(arg_b):
        ref_a, ref_b = parse_hub_argument(arg_a), parse_hub_argument(arg_b)
        if ref_a.file is None and ref_b.file is None:
            return _run_hub_pair(ref_a, ref_b, rank=rank, rank_cap=rank_cap, features=features)
    measured = None
    comparison = None
    ckpt_a, about_a = _open_checkpoint(arg_a)
    with ckpt_a:
        ckpt_b, about_b = _open_checkpoint(arg_b)
        with ckpt_b:
            digests_a = tensor_digests(ckpt_a)
            digests_b = tensor_digests(ckpt_b)
            if is_quantization_pair(ckpt_a, ckpt_b):
                diffs, comparison = compare_quantized(
                    ckpt_a, ckpt_b, digests_a, digests_b, rank=rank, rank_cap=rank_cap
                )
            else:
                diffs = diff_checkpoints(
                    ckpt_a, ckpt_b, digests_a, digests_b, rank=rank, rank_cap=rank_cap
                )
                if features:
                    measured = compute_features(ckpt_a, ckpt_b, diffs, rank_cap)
    if comparison is not None and ("hf" in about_a or "hf" in about_b):
        comparison = {**comparison, **_hub_comparison((ckpt_a, about_a), (ckpt_b, about_b))}
    a = {**about_a, "root": build_manifest(ckpt_a, digests_a)["root"], **_gguf_side(ckpt_a)}
    b = {**about_b, "root": build_manifest(ckpt_b, digests_b)["root"], **_gguf_side(ckpt_b)}
    header = [] if comparison is None else [render_comparison(comparison)]
    files = _unchanged_files(ckpt_b, diffs)
    return PairRun(diffs, a, b, header, measured, comparison, files)


def cmd_diff(args: argparse.Namespace) -> int:
    if args.features and not args.classify:
        args.usage_error("--features applies with --classify")
    _diff_format(args)  # refuse two formats before anything is read
    run = run_pair(args.a, args.b, rank=args.rank, rank_cap=args.rank_cap, features=args.features)
    return _report_diff(args, run)


def _gguf_side(ckpt: Checkpoint) -> dict[str, Any]:
    """The ``gguf`` key of a diff side that is a GGUF file; nothing for safetensors."""
    files = gguf_files(ckpt)
    return {"gguf": gguf_about(files)} if files else {}


def _hub_comparison(*sides: tuple[Checkpoint, dict[str, Any]]) -> dict[str, Any]:
    """What a quantization comparison with a Hub side compared: files, commits, file types.

    Per side: the files read (as the repository names them, or by base name for a path), the
    commit sha (None for a path), and a GGUF side's ``general.file_type`` (None otherwise).
    """
    files, revisions, quantizations = {}, {}, {}
    for label, (ckpt, about) in zip("ab", sides, strict=True):
        hf = about.get("hf")
        listed = hf.get("files") if hf else None
        files[label] = (
            [file["name"] for file in listed] if listed else [f.path.name for f in ckpt.files]
        )
        revisions[label] = hf["sha"] if hf else None
        parts = gguf_files(ckpt)
        quantizations[label] = parts[0].file_type if parts else None
    return {"files": files, "revisions": revisions, "quantizations": quantizations}


def render_comparison(comparison: dict) -> str:
    """One line saying that values were compared across a quantization, and how."""
    sides = " and ".join(comparison["gguf"])
    mapping = comparison["mapping"]
    names = f"names mapped by the {mapping} table" if mapping else "names compared as stored"
    return (
        f"quantization comparison: {sides} GGUF; {names}; values dequantised to "
        f"{comparison['values']} where bytes cannot be compared"
    )


def cmd_verify(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)  # before any fetch: a bad manifest costs no download
    ckpt, about = _open_checkpoint(args.checkpoint)
    with ckpt:
        report = verify_checkpoint(ckpt, manifest)
    if args.json:
        payload = {
            "checkpoint": str(args.checkpoint),
            **{key: about[key] for key in ("hf",) if key in about},
            **report.to_dict(include_ok=args.all),
        }
        emit_json(payload)
    else:
        print(render_verify_summary(report))
        if any(t.status != "ok" for t in report.tensors) or args.all:
            print()
            print(render_verify_table(report, show_all=args.all))
    return 0 if report.passed else 1


def render_dropped_extras(result: CommitResult, *, no_extras: bool) -> str | None:
    """The stderr line when the parent records extras and this commit records none, else None."""
    if result.extras or not result.parent_extras:
        return None
    flag = " (--no-extras)" if no_extras else ""
    return f"extras: the parent recorded {result.parent_extras}; this commit records none{flag}"


def render_skipped(skipped: Sequence[str]) -> str:
    """The stderr line naming the directory entries a commit did not record, with reasons."""
    return "skipped: " + ", ".join(printable(entry) for entry in skipped)


def _subject(message: object) -> str:
    """The first line of a commit message."""
    lines = str(message).splitlines()
    return lines[0] if lines else ""


def _log_row(store: ObjectStore, commit_id: str, record: dict) -> tuple[str, ...]:
    count = tensor_count(store.get_manifest(record["manifest"]))
    return (
        short_id(commit_id),
        str(record.get("time", "-")),
        record["operation"]["kind"],
        plural(count, "tensor"),
        _subject(record.get("message", "")),
    )


def _fmt_message(message: object) -> str:
    """'message: ' and the first line, then further lines indented (blank lines stay empty)."""
    first, *rest = printable(message).split("\\x0a")
    return "\n".join(
        [f"message: {first}".rstrip(), *(f"  {line}" if line else "" for line in rest)]
    )


def _fmt_operation(operation: dict) -> str:
    """The operation kind, followed by any extra keys as JSON."""
    extra = {key: value for key, value in operation.items() if key != "kind"}
    return operation["kind"] + (f" {json.dumps(extra, sort_keys=True)}" if extra else "")


def _fmt_tensor_counts(result: ShowResult) -> str:
    comparison = result.comparison
    if comparison is None or result.parent_id is None:
        return f"tensors: {result.tensor_count} (no parent)"
    counts = ", ".join(f"{count} {status}" for status, count in comparison.counts.items())
    return f"tensors: {result.tensor_count} ({counts} against {short_id(result.parent_id)})"


def _comparison_table(comparison: ManifestComparison | None) -> str:
    """Changed, added, and removed tensors, one per row; empty when there are none."""
    if comparison is None:
        return ""
    rows = [
        *(("changed", name) for name in comparison.changed),
        *(("added", name) for name in comparison.added),
        *(("removed", name) for name in comparison.removed),
    ]
    return format_table([("status", "tensor"), *rows]) if rows else ""


def render_show(result: ShowResult) -> str:
    record = result.record
    lines = [
        f"commit {result.commit_id}",
        f"parents: {' '.join(record['parents']) or 'none'}",
        f"manifest: {record['manifest']}",
        _fmt_tensor_counts(result),
        f"operation: {_fmt_operation(record['operation'])}",
        f"author: {printable(record.get('author', '-'))}",
        f"time: {printable(record.get('time', '-'))}",
        _fmt_message(record.get("message", "")),
        *(
            f"file: {entry['name']} ({entry['nbytes']:,} bytes, "
            f"{plural(len(entry['tensors']), 'tensor')}, sha256 {entry['sha256']})"
            for entry in record["files"]
        ),
        *(
            f"extra: {printable(entry['name'])} ({entry['nbytes']:,} bytes)"
            for entry in record.get("extras", [])
        ),
    ]
    table = _comparison_table(result.comparison)
    return "\n".join(lines) + (f"\n\n{table}" if table else "")


def render_checkout(result: CheckoutResult) -> str:
    extras = f" and {plural(len(result.extras), 'extra')}" if result.extras else ""
    count = len(result.replaced)
    replaced = f", replacing {plural(count, 'existing file')}" if count else ""
    head = (
        f"checked out {short_id(result.commit_id)} into {result.out_dir}: "
        f"{plural(len(result.files), 'file')}{extras}, {result.nbytes:,} bytes "
        f"in {result.seconds:.2f} s{replaced}"
    )
    written = (*result.files, *result.extras)
    rows = [
        (file.name, f"{file.nbytes:,} bytes", "sha256 ok", "replaced" if file.replaced else "")
        for file in written
    ]
    root = f"manifest root {result.manifest_root} verified against the files"
    return head + (f"\n{format_table(rows)}" if rows else "") + f"\n{root}"


def cmd_init(args: argparse.Namespace) -> int:
    repo = init_repository(args.dir)
    print(f"initialised an empty wvc repository in {repo / '.wvc'}")
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    store = open_repository(args.repo)
    result = commit_checkpoint(
        store,
        args.checkpoint,
        message=args.message,
        operation_kind=args.op,
        author=default_author() if args.author is None else args.author,
        parent_ref=args.parent,
        branch=args.branch,
        source=args.source,
        extras=not args.no_extras,
    )
    print(render_commit(result))
    if result.skipped:
        print(render_skipped(result.skipped), file=sys.stderr)
    dropped = render_dropped_extras(result, no_extras=args.no_extras)
    if dropped is not None:
        print(dropped, file=sys.stderr)
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    store = open_repository(args.repo)
    rows = [_log_row(store, commit_id, rec) for commit_id, rec in log_entries(store, limit=args.n)]
    if rows:
        print(format_table(rows))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    result = show_commit(open_repository(args.repo), args.ref)
    if args.json:
        emit_json(result.to_dict())
    else:
        print(render_show(result))
    return 0


def cmd_checkout(args: argparse.Namespace) -> int:
    result = checkout_commit(
        open_repository(args.repo), args.ref, args.out, force=args.force, overwrite=args.overwrite
    )
    print(render_checkout(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wvc",
        description="Version control primitives for model weights.",
        epilog=EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"wvc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_hash = sub.add_parser(
        "hash", help="print a tensor-level manifest (per-tensor SHA-256 and a root)"
    )
    p_hash.add_argument("path", help=CHECKPOINT_HELP)
    p_hash.add_argument("--root-only", action="store_true", help="print only the manifest root")
    p_hash.add_argument(
        "--format",
        choices=MANIFEST_FORMATS,
        default="json",
        help="json: the wvc manifest (default); gguf-hash: llama.cpp gguf-hash manifest lines",
    )
    p_hash.set_defaults(func=cmd_hash, usage_error=p_hash.error)

    p_diff = sub.add_parser("diff", help="compare two checkpoints tensor by tensor")
    p_diff.add_argument("a", help=f"base checkpoint: {CHECKPOINT_HELP}")
    p_diff.add_argument("b", help=f"other checkpoint: {CHECKPOINT_HELP}")
    p_diff.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p_diff.add_argument(
        "--format",
        choices=DIFF_FORMATS,
        help="table (default), json (as --json), or agent: a compact Markdown report for LLM "
        f"agents, the stable format {AGENT_FORMAT} (docs/agents-reading-diffs.md)",
    )
    p_diff.add_argument("--all", action="store_true", help="list unchanged tensors too")
    p_diff.add_argument("--rank", action="store_true", help="estimate the rank of each change")
    p_diff.add_argument(
        "--rank-cap",
        type=_count_arg,
        default=DEFAULT_RANK_CAP,
        help="skip the rank estimate when the touched submatrix has more elements than this",
    )
    p_diff.add_argument(
        "--classify",
        action="store_true",
        help="say what kind of change this is (see 'wvc classify'); added to the JSON record",
    )
    p_diff.add_argument("--features", action="store_true", help=FEATURES_HELP)
    p_diff.set_defaults(func=cmd_diff, usage_error=p_diff.error)
    add_classify_command(sub, run_pair, CHECKPOINT_HELP, _count_arg)

    p_verify = sub.add_parser(
        "verify", help="check a checkpoint against a manifest written by 'wvc hash'"
    )
    p_verify.add_argument("checkpoint", help=CHECKPOINT_HELP)
    p_verify.add_argument("manifest", help="manifest JSON written by 'wvc hash'")
    p_verify.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p_verify.add_argument("--all", action="store_true", help="list tensors that verify ok too")
    p_verify.set_defaults(func=cmd_verify)
    _add_repository_commands(sub)
    add_bisect_command(sub)
    add_fsck_command(sub)
    add_peft_commands(sub)
    add_remote_commands(sub)
    return parser


def _count_arg(text: str) -> int:
    """argparse type for a non-negative integer."""
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {text!r}")
    return value


def _add_repository_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_init = sub.add_parser("init", help="create a wvc repository (a .wvc directory)")
    p_init.add_argument("dir", nargs="?", default=".", help="directory (default: current)")
    p_init.set_defaults(func=cmd_init)

    p_commit = sub.add_parser("commit", help="store a checkpoint as a commit on the branch")
    p_commit.add_argument("checkpoint", help="a .safetensors file or a directory of shards")
    p_commit.add_argument("-m", "--message", required=True, help="commit message")
    p_commit.add_argument(
        "--op",
        default="unknown",
        choices=sorted(OPERATION_KINDS),
        metavar="KIND",
        help=f"operation kind, one of {', '.join(sorted(OPERATION_KINDS))} (default: unknown)",
    )
    p_commit.add_argument(
        "--parent", metavar="REF", help=f"parent commit (default: HEAD): {REF_HELP}"
    )
    p_commit.add_argument(
        "--branch", metavar="NAME", help="commit onto NAME (new or fast-forward); HEAD moves there"
    )
    p_commit.add_argument(
        "--author", metavar="NAME", help=f"author (default: ${AUTHOR_ENV}; one of them is needed)"
    )
    p_commit.add_argument(
        "--source",
        type=source_arg,
        metavar="SPEC",
        help="where the checkpoint came from: hf:OWNER/NAME[@REVISION], or other text as a path",
    )
    p_commit.add_argument(
        "--no-extras",
        action="store_true",
        help="record no extras: none of a directory's non-tensor files (config, tokenizer, "
        "licence, ...), and for a single file none carried from the parent",
    )
    add_repo_option(p_commit)
    p_commit.set_defaults(func=cmd_commit)

    p_log = sub.add_parser("log", help="list commits from HEAD along first parents")
    p_log.add_argument("-n", type=_count_arg, metavar="N", help="show at most N commits")
    add_repo_option(p_log)
    p_log.set_defaults(func=cmd_log)

    p_show = sub.add_parser("show", help="print a commit and compare it with its first parent")
    p_show.add_argument("ref", help=REF_HELP)
    p_show.add_argument("--json", action="store_true", help="emit JSON instead of text")
    add_repo_option(p_show)
    p_show.set_defaults(func=cmd_show)

    p_checkout = sub.add_parser("checkout", help="write a commit's files byte-for-byte")
    p_checkout.add_argument("ref", help=REF_HELP)
    p_checkout.add_argument("--out", required=True, metavar="DIR", help="output directory")
    p_checkout.add_argument(
        "--force",
        action="store_true",
        help="write into a non-empty DIR, replacing same-named shards and extras that an earlier "
        "checkout of this store wrote",
    )
    p_checkout.add_argument(
        "--overwrite",
        action="append",
        default=[],
        metavar="NAME",
        help="with --force, also replace the existing file NAME, whatever it holds (repeatable)",
    )
    add_repo_option(p_checkout)
    p_checkout.set_defaults(func=cmd_checkout)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; return its exit code.

    A wvc error, an OS error, running out of memory, and too deep a recursion print one
    ``wvc: ...`` line and exit 2; with ``WVC_DEBUG`` set they propagate with their traceback.
    Standard output closed by its reader (``wvc hash X | head``) exits 0 quietly, and Ctrl-C
    exits 130. Any other exception is a bug in wvc and propagates.
    """
    configure_streams()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError as exc:  # before OSError, which it derives from
        if _stdout_is_pipe():
            _silence_stdout()
            return 0
        if _debugging():  # standard output is no pipe, so some other pipe was closed
            raise
        return _report(exc)
    except KeyboardInterrupt:
        if _debugging():
            raise
        print("wvc: interrupted", file=sys.stderr)
        return INTERRUPTED_EXIT
    except MemoryError:  # a record may claim a tensor larger than this machine's memory
        if _debugging():
            raise
        print(f"wvc: out of memory while running {args.command}", file=sys.stderr)
        return 2
    except (WvcError, OSError, RecursionError) as exc:
        if _debugging():
            raise
        return _report(exc)


def _debugging() -> bool:
    """True when ``WVC_DEBUG`` is set to a non-empty value."""
    return bool(os.environ.get(DEBUG_ENV))


def _report(exc: BaseException) -> int:
    print(f"wvc: {printable(exc)}", file=sys.stderr)
    return 2


def _stdout_is_pipe() -> bool:
    """True when standard output is a pipe or a socket, which a reader can close."""
    try:
        mode = os.fstat(sys.stdout.fileno()).st_mode
    except (AttributeError, OSError, ValueError):  # a replaced stream, such as a StringIO
        return False
    return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)


def _silence_stdout() -> None:
    """Point standard output at the null device, so the interpreter's last flush cannot fail."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, sys.stdout.fileno())
    finally:
        os.close(devnull)
