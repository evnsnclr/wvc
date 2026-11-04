"""``wvc classify`` and what ``wvc diff --classify`` shares with it (docs/classification.md).

The checkpoint pair is opened and diffed by a runner ``wvc.cli`` passes in, so this module stays
a leaf among the CLI modules: it never imports ``wvc.cli``.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wvc.classify import Classification, classification_lines, classify_record
from wvc.classify_input import ClassifyError, load_record
from wvc.diff import DEFAULT_RANK_CAP, TensorDiff, summarize
from wvc.output import emit_json
from wvc.quote import printable

SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
REPO_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
FEATURES_HELP = (
    "read the changed tensors again and measure their spectra against the rounding floor "
    "(the rank of a LoRA merge, and whether the layers share one direction)"
)


@dataclass(frozen=True)
class PairRun:
    """A diffed pair: the diff, the JSON sides, the text header, and the features if measured."""

    diffs: list[TensorDiff]
    a: dict[str, Any]
    b: dict[str, Any]
    header: list[str]
    features: tuple[dict[str, dict], dict] | None = None
    comparison: dict | None = None  # a quantization comparison's record (spec 2.1)
    unchanged_files: dict | None = None  # for the agent view


PairRunner = Callable[..., PairRun]  # (a, b, *, rank, rank_cap, features) -> PairRun


def tensor_dicts(run: PairRun, *, include_unchanged: bool) -> list[dict]:
    """The record's tensor entries, each changed matrix with its ``features`` when measured."""
    measured = run.features[0] if run.features is not None else {}
    entries = []
    for item in run.diffs:
        if not include_unchanged and item.status == "unchanged":
            continue
        entry = item.to_dict()
        if item.name in measured:
            entry["features"] = measured[item.name]
        entries.append(entry)
    return entries


def classify_run(run: PairRun) -> Classification:
    """Classify a pair from every tensor of its diff, whether or not ``--all`` was given."""
    record: dict[str, Any] = {
        "summary": summarize(run.diffs),
        "tensors": tensor_dicts(run, include_unchanged=True),
    }
    if run.features is not None:
        record["features"] = run.features[1]
    if run.comparison is not None:
        record["comparison"] = run.comparison
    return classify_record(record, source="diff")


def side_argument(record: dict, key: str) -> str:
    """The checkpoint argument a record's side names: its path, or ``hf:REPO@SHA``."""
    side = record.get(key)
    if isinstance(side, dict) and isinstance(side.get("path"), str):
        return str(Path(side["path"]).expanduser())
    hub = side.get("hf") if isinstance(side, dict) else None
    if isinstance(hub, dict):
        repo, sha = hub.get("repo"), hub.get("sha")
        repo_ok = isinstance(repo, str) and REPO_PATTERN.fullmatch(repo) is not None
        if repo_ok and isinstance(sha, str) and SHA_PATTERN.fullmatch(sha):
            return f"hf:{repo}@{sha}"
    raise ClassifyError(f"diff record.{key}: names no checkpoint path or pinned Hub commit")


def cmd_classify(args: argparse.Namespace) -> int:
    run_pair: PairRunner = args.run_pair
    if len(args.inputs) == 1:
        record = load_record(args.inputs[0])
        if args.features:
            pair = (side_argument(record, "a"), side_argument(record, "b"))
            run = run_pair(*pair, rank=True, rank_cap=args.rank_cap, features=True)
            result = classify_run(run)
        else:
            result = classify_record(record, source=str(args.inputs[0]))
    elif len(args.inputs) == 2:
        a, b = args.inputs
        result = classify_run(
            run_pair(a, b, rank=True, rank_cap=args.rank_cap, features=args.features)
        )
    else:
        args.usage_error("give one diff record, or two checkpoints")
    if args.json:
        emit_json(result.to_dict(), allow_nan=False)
    else:
        for line in classification_lines(result):
            print(printable(line))
    return 0


def add_classify_command(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    run_pair: PairRunner,
    checkpoint_help: str,
    count_arg: Callable[[str], int],
) -> None:
    parser = sub.add_parser(
        "classify",
        help="say what kind of change a diff shows, with the evidence and the rule that fired",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help=f"a diff JSON record (wvc diff --rank --json), or two checkpoints: {checkpoint_help}",
    )
    parser.add_argument("--json", action="store_true", help="emit the classification as JSON")
    parser.add_argument(
        "--features",
        action="store_true",
        help=FEATURES_HELP + "; with a record, the pair it names is read again",
    )
    parser.add_argument(
        "--rank-cap",
        type=count_arg,
        default=DEFAULT_RANK_CAP,
        help="skip rank estimates and keep no whole delta above this many elements",
    )
    parser.set_defaults(func=cmd_classify, run_pair=run_pair, usage_error=parser.error)
