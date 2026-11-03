"""The ``wvc fsck`` command."""

from __future__ import annotations

import argparse

from wvc.cli_common import add_repo_option
from wvc.fsck import FsckReport, fsck
from wvc.output import emit_json
from wvc.quote import printable
from wvc.repo import open_repository


def render_fsck(report: FsckReport) -> str:
    head = (
        f"fsck: {report.payloads} payloads, {report.objects} objects "
        f"({report.tensors_decoded} decoded), {report.manifests} manifests, "
        f"{report.commits} commits, {report.refs} refs; "
        f"{'ok' if report.ok else f'{len(report.problems)} problems'}"
    )
    return "\n".join([head, *map(printable, report.problems)])  # ref names are file names


def cmd_fsck(args: argparse.Namespace) -> int:
    report = fsck(open_repository(args.repo), decode=not args.no_decode)
    if args.json:
        emit_json(report.to_dict())
    else:
        print(render_fsck(report))
    return 0 if report.ok else 1


def add_fsck_command(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser("fsck", help="check every payload, object, manifest, commit, and ref")
    add_repo_option(parser)
    parser.add_argument(
        "--no-decode", action="store_true", help="skip rebuilding tensors through their chains"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.set_defaults(func=cmd_fsck)
