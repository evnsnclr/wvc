"""Command-line entry points for model cards and remotes (spec sections 5.1 and 5.2).

``wvc card``, ``wvc remote add``, ``wvc remote list``, ``wvc push``, ``wvc pull``, and
``wvc clone``, plus the argument type of ``wvc commit --source``. ``wvc.cli`` registers the
commands through ``add_remote_commands`` and turns errors into exit code 2.
"""

from __future__ import annotations

import argparse

from wvc.card import model_card, write_manifest
from wvc.cli_common import REF_HELP, add_repo_option, plural, short_id
from wvc.output import emit_text
from wvc.quote import printable
from wvc.records import StoreError
from wvc.remote import DEFAULT_REMOTE, TransferResult, add_remote, clone, list_remotes, pull, push
from wvc.repo import open_repository
from wvc.sources import parse_source

URL_HELP = "a directory path or hf://OWNER/NAME[@REVISION]"


def source_arg(text: str) -> dict:
    """argparse type of ``wvc commit --source``: the parsed source, or a usage error."""
    try:
        return parse_source(text)
    except StoreError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def render_transfer(result: TransferResult, *, verb: str, ref: str) -> str:
    """Two lines: what moved, then how the receiving side's ``ref`` changed."""
    direction = "to" if verb == "pushed" else "from"
    counts = ", ".join(
        (
            plural(result.commits, "commit"),
            plural(result.objects, "object"),
            plural(result.payloads, "payload"),
        )
    )
    moved = f"{result.nbytes:,} bytes in {result.seconds:.2f} s"
    change = (
        f"{short_id(result.previous)} -> {short_id(result.head)}"
        if result.updated
        else f"up to date at {short_id(result.head)}"
    )
    first = f"{verb} {result.branch} {direction} {result.remote}: {counts}, {moved}"
    return f"{first}\n{ref}: {change}"


def cmd_remote_add(args: argparse.Namespace) -> int:
    url = add_remote(open_repository(args.repo), args.name, args.url)
    print(f"added remote {printable(args.name)}: {printable(url)}")
    return 0


def cmd_remote_list(args: argparse.Namespace) -> int:
    rows = [
        (printable(name), printable(url))  # config.json may be edited by hand
        for name, url in list_remotes(open_repository(args.repo)).items()
    ]
    width = max((len(name) for name, _ in rows), default=0)
    for name, url in rows:
        print(f"{name.ljust(width)}  {url}")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    result = push(open_repository(args.repo), args.remote, args.branch)
    print(render_transfer(result, verb="pushed", ref=f"{result.remote} {result.branch}"))
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    result = pull(open_repository(args.repo), args.remote, args.branch)
    print(render_transfer(result, verb="pulled", ref=result.branch))
    return 0


def cmd_clone(args: argparse.Namespace) -> int:
    result = clone(args.url, args.dir)
    print(f"cloned {printable(args.url)} into {printable(args.dir)}")
    print(render_transfer(result, verb="pulled", ref=result.branch))
    return 0


def cmd_card(args: argparse.Namespace) -> int:
    store = open_repository(args.repo)
    result = model_card(store, args.ref)
    if args.manifest_out is not None:
        write_manifest(store, result.manifest_root, args.manifest_out)
    emit_text(result.yaml)
    return 0


def add_remote_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register card, remote (add, list), push, pull, and clone."""
    p_card = sub.add_parser("card", help="print model-card YAML front matter for a commit")
    p_card.add_argument("ref", help=REF_HELP)
    p_card.add_argument(
        "--manifest-out",
        metavar="PATH",
        help="also write the manifest JSON, as 'wvc hash' prints it, to PATH",
    )
    add_repo_option(p_card)
    p_card.set_defaults(func=cmd_card)
    _add_remote_parser(sub)
    for name, func, text in (
        ("push", cmd_push, "send a branch's new commits to a remote; fast-forward its branch"),
        ("pull", cmd_pull, "fetch a branch's new commits from a remote; fast-forward locally"),
    ):
        parser = sub.add_parser(name, help=text)
        parser.add_argument(
            "remote", nargs="?", default=DEFAULT_REMOTE, help=f"default: {DEFAULT_REMOTE}"
        )
        parser.add_argument("branch", nargs="?", help="default: the current branch")
        add_repo_option(parser)
        parser.set_defaults(func=func)
    p_clone = sub.add_parser("clone", help="init DIR, add URL as remote origin, and pull main")
    p_clone.add_argument("url", help=URL_HELP)
    p_clone.add_argument("dir", help="directory for the new repository")
    p_clone.set_defaults(func=cmd_clone)


def _add_remote_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_remote = sub.add_parser("remote", help="add or list remotes")
    actions = p_remote.add_subparsers(dest="remote_command", required=True)
    p_add = actions.add_parser("add", help="record a remote in .wvc/config.json")
    p_add.add_argument("name", help="remote name, for example origin")
    p_add.add_argument("url", help=URL_HELP)
    add_repo_option(p_add)
    p_add.set_defaults(func=cmd_remote_add)
    p_list = actions.add_parser("list", help="print each remote's name and URL")
    add_repo_option(p_list)
    p_list.set_defaults(func=cmd_remote_list)
