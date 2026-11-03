"""Command-line entry point ``wvc merge-adapter`` (spec section 5.3).

``wvc.cli.build_parser`` registers the command through ``add_peft_commands``. This module must
not import ``wvc.cli``, which imports it; what they share is in ``wvc.cli_common``.
"""

from __future__ import annotations

import argparse

from wvc.cli_common import REF_HELP, add_repo_option, render_commit
from wvc.merge_adapter import merge_loaded_adapter
from wvc.peft import CONFIG_NAME, WEIGHTS_NAME, Adapter, load_adapter
from wvc.repo import AUTHOR_ENV, default_author, open_repository


def render_adapter(adapter: Adapter) -> str:
    """The line ``merge-adapter`` prints after the commit summary."""
    return (
        f"adapter targets merged: {len(adapter.targets)} "
        f"(r {adapter.r}, lora_alpha {adapter.lora_alpha}, scale {adapter.scale!r})"
    )


def cmd_merge_adapter(args: argparse.Namespace) -> int:
    store = open_repository(args.repo)
    adapter = load_adapter(args.adapter_dir)
    result = merge_loaded_adapter(
        store,
        args.base_ref,
        adapter,
        args.out,
        message=args.message,
        author=default_author() if args.author is None else args.author,
        branch=args.branch,
    )
    print(render_commit(result))
    print(render_adapter(adapter))
    return 0


def add_peft_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``merge-adapter`` on the ``wvc`` subcommand parsers."""
    parser = subparsers.add_parser(
        "merge-adapter",
        help="merge a PEFT LoRA adapter into a commit and commit the merged checkpoint",
    )
    parser.add_argument("base_ref", metavar="BASE_REF", help=f"the base commit: {REF_HELP}")
    parser.add_argument(
        "adapter_dir",
        metavar="ADAPTER_DIR",
        help=f"a PEFT adapter directory holding {CONFIG_NAME} and {WEIGHTS_NAME}",
    )
    parser.add_argument(
        "--out", required=True, metavar="DIR", help="a new or empty directory for the merge"
    )
    parser.add_argument("-m", "--message", required=True, help="commit message")
    parser.add_argument(
        "--author",
        metavar="NAME",
        help=f"author (default: ${AUTHOR_ENV}; one of the two is required)",
    )
    parser.add_argument(
        "--branch",
        metavar="NAME",
        help="commit onto branch NAME (new, or one the merge fast-forwards) and switch HEAD to it; "
        "needed when BASE_REF is not the head of the current branch",
    )
    add_repo_option(parser)
    parser.set_defaults(func=cmd_merge_adapter)
