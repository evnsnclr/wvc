"""``wvc bisect`` on the command line (spec section 4.1); ``wvc.cli`` registers it.

This module must not import ``wvc.cli``, which imports it.
"""

from __future__ import annotations

import argparse

from wvc.bisect import BisectResult, bisect, sweep
from wvc.cli_common import REF_HELP, add_repo_option, format_table, short_id
from wvc.output import emit_json
from wvc.repo import open_repository

TEST_HELP = (
    "shell command run on each tested checkout, with the checkout directory appended as its "
    "last argument and in $WVC_CHECKOUT, and the commit id in $WVC_COMMIT; exit 0 means good, "
    "1 means bad, and any other exit stops the run; its output goes to standard error"
)
NOT_MONOTONE = "monotone: no (a bad commit is followed by a good one)"


def add_bisect_command(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``wvc bisect GOOD BAD --test CMD`` and its options."""
    parser = sub.add_parser(
        "bisect", help="find the first bad commit between GOOD and BAD with a test command"
    )
    parser.add_argument("good", metavar="GOOD", help=f"a good commit: {REF_HELP}")
    parser.add_argument("bad", metavar="BAD", help=f"a descendant of GOOD: {REF_HELP}")
    parser.add_argument("--test", required=True, metavar="CMD", help=TEST_HELP)
    add_repo_option(parser)
    parser.add_argument(
        "--workdir",
        metavar="DIR",
        help="directory for the checkouts (default: a new temporary directory)",
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep every checkout and the working directory"
    )
    parser.add_argument(
        "--sweep", action="store_true", help="test every candidate in order instead of bisecting"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.set_defaults(func=cmd_bisect)


def cmd_bisect(args: argparse.Namespace) -> int:
    """Run a bisect or a sweep and print what it found; 0 whatever it found."""
    store = open_repository(args.repo)
    search = sweep if args.sweep else bisect
    result = search(store, args.good, args.bad, args.test, workdir=args.workdir, keep=args.keep)
    if args.json:
        emit_json(result.to_dict())
    else:
        print(render_bisect(result, keep=args.keep))
    return 0


def render_bisect(result: BisectResult, *, keep: bool) -> str:
    """One line per test, the first bad commit, the number of tests, and for a sweep whether
    the curve is monotone; with ``keep``, where the checkouts are."""
    rows = [
        (
            step.result,
            short_id(step.commit),
            f"test {step.test_seconds:.2f} s",
            f"checkout {step.checkout_seconds:.2f} s",
        )
        for step in result.steps
    ]
    found = (
        "every candidate is good" if result.first_bad is None else f"first bad: {result.first_bad}"
    )
    tests = f"tests: {result.tests} of {len(result.candidates)} candidates"
    monotone = (
        [] if result.monotone is None else ["monotone: yes" if result.monotone else NOT_MONOTONE]
    )
    kept = [f"checkouts kept in {result.workdir}"] if keep else []
    table = [format_table(rows)] if rows else []
    return "\n".join([*table, found, tests, *monotone, *kept])
