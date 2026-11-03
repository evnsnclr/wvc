"""What the command-line modules share: help text, the ``--repo`` option, and text rendering.

A leaf among the CLI modules: it must not import ``wvc.cli`` or any ``wvc.cli_*`` module, so
that each of them can import it at module top.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import TYPE_CHECKING

from wvc.ids import short
from wvc.quote import printable

if TYPE_CHECKING:
    from wvc.repo import CommitResult

REF_HELP = "HEAD, a branch, a commit id, or a commit id prefix of at least 7 hex characters"
REPO_HELP = "repository directory (default: the nearest one at or above the current directory)"


def add_repo_option(parser: argparse.ArgumentParser) -> None:
    """Add ``--repo DIR`` to a command that works on a repository."""
    parser.add_argument("--repo", metavar="DIR", help=REPO_HELP)


def format_table(rows: Sequence[Sequence[str]]) -> str:
    """Left-align columns two spaces apart, escaped, with no trailing whitespace; '' if empty."""
    if not rows:
        return ""
    cells = [[printable(cell) for cell in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(cells[0]))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in cells
    )


def plural(count: int, noun: str) -> str:
    """``count`` with thousands separators and ``noun``, with an 's' unless the count is 1."""
    return f"{count:,} {noun}" + ("" if count == 1 else "s")


short_id = short  # the CLI's name for ``wvc.ids.short``


def render_commit(result: CommitResult) -> str:
    """What ``commit`` and ``merge-adapter`` print about the commit they made."""
    parent = "no parent" if result.parent_id is None else f"parent {short_id(result.parent_id)}"
    forms = ", ".join(f"{count} {form}" for form, count in result.stored_delta.items())
    delta = f"{sum(result.stored_delta.values())} stored as delta" + (f": {forms}" if forms else "")
    carried = " carried from the parent" if result.extras_carried else ""
    extras = f"extras: {plural(result.extras, 'file')}{carried}, {result.extras_bytes:,} bytes"
    return "\n".join(
        (
            f"commit {result.commit_id} ({result.branch}, {parent})",
            f"tensors: {result.tensors} ({result.reused} reused by hash, "
            f"{result.stored_full} stored full, {delta})",
            *((extras,) if result.extras else ()),
            f"payload bytes written: {result.payload_bytes:,} in {result.seconds:.2f} s",
        )
    )
