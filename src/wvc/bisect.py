"""``wvc bisect``: find the first bad commit on a first-parent chain (spec section 4.1).

The candidates are the commits after GOOD up to and including BAD, oldest first, found by
walking first parents back from BAD, which must descend from GOOD. Each tested commit is
checked out byte-for-byte into ``WORKDIR/<first 12 hex characters of its id>`` and a test
command runs on it through the shell: exit 0 means good, exit 1 means bad, and any other exit
stops the run. ``bisect`` binary-searches the candidates, assuming one good-to-bad transition;
``sweep`` tests every candidate in order and reports whether the curve is monotone, which is
how that assumption is checked. A checkout is removed after its test, and a temporary WORKDIR
at the end of the run, unless ``keep``.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

from wvc.checkout import checkout_commit
from wvc.ids import short
from wvc.quote import printable
from wvc.records import StoreError
from wvc.repo import log_entries
from wvc.store import ObjectStore

GOOD, BAD = "good", "bad"
RESULT_OF_EXIT = {0: GOOD, 1: BAD}  # any other exit code stops the run
CHECKOUT_ENV, COMMIT_ENV = "WVC_CHECKOUT", "WVC_COMMIT"
WORKDIR_PREFIX = "wvc-bisect-"
STDERR_FD = 2  # where a test's standard output goes (see ``run_test``)
Shell = Literal["sh", "cmd"]
Mode = Literal["bisect", "sweep"]
# What cmd.exe cannot take literally inside double quotes: '"' ends them, '%' expands variables
# there (and '!' does when delayed expansion is on), and a line break ends the command.
CMD_REFUSED = frozenset('"%!\r\n')


class BisectError(StoreError):
    """A bisect cannot start or go on.

    Raised when BAD does not descend from GOOD, when a test exits with neither 0 nor 1, and when
    the test command or the working directory is unusable.
    """


@dataclass(frozen=True)
class BisectStep:
    """One tested commit."""

    commit: str
    result: str  # GOOD or BAD
    exit_code: int
    test_seconds: float
    checkout_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BisectResult:
    """What a bisect or a sweep found, with every test it ran."""

    first_bad: str | None  # None when every candidate is good
    candidates: tuple[str, ...]  # oldest first
    steps: tuple[BisectStep, ...]  # in the order the tests ran
    mode: Mode
    workdir: str | None  # None when the temporary workdir was removed
    monotone: bool | None = None  # sweep only: no bad commit is followed by a good one

    @property
    def tests(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "first_bad": self.first_bad,
            "tests": self.tests,
            "monotone": self.monotone,
            "candidates": list(self.candidates),
            "steps": [step.to_dict() for step in self.steps],
            "workdir": self.workdir,
        }


def candidates(store: ObjectStore, good_ref: str, bad_ref: str) -> list[str]:
    """The commits after GOOD up to and including BAD along BAD's first parents, oldest first.

    BisectError when GOOD and BAD are the same commit, or when the first-parent chain from BAD
    ends without reaching GOOD.
    """
    good, bad = store.resolve(good_ref), store.resolve(bad_ref)
    if good == bad:
        raise BisectError(f"GOOD and BAD are the same commit {good}; BAD must descend from GOOD")
    newest_first: list[str] = []
    for commit_id, _ in log_entries(store, bad):
        if commit_id == good:
            return newest_first[::-1]
        newest_first.append(commit_id)
    raise BisectError(
        f"GOOD {good_ref!r} is not an ancestor of BAD {bad_ref!r} along first parents; "
        "BAD must descend from GOOD"
    )


def quote_argument(value: str, shell: Shell | None = None) -> str:
    """``value`` as one argument of ``shell`` (default: the platform shell, ``cmd`` on Windows).

    ``sh``: ``shlex.quote``. ``cmd``: always in double quotes, inside which ``cmd.exe`` takes
    ``&``, ``|``, ``<``, ``>``, ``^``, ``(``, and ``)`` literally, with trailing backslashes
    doubled so that the program's argument parser does not read the closing quote as escaped.
    BisectError for a value ``cmd.exe`` cannot be given: one holding ``"`` (no escape exists
    inside quotes), ``%`` (expands even inside quotes, as ``!`` does under delayed expansion),
    ``!``, or a line break.
    """
    if shell is None:
        shell = "cmd" if os.name == "nt" else "sh"
    if shell == "sh":
        return shlex.quote(value)
    if shell != "cmd":
        raise ValueError(f"unknown shell {shell!r}; expected 'sh' or 'cmd'")
    refused = sorted(set(value) & CMD_REFUSED)
    if refused:
        listed = ", ".join(repr(ch) for ch in refused)
        raise BisectError(
            f"cannot pass the checkout directory {printable(value)} to cmd.exe: it holds {listed}, "
            "which cmd.exe does not take literally even inside double quotes; "
            "choose a --workdir without them"
        )
    trailing = len(value) - len(value.rstrip("\\"))
    return '"' + value + "\\" * trailing + '"'


def run_test(command: str, checkout_dir: str | Path, commit_id: str) -> tuple[int, float]:
    """Run ``command`` through the shell on one checkout; return its exit code and seconds.

    The command gets the parent environment plus ``WVC_CHECKOUT`` (the directory) and
    ``WVC_COMMIT`` (the id), and the directory, quoted for the platform shell by
    ``quote_argument``, as its last argument. Its standard output goes to
    standard error, so that what wvc prints itself (such as ``--json``) stays parseable. A
    non-zero exit code is returned, never raised.
    """
    directory = str(checkout_dir)
    env = {**os.environ, CHECKOUT_ENV: directory, COMMIT_ENV: commit_id}
    started = time.perf_counter()
    # The test command is the user's own, run through their shell.
    completed = subprocess.run(  # noqa: S602
        f"{command} {quote_argument(directory)}",
        shell=True,
        env=env,
        stdout=STDERR_FD,
        check=False,
    )
    return completed.returncode, time.perf_counter() - started


def bisect(
    store: ObjectStore,
    good_ref: str,
    bad_ref: str,
    command: str,
    *,
    workdir: str | Path | None = None,
    keep: bool = False,
) -> BisectResult:
    """Binary-search the candidates for the first bad commit.

    Invariant: the candidates below ``lo`` are good and those at or above ``hi`` are bad. ``hi``
    starts past BAD, so BAD itself is tested when every earlier test finds good, and a run that
    finds no bad candidate reports ``first_bad`` None. Assumes one good-to-bad transition;
    ``sweep`` checks that assumption.
    """
    commits = _prepare(store, good_ref, bad_ref, command)
    steps: list[BisectStep] = []
    with _workspace(store, workdir, keep=keep) as root:
        lo, hi = 0, len(commits)
        while lo < hi:
            mid = (lo + hi) // 2
            step = _check_out_and_test(store, command, root, commits[mid], keep=keep)
            steps.append(step)
            if step.result == GOOD:
                lo = mid + 1  # every candidate up to mid is good
            else:
                hi = mid  # every candidate from mid on is bad
    first_bad = commits[lo] if lo < len(commits) else None
    workdir_left = _reported(root, workdir, keep=keep)
    return BisectResult(first_bad, tuple(commits), tuple(steps), "bisect", workdir_left)


def sweep(
    store: ObjectStore,
    good_ref: str,
    bad_ref: str,
    command: str,
    *,
    workdir: str | Path | None = None,
    keep: bool = False,
) -> BisectResult:
    """Test every candidate, oldest first.

    Reports the first bad candidate and whether the curve is monotone, that is whether no bad
    commit is followed by a good one.
    """
    commits = _prepare(store, good_ref, bad_ref, command)
    with _workspace(store, workdir, keep=keep) as root:
        steps = tuple(
            _check_out_and_test(store, command, root, commit_id, keep=keep) for commit_id in commits
        )
    first_bad = next((step.commit for step in steps if step.result == BAD), None)
    monotone = not any(a.result == BAD and b.result == GOOD for a, b in pairwise(steps))
    workdir_left = _reported(root, workdir, keep=keep)
    return BisectResult(first_bad, tuple(commits), steps, "sweep", workdir_left, monotone)


def _prepare(store: ObjectStore, good_ref: str, bad_ref: str, command: object) -> list[str]:
    """Check the test command, then return the candidates."""
    if not isinstance(command, str) or not command.strip():
        raise BisectError("the test command is empty")
    return candidates(store, good_ref, bad_ref)


@contextmanager
def _workspace(
    store: ObjectStore, workdir: str | Path | None, *, keep: bool
) -> Generator[Path, None, None]:
    """The directory checkouts go into.

    ``workdir`` when given (created when missing, never removed), else a new temporary
    directory, which is removed on the way out unless ``keep``. BisectError, before any
    checkout, when the platform shell cannot be given the directory (see ``quote_argument``).
    """
    if workdir is None:
        root = Path(tempfile.mkdtemp(prefix=WORKDIR_PREFIX))
    else:
        root = _given_workdir(store, Path(workdir))
    try:
        quote_argument(str(root))  # a checkout directory adds only hex characters to this
        yield root
    finally:
        if workdir is None and not keep and root.is_dir():
            shutil.rmtree(root)


def _given_workdir(store: ObjectStore, workdir: Path) -> Path:
    """``workdir`` made absolute and created when missing.

    Refused inside the store and when it exists but is not a directory.
    """
    root = workdir.absolute()
    quote_argument(str(root))  # refused here, before the directory is created
    if root.resolve().is_relative_to(store.dir.resolve()):
        raise BisectError(f"refusing to check out into the store itself ({root})")
    if os.path.lexists(root) and not root.is_dir():
        raise BisectError(f"workdir {root} exists and is not a directory")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _check_out_and_test(
    store: ObjectStore, command: str, root: Path, commit_id: str, *, keep: bool
) -> BisectStep:
    """Check out one commit into a new directory under ``root`` and run the test on it.

    The directory is removed after the test unless ``keep``; then BisectError when the test
    exited with neither 0 nor 1. A directory that already exists is refused, so that a run
    never removes anything it did not create.
    """
    directory = root / short(commit_id)
    if os.path.lexists(directory):
        raise BisectError(f"{directory} already exists; bisect checks out into new directories")
    try:
        checkout = checkout_commit(store, commit_id, directory)
        exit_code, test_seconds = run_test(command, directory, commit_id)
    finally:
        if not keep and directory.is_dir():
            shutil.rmtree(directory)
    result = RESULT_OF_EXIT.get(exit_code)
    if result is None:
        raise BisectError(
            f"the test command exited with {exit_code} on commit {commit_id}; "
            "0 means good, 1 means bad, and any other exit stops the run"
        )
    return BisectStep(commit_id, result, exit_code, test_seconds, checkout.seconds)


def _reported(root: Path, workdir: str | Path | None, *, keep: bool) -> str | None:
    """The workdir a result names: None for a temporary one that was removed."""
    return None if workdir is None and not keep else str(root)
