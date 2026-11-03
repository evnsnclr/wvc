"""wvc bisect over a first-parent chain: binary search, sweep, cleanup, and the CLI."""

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from helpers import commit, write_safetensors

from wvc import bisect as bisect_module
from wvc.bisect import (
    BisectError,
    BisectResult,
    BisectStep,
    bisect,
    candidates,
    quote_argument,
    run_test,
    sweep,
)
from wvc.cli import main
from wvc.cli_bisect import render_bisect
from wvc.records import StoreError

FILE = "model.safetensors"
THRESHOLD = 4  # the check script calls a checkout bad when w[0] exceeds this
FIELDS = {"commit", "result", "exit_code", "test_seconds", "checkout_seconds"}
KEYS = ["mode", "first_bad", "tests", "monotone", "candidates", "steps", "workdir"]
STEP_LINE = re.compile(r"(good|bad)\s+([0-9a-f]{12})  test \d+\.\d\d s\s+checkout \d+\.\d\d s")

# The test command of these tests. Its arguments are a threshold and the checkout directory,
# which bisect appends; it exits 1 when w[0] in the checkout exceeds the threshold, 0 when not,
# and 3 when given a flag. With BISECT_LOG set, it appends WVC_COMMIT, WVC_CHECKOUT, and its
# last argument to that file, separated by tabs.
SCRIPT = """\
import os
import sys

from wvc.safetensors_io import load_checkpoint

*flags, threshold, checkout = sys.argv[1:]
if flags:
    sys.exit(3)
if "BISECT_LOG" in os.environ:
    with open(os.environ["BISECT_LOG"], "a", encoding="utf-8") as log:
        print(os.environ["WVC_COMMIT"], os.environ["WVC_CHECKOUT"], checkout, sep="\\t", file=log)
value = float(load_checkpoint(checkout).as_float32("w").reshape(-1)[0])
sys.exit(1 if value > float(threshold) else 0)
"""


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "check_w.py"
    path.write_text(SCRIPT, encoding="utf-8")
    return path


@pytest.fixture
def chain(store, tmp_path):
    """GOOD (w[0] = 0), then seven commits in which w[0] grows by one, from 1 to 7."""
    return commit_values(store, tmp_path / "revisions", range(8))


@pytest.fixture
def temporary(tmp_path, monkeypatch):
    """The directory in which tempfile, and so the default workdir, creates directories."""
    path = tmp_path / "tmp"
    path.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(path))
    return path


def join(argv):
    """``argv`` as one command line of the platform shell (cmd.exe on Windows)."""
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def command(script, *args):
    """A shell command that runs ``script`` with ``args`` under this interpreter."""
    return join([sys.executable, str(script), *map(str, args)])


def commit_value(store, directory, value, parent=None, branch=None):
    """Commit a tiny checkpoint in which w[0] = value; return the commit id."""
    directory.mkdir(parents=True)
    tensors = {
        "w": ("F32", np.array([value, 0.5, -0.5, 2.0], dtype=np.float32)),
        "b": ("F32", np.ones(3, dtype=np.float32)),
    }
    path = write_safetensors(directory / FILE, tensors)
    return commit(store, path, f"w[0] = {value}", parent_ref=parent, branch=branch).commit_id


def commit_values(store, root, values):
    """One commit per value on one first-parent chain; the ids, oldest first."""
    return [commit_value(store, root / f"rev-{i}", value) for i, value in enumerate(values)]


def outcomes(result):
    return [(step.commit, step.result, step.exit_code) for step in result.steps]


def run(capsys, *argv):
    """Run the CLI; return its exit code, stdout, and stderr."""
    code = main([str(arg) for arg in argv])
    out, err = capsys.readouterr()
    return code, out, err


def test_candidates_follow_first_parents_from_bad_back_to_good(store, chain):
    assert candidates(store, chain[0], chain[-1]) == chain[1:]
    assert candidates(store, chain[2][:7], "HEAD") == chain[3:]
    assert candidates(store, chain[6], "main") == [chain[7]]


def test_bad_must_descend_from_good(store, chain, script, tmp_path):
    side = commit_value(store, tmp_path / "side", 9, parent=chain[1], branch="side")
    assert candidates(store, chain[1], side) == [side]
    for good, bad in ((chain[5], chain[2]), (chain[3], side)):
        with pytest.raises(BisectError, match="is not an ancestor of BAD"):
            candidates(store, good, bad)
    with pytest.raises(BisectError, match="the same commit"):
        candidates(store, chain[2], chain[2][:7])
    work = tmp_path / "work"
    for search in (bisect, sweep):
        with pytest.raises(BisectError, match="is not an ancestor of BAD"):
            search(store, chain[3], side, command(script, THRESHOLD), workdir=work)
        with pytest.raises(BisectError, match="command is empty"):
            search(store, chain[0], chain[1], " ", workdir=work)
    assert not work.exists()  # refused before anything was checked out


def test_bisect_finds_the_first_bad_commit_in_three_tests(store, chain, script, temporary):
    result = bisect(store, chain[0], chain[-1], command(script, THRESHOLD))
    assert (result.mode, result.first_bad, result.candidates) == ("bisect", chain[5], (*chain[1:],))
    assert result.tests == len(result.steps) <= 3
    assert outcomes(result) == [(chain[4], "good", 0), (chain[6], "bad", 1), (chain[5], "bad", 1)]
    assert all(step.test_seconds > 0 and step.checkout_seconds > 0 for step in result.steps)
    assert (result.monotone, result.workdir) == (None, None)
    assert list(temporary.iterdir()) == []  # the temporary workdir is gone


def test_bisect_when_every_candidate_is_good_or_bad(store, chain, script, temporary):
    every_good = bisect(store, chain[0], chain[-1], command(script, 100))
    assert every_good.first_bad is None
    assert every_good.tests == 3
    assert outcomes(every_good)[-1] == (chain[7], "good", 0)  # BAD itself tested good
    every_bad = bisect(store, chain[0], chain[-1], command(script, 0))
    assert every_bad.first_bad == chain[1]
    assert every_bad.tests == 3
    one = bisect(store, chain[6], chain[7], command(script, THRESHOLD))
    assert (one.first_bad, outcomes(one)) == (chain[7], [(chain[7], "bad", 1)])


def test_sweep_tests_every_candidate_and_agrees_with_bisect(store, chain, script, temporary):
    test = command(script, THRESHOLD)
    result = sweep(store, chain[0], chain[-1], test)
    assert (result.mode, result.first_bad, result.monotone) == ("sweep", chain[5], True)
    assert result.tests == 7
    assert result.candidates == (*chain[1:],)
    assert outcomes(result) == [
        *((commit_id, "good", 0) for commit_id in chain[1:5]),
        *((commit_id, "bad", 1) for commit_id in chain[5:]),
    ]
    assert result.first_bad == bisect(store, chain[0], chain[-1], test).first_bad
    assert result.workdir is None
    assert list(temporary.iterdir()) == []
    every_bad = sweep(store, chain[5], chain[7], command(script, 0))
    assert (every_bad.first_bad, every_bad.monotone, every_bad.tests) == (chain[6], True, 2)


def test_sweep_flags_a_curve_that_goes_up_down_and_up_again(store, script, tmp_path, temporary):
    ids = commit_values(store, tmp_path / "revisions", [0, 2, 6, 3, 7])
    test = command(script, THRESHOLD)
    result = sweep(store, ids[0], ids[-1], test)
    assert [step.result for step in result.steps] == ["good", "bad", "good", "bad"]
    assert (result.first_bad, result.monotone, result.tests) == (ids[2], False, 4)
    # Binary search assumes one transition, so on this chain it lands elsewhere.
    assert bisect(store, ids[0], ids[-1], test).first_bad == ids[4]


def test_any_other_exit_code_stops_the_run_and_cleans_up(store, chain, script, tmp_path, temporary):
    bogus = command(script, "--bogus", THRESHOLD)
    with pytest.raises(BisectError, match=f"exited with 3 on commit {chain[4]}"):
        bisect(store, chain[0], chain[-1], bogus)
    assert list(temporary.iterdir()) == []  # the temporary workdir and its checkout are gone
    work = tmp_path / "work"
    with pytest.raises(BisectError, match=f"exited with 3 on commit {chain[1]}"):
        sweep(store, chain[0], chain[-1], bogus, workdir=work)
    assert list(work.iterdir()) == []  # a given workdir stays, without checkouts
    with pytest.raises(BisectError, match="exited with 3"):
        bisect(store, chain[0], chain[-1], bogus, workdir=work, keep=True)
    assert [path.name for path in work.iterdir()] == [chain[4][:12]]  # kept, as asked


def test_a_workdir_in_the_store_or_on_a_file_is_refused(store, chain, script, tmp_path):
    test = command(script, THRESHOLD)
    with pytest.raises(BisectError, match="store itself"):
        bisect(store, chain[0], chain[-1], test, workdir=store.dir / "work")
    assert not (store.dir / "work").exists()  # nothing was created inside the store
    a_file = tmp_path / "a-file"
    a_file.write_text("mine", encoding="utf-8")
    with pytest.raises(BisectError, match="not a directory"):
        sweep(store, chain[0], chain[-1], test, workdir=a_file)
    assert a_file.read_text(encoding="utf-8") == "mine"


def test_keep_leaves_every_checkout(store, chain, script, tmp_path, temporary):
    test = command(script, THRESHOLD)
    result = bisect(store, chain[0], chain[-1], test, keep=True)
    work = Path(result.workdir)
    assert work.parent == temporary
    assert work.name.startswith("wvc-bisect-")
    tested = [step.commit for step in result.steps]
    assert sorted(path.name for path in work.iterdir()) == sorted(c[:12] for c in tested)
    for commit_id in tested:
        source = tmp_path / "revisions" / f"rev-{chain.index(commit_id)}" / FILE
        assert [path.name for path in (work / commit_id[:12]).iterdir()] == [FILE]
        assert (work / commit_id[:12] / FILE).read_bytes() == source.read_bytes()
    given = tmp_path / "work"
    kept = sweep(store, chain[0], chain[2], test, workdir=given, keep=True)
    assert kept.workdir == str(given)
    names = sorted(commit_id[:12] for commit_id in chain[1:3])
    assert sorted(path.name for path in given.iterdir()) == names
    # A second run into the same workdir refuses to touch the checkouts already there.
    with pytest.raises(BisectError, match="already exists"):
        sweep(store, chain[0], chain[2], test, workdir=given)
    assert sorted(path.name for path in given.iterdir()) == names
    assert (given / names[0] / FILE).is_file()


def test_the_command_gets_the_environment_and_the_checkout_as_last_argument(
    store, chain, script, tmp_path, monkeypatch
):
    log = tmp_path / "log.tsv"
    monkeypatch.setenv("BISECT_LOG", str(log))  # the parent environment reaches the command
    work = tmp_path / "work"
    sweep(store, chain[0], chain[3], command(script, THRESHOLD), workdir=work)
    rows = [line.split("\t") for line in log.read_text(encoding="utf-8").splitlines()]
    assert rows == [[c, str(work / c[:12]), str(work / c[:12])] for c in chain[1:4]]


@pytest.mark.parametrize("name", ["it's a dir; $HOME", "A&B ^(x)", "A&B^", "trailing-slash"])
def test_run_test_quotes_the_directory_and_returns_any_exit_code(tmp_path, name):
    """The platform shell hands the program the directory unchanged (sh, or cmd on Windows)."""
    odd = tmp_path / name
    odd.mkdir()
    check = (
        "import os, sys; "
        "sys.exit(7 if sys.argv[1:] == [os.environ['WVC_CHECKOUT']] "
        "and os.environ['WVC_COMMIT'] == 'c' * 64 else 9)"
    )
    directory = str(odd) + os.sep if name == "trailing-slash" else odd
    code, seconds = run_test(join([sys.executable, "-c", check]), directory, "c" * 64)
    assert code == 7
    assert seconds > 0


# quote_argument for cmd.exe, checked on every platform: the value, then the exact argument.
CMD_QUOTED = [
    ("C:\\Users\\A B\\Temp\\wvc-bisect-x", '"C:\\Users\\A B\\Temp\\wvc-bisect-x"'),
    ("C:\\Users\\A&B\\Temp", '"C:\\Users\\A&B\\Temp"'),
    ("C:\\a^b|c<d>e", '"C:\\a^b|c<d>e"'),
    ("C:\\Program Files (x86)\\w", '"C:\\Program Files (x86)\\w"'),
    ("C:\\work\\", '"C:\\work\\\\"'),  # the trailing backslash must not escape the quote
    ("C:\\A B\\work\\\\", '"C:\\A B\\work\\\\\\\\"'),
    ("", '""'),
]


@pytest.mark.parametrize(("value", "quoted"), CMD_QUOTED)
def test_quote_argument_for_cmd_wraps_the_value_in_double_quotes(value, quoted):
    assert quote_argument(value, shell="cmd") == quoted  # noqa: S604  shell= names a quoting dialect; nothing runs
    if not value or " " in value:  # where list2cmdline quotes too, the C runtime parses it back
        assert quoted == subprocess.list2cmdline([value])


@pytest.mark.parametrize(
    "value", ['C:\\a"b', "C:\\100%", "C:\\%TEMP%\\x", "C:\\wow!", "C:\\a\nb", "C:\\a\rb"]
)
def test_quote_argument_for_cmd_refuses_what_cmd_expands_inside_quotes(value):
    with pytest.raises(StoreError, match=r"cannot pass the checkout directory .* to cmd\.exe"):
        quote_argument(value, shell="cmd")  # noqa: S604  shell= names a quoting dialect; nothing runs


@pytest.mark.parametrize(
    "value", ["/tmp/plain", "/tmp/a b", "/tmp/it's; $(x) `y` & z", "/tmp/50% off!", '/tmp/"q"', ""]
)
def test_quote_argument_for_sh_is_shlex_quote(value):
    assert quote_argument(value, shell="sh") == shlex.quote(value)  # noqa: S604  shell= names a quoting dialect; nothing runs
    assert shlex.split(quote_argument(value, shell="sh")) == [value]  # noqa: S604  shell= names a quoting dialect; nothing runs


def test_quote_argument_defaults_to_the_platform_shell():
    platform = "cmd" if os.name == "nt" else "sh"
    assert quote_argument("C:\\A B\\") == quote_argument("C:\\A B\\", shell=platform)
    with pytest.raises(ValueError, match="unknown shell"):
        quote_argument("x", shell="fish")  # noqa: S604  shell= names a quoting dialect; nothing runs


def test_a_workdir_cmd_cannot_take_is_refused_before_anything_is_created(
    store, chain, script, tmp_path, monkeypatch
):
    """With cmd quoting forced, a workdir holding '%' stops the run before it starts."""
    cmd_quoting = partial(quote_argument, shell="cmd")  # noqa: S604  shell= names a quoting dialect; nothing runs
    monkeypatch.setattr(bisect_module, "quote_argument", cmd_quoting)
    test = command(script, THRESHOLD)
    given = tmp_path / "100%" / "work"
    with pytest.raises(BisectError, match=re.escape("to cmd.exe: it holds '%'")):
        bisect(store, chain[0], chain[-1], test, workdir=given)
    assert not given.parent.exists()
    temporary = tmp_path / "50%"
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    with pytest.raises(BisectError, match=re.escape("to cmd.exe: it holds '%'")):
        sweep(store, chain[0], chain[-1], test)
    assert list(temporary.iterdir()) == []  # the temporary workdir was removed


def test_the_commands_output_goes_to_stderr(store, chain, capfd, temporary):
    chatty = join([sys.executable, "-c", "print('chatty')"])  # ignores the directory
    argv = ["bisect", chain[0], chain[2], "--test", chatty, "--json"]
    code = main([*argv, "--repo", str(store.repo)])
    out, err = capfd.readouterr()
    assert code == 0
    assert json.loads(out)["first_bad"] is None  # stdout holds only the JSON
    assert err.splitlines() == ["chatty"]


def test_cli_prints_each_step_then_the_first_bad_commit(store, chain, script, capsys, temporary):
    test = command(script, THRESHOLD)
    code, out, err = run(
        capsys, "bisect", chain[0][:7], "HEAD", "--test", test, "--repo", store.repo
    )
    lines = out.splitlines()
    assert (code, err) == (0, "")
    assert [STEP_LINE.fullmatch(line).groups() for line in lines[:3]] == [
        ("good", chain[4][:12]),
        ("bad", chain[6][:12]),
        ("bad", chain[5][:12]),
    ]
    assert lines[3:] == [f"first bad: {chain[5]}", "tests: 3 of 7 candidates"]


def test_cli_sweep_reports_the_curve_and_kept_checkouts(store, chain, script, tmp_path, capsys):
    work = tmp_path / "work"
    code, out, _ = run(
        capsys, "bisect", chain[0], chain[2], "--test", command(script, 100),
        "--sweep", "--keep", "--workdir", work, "--repo", store.repo,
    )  # fmt: skip
    lines = out.splitlines()
    assert code == 0
    assert [STEP_LINE.fullmatch(line).groups() for line in lines[:2]] == [
        ("good", chain[1][:12]),
        ("good", chain[2][:12]),
    ]
    assert lines[2:] == [
        "every candidate is good",
        "tests: 2 of 2 candidates",
        "monotone: yes",
        f"checkouts kept in {work}",
    ]
    assert sorted(path.name for path in work.iterdir()) == sorted(c[:12] for c in chain[1:3])


def test_render_names_a_curve_that_is_not_monotone():
    first, second = "a" * 64, "b" * 64
    steps = (BisectStep(first, "bad", 1, 0.5, 0.25), BisectStep(second, "good", 0, 12.5, 0.75))
    result = BisectResult(first, (first, second), steps, "sweep", None, monotone=False)
    assert render_bisect(result, keep=False).splitlines() == [
        "bad   aaaaaaaaaaaa  test 0.50 s   checkout 0.25 s",
        "good  bbbbbbbbbbbb  test 12.50 s  checkout 0.75 s",
        f"first bad: {first}",
        "tests: 2 of 2 candidates",
        "monotone: no (a bad commit is followed by a good one)",
    ]
    empty = BisectResult(None, (), (), "bisect", None)  # never produced; still rendered
    assert render_bisect(empty, keep=False).splitlines() == [
        "every candidate is good",
        "tests: 0 of 0 candidates",
    ]


def test_cli_json(store, script, tmp_path, capsys, temporary):
    ids = commit_values(store, tmp_path / "revisions", [0, 2, 6, 3, 7])
    test = command(script, THRESHOLD)
    argv = ["bisect", ids[0], ids[-1], "--test", test, "--json", "--repo", store.repo]
    code, out, _ = run(capsys, *argv, "--sweep")
    payload = json.loads(out)
    assert code == 0
    assert list(payload) == KEYS
    summary = payload["mode"], payload["first_bad"], payload["tests"], payload["monotone"]
    assert summary == ("sweep", ids[2], 4, False)
    assert (payload["candidates"], payload["workdir"]) == (ids[1:], None)
    assert [(s["commit"], s["result"], s["exit_code"]) for s in payload["steps"]] == [
        (ids[1], "good", 0),
        (ids[2], "bad", 1),
        (ids[3], "good", 0),
        (ids[4], "bad", 1),
    ]
    assert all(set(step) == FIELDS for step in payload["steps"])
    code, out, _ = run(capsys, *argv)
    payload = json.loads(out)
    summary = payload["mode"], payload["first_bad"], payload["tests"], payload["monotone"]
    assert code == 0
    assert summary == ("bisect", ids[4], 2, None)


def test_cli_refusals_exit_2_with_a_message(store, chain, script, capsys, temporary):
    def refused(*argv):
        code, out, err = run(capsys, "bisect", *argv, "--repo", store.repo)
        assert (code, out) == (2, ""), (argv, code, out, err)
        assert err.startswith("wvc: "), (argv, code, out, err)
        return err

    test, bogus = command(script, THRESHOLD), command(script, "--bogus", THRESHOLD)
    assert "exited with 3" in refused(chain[0], chain[-1], "--test", bogus)
    assert "not an ancestor" in refused(chain[-1], chain[0], "--test", test)
    assert "the same commit" in refused("HEAD", "main", "--test", test)
    assert "unknown ref" in refused("nope!", "HEAD", "--test", test)
    assert "command is empty" in refused(chain[0], "HEAD", "--test", "")
    assert list(temporary.iterdir()) == []
    with pytest.raises(SystemExit) as exit_info:
        main(["bisect", chain[0], chain[-1], "--repo", str(store.repo)])
    assert exit_info.value.code == 2
    assert "--test" in capsys.readouterr().err
