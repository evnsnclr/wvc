"""The E009 census's ``--pairs`` rerun (experiments/E009/census.py): no network, no diff."""

import csv
import runpy
from pathlib import Path

import pytest

E009 = Path(__file__).resolve().parents[1] / "experiments" / "E009"
PAIR = {  # a pair from the committed re-diff list, with its saved selection
    "base": "Qwen/Qwen3-0.6B",
    "base_sha": "c1899de289a04d12100db370d81485cdf75e47ca",
    "upload": "jinaai/jina-reranker-v3",
    "upload_sha": "d7d7e73b6ea138ced340b83865931b5dfb6c97aa",
}


@pytest.fixture
def census():
    return runpy.run_path(str(E009 / "census.py"))


def _args(census, tmp_path, *extra):
    pairs = tmp_path / "pairs.csv"
    with pairs.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PAIR))
        writer.writeheader()
        writer.writerow(PAIR)
    out = tmp_path / "out"
    out.mkdir()
    argv = ["--cache", str(tmp_path / "cache"), "--out", str(out), "--pairs", str(pairs)]
    return census["parse_args"]([*argv, "--max-bytes", "1", "--min-free", "1", *extra])


def _fail_before(census, args):
    row = dict.fromkeys(census["CSV_FIELDS"], "") | {
        "base": PAIR["base"],
        "upload": PAIR["upload"],
        "outcome": "failed",
        "reason": "timeout",
    }
    census["append_row"](args.out, row)


def test_pairs_cannot_be_combined_with_select_only(census, tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        _args(census, tmp_path, "--select-only")
    assert raised.value.code == 2
    assert "--select-only" in capsys.readouterr().err


@pytest.mark.parametrize("retry", [False, True])
def test_a_failed_listed_pair_reruns_only_with_retry_failed(census, tmp_path, retry):
    args = _args(census, tmp_path, *(["--retry-failed"] if retry else []))
    _fail_before(census, args)
    ran = []

    def fake_run_pair(args, selection, entry, state):
        ran.append(entry["upload"])
        return True

    census["run_listed"].__globals__["run_pair"] = fake_run_pair  # run_path returns a copy
    assert census["run_listed"](args, {"bytes_new": 0}) is True
    assert ran == ([PAIR["upload"]] if retry else [])


def test_a_listed_pair_pinned_elsewhere_is_refused(census):
    with pytest.raises(SystemExit, match="another commit"):
        census["listed_pair"]({**PAIR, "base_sha": "0" * 40})
    with pytest.raises(SystemExit, match="not an eligible upload"):
        census["listed_pair"]({**PAIR, "upload_sha": "0" * 40})
