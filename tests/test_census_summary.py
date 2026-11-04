"""The E009 census summary on receipts with and without tensor ids (experiments/E009)."""

import csv
import json
import runpy
from pathlib import Path

import pytest

E009 = Path(__file__).resolve().parents[1] / "experiments" / "E009"
BASE = "org/base"


@pytest.fixture(scope="module")
def summary():
    return runpy.run_path(str(E009 / "summarize.py"))


def _run(summary, capsysbinary, *argv) -> str:
    assert summary["main"](["summarize.py", *map(str, argv)]) == 0
    return capsysbinary.readouterr().out.decode("utf-8")


def test_receipts_without_ids_reproduce_the_committed_summary(summary, capsysbinary):
    text = _run(summary, capsysbinary, E009 / "census.csv")
    assert text == (E009 / "summary.md").read_text(encoding="utf-8")
    assert "renamed, unchanged" not in text


def test_the_rediff_list_names_the_pairs_with_added_and_removed_tensors(summary, capsysbinary):
    text = _run(summary, capsysbinary, "--rediff", E009 / "census.csv")
    rows = [line for line in text.splitlines() if line.startswith("| ") and "---" not in line]
    listed = rows[1:]  # the header first
    assert text.startswith(f"{len(listed)} diffed pairs on ")
    assert listed
    assert all(int(r.split(" | ")[4]) and int(r.split(" | ")[5][:-2]) for r in listed)
    assert any("| vllm-sr/d3-edge |" in r for r in listed)


def _tensor(name, status, **fields):
    entry = {"name": name, "status": status, "changed": None, "numel": 10, "rank": None}
    return {**entry, **fields}


def _receipt(renamed, tensors):
    summary = {"tensors": 10, "by_status": {}, "fraction_changed": None}
    if renamed is not None:
        summary["renamed"] = renamed
    return {"format": "test", "census": {}, "diff": {"summary": summary, "tensors": tensors}}


PAIRS = {  # upload: (unchanged, changed, added, removed, receipt's renamed, listed tensors)
    "org/renamed-same": (10, 0, 0, 0, 10, []),
    "org/renamed-edit": (
        9,
        1,
        0,
        0,
        10,
        [_tensor("model.w", "changed", changed=3, renamed_to="lm.model.w")],
    ),
    "org/old-layout": (4, 0, 6, 6, None, []),
    "org/packed": (4, 0, 6, 6, 0, []),
}


def _census(tmp_path: Path) -> Path:
    with (E009 / "census.csv").open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    rows = []
    for rank, (upload, (unchanged, changed, added, removed, renamed, tensors)) in enumerate(
        PAIRS.items(), start=1
    ):
        values = {"unchanged": unchanged, "changed": changed, "added": added, "removed": removed}
        row = dict.fromkeys(header, "") | {
            "host": "test",
            "base": BASE,
            "upload": upload,
            "listing_rank": rank,
            "base_sha": "a" * 40,
            "upload_sha": "b" * 40,
            "relation_tag": "finetune",
            "outcome": "diffed",
            "tensors": 10,
            "reshaped": 0,
            "bytes_new": 0,
            **values,
        }
        rows.append(row)
        receipt = tmp_path / "pairs" / f"{upload.replace('/', '__')}.json"
        receipt.parent.mkdir(exist_ok=True)
        receipt.write_text(json.dumps(_receipt(renamed, tensors)), encoding="utf-8")
    (tmp_path / "selection").mkdir()
    walk = {"base": {"repo": BASE}, "k": 10, "walked": [{"eligible": True, "reason": "ok"}] * 4}
    (tmp_path / "selection" / "org__base.json").write_text(json.dumps(walk), encoding="utf-8")
    path = tmp_path / "census.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_receipts_with_ids_split_the_renamed_layout(summary, tmp_path):
    rows = summary["with_receipts"](tmp_path, [])
    assert rows == []
    _census(tmp_path)
    with (tmp_path / "census.csv").open(encoding="utf-8", newline="") as handle:
        rows = summary["with_receipts"](tmp_path, list(csv.DictReader(handle)))
    kinds = {row["upload"]: summary["category"](row) for row in rows}
    assert kinds == {
        "org/renamed-same": "renamed, unchanged",
        "org/renamed-edit": "renamed, changed",
        "org/old-layout": "renamed layout",
        "org/packed": "renamed layout",
    }
    edit = next(row for row in rows if row["upload"] == "org/renamed-edit")
    assert (edit["renamed"], edit["renamed_unchanged"], edit["renamed_changed"]) == ("10", "9", "1")
    order = summary["categories"](rows)
    assert order.index("renamed layout") + 1 == order.index("renamed, unchanged")
    assert [row["upload"] for row in summary["needs_rediff"](rows)] == ["org/old-layout"]


def test_the_summary_of_receipts_with_ids_counts_renames(summary, tmp_path, capsysbinary):
    text = _run(summary, capsysbinary, _census(tmp_path))
    assert "| renamed layout | renamed, unchanged | renamed, changed | dtype rewrite |" in text
    assert "## Renamed tensors paired by the diff" in text
    assert "| org/base | org/renamed-edit | renamed, changed | 10 | 9 | 1 | 0 | 0 |" in text
    assert "renamed layout: 4 of 4 (100%), 0 on Qwen3.5 bases" in text
    assert "(renamed, unchanged 1, renamed, changed 1)" in text
    rediff = _run(summary, capsysbinary, "--rediff", tmp_path / "census.csv")
    assert rediff.startswith("1 diffed pairs on 1 bases")
    assert "| org/old-layout |" in rediff
    assert "org/packed" not in rediff
