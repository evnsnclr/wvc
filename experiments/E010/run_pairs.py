"""Run E010: one ``wvc diff --rank --all --json --classify --features`` receipt per labelled pair.

Downloads go to a dedicated Hugging Face cache (``--cache``), never the shared one, so deleting
them touches nothing else. After each pair the derivative's repository and its blobs are removed
from that cache; a base is removed after the last pair that uses it. (The first run removed the
repository directories but not the shared blobs they link to; see the page's Deviations.)
A pair whose receipt exists is skipped, so the run resumes. Each pair appends one JSON line to
``run.log``: UTC start and end, wall seconds, exit status, bytes the cache grew by (the
download), and free disk.

Run: ``uv run python experiments/E010/run_pairs.py experiments/E010/labels.json
experiments/E010/pairs --cache DIR [--only ID ...]``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

MIN_FREE = 50 * 10**9  # stop before a pair when less than this is free


def blob_bytes(cache: Path) -> int:
    total = 0
    for root, _, files in os.walk(cache):
        if Path(root).name == "blobs":
            total += sum((Path(root) / name).stat().st_size for name in files)
    return total


def repo_dir(cache: Path, repo: str) -> Path:
    return cache / ("models--" + repo.replace("/", "--"))


def linked_blobs(directory: Path) -> set[Path]:
    """The blobs a repository's snapshot links point to (huggingface_hub 2.x shares one
    top-level ``blobs/`` directory, so removing the repository directory frees nothing)."""
    found = set()
    for root, _, files in os.walk(directory):
        for name in files:
            path = Path(root) / name
            if path.is_symlink():
                found.add(path.resolve())
    return found


def remove_repo(cache: Path, repo: str) -> None:
    """Remove a repository and every blob no other cached repository links to."""
    target = repo_dir(cache, repo)
    mine = linked_blobs(target)
    others = set()
    for directory in cache.glob("models--*"):
        if directory != target:
            others |= linked_blobs(directory)
    shutil.rmtree(target, ignore_errors=True)
    for blob in mine - others:
        blob.unlink(missing_ok=True)


def _parses(path: Path) -> bool:
    """An uncaught exception also exits 1, so a receipt counts only when it is a JSON record."""
    try:
        return "summary" in json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_one(row: dict, out: Path, cache: Path) -> dict:
    base, derivative = row["base"], row["derivative"]
    argv = [
        sys.executable,
        "-m",
        "wvc",
        "diff",
        "--rank",
        "--all",
        "--json",
        "--classify",
        "--features",
        f"hf:{base['repo']}@{base['sha']}",
        f"hf:{derivative['repo']}@{derivative['sha']}",
    ]
    env = {**os.environ, "HF_HUB_CACHE": str(cache), "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    before, start, started = blob_bytes(cache), time.monotonic(), now()
    with (out / f"{row['id']}.json").open("wb") as stdout:
        done = subprocess.run(argv, stdout=stdout, stderr=subprocess.PIPE, env=env, check=False)
    if done.stderr:
        (out / f"{row['id']}.stderr.txt").write_bytes(done.stderr)
    if done.returncode not in (0, 1) or not _parses(out / f"{row['id']}.json"):
        (out / f"{row['id']}.json").rename(out / f"{row['id']}.failed.json")
    return {
        "id": row["id"],
        "start": started,
        "end": now(),
        "seconds": round(time.monotonic() - start, 1),
        "exit": done.returncode,
        "downloaded_bytes": blob_bytes(cache) - before,
        "free_bytes": shutil.disk_usage(cache).free,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    rows = json.loads(args.labels.read_text(encoding="utf-8"))
    rows = [r for r in rows if not args.only or r["id"] in args.only]
    rows.sort(key=lambda r: (r["base"]["repo"], r["id"]))
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    log = args.out.parent / "run.log"
    for index, row in enumerate(rows):
        done = [args.out / f"{row['id']}{suffix}.json" for suffix in ("", ".failed")]
        if any(path.exists() for path in done):
            continue
        if shutil.disk_usage(args.cache).free < MIN_FREE:
            print(f"stopping before {row['id']}: under {MIN_FREE:,} bytes free", flush=True)
            break
        entry = run_one(row, args.out, args.cache)
        with log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
        later = {r["base"]["repo"] for r in rows[index + 1 :]}
        if row["derivative"]["repo"] not in later:  # F01's derivative is A01's base
            remove_repo(args.cache, row["derivative"]["repo"])
        if row["base"]["repo"] not in later:
            remove_repo(args.cache, row["base"]["repo"])


if __name__ == "__main__":
    main()
