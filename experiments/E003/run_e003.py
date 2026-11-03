"""Run E003 (experiments/E003-astra-edit-commit.md) and write experiments/E003/receipt.json.

Usage, from the repository root once M3 is merged:
    uv run python experiments/E003/run_e003.py
Inputs are the E001 files in the Hugging Face cache; nothing is downloaded. The script
runs the wvc CLI through subprocess, times every step, reads commit ids from
`.wvc/refs/heads/main`, measures the store, hashes the checkouts, and compares them with
the Hub blob hashes recorded in E001. It records; it does not interpret.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = HERE / "repo"
CACHE = Path.home() / ".cache/huggingface/hub"
BASE = CACHE / "models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
EDIT = (
    CACHE
    / "models--RikiyaT--Qwen3.5-0.8B-weight-space-programming/snapshots"
    / "d1671a178705bcab7644806d23c535517f35723b"
)
SHARD = "model.safetensors-00001-of-00001.safetensors"
EXPECTED = {
    "base": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    "edit": "fd1e42acdd118ac1b300d37500fd2885e75f5a09de3a33010d802bf7f737b553",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run(args: list[str], cwd: Path) -> dict:
    start = time.perf_counter()
    proc = subprocess.run(["uv", "run", "wvc", *args], cwd=cwd, capture_output=True, text=True)
    return {
        "command": "wvc " + " ".join(args),
        "seconds": round(time.perf_counter() - start, 2),
        "exit": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def head_commit() -> str:
    return (REPO / ".wvc/refs/heads/main").read_text().strip()


def main() -> int:
    os.environ.setdefault(
        "WVC_AUTHOR", "evansmith"
    )  # commits need an author since the integrity fixes
    for path in (BASE / SHARD, EDIT / SHARD):
        if not path.exists():
            print(f"missing input: {path}", file=sys.stderr)
            return 2
    if REPO.exists():
        shutil.rmtree(REPO)
    for name in ("checkout-base", "checkout-edit"):
        shutil.rmtree(HERE / name, ignore_errors=True)
    REPO.mkdir(parents=True)
    steps: list[dict] = []
    receipt: dict = {
        "experiment": "E003",
        "started": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wvc_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "inputs": {"base": str(BASE / SHARD), "edit": str(EDIT / SHARD)},
        "steps": steps,
    }
    steps.append(run(["init", "."], REPO))
    steps.append(
        run(
            [
                "commit",
                str(BASE),
                "-m",
                "Qwen/Qwen3.5-0.8B at 2fc0636",
                "--op",
                "pretrain-checkpoint",
            ],
            REPO,
        )
    )
    first = head_commit()
    receipt["store_bytes_after_first_commit"] = tree_bytes(REPO / ".wvc")
    steps.append(
        run(
            [
                "commit",
                str(EDIT),
                "-m",
                "Weight-Space Programming edit (Takehi et al.)",
                "--op",
                "direct-edit",
            ],
            REPO,
        )
    )
    second = head_commit()
    receipt["store_bytes_after_second_commit"] = tree_bytes(REPO / ".wvc")
    receipt["commits"] = {"first": first, "second": second}
    steps.append(run(["log"], REPO))
    steps.append(run(["show", "HEAD"], REPO))
    steps.append(run(["show", "HEAD", "--json"], REPO))
    steps.append(run(["checkout", second, "--out", str(HERE / "checkout-edit")], REPO))
    steps.append(run(["checkout", first, "--out", str(HERE / "checkout-base")], REPO))
    receipt["checkouts"] = {}
    for label, folder in (("base", "checkout-base"), ("edit", "checkout-edit")):
        out = HERE / folder / SHARD
        digest = sha256_file(out) if out.exists() else None
        receipt["checkouts"][label] = {
            "path": str(out),
            "sha256": digest,
            "nbytes": out.stat().st_size if out.exists() else None,
            "matches_hub_blob": digest == EXPECTED[label],
        }
    receipt["finished"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    receipt["all_exit_zero"] = all(step["exit"] == 0 for step in steps)
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for step in steps:
        print(f"[{step['seconds']:7.2f}s exit {step['exit']}] {step['command']}")
    print(
        "store bytes:",
        receipt["store_bytes_after_first_commit"],
        "->",
        receipt["store_bytes_after_second_commit"],
    )
    print("checkouts:", {k: v["matches_hub_blob"] for k, v in receipt["checkouts"].items()})
    return (
        0
        if receipt["all_exit_zero"]
        and all(v["matches_hub_blob"] for v in receipt["checkouts"].values())
        else 1
    )


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
