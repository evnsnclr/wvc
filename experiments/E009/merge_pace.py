"""Merge a PACE census output directory into this one: new receipts and new CSV rows only.

    uv run python experiments/E009/merge_pace.py <fetched PACE out dir>

Idempotent: a receipt already here is left alone, and a row is appended only when its
(run_at, base, upload) is not in the local CSV yet.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(source: Path) -> int:
    copied = 0
    for kind in ("pairs", "failures"):
        for receipt in sorted((source / kind).glob("*.json")) if (source / kind).exists() else ():
            target = HERE / kind / receipt.name
            if not target.exists():
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(receipt, target)
                copied += 1
    local = rows(HERE / "census.csv")
    seen = {(r["run_at"], r["base"], r["upload"]) for r in local}
    fields = list(local[0].keys())
    new = [
        r for r in rows(source / "census.csv") if (r["run_at"], r["base"], r["upload"]) not in seen
    ]
    with (HERE / "census.csv").open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writerows(new)
    print(f"copied {copied} receipts, appended {len(new)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
