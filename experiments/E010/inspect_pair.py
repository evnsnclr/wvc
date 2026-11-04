"""Print the evidence behind E010 pairs' classifications, by module (for the miss list).

Run: ``uv run python experiments/E010/inspect_pair.py experiments/E010/pairs/A05.json [...]``.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from wvc.tensor_names import layer_of

COLUMNS = (
    ("rank", lambda t, f: t.get("rank")),
    ("top1", lambda t, f: t.get("top1_energy")),
    ("sig", lambda t, f: f.get("signal_rank")),
    ("sigtop1", lambda t, f: f.get("signal_top1")),
    ("snr", lambda t, f: f.get("snr")),
    ("align", lambda t, f: f.get("direction_alignment")),
    ("rel", lambda t, f: t.get("rel_delta_norm")),
)


def median(values: list) -> float | None:
    present = [v for v in values if v is not None]
    return round(statistics.median(present), 3) if present else None


def show(path: Path) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    result = record["classification"]
    evidence = result["evidence"]
    print(f"== {path.name}: {result['kind']} {result['tier']} {result['rule']}")
    print(f"   {result['reason'][:200]}")
    print("   status", record["summary"]["by_status"], "pair", record.get("features"))
    print("   failed", evidence.get("failed"))
    print("   casts", evidence.get("casts"), "lossless", evidence.get("cast_lossless"))
    by_module: dict[str, list] = defaultdict(list)
    dtypes: dict[str, str] = {}
    for tensor in record["tensors"]:
        if tensor["status"] in ("changed", "reshaped") and tensor.get("changed"):
            _, module = layer_of(tensor["name"])
            found = tensor.get("features") or {}
            by_module[module].append([get(tensor, found) for _, get in COLUMNS])
            dtypes[module] = f"{tensor['dtype_a']}->{tensor['dtype_b']}"
    for module, rows in sorted(by_module.items()):
        cells = " ".join(
            f"{name}={median([row[i] for row in rows])}" for i, (name, _) in enumerate(COLUMNS)
        )
        print(f"   {module[:40]:40} n={len(rows):3} {cells} {dtypes[module]}")
    for status in ("added", "removed"):
        names = [t["name"] for t in record["tensors"] if t["status"] == status]
        if names:
            print(f"   {status} {len(names)}: {names[:3]}")


def main() -> None:
    for argument in sys.argv[1:]:
        show(Path(argument))


if __name__ == "__main__":
    main()
