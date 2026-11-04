"""Classify the E009 census receipts from their records alone (no download, no features).

Each census receipt (``wvc-census-pair/1``) wraps a ``wvc diff --rank --json`` record under
``diff``; a pair the census skipped or failed has none. Writes one row per receipt with the
Hub relation tag the upload declares, and prints the counts of kind by relation tag.

Run: ``uv run python experiments/E010/classify_census.py experiments/E009/pairs
experiments/E010/census-kinds.json``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from wvc.classify import RULES_VERSION, classify_record
from wvc.errors import WvcError


def main() -> None:
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for path in sorted(source.glob("*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        census, record = receipt.get("census", {}), receipt.get("diff")
        row = {
            "receipt": path.name,
            "base": census.get("base"),
            "upload": census.get("upload"),
            "relation_tag": census.get("relation_tag"),
        }
        if not isinstance(record, dict):
            rows.append({**row, "kind": None, "reason": "no diff record in the receipt"})
            continue
        try:
            result = classify_record(record, source=path.name)
        except WvcError as exc:
            rows.append({**row, "kind": None, "reason": f"refused: {exc}"})
            continue
        rows.append(
            {
                **row,
                "kind": result.kind,
                "tier": result.tier,
                "rule": result.rule,
                "reason": result.reason,
                "layers_known": result.evidence.get("layers_known"),
            }
        )
    document = {"rules": RULES_VERSION, "features": False, "pairs": rows}
    out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    classified = [r for r in rows if r["kind"]]
    print(f"{len(rows)} receipts; {len(classified)} with a diff record; rules {RULES_VERSION}")
    print("kinds:", dict(Counter(r["kind"] for r in classified).most_common()))
    for tag in sorted({str(r["relation_tag"]) for r in classified}):
        kinds = Counter(r["kind"] for r in classified if str(r["relation_tag"]) == tag)
        print(f"  {tag}: {dict(kinds.most_common())}")


if __name__ == "__main__":
    main()
