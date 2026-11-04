"""Score E010: claimed method against predicted kind, with features and from the record alone.

The with-features kind is the ``classification`` each receipt carries. The record-only kind is
computed here from the same receipt with every ``features`` key removed. ``--rules`` names
the classifier to score: ``receipt`` (the classification the run wrote, rules v1) or
``current`` (re-run ``wvc.classify`` as it is now on both inputs, for a later rules version).

Run: ``uv run python experiments/E010/score.py experiments/E010 [--rules current] [--out NAME]``.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from wvc.classify import KINDS, RULES_VERSION, classify_record

SUMMARY_KEYS = ("r", "r_source", "direction_coherence", "projections", "rank", "casts")


def strip_features(record: dict) -> dict:
    stripped = copy.deepcopy(record)
    stripped.pop("features", None)
    stripped.pop("classification", None)
    for entry in stripped["tensors"]:
        entry.pop("features", None)
    return stripped


def classify_both(record: dict, rules: str) -> tuple[dict, dict]:
    if rules == "receipt":
        with_features = record["classification"]
    else:
        clean = copy.deepcopy(record)
        clean.pop("classification", None)
        with_features = classify_record(clean).to_dict()
    return with_features, classify_record(strip_features(record)).to_dict()


def per_kind(rows: list[dict], key: str) -> dict:
    stats = {}
    for kind in KINDS:
        claimed = [r for r in rows if r["label"] == kind]
        predicted = [r for r in rows if r[key]["kind"] == kind]
        hits = [r for r in claimed if r[key]["kind"] == kind]
        if claimed or predicted:
            stats[kind] = {
                "claimed": len(claimed),
                "predicted": len(predicted),
                "agree": len(hits),
                "precision": len(hits) / len(predicted) if predicted else None,
                "recall": len(hits) / len(claimed) if claimed else None,
            }
    return stats


def confusion(rows: list[dict], key: str) -> dict:
    table: dict[str, Counter] = {}
    for row in rows:
        table.setdefault(row["label"], Counter())[row[key]["kind"]] += 1
    return {label: dict(counts) for label, counts in table.items()}


def evidence_digest(result: dict) -> dict:
    evidence = result["evidence"]
    digest = {k: evidence[k] for k in SUMMARY_KEYS if k in evidence}
    for key in ("tensors", "changed", "cast", "cast_lossless", "added", "removed", "layers"):
        digest[key] = evidence.get(key)
    digest["layers_touched"] = len(evidence.get("layers_touched", []))
    digest["vectors_changed"] = evidence.get("vectors_changed")
    digest["modules_changed"] = evidence.get("modules_changed")
    digest["resized"] = len(evidence.get("resized", []))
    return digest


def load_rows(root: Path, rules: str) -> tuple[list[dict], list[dict]]:
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    rows, failures = [], []
    for label in labels:
        receipt = root / "pairs" / f"{label['id']}.json"
        if not receipt.exists():
            failures.append({"id": label["id"], "label": label["label"], "receipt": "missing"})
            continue
        record = json.loads(receipt.read_text(encoding="utf-8"))
        features, record_only = classify_both(record, rules)
        rows.append({**label, "features": features, "record": record_only})
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--rules", choices=("receipt", "current"), default="receipt")
    parser.add_argument("--out", default="scores.json")
    args = parser.parse_args()
    rows, failures = load_rows(args.root, args.rules)
    report = {
        "rules": "v1 (as written in the receipts)" if args.rules == "receipt" else RULES_VERSION,
        "pairs": len(rows),
        "failures": failures,
        "agree_features": sum(r["features"]["kind"] == r["label"] for r in rows),
        "agree_record": sum(r["record"]["kind"] == r["label"] for r in rows),
        "per_kind_features": per_kind(rows, "features"),
        "per_kind_record": per_kind(rows, "record"),
        "confusion_features": confusion(rows, "features"),
        "confusion_record": confusion(rows, "record"),
        "pairs_detail": [
            {
                "id": r["id"],
                "label": r["label"],
                "r_stated": r["r"],
                "derivative": r["derivative"]["repo"],
                "features": {k: r["features"][k] for k in ("kind", "tier", "rule", "reason")},
                "record": {k: r["record"][k] for k in ("kind", "tier", "rule", "reason")},
                "evidence": evidence_digest(r["features"]),
            }
            for r in rows
        ],
    }
    (args.root / args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(
        f"{report['agree_features']} of {len(rows)} agree with features; "
        f"{report['agree_record']} from the record alone; {len(failures)} without a receipt"
    )
    for row in rows:
        mark = "" if row["features"]["kind"] == row["label"] else "  MISS"
        print(
            f"{row['id']} {row['label']:18} -> {row['features']['kind']:18} "
            f"{row['features']['tier']:6} {row['features']['rule']:4} | record "
            f"{row['record']['kind']} {row['record']['rule']}{mark}"
        )


if __name__ == "__main__":
    main()
