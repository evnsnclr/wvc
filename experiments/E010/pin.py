"""Pin every repository of the E010 labelled set to its current commit; write labels.json.

Reads the Hub API only (model_info with file metadata); downloads nothing.
Run: ``uv run python experiments/E010/pin.py experiments/E010/labels.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels_draft import PAIRS  # noqa: E402


def describe(api: HfApi, repo: str, cache: dict) -> dict:
    if repo not in cache:
        info = api.model_info(repo, files_metadata=True)
        shards = {
            s.rfilename: s.size
            for s in info.siblings or []
            if s.rfilename.endswith(".safetensors") and "/" not in s.rfilename
        }
        cache[repo] = {
            "repo": repo,
            "sha": info.sha,
            "safetensors_bytes": sum(shards.values()),
            "shards": shards,
        }
    return cache[repo]


def main() -> None:
    api, cache, rows = HfApi(), {}, []
    for pair_id, label, base, derivative, source, quote, rank in PAIRS:
        rows.append(
            {
                "id": pair_id,
                "label": label,
                "r": rank,
                "label_source": source,
                "quote": quote,
                "base": describe(api, base, cache),
                "derivative": describe(api, derivative, cache),
            }
        )
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    bases = {row["base"]["repo"]: row["base"]["safetensors_bytes"] for row in rows}
    derived = sum(row["derivative"]["safetensors_bytes"] for row in rows)
    largest = max(
        r["base"]["safetensors_bytes"] + r["derivative"]["safetensors_bytes"] for r in rows
    )
    print(f"pairs {len(rows)}; bases {len(bases)}, {sum(bases.values()):,} bytes")
    print(f"derivatives {derived:,} bytes; largest pair {largest:,} bytes")


if __name__ == "__main__":
    main()
