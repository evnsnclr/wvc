"""Fetch the E004 model files into the Hugging Face cache and write results/models.json.

The base's weights (`model.safetensors` of Qwen/Qwen2.5-0.5B-Instruct) are already cached by
E002; this adds its config and tokenizer files at the same pinned revision, and the judge
Qwen/Qwen3Guard-Gen-0.6B at a pinned revision. Every file's size and SHA-256 is recorded, as
are the bytes this run downloaded. Run through `uv run --with ...` (see run.sh).
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from e004_config import (  # noqa: E402
    BASE_FILES,
    BASE_MODEL,
    BASE_REVISION,
    JUDGE_FILES,
    JUDGE_MODEL,
    JUDGE_REVISION,
    RESULTS,
    WEIGHTS,
    check_disk,
    snapshot_dir,
    utc_now,
    write_json,
)
from huggingface_hub import hf_hub_download  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(repo_id: str, revision: str, filenames: tuple[str, ...]) -> tuple[list[dict], int]:
    """Download missing files of one pinned revision; return file records and bytes fetched."""
    records, fetched = [], 0
    snapshot = snapshot_dir(repo_id, revision)
    for name in filenames:
        cached = (snapshot / name).exists()
        path = Path(hf_hub_download(repo_id, name, revision=revision))
        size = path.stat().st_size
        fetched += 0 if cached else size
        records.append(
            {
                "file": name,
                "bytes": size,
                "sha256": sha256_file(path),
                "downloaded_by_this_run": not cached,
            }
        )
    return records, fetched


def main() -> int:
    started = time.perf_counter()
    check_disk("model downloads")
    base_weights = snapshot_dir(BASE_MODEL, BASE_REVISION) / WEIGHTS
    if not base_weights.exists():
        print(f"missing cached base weights: {base_weights}", file=sys.stderr)
        return 2
    base_files, base_fetched = fetch(BASE_MODEL, BASE_REVISION, BASE_FILES)
    blob_name = base_weights.resolve().name
    weights_record = {
        "file": WEIGHTS,
        "bytes": base_weights.stat().st_size,
        "sha256": sha256_file(base_weights),
        "downloaded_by_this_run": False,
        "cache_blob": blob_name,
    }
    judge_files, judge_fetched = fetch(JUDGE_MODEL, JUDGE_REVISION, JUDGE_FILES)
    payload = {
        "written": utc_now(),
        "base": {
            "repo": BASE_MODEL,
            "revision": BASE_REVISION,
            "snapshot": str(snapshot_dir(BASE_MODEL, BASE_REVISION)),
            "files": [weights_record, *base_files],
        },
        "judge": {
            "repo": JUDGE_MODEL,
            "revision": JUDGE_REVISION,
            "license": "apache-2.0 (Hub model card metadata)",
            "snapshot": str(snapshot_dir(JUDGE_MODEL, JUDGE_REVISION)),
            "files": judge_files,
        },
        "bytes_downloaded_by_this_run": base_fetched + judge_fetched,
        "seconds": round(time.perf_counter() - started, 2),
    }
    write_json(RESULTS / "models.json", payload)
    print(
        f"base files ok; judge files ok; downloaded {payload['bytes_downloaded_by_this_run']:,} B"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
