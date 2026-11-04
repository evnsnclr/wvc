"""The Hub census: diff the most downloaded derivative uploads of fixed bases against them.

For each base in ``BASES``, the derivatives whose card ``base_model`` points at it are listed
through the Hub API in order of downloads over the last 30 days. The list is walked in that
order until ``k`` eligible uploads are taken; every upload passed over gets a row saying why
(``adapter``, ``no-safetensors``, ``size-cap``, ``api-error``). The walk is saved once per base
under ``selection/`` and reused, so a resumed run diffs the same uploads.

Each eligible pair is diffed by the installed package, ``python -m wvc diff --rank --json
hf:BASE@SHA hf:UPLOAD@SHA``, with the Hugging Face cache pointed at a dedicated directory.
A diff that exits 0 or 1 with a JSON document becomes a receipt under
``pairs/<owner>__<name>.json``; anything else becomes a record under ``failures/`` and a
``failed`` row. A pair with a receipt is never rerun; a failed pair is rerun only with
``--retry-failed``. After each pair the upload's files are removed from the cache, and so is
every blob no remaining snapshot references, so only the bases stay. Every pair, diffed or
not, appends one row to ``census.csv``. No machine path enters a receipt.

    uv run python experiments/E009/census.py --cache ~/.cache/wvc-census/hub \
        --group small --max-bytes 300e9 --min-free 100e9
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RECEIPT_FORMAT = "wvc-census-pair/1"
FAILURE_FORMAT = "wvc-census-failure/1"
SELECTION_FORMAT = "wvc-census-selection/1"
LISTING_LIMIT = 300  # candidates read per base; the walk stops long before on every base
SIZE_CAP_FACTOR = 2.5  # an upload may hold at most this many times the base's safetensors bytes
SIZE_CAP_ABSOLUTE = 45 * 10**9
PAIR_TIMEOUT_SECONDS = 2 * 3600
LOW_RANK_MAX = 64
STDERR_KEEP = 2000  # characters of a failure's stderr kept in its record

# base -> (group, uploads to take). Fixed before the first run (E009, 2026-11-03).
BASES: dict[str, tuple[str, int]] = {
    "Qwen/Qwen3.5-0.8B": ("small", 10),
    "Qwen/Qwen3-0.6B": ("small", 10),
    "Qwen/Qwen2.5-0.5B-Instruct": ("small", 10),
    "Qwen/Qwen2.5-1.5B-Instruct": ("small", 10),
    "Qwen/Qwen3-1.7B": ("small", 10),
    "meta-llama/Llama-3.2-1B-Instruct": ("small", 10),
    "google/gemma-3-1b-it": ("small", 10),
    "Qwen/Qwen3.5-9B": ("large-laptop", 6),
    "Qwen/Qwen3-8B": ("large-laptop", 6),
    "meta-llama/Llama-3.1-8B-Instruct": ("large-laptop", 6),
    "google/gemma-2-9b-it": ("large-pace", 6),
}

CSV_FIELDS = [
    "run_at",
    "host",
    "base",
    "upload",
    "listing_rank",
    "base_sha",
    "upload_sha",
    "relation_tag",
    "card_base_model_relation",
    "downloads_30d",
    "created_at",
    "upload_safetensors_bytes",
    "outcome",
    "reason",
    "tensors",
    "unchanged",
    "changed",
    "added",
    "removed",
    "reshaped",
    "fraction_changed",
    "rank_measured",
    "rank_max",
    "rank_median",
    "rank_le_64",
    "shards_a",
    "shards_b",
    "shards_skipped",
    "fetched_a",
    "fetched_b",
    "bytes_new",
    "wall_seconds",
    "wvc_version",
]


@dataclass(frozen=True)
class Shard:
    name: str
    nbytes: int
    sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "nbytes": self.nbytes, "sha256": self.sha256}


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def gb(nbytes: float) -> str:
    return f"{nbytes / 1e9:.2f} GB"


def slug(repo_id: str) -> str:
    return repo_id.replace("/", "__")


# ---------------------------------------------------------------- Hub listing and selection


def top_level_shards(info: Any) -> list[Shard]:
    shards = []
    for sibling in info.siblings or ():
        name = sibling.rfilename
        if name.endswith(".safetensors") and "/" not in name and not name.startswith("."):
            lfs = sibling.lfs
            sha = (
                (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None))
                if lfs
                else None
            )
            shards.append(Shard(name, int(sibling.size or 0), sha))
    return sorted(shards, key=lambda shard: shard.name)


def relation_of(tags: list[str], base: str) -> str | None:
    """The relation the Hub's ``base_model:<relation>:<base>`` tag names, if any."""
    for tag in tags:
        parts = tag.split(":", 2)
        if len(parts) == 3 and parts[0] == "base_model" and parts[2] == base:
            return parts[1]
    return None


def card_fields(card: Any) -> tuple[Any, Any]:
    if card is None:
        return None, None
    get = card.get if hasattr(card, "get") else lambda key: getattr(card, key, None)
    return get("base_model"), get("base_model_relation")


def describe_base(api: Any, base: str) -> dict[str, Any]:
    info = api.model_info(base, files_metadata=True)
    shards = top_level_shards(info)
    return {
        "repo": base,
        "sha": info.sha,
        "shards": [shard.to_dict() for shard in shards],
        "safetensors_bytes": sum(shard.nbytes for shard in shards),
    }


def walk_candidates(api: Any, base: str, k: int, base_bytes: int) -> list[dict[str, Any]]:
    cap = min(int(SIZE_CAP_FACTOR * base_bytes), SIZE_CAP_ABSOLUTE)
    listing = api.list_models(
        filter=f"base_model:{base}",
        sort="downloads",
        limit=LISTING_LIMIT,
        expand=["downloads", "createdAt", "tags", "cardData"],
    )
    walked, taken = [], 0
    for rank, model in enumerate(listing, start=1):
        if taken >= k:
            break
        card_base, card_relation = card_fields(model.card_data)
        entry = {
            "listing_rank": rank,
            "upload": model.id,
            "downloads_30d": model.downloads,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "relation_tag": relation_of(list(model.tags or ()), base),
            "card_base_model": card_base,
            "card_base_model_relation": card_relation,
        }
        entry.update(classify_candidate(api, model.id, entry["relation_tag"], cap))
        taken += entry["eligible"]
        walked.append(entry)
    return walked


def classify_candidate(api: Any, repo_id: str, relation: str | None, cap: int) -> dict[str, Any]:
    if relation == "adapter":
        return {"eligible": False, "reason": "adapter", "sha": None, "shards": []}
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception as exc:  # gated, removed, or a Hub error: recorded, never guessed around
        return {
            "eligible": False,
            "reason": f"api-error: {type(exc).__name__}",
            "sha": None,
            "shards": [],
        }
    shards = top_level_shards(info)
    result = {"sha": info.sha, "shards": [shard.to_dict() for shard in shards]}
    total = sum(shard.nbytes for shard in shards)
    if not shards:
        return {**result, "eligible": False, "reason": "no-safetensors"}
    if total > cap:
        return {**result, "eligible": False, "reason": f"size-cap: {total} > {cap}"}
    return {**result, "eligible": True, "reason": ""}


def load_or_select(api: Any, out: Path, base: str, k: int, host: str) -> dict[str, Any]:
    path = out / "selection" / f"{slug(base)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    described = describe_base(api, base)
    selection = {
        "format": SELECTION_FORMAT,
        "listed_at": now(),
        "base": described,
        "k": k,
        "size_cap_bytes": min(
            int(SIZE_CAP_FACTOR * described["safetensors_bytes"]), SIZE_CAP_ABSOLUTE
        ),
        "walked": walk_candidates(api, base, k, described["safetensors_bytes"]),
    }
    write_json(path, selection)
    for entry in selection["walked"]:
        if not entry["eligible"]:
            append_row(
                out,
                base_row(selection, entry, host)
                | {"outcome": "skipped", "reason": entry["reason"]},
            )
    return selection


# ---------------------------------------------------------------- cache accounting


def cache_files(cache: Path) -> dict[str, int]:
    """Every regular file under the cache, by path relative to it, with its size."""
    files = {}
    for root, _, names in os.walk(cache):
        for name in names:
            path = Path(root) / name
            if not path.is_symlink() and path.is_file():
                files[str(path.relative_to(cache))] = path.stat().st_size
    return files


def collect_garbage(cache: Path, upload: str) -> int:
    """Remove the upload's repository and every blob no remaining snapshot points at."""
    repo_dir = cache / f"models--{upload.replace('/', '--')}"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    referenced = set()
    for snapshots in cache.glob("models--*/snapshots"):
        for link in snapshots.rglob("*"):
            if link.is_symlink():
                referenced.add(link.resolve())
    freed = 0
    for blobs in [cache / "blobs", *cache.glob("models--*/blobs")]:
        for blob in blobs.rglob("*") if blobs.exists() else ():
            if (
                blob.is_file()
                and not blob.name.startswith(".")
                and blob.resolve() not in referenced
            ):
                freed += blob.stat().st_size
                blob.unlink()
    xet = cache / "xet"
    if xet.exists():
        shutil.rmtree(xet)
    return freed


def is_cached(cache: Path, repo: str, sha: str, name: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(repo, name, cache_dir=str(cache), revision=sha)
    return isinstance(found, str)


def estimate_fetch(cache: Path, selection: dict[str, Any], entry: dict[str, Any]) -> int:
    """Bytes the diff will download: shards that differ on the Hub and are not cached yet."""
    base = selection["base"]
    base_by_name = {shard["name"]: shard for shard in base["shards"]}
    skipped = {
        shard["name"]
        for shard in entry["shards"]
        if shard["sha256"]
        and (other := base_by_name.get(shard["name"])) is not None
        and (other["sha256"], other["nbytes"]) == (shard["sha256"], shard["nbytes"])
    }
    upload_bytes = sum(shard["nbytes"] for shard in entry["shards"] if shard["name"] not in skipped)
    base_bytes = sum(
        shard["nbytes"]
        for shard in base["shards"]
        if shard["name"] not in skipped
        and not is_cached(cache, base["repo"], base["sha"], shard["name"])
    )
    return upload_bytes + base_bytes


# ---------------------------------------------------------------- one pair


def run_wvc(
    base: str, base_sha: str, upload: str, upload_sha: str, env: dict[str, str]
) -> tuple[subprocess.CompletedProcess | None, float]:
    command = [
        sys.executable,
        "-m",
        "wvc",
        "diff",
        "--rank",
        "--json",
        f"hf:{base}@{base_sha}",
        f"hf:{upload}@{upload_sha}",
    ]
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command, capture_output=True, env=env, timeout=PAIR_TIMEOUT_SECONDS, check=False
        )
    except subprocess.TimeoutExpired:
        return None, time.monotonic() - start
    return proc, time.monotonic() - start


def wvc_command(base: str, base_sha: str, upload: str, upload_sha: str) -> str:
    return f"wvc diff --rank --json hf:{base}@{base_sha} hf:{upload}@{upload_sha}"


def parse_document(proc: subprocess.CompletedProcess) -> dict[str, Any] | None:
    if proc.returncode not in (0, 1):
        return None
    try:
        document = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) and "summary" in document else None


def failure_reason(proc: subprocess.CompletedProcess | None) -> str:
    if proc is None:
        return f"timeout after {PAIR_TIMEOUT_SECONDS} s"
    stderr = proc.stderr.decode("utf-8", "replace").strip()
    lines = [line for line in stderr.splitlines() if line.strip()]
    if "Traceback (most recent call last)" in stderr:
        return f"crash: {lines[-1][:300] if lines else 'no message'}"
    if proc.returncode == 2:
        return f"refused: {lines[0][:300] if lines else 'no message'}"
    return f"exit {proc.returncode}: {lines[-1][:300] if lines else 'no JSON document'}"


def rank_stats(document: dict[str, Any]) -> dict[str, Any]:
    ranks = [
        t["rank"]
        for t in document.get("tensors", ())
        if t.get("status") == "changed" and t.get("rank") is not None
    ]
    return {
        "rank_measured": len(ranks),
        "rank_max": max(ranks) if ranks else "",
        "rank_median": statistics.median(ranks) if ranks else "",
        "rank_le_64": sum(1 for rank in ranks if rank <= LOW_RANK_MAX),
    }


def diff_row(document: dict[str, Any]) -> dict[str, Any]:
    summary, counts = document["summary"], document["summary"]["by_status"]
    files_a, files_b = document["a"]["files"], document["b"]["files"]
    return {
        "tensors": summary["tensors"],
        **{
            status: counts[status]
            for status in ("unchanged", "changed", "added", "removed", "reshaped")
        },
        "fraction_changed": summary["fraction_changed"]
        if summary["fraction_changed"] is not None
        else "",
        **rank_stats(document),
        "shards_skipped": len(files_a["skipped"]),
        "fetched_a": len(files_a["fetched"]),
        "fetched_b": len(files_b["fetched"]),
    }


def base_row(selection: dict[str, Any], entry: dict[str, Any], host: str) -> dict[str, Any]:
    return {
        "run_at": now(),
        "host": host,
        "base": selection["base"]["repo"],
        "upload": entry["upload"],
        "listing_rank": entry["listing_rank"],
        "base_sha": selection["base"]["sha"],
        "upload_sha": entry["sha"] or "",
        "relation_tag": entry["relation_tag"] or "",
        "card_base_model_relation": entry["card_base_model_relation"] or "",
        "downloads_30d": entry["downloads_30d"],
        "created_at": entry["created_at"] or "",
        "upload_safetensors_bytes": sum(shard["nbytes"] for shard in entry["shards"]),
        "shards_a": len(selection["base"]["shards"]),
        "shards_b": len(entry["shards"]),
        "wvc_version": version("wvc"),
    }


def provenance() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "src"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not commit:  # a synced copy without .git (PACE): the submitter names the commit
        commit = os.environ.get("WVC_CENSUS_SOURCE_COMMIT", "")
        dirty = os.environ.get("WVC_CENSUS_SOURCE_DIRTY", "")
    return {
        "wvc_version": version("wvc"),
        "wvc_source_commit": commit or None,
        "wvc_source_dirty": bool(dirty),
        "huggingface_hub": version("huggingface_hub"),
        "numpy": version("numpy"),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
    }


def census_block(selection: dict[str, Any], entry: dict[str, Any], host: str) -> dict[str, Any]:
    base = selection["base"]
    return {
        "experiment": "E009",
        "host": host,
        "base": base["repo"],
        "base_sha": base["sha"],
        "upload": entry["upload"],
        "upload_sha": entry["sha"],
        "listing_rank": entry["listing_rank"],
        "listed_at": selection["listed_at"],
        "relation_tag": entry["relation_tag"],
        "card_base_model": entry["card_base_model"],
        "card_base_model_relation": entry["card_base_model_relation"],
        "downloads_30d": entry["downloads_30d"],
        "created_at": entry["created_at"],
        "upload_shards": entry["shards"],
        "command": wvc_command(base["repo"], base["sha"], entry["upload"], entry["sha"]),
        **provenance(),
    }


def fetched_shards(
    document: dict[str, Any], selection: dict[str, Any], entry: dict[str, Any]
) -> dict[str, list]:
    by_name_a = {shard["name"]: shard for shard in selection["base"]["shards"]}
    by_name_b = {shard["name"]: shard for shard in entry["shards"]}
    return {
        "a": [by_name_a.get(name, {"name": name}) for name in document["a"]["files"]["fetched"]],
        "b": [by_name_b.get(name, {"name": name}) for name in document["b"]["files"]["fetched"]],
    }


def run_pair(
    args: argparse.Namespace,
    selection: dict[str, Any],
    entry: dict[str, Any],
    state: dict[str, Any],
) -> bool:
    """Diff one pair; False when a budget would be exceeded and the run must stop."""
    cache: Path = args.cache
    estimate = estimate_fetch(cache, selection, entry)
    free = shutil.disk_usage(cache).free
    if state["bytes_new"] + estimate > args.max_bytes:
        print(
            f"stop: {entry['upload']} needs ~{gb(estimate)}; "
            f"run total {gb(state['bytes_new'])} of {gb(args.max_bytes)}"
        )
        return False
    if free - estimate < args.min_free:
        print(
            f"stop: {entry['upload']} needs ~{gb(estimate)} with {gb(free)} free; "
            f"floor {gb(args.min_free)}"
        )
        return False
    base = selection["base"]
    before = cache_files(cache)
    proc, wall = run_wvc(base["repo"], base["sha"], entry["upload"], entry["sha"], state["env"])
    after = cache_files(cache)
    bytes_new = sum(size for path, size in after.items() if before.get(path) != size)
    state["bytes_new"] += bytes_new
    block = census_block(selection, entry, args.host) | {
        "exit_code": proc.returncode if proc else None,
        "wall_seconds": round(wall, 1),
        "bytes_new": bytes_new,
        "run_at": now(),
    }
    row = base_row(selection, entry, args.host) | {
        "bytes_new": bytes_new,
        "wall_seconds": round(wall, 1),
    }
    document = parse_document(proc) if proc else None
    name = f"{slug(entry['upload'])}.json"
    if document is not None:
        block["shards_fetched"] = fetched_shards(document, selection, entry)
        write_json(
            args.out / "pairs" / name, {"format": RECEIPT_FORMAT, "census": block, "diff": document}
        )
        append_row(args.out, row | {"outcome": "diffed"} | diff_row(document))
    else:
        reason = failure_reason(proc)
        stderr = proc.stderr.decode("utf-8", "replace")[-STDERR_KEEP:] if proc else ""
        write_json(
            args.out / "failures" / name,
            {
                "format": FAILURE_FORMAT,
                "census": block,
                "reason": reason,
                "stderr_tail": scrub(stderr),
            },
        )
        append_row(args.out, row | {"outcome": "failed", "reason": reason})
    freed = collect_garbage(cache, entry["upload"])
    outcome = "diffed" if document is not None else "FAILED"
    print(
        f"{outcome} {entry['upload']}: {wall:.0f} s, +{gb(bytes_new)} "
        f"(run total {gb(state['bytes_new'])}), freed {gb(freed)}"
    )
    return True


def scrub(text: str) -> str:
    """Drop machine paths from a failure's stderr: home and cache prefixes become placeholders."""
    for prefix, placeholder in ((str(Path.home()), "~"), (str(REPO), "<repo>")):
        text = text.replace(prefix, placeholder)
    return text


# ---------------------------------------------------------------- files


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n"
    )
    tmp.replace(path)


def append_row(out: Path, row: dict[str, Any]) -> None:
    path = out / "census.csv"
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def failed_before(out: Path) -> set[tuple[str, str]]:
    path = out / "census.csv"
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["base"], row["upload"])
            for row in csv.DictReader(handle)
            if row["outcome"] == "failed"
        }


# ---------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cache", type=Path, required=True, help="a dedicated Hugging Face hub cache"
    )
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument(
        "--group", action="append", help="small, large-laptop, large-pace (repeatable)"
    )
    parser.add_argument("--base", action="append", help="limit to these bases (repeatable)")
    parser.add_argument(
        "--max-bytes", type=float, required=True, help="stop before a pair would pass this"
    )
    parser.add_argument(
        "--min-free", type=float, required=True, help="stop before free disk would fall under this"
    )
    parser.add_argument("--host", default="laptop")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--select-only", action="store_true", help="write the selections and stop")
    parser.add_argument("--purge-bases", action="store_true", help="empty the cache at the end")
    args = parser.parse_args(argv)
    args.cache = args.cache.expanduser().resolve()
    return args


def chosen_bases(args: argparse.Namespace) -> list[tuple[str, int]]:
    return [
        (base, k)
        for base, (group, k) in BASES.items()
        if (not args.group or group in args.group) and (not args.base or base in args.base)
    ]


def main(argv: list[str] | None = None) -> int:
    from huggingface_hub import HfApi

    args = parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "HF_HUB_CACHE": str(args.cache),
        "HF_XET_CACHE": str(args.cache / "xet"),
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }
    os.environ["HF_HUB_CACHE"] = str(args.cache)
    api = HfApi()
    state = {"bytes_new": 0, "env": env}
    retry = failed_before(args.out) if args.retry_failed else set()
    done_failed = failed_before(args.out) - retry
    start = time.monotonic()
    for base, k in chosen_bases(args):
        selection = load_or_select(api, args.out, base, k, args.host)
        print(
            f"{base}: {sum(e['eligible'] for e in selection['walked'])} eligible "
            f"of {len(selection['walked'])} walked"
        )
        if args.select_only:
            continue
        for entry in selection["walked"]:
            if (
                not entry["eligible"]
                or (args.out / "pairs" / f"{slug(entry['upload'])}.json").exists()
            ):
                continue
            if (base, entry["upload"]) in done_failed:
                continue
            if not run_pair(args, selection, entry, state):
                print(
                    f"stopped on budget after {time.monotonic() - start:.0f} s; "
                    f"{gb(state['bytes_new'])} new"
                )
                return 3
    if args.purge_bases:
        shutil.rmtree(args.cache)
    print(
        f"done in {time.monotonic() - start:.0f} s; {gb(state['bytes_new'])} written to the cache"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
