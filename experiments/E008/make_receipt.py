"""Build receipt.json for E008 from the run log, the diff outputs, hub-plan.txt and the cache.

Usage: uv run python -I make_receipt.py RUN_LOG HUB_PLAN OUT_DIR HF_HUB_CACHE VERSIONS_JSON
"""

import json
import re
import sys
from pathlib import Path

from wvc.safetensors_io import read_header

SHARD2 = "model.safetensors-00002-of-00004.safetensors"
PLANNED_RANK_CAP = 1 << 24  # the --rank-cap default at commit 48c824b, where the run happened
RAISED_RANK_CAP = 1 << 26
LOW_RANK = 4
P2_FRACTION = 0.1
P5_SECONDS = 300
HEADER_LINE = "3 byte-identical on the Hub and skipped; fetched 1 from a and 1 from b"


def parse_log(path: Path) -> tuple[dict[str, str | int], list[dict[str, str | int]]]:
    log: dict[str, str | int] = {}
    shards: list[dict[str, str | int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "shard":
            shards.append({"snapshot_path": parts[1], "bytes": int(parts[3]), "sha256": parts[5]})
        elif re.fullmatch(r"run\d_rc", parts[0]):
            log[parts[0]] = int(parts[1])
            log[parts[2]] = int(parts[3])
        elif len(parts) >= 2:
            log[parts[0]] = parts[1]
    return log, shards


def parse_plan(path: Path) -> dict[str, dict]:
    expected: dict[str, dict] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"hf:(\S+)@main\s+commit\s+([0-9a-f]{40})", line)
        if m:
            current = m.group(1)
            expected[current] = {"commit": m.group(2), "files": {}}
            continue
        m = re.match(r"\s+(\S+\.safetensors)\s+([\d,]+) bytes\s+sha256 ([0-9a-f]{64})", line)
        if m and current:
            expected[current]["files"][m.group(1)] = {
                "nbytes": int(m.group(2).replace(",", "")),
                "sha256": m.group(3),
            }
    return expected


def cache_dir_for(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def without_rank(tensor: dict) -> dict:
    return {k: v for k, v in tensor.items() if k not in ("rank", "top1_energy")}


def side_record(key: str, diff: dict, expected: dict, shards: list, hub_cache: Path) -> dict:
    side = diff[key]
    repo = side["hf"]["repo"]
    exp = expected[repo]
    blob = next(s for s in shards if str(s["snapshot_path"]).startswith(cache_dir_for(repo) + "/"))
    want = exp["files"][SHARD2]
    snapshot = hub_cache / cache_dir_for(repo) / "snapshots" / exp["commit"] / SHARD2
    header = read_header(snapshot)
    return {
        "repo": repo,
        "revision": side["hf"]["revision"],
        "commit": side["hf"]["sha"],
        "commit_matches_plan": side["hf"]["sha"] == exp["commit"],
        "skipped": [f["name"] for f in side["files"]["skipped"]],
        "skipped_match_plan": all(
            exp["files"][f["name"]] == {"nbytes": f["nbytes"], "sha256": f["sha256"]}
            for f in side["files"]["skipped"]
        ),
        "fetched": list(side["files"]["fetched"]),
        "downloaded_shard": {
            "name": SHARD2,
            "bytes": blob["bytes"],
            "sha256": blob["sha256"],
            "hub_lfs_bytes": want["nbytes"],
            "hub_lfs_sha256": want["sha256"],
            "matches_hub": blob["sha256"] == want["sha256"] and blob["bytes"] == want["nbytes"],
            "tensors_in_shard": len(header.tensors),
            "tensor_names_in_shard": sorted(header.tensors),
        },
        "root": side.get("root"),
    }


def main() -> None:
    log_path, plan_path, out_dir, hub_cache = (Path(p) for p in sys.argv[1:5])
    versions = json.loads(sys.argv[5])
    log, shards = parse_log(log_path)
    expected = parse_plan(plan_path)
    diff = json.loads((out_dir / "diff.json").read_text(encoding="utf-8"))
    diff_txt = (out_dir / "diff.txt").read_text(encoding="utf-8")
    ranked = json.loads((out_dir / "diff-rankcap.json").read_text(encoding="utf-8"))

    sides = {k: side_record(k, diff, expected, shards, hub_cache) for k in "ab"}
    shard2_names = set(sides["a"]["downloaded_shard"]["tensor_names_in_shard"])
    changed = [t for t in diff["tensors"] if t["status"] == "changed"]
    ranked_changed = {t["name"]: t for t in ranked["tensors"] if t["status"] == "changed"}
    summary = diff["summary"]
    by_status = summary["by_status"]
    same_tensors = ranked["summary"] == summary and [
        without_rank(t) for t in ranked["tensors"]
    ] == [without_rank(t) for t in diff["tensors"]]

    p1_held = (
        HEADER_LINE in diff_txt
        and all(sides[k]["fetched"] == [SHARD2] and len(sides[k]["skipped"]) == 3 for k in "ab")
        and all(sides[k]["downloaded_shard"]["matches_hub"] for k in "ab")
        and all(sides[k]["skipped_match_plan"] for k in "ab")
    )
    in_shard2 = all(t["name"] in shard2_names for t in changed)
    fraction = len(changed) / len(shard2_names)
    p2_held = in_shard2 and fraction < P2_FRACTION
    p3_held = all(by_status[key] == 0 for key in ("added", "removed", "reshaped"))
    ranks = [ranked_changed[t["name"]]["rank"] for t in changed]
    top1 = [ranked_changed[t["name"]]["top1_energy"] for t in changed]
    full = [min(t["shape_a"]) for t in changed]
    if changed and all(r is not None and r <= LOW_RANK for r in ranks):
        shape = (
            f"rank <= {LOW_RANK} in every changed tensor at the 1e-2 tolerance: a few directions; "
            "rank alone does not separate a baked steering direction, a low-r LoRA merge "
            "and a rank-one edit"
        )
    elif changed and all(r == f for r, f in zip(ranks, full, strict=True)):
        shape = "full rank in every changed tensor (dense)"
    elif changed and len(set(ranks)) == 1:
        shape = f"uniform rank {ranks[0]} below full rank"
    else:
        shape = "mixed or unreported ranks"
    run2 = log["run2_seconds"]
    p5_held = isinstance(run2, int) and run2 < P5_SECONDS

    receipt = {
        "experiment": "E008",
        "started_utc": log["start_utc"],
        "ended_utc": log["end_utc"],
        "wvc_commit": log["code_rev"],
        "software": versions,
        "inputs": {
            k: {key: sides[k][key] for key in ("repo", "revision", "commit", "commit_matches_plan")}
            for k in "ab"
        },
        "shards": {
            k: {
                key: sides[k][key]
                for key in ("skipped", "skipped_match_plan", "fetched", "downloaded_shard")
            }
            for k in "ab"
        },
        "download_bytes": sum(sides[k]["downloaded_shard"]["bytes"] for k in "ab"),
        "disk_free_gb": {"before": int(log["free_gb_before"]), "after": int(log["free_gb_after"])},
        "runs": {
            "run1": {
                "command": "wvc diff --rank A B (download + diff)",
                "seconds": log["run1_seconds"],
                "exit": log["run1_rc"],
            },
            "run2": {
                "command": "wvc diff --rank --json A B (cache)",
                "seconds": run2,
                "exit": log["run2_rc"],
            },
            "run3": {
                "command": f"wvc diff --rank --rank-cap {RAISED_RANK_CAP} A B (cache)",
                "seconds": log["run3_seconds"],
                "exit": log["run3_rc"],
            },
            "run4": {
                "command": f"wvc diff --rank --rank-cap {RAISED_RANK_CAP} --json A B (cache)",
                "seconds": log["run4_seconds"],
                "exit": log["run4_rc"],
            },
        },
        "deviation_from_plan": (
            f"the planned command skips the SVD when a touched block exceeds "
            f"{PLANNED_RANK_CAP} elements "
            "(--rank-cap default); the changed tensor's touched block is 4096 x 12288 = "
            "50,331,648, so runs 3 and 4 "
            f"repeat the diff from cache with --rank-cap {RAISED_RANK_CAP} to obtain the rank. "
            "Exit code 1 is the "
            "diff command's 'differences found' status."
        ),
        "rank_run_agrees_with_planned_run": same_tensors,
        "summary": summary,
        "roots": {k: sides[k]["root"] for k in "ab"},
        "changed_tensors": [ranked_changed[t["name"]] for t in changed],
        "predictions": {
            "P1": {
                "predicted": "downloads exactly the two copies of shard 2; header line "
                + repr(HEADER_LINE),
                "observed": {
                    k: {
                        "skipped": len(sides[k]["skipped"]),
                        "fetched": sides[k]["fetched"],
                        "blob_matches_hub": sides[k]["downloaded_shard"]["matches_hub"],
                    }
                    for k in "ab"
                },
                "held": p1_held,
            },
            "P2": {
                "predicted": "every changed tensor lives in shard 2 and fewer than "
                f"{P2_FRACTION:.0%} of shard 2's tensors change (60% credence)",
                "observed": {
                    "changed": len(changed),
                    "tensors_in_shard2": len(shard2_names),
                    "fraction": fraction,
                    "all_in_shard2": in_shard2,
                },
                "held": p2_held,
            },
            "P3": {
                "predicted": "nothing added, removed or reshaped",
                "observed": {key: by_status[key] for key in ("added", "removed", "reshaped")},
                "held": p3_held,
            },
            "P4": {
                "predicted": "recorded, not scored: rank <= 4 for a baked steering direction, "
                "uniform r for a merged LoRA, full rank for a dense fine-tune",
                "observed": {
                    "ranks": ranks,
                    "top1_energy": top1,
                    "full_rank_would_be": full,
                    "shape": shape,
                },
                "held": None,
            },
            "P5": {
                "predicted": f"wall time after the download under {P5_SECONDS} s",
                "observed": {"run2_seconds": run2, "run3_seconds_with_svd": log["run3_seconds"]},
                "held": p5_held,
            },
        },
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("predictions", {k: v["held"] for k, v in receipt["predictions"].items()})
    print("shape", shape, "ranks", ranks, "top1", top1)
    print(
        "changed", len(changed), "of", len(shard2_names), "in shard 2; fraction", round(fraction, 4)
    )
    print("rank run agrees", same_tensors)


if __name__ == "__main__":
    main()
