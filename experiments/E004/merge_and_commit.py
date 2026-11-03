"""Merge each saved E004 adapter into the base and commit it as one wvc revision.

The repository is initialised at experiments/E004/repo, and the base itself is the first
commit (`--op pretrain-checkpoint`, message `step 0`). Then, for every checkpoints/step-<n>/
in step order: the base `model.safetensors` is copied into merged/step-<n>/, each target
tensor's byte range is overwritten in place with the merged tensor of `lowrank-merge/0`
(spec section 3.2: float64, W + scale * sum_k outer(B[:, k], A[k, :]) by elementwise numpy
operations, float32 by round-to-nearest-even, then bf16 by the round-to-nearest-even bit
trick), so the header and every other byte stay the base's; the directory is committed with
`uv run wvc commit merged/step-<n> -m "step <n>" --op adapter-merge --repo ...`; and the merged
file is deleted. Commit ids, the counts each commit prints, the counts `wvc show` gives
against the parent, and the store size after each commit go to results/commits.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os

os.environ.setdefault("WVC_AUTHOR", "evansmith")
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from e004_config import (
    BASE_MODEL,
    BASE_REVISION,
    CHECKPOINTS,
    MERGED,
    REPO,
    RESULTS,
    ROOT,
    WEIGHTS,
    check_disk,
    snapshot_dir,
    utc_now,
    write_json,
)

from wvc.safetensors_io import load_checkpoint, read_header

ADAPTER_KEY = re.compile(r"^base_model\.model\.(?P<path>.+)\.lora_(?P<factor>[AB])\.weight$")
COMMIT_LINE = re.compile(
    r"^commit (?P<id>[0-9a-f]{64}) \((?P<branch>[^,]+), "
    r"(?:no parent|parent (?P<parent>[0-9a-f]+))\)$"
)
COUNTS_LINE = re.compile(
    r"^tensors: (?P<tensors>\d+) \((?P<reused>\d+) reused by hash, (?P<full>\d+) stored full, "
    r"(?P<delta>\d+) stored as delta(?:: (?P<forms>.*))?\)$"
)
PAYLOAD_LINE = re.compile(r"^payload bytes written: (?P<bytes>[\d,]+) in (?P<seconds>[\d.]+) s$")
REFUSED_CONFIG = ("use_dora", "fan_in_fan_out", "rank_pattern", "alpha_pattern", "modules_to_save")


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 values to bfloat16 (round-to-nearest-even) and return the 16-bit patterns.

    Copied from tests/conftest.py.
    """
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_f64(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)


def lowrank_merge(w_bits: np.ndarray, a: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    """`lowrank-merge/0` of spec 3.2 for a bf16 base tensor; returns bf16 bit patterns."""
    w = bf16_bits_to_f64(w_bits)
    d = np.zeros_like(w)
    for k in range(a.shape[0]):  # increasing k; outer products by broadcasting, no BLAS
        d += b[:, k][:, np.newaxis] * a[k, :][np.newaxis, :]
    return f32_to_bf16_bits((w + scale * d).astype(np.float32))


def read_adapter(adapter_dir: Path) -> tuple[float, dict]:
    """The scale and, per base tensor name, the factors (A [r, in], B [out, r]) in float64."""
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or any(config.get(key) for key in REFUSED_CONFIG):
        raise SystemExit(f"{adapter_dir}: unsupported adapter configuration")
    if config.get("layers_pattern") or config.get("layers_to_transform"):
        raise SystemExit(f"{adapter_dir}: layers_pattern and layers_to_transform are refused")
    r, alpha = config["r"], config["lora_alpha"]
    scale = alpha / math.sqrt(r) if config.get("use_rslora") else alpha / r
    ckpt = load_checkpoint(adapter_dir / "adapter_model.safetensors")
    factors: dict[str, dict[str, np.ndarray]] = {}
    for name in ckpt.tensors:
        match = ADAPTER_KEY.match(name)
        if match is None:
            raise SystemExit(f"{adapter_dir}: unexpected adapter tensor {name!r}")
        target = f"{match['path']}.weight"
        values = ckpt.as_float32(name).astype(np.float64)  # exact for F32 and BF16
        factors.setdefault(target, {})[match["factor"]] = values
    if any(set(pair) != {"A", "B"} for pair in factors.values()):
        raise SystemExit(f"{adapter_dir}: every target needs both lora_A and lora_B")
    return float(scale), {name: (pair["A"], pair["B"]) for name, pair in factors.items()}


def write_merged(base_file: Path, out_file: Path, scale: float, pairs: dict) -> dict:
    """Copy the base file and overwrite each target tensor's bytes with its merged tensor."""
    header = read_header(base_file)
    base = load_checkpoint(base_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base_file, out_file)  # follows the cache symlink; the copy is writable
    changed_elements, changed_tensors = 0, 0
    with out_file.open("r+b") as handle:
        for name, (a, b) in sorted(pairs.items()):
            info = header.tensors[name]
            out_dim, in_dim = info.shape
            if info.dtype != "BF16" or a.shape[1] != in_dim or b.shape != (out_dim, a.shape[0]):
                raise SystemExit(f"{name}: shapes or dtype do not fit a [out, in] BF16 merge")
            w_bits = np.frombuffer(base.raw(name), dtype="<u2").reshape(info.shape)
            merged = lowrank_merge(w_bits, a, b, scale)
            payload = merged.astype("<u2").tobytes()
            if len(payload) != info.nbytes:
                raise SystemExit(f"{name}: merged tensor has {len(payload)} bytes")
            handle.seek(header.data_start + info.begin)
            handle.write(payload)
            differs = int(np.count_nonzero(merged != w_bits))
            changed_elements += differs
            changed_tensors += differs > 0
    with base_file.open("rb") as a_file, out_file.open("rb") as b_file:
        if a_file.read(header.data_start) != b_file.read(header.data_start):
            raise SystemExit(f"{out_file}: header differs from the base header")
    if out_file.stat().st_size != base_file.stat().st_size:
        raise SystemExit(f"{out_file}: size differs from the base")
    return {
        "targets": len(pairs),
        "scale": scale,
        "tensors_changed_vs_base": changed_tensors,
        "elements_changed_vs_base": changed_elements,
    }


def wvc(*args: str) -> subprocess.CompletedProcess:
    """Run `uv run wvc ...` from the repository root with the project's own environment."""
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    return subprocess.run(
        ["uv", "run", "wvc", *args], cwd=ROOT, env=env, capture_output=True, text=True
    )


def parse_commit(stdout: str) -> dict:
    lines = stdout.strip().splitlines()
    head, counts, payload = (
        COMMIT_LINE.match(lines[0]),
        COUNTS_LINE.match(lines[1]),
        PAYLOAD_LINE.match(lines[2]),
    )
    if not (head and counts and payload):
        raise SystemExit(f"cannot parse the commit output:\n{stdout}")
    forms = {}
    for part in filter(None, (counts["forms"] or "").split(", ")):
        number, form = part.split(" ", 1)
        forms[form] = int(number)
    return {
        "commit": head["id"],
        "branch": head["branch"],
        "parent_prefix": head["parent"],
        "tensors": int(counts["tensors"]),
        "reused_by_hash": int(counts["reused"]),
        "stored_full": int(counts["full"]),
        "stored_as_delta": int(counts["delta"]),
        "delta_forms": forms,
        "payload_bytes_written": int(payload["bytes"].replace(",", "")),
        "wvc_seconds": float(payload["seconds"]),
    }


def show_counts(repo: Path, commit: str) -> dict:
    proc = wvc("show", commit, "--json", "--repo", str(repo))
    if proc.returncode != 0:
        raise SystemExit(f"wvc show {commit} failed:\n{proc.stderr}")
    comparison = json.loads(proc.stdout)["comparison"]
    if comparison is None:
        return {"against_parent": None}
    changed = comparison["changed"]
    return {
        "against_parent": comparison["counts"],
        "changed_are_all_down_proj": all(name.endswith("mlp.down_proj.weight") for name in changed),
    }


def store_bytes(repo: Path) -> int:
    return sum(path.stat().st_size for path in (repo / ".wvc").rglob("*") if path.is_file())


def commit(repo: Path, checkpoint: Path, message: str, op: str, previous: int) -> dict:
    started = time.perf_counter()
    proc = wvc("commit", str(checkpoint), "-m", message, "--op", op, "--repo", str(repo))
    seconds = round(time.perf_counter() - started, 2)
    if proc.returncode != 0:
        raise RuntimeError(f"wvc commit {checkpoint} exited {proc.returncode}:\n{proc.stderr}")
    record = parse_commit(proc.stdout)
    size = store_bytes(repo)
    return {
        **record,
        "message": message,
        "operation": op,
        "wall_seconds": seconds,
        "store_bytes": size,
        "store_growth_bytes": size - previous,
        "show": show_counts(repo, record["commit"]),
    }


def adapter_steps(checkpoints: Path) -> list[int]:
    found = [
        int(path.name.removeprefix("step-"))
        for path in checkpoints.glob("step-*")
        if (path / "adapter_model.safetensors").is_file()
    ]
    return sorted(found)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
    parser.add_argument("--merged", type=Path, default=MERGED)
    parser.add_argument("--out", type=Path, default=RESULTS / "commits.json")
    parser.add_argument("--fresh", action="store_true", help="delete an existing repository")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started, started_utc = time.perf_counter(), utc_now()
    check_disk("merging and committing")
    base_dir = snapshot_dir(BASE_MODEL, BASE_REVISION)
    steps = adapter_steps(args.checkpoints)
    if not steps:
        print(f"no adapters under {args.checkpoints}", file=sys.stderr)
        return 2
    if args.repo.exists():
        if not args.fresh:
            print(f"{args.repo} exists; pass --fresh to replace it", file=sys.stderr)
            return 2
        shutil.rmtree(args.repo)
    args.repo.mkdir(parents=True)
    init = wvc("init", str(args.repo))
    if init.returncode != 0:
        print(init.stderr, file=sys.stderr)
        return 1
    revisions: list[dict] = []
    payload = {"started": started_utc, "status": "running", "base": str(base_dir)}
    payload["revisions"] = revisions
    try:
        revisions.append(
            {"step": 0, **commit(args.repo, base_dir, "step 0", "pretrain-checkpoint", 0)}
        )
        for step in steps:
            check_disk(f"merging step {step}")
            t0 = time.perf_counter()
            scale, pairs = read_adapter(args.checkpoints / f"step-{step}")
            out_dir = args.merged / f"step-{step}"
            merge = write_merged(base_dir / WEIGHTS, out_dir / WEIGHTS, scale, pairs)
            merge["seconds"] = round(time.perf_counter() - t0, 2)
            previous = revisions[-1]["store_bytes"]
            record = commit(args.repo, out_dir, f"step {step}", "adapter-merge", previous)
            revisions.append({"step": step, "merge": merge, **record})
            shutil.rmtree(out_dir)
            print(
                f"step {step}: {record['commit'][:12]} reused {record['reused_by_hash']}, "
                f"delta {record['stored_as_delta']} {record['delta_forms']}, "
                f"store +{record['store_growth_bytes']:,} B",
                flush=True,
            )
            payload["seconds"] = round(time.perf_counter() - started, 1)
            write_json(args.out, payload)
        payload["status"] = "complete"
    except (RuntimeError, SystemExit) as exc:
        payload["status"] = "failed"
        payload["error"] = str(exc)
        print(exc, file=sys.stderr)
    log = wvc("log", "--repo", str(args.repo))
    payload.update(
        finished=utc_now(),
        seconds=round(time.perf_counter() - started, 1),
        wvc_log=log.stdout.splitlines(),
        store_bytes_final=store_bytes(args.repo),
    )
    write_json(args.out, payload)
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
