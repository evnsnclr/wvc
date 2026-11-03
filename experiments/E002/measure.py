"""E002: bytes needed to store checkpoint B when checkpoint A is already stored.

Runs the frozen plan in experiments/E002-storage-measurement.md on four revision pairs and
writes results/<pair>.json, results/summary.md, and receipt.json. Strategies live in
e002_strategies.py, the P4 generator in e002_synthetic.py, the self-test in e002_selftest.py.
Single process, numpy only, no model execution.

    uv run --with zstandard python experiments/E002/measure.py --selftest
    uv run --with zstandard python experiments/E002/measure.py --pair all \
        --out experiments/E002/results/

Timing: time.perf_counter wall time in this process, after an untimed pass over both files
(input hashing) has warmed the page cache. Selection strategies (S2, S3, S4u, S4) are timed
including the hashing or byte comparison of both A and B they need; their z variant adds the
zstd time of the selected bytes; S1 is S0 plus the zstd time of every tensor. S5 and S6 are
timed without their reconstruction check (verify_seconds). S7 times its S2z part plus the bf16
cast of the known factors.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zstandard
from e002_selftest import selftest
from e002_strategies import (
    INT_VIEWS,
    counterpart,
    payload_bytes,
    range_bytes,
    s0_all_tensors,
    s2_content_addressed_bytes,
    s2_tensor_dedup,
    s3_fixed_chunk_dedup,
    s4_cdc_dedup,
    s4u_cdc_upper_bound,
    s5_xor_delta,
    s6_sparse_delta,
    s7_low_rank_bound,
    unchanged_names,
    verify_payloads,
    zstd_size,
)
from e002_synthetic import generate_p4

from wvc.hashing import digest_bytes, tensor_digests
from wvc.safetensors_io import ITEMSIZES, Checkpoint, load_checkpoint

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRATCH = HERE / "scratch"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
CODE_FILES = ("measure.py", "e002_strategies.py", "e002_synthetic.py", "e002_selftest.py")

MIN_FREE_BYTES = 20 * 1024**3  # stop rule "20 GB", read conservatively as 20 GiB
S4_BUDGET_SECONDS = 30 * 60
P4_EXPECTED_TARGETS = 48
P3_EXPECTED_CHANGED = (
    "model.language_model.layers.15.mlp.down_proj.weight",
    "model.language_model.layers.19.mlp.down_proj.weight",
)
P3_EXPECTED_ELEMENTS = 130_188
ORDER = ("S0", "S1", "S2", "S2z", "S3", "S3z", "S4u", "S4uz", "S4", "S4z", "S5", "S6", "S7")
CROSS = ("S1", "S2z", "S3z", "S4uz", "S5", "S6", "S7")


class StopRule(RuntimeError):
    """A stop rule of the frozen plan fired (disk space, input sanity, reconstruction)."""


# ---------------------------------------------------------------- inputs


@dataclass(frozen=True)
class HubFile:
    repo: str
    filename: str
    revision: str
    download: bool = True

    def cache_dir(self) -> Path:
        return HF_HUB / ("models--" + self.repo.replace("/", "--"))


QWEN25 = HubFile("Qwen/Qwen2.5-0.5B", "model.safetensors", "main")
P3_FILE = "model.safetensors-00001-of-00001.safetensors"
P3_A = HubFile("Qwen/Qwen3.5-0.8B", P3_FILE, "2fc06364715b967f1860aea9cf38778875588b17", False)
P3_B = HubFile(
    "RikiyaT/Qwen3.5-0.8B-weight-space-programming",
    P3_FILE,
    "d1671a178705bcab7644806d23c535517f35723b",
    False,
)
PYTHIA = "EleutherAI/pythia-160m"
PAIRS: dict[str, tuple[str, HubFile, HubFile | None]] = {
    "P1": (
        "consecutive pretraining checkpoints, 1,000 steps apart",
        HubFile(PYTHIA, "model.safetensors", "step142000"),
        HubFile(PYTHIA, "model.safetensors", "step143000"),
    ),
    "P2": (
        "dense post-training fine-tune",
        QWEN25,
        HubFile("Qwen/Qwen2.5-0.5B-Instruct", "model.safetensors", "main"),
    ),
    "P3": ("targeted sparse rank-1 edit (E001)", P3_A, P3_B),
    "P4": ("adapter merge, synthetic: rank-8 update merged into q_proj and v_proj", QWEN25, None),
}


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def free_disk_bytes() -> int:
    return int(sh("df", "-k", "/System/Volumes/Data").splitlines()[-1].split()[3]) * 1024


@dataclass
class RunLog:
    command: str
    started: str = field(default_factory=now_iso)
    t0: float = field(default_factory=time.perf_counter)
    disk_checks: list[dict] = field(default_factory=list)
    downloads: list[dict] = field(default_factory=list)
    fetched: dict[HubFile, Path] = field(default_factory=dict)

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def check_disk(self, label: str) -> int:
        free = free_disk_bytes()
        self.disk_checks.append({"when": label, "at": now_iso(), "free_bytes": free})
        if free < MIN_FREE_BYTES:
            raise StopRule(f"free disk {free:,} B is below {MIN_FREE_BYTES:,} B at {label!r}")
        return free


def fetch(hub: HubFile, log: RunLog) -> Path:
    """Cached snapshot path of a Hub file, downloaded first unless the plan says it is cached."""
    if hub not in log.fetched:
        if hub.download:
            log.fetched[hub] = download(hub, log)
        else:
            path = hub.cache_dir() / "snapshots" / hub.revision / hub.filename
            if not path.is_file():
                raise FileNotFoundError(f"{path}: expected in the Hugging Face cache")
            log.fetched[hub] = path
    return log.fetched[hub]


def downloaded_path(stdout: str, filename: str) -> Path:
    """The cached path printed by `hf download` (hf 2.x prints it as `path=<path>`)."""
    for line in reversed(stdout.splitlines()):
        text = line.strip().removeprefix("path=")
        if text.endswith("/" + filename) and Path(text).is_file():
            return Path(text)
    raise RuntimeError(f"no cached path for {filename} in hf download output: {stdout!r}")


def download(hub: HubFile, log: RunLog) -> Path:
    before = {str(p) for p in hub.cache_dir().glob(f"snapshots/*/{hub.filename}")}
    free_before = log.check_disk(f"before download {hub.repo}@{hub.revision}")
    cmd = ["uvx", "--from", "huggingface_hub", "hf", "download", hub.repo, hub.filename]
    cmd += ["--revision", hub.revision]
    t0 = time.perf_counter()
    stdout = sh(*cmd)
    seconds = time.perf_counter() - t0
    path = downloaded_path(stdout, hub.filename)
    blob_mtime = dt.datetime.fromtimestamp(path.resolve().stat().st_mtime, dt.UTC)
    log.downloads.append(
        {
            "repo": hub.repo,
            "file": hub.filename,
            "revision": hub.revision,
            "resolved_commit": path.parent.name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "already_cached": str(path) in before,
            "blob_written_at": blob_mtime.isoformat(timespec="seconds"),
            "seconds": round(seconds, 3),
            "free_bytes_before": free_before,
            "free_bytes_after": log.check_disk(f"after download {hub.repo}@{hub.revision}"),
            "at": now_iso(),
            "command": " ".join(cmd),
            "hf_cli": sh("uvx", "--from", "huggingface_hub", "hf", "version"),
        }
    )
    return path


def describe_input(hub: HubFile | None, path: Path, ckpt: Checkpoint) -> dict:
    """Identity of one input file; its content SHA-256 must equal the cache blob name."""
    resolved = path.resolve()
    blob = Path(os.readlink(path)).name if path.is_symlink() else None
    sha = digest_bytes(np.memmap(resolved, dtype=np.uint8, mode="r"))
    if blob is not None and blob != sha:
        raise StopRule(f"{path}: content SHA-256 {sha} differs from its blob name {blob}")
    return {
        "repo": hub.repo if hub else None,
        "revision": hub.revision if hub else None,
        "resolved_commit": path.parent.name if hub else None,
        "file": path.name,
        "path": str(path),
        "resolved_path": str(resolved),
        "blob_sha256": blob,
        "sha256": sha,
        "bytes": resolved.stat().st_size,
        "tensor_count": len(ckpt.tensors),
        "tensor_bytes": sum(info.nbytes for info in ckpt.tensors.values()),
    }


# ---------------------------------------------------------------- measurement

SELECTIONS = (
    (("S0", "S1"), s0_all_tensors, False, "one frame per tensor"),
    (("S2", "S2z"), s2_tensor_dedup, False, "one frame per new tensor"),
    (("S3", "S3z"), s3_fixed_chunk_dedup, False, "one frame per new chunk"),
    (("S4u", "S4uz"), s4u_cdc_upper_bound, True, "one frame per tensor, ranges joined"),
)


def timed(fn: Callable, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def run_selection(keys, fn, a: Checkpoint, b: Checkpoint, per_tensor: bool, frames: str) -> dict:
    raw_key, z_key = keys
    (ranges, stats), seconds = timed(fn, a, b)
    zbytes, zseconds = timed(zstd_size, b, ranges, per_tensor)
    return {
        raw_key: {"bytes": range_bytes(ranges), "seconds": seconds, "details": stats},
        z_key: {
            "bytes": zbytes,
            "seconds": seconds + zseconds,
            "details": {"zstd_seconds": round(zseconds, 4), "zstd_frames": frames},
        },
    }


def run_delta(fn, a: Checkpoint, b: Checkpoint, da: dict, db: dict) -> dict:
    payloads, seconds = timed(fn, a, b)
    (ok, failures), verify_seconds = timed(verify_payloads, a, b, payloads, db)
    dz = zstandard.ZstdDecompressor()
    sparse = [frames for kind, frames in payloads.values() if kind == "sparse"]
    unchanged = unchanged_names(a, b, da, db)
    details = {
        "tensors_by_encoding": dict(Counter(kind for kind, _ in payloads.values())),
        "changed_elements_in_sparse_tensors": sum(len(dz.decompress(f[0])) // 4 for f in sparse),
        "bytes_from_unchanged_tensors": payload_bytes(payloads[n] for n in unchanged),
        "reconstruction_failures": failures,
    }
    return {
        "bytes": payload_bytes(payloads.values()),
        "seconds": seconds,
        "reconstruction_ok": ok,
        "verify_seconds": round(verify_seconds, 4),
        "details": details,
    }


def with_ratios(results: dict, s0: int) -> dict:
    """Entries in ORDER, with ratio_to_S0 and rounded seconds for every measured entry."""
    out = {}
    for key in ORDER:
        if key in results:
            entry = dict(results[key])
            if "bytes" in entry:
                entry["ratio_to_S0"] = round(entry["bytes"] / s0, 8)
                entry["seconds"] = round(entry["seconds"], 4)
            out[key] = entry
    return out


def measure_strategies(a: Checkpoint, b: Checkpoint, da: dict, db: dict, factors) -> dict:
    results: dict[str, dict] = {}
    for keys, fn, per_tensor, frames in SELECTIONS:
        results.update(run_selection(keys, fn, a, b, per_tensor, frames))
    variant = s2_content_addressed_bytes(a, b, da, db)
    results["S2"]["details"]["content_addressed_bytes"] = variant
    results["S2"]["details"]["content_addressed_equals_by_name"] = variant == results["S2"]["bytes"]
    results["S5"] = run_delta(s5_xor_delta, a, b, da, db)
    results["S6"] = run_delta(s6_sparse_delta, a, b, da, db)
    if factors is not None:
        (nbytes, details), seconds = timed(s7_low_rank_bound, a, b, factors)
        results["S7"] = {"bytes": nbytes, "seconds": seconds, "details": details}
    return results


def pairing_summary(a: Checkpoint, b: Checkpoint, da: dict, db: dict) -> dict:
    common = [n for n in b.tensors if n in a.tensors]
    matched = [n for n in common if counterpart(a, b, n)]
    unchanged = unchanged_names(a, b, da, db)
    return {
        "tensors_a": len(a.tensors),
        "tensors_b": len(b.tensors),
        "same_name_dtype_shape": len(matched),
        "unchanged": len(unchanged),
        "changed": len(matched) - len(unchanged),
        "dtype_or_shape_differs": len(common) - len(matched),
        "only_in_b": len(b.tensors) - len(common),
        "only_in_a": len(set(a.tensors) - set(b.tensors)),
    }


def changed_elements(a: Checkpoint, b: Checkpoint, name: str) -> int:
    view = INT_VIEWS[ITEMSIZES[b.tensors[name].dtype]]
    return int(np.count_nonzero(a.raw(name).view(view) != b.raw(name).view(view)))


def check_p3(a: Checkpoint, b: Checkpoint, da: dict, db: dict) -> dict:
    changed = sorted(n for n in b.tensors if counterpart(a, b, n) and da[n] != db[n])
    same_set = set(a.tensors) == set(b.tensors) and all(counterpart(a, b, n) for n in b.tensors)
    elements = sum(changed_elements(a, b, n) for n in changed)
    record = {"changed_tensors": changed, "changed_elements": elements, "same_tensors": same_set}
    record["passed"] = (
        same_set and tuple(changed) == P3_EXPECTED_CHANGED and elements == P3_EXPECTED_ELEMENTS
    )
    if not record["passed"]:
        raise StopRule(f"P3 sanity check failed: {record}")
    return record


def check_p4(a: Checkpoint, b: Checkpoint, da: dict, db: dict, targets: list[str]) -> dict:
    changed = sorted(n for n in b.tensors if da.get(n) != db[n])
    fa, fb = a.files[0], b.files[0]
    with fa.path.open("rb") as ha, fb.path.open("rb") as hb:
        header_same = ha.read(fa.data_start) == hb.read(fb.data_start)
    record = {"targets": len(targets), "changed_tensors": len(changed)}
    record["header_bytes_identical"] = header_same
    record["passed"] = (
        len(targets) == P4_EXPECTED_TARGETS
        and changed == sorted(targets)
        and header_same
        and set(a.tensors) == set(b.tensors)
        and all(counterpart(a, b, n) for n in b.tensors)
    )
    if not record["passed"]:
        raise StopRule(f"P4 synthetic file check failed: {record}")
    return record


# ---------------------------------------------------------------- per-pair run


def load_pair(pair: str, log: RunLog) -> tuple[Path, Path, dict | None, dict | None]:
    _, hub_a, hub_b = PAIRS[pair]
    path_a = fetch(hub_a, log)
    if hub_b is not None:
        return path_a, fetch(hub_b, log), None, None
    log.check_disk("before generating P4 B")
    out = generate_p4(path_a, SCRATCH / "P4-B.safetensors", SCRATCH / "P4-factors.npz")
    log.check_disk("after generating P4 B")
    return path_a, *out


def run_pair(pair: str, log: RunLog, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    log.check_disk(f"before {pair}")
    kind, hub_a, hub_b = PAIRS[pair]
    path_a, path_b, factors, generation = load_pair(pair, log)
    a, b = load_checkpoint(path_a), load_checkpoint(path_b)
    inputs = {"a": describe_input(hub_a, path_a, a), "b": describe_input(hub_b, path_b, b)}
    if generation is not None:
        inputs["b"]["synthetic"] = generation
    da, db = tensor_digests(a), tensor_digests(b)
    sanity = check_p3(a, b, da, db) if pair == "P3" else None
    if pair == "P4":
        sanity = check_p4(a, b, da, db, list(factors or {}))
    s0 = sum(info.nbytes for info in b.tensors.values())
    strategies = with_ratios(measure_strategies(a, b, da, db, factors), s0)
    record = {
        "experiment": "E002",
        "pair": pair,
        "kind": kind,
        "inputs": inputs,
        "pairing": pairing_summary(a, b, da, db),
        "sanity_check": sanity,
        "strategies": strategies,
        "reconstruction": {k: strategies[k]["reconstruction_ok"] for k in ("S5", "S6")},
        "notes": pair_notes(pair),
        "pair_seconds": round(time.perf_counter() - t0, 3),
        "measured_at": now_iso(),
    }
    write_json(out_dir / f"{pair}.json", record)
    if not all(record["reconstruction"].values()):
        raise StopRule(f"{pair}: reconstruction failed: {record['reconstruction']}")
    return record


def pair_notes(pair: str) -> list[str]:
    notes = [
        "S2 pairs tensors by name: a tensor of B is free when the same-named tensor of A has "
        "equal dtype, shape, and SHA-256; details.content_addressed_bytes ignores names.",
        "S3 and S4 count every chunk of B whose SHA-256 is absent from A's chunk set, "
        "including repeats within B (details.new_distinct_bytes counts repeats once).",
        "S5 and S6 include the zstd frames of tensors that did not change "
        "(details.bytes_from_unchanged_tensors).",
    ]
    if pair == "P4":
        notes.append(
            "B is synthetic. S7 is a bound from the known factors (U and V cast to bf16, "
            "scale folded into U) plus S2z of the other tensors; it does not rebuild B exactly."
        )
    else:
        notes.append("S7 applies to P4 only.")
    return notes


def add_s4(record: dict, out_dir: Path) -> None:
    a = load_checkpoint(Path(record["inputs"]["a"]["path"]))
    b = load_checkpoint(Path(record["inputs"]["b"]["path"]))
    frames = "one frame per new chunk"
    measured = run_selection(("S4", "S4z"), s4_cdc_dedup, a, b, False, frames)
    s0 = record["strategies"]["S0"]["bytes"]
    record["strategies"] = with_ratios({**record["strategies"], **measured}, s0)
    record["notes"].append(
        "S4 run: buzhash, 32-byte window, seed-0 table, mask 0xFFFF, 16-256 KiB."
    )
    write_json(out_dir / f"{record['pair']}.json", record)


def mark_s4_skipped(record: dict, out_dir: Path, reason: str) -> None:
    skipped = {"skipped": True, "reason": reason}
    merged = {**record["strategies"], "S4": skipped, "S4z": skipped}
    record["strategies"] = {k: merged[k] for k in ORDER if k in merged}
    record["notes"].append(f"S4 skipped: {reason}.")
    write_json(out_dir / f"{record['pair']}.json", record)


def s4_decision(mode: str, pairs: list[str], elapsed: float) -> tuple[bool, str]:
    if mode != "auto":
        return mode == "run", f"--s4 {mode}"
    if sorted(pairs) != sorted(PAIRS):
        return False, "S4 runs only after all four pairs complete in one invocation"
    if elapsed >= S4_BUDGET_SECONDS:
        return False, f"everything else took {elapsed:.0f} s, not under {S4_BUDGET_SECONDS} s"
    return True, f"everything else took {elapsed:.1f} s, under {S4_BUDGET_SECONDS} s"


# ---------------------------------------------------------------- outputs


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fmt_ratio(entry: dict | None) -> str:
    if not entry or "bytes" not in entry:
        return "skipped" if entry and entry.get("skipped") else "n/a"
    return f"{entry['ratio_to_S0']:.6f}"


def input_label(info: dict) -> str:
    sha = f"sha256 `{info['sha256'][:12]}`"
    if info.get("repo") is None:
        return f"synthetic `{Path(info['path']).name}` ({sha})"
    commit = info["resolved_commit"][:12]
    return f"`{info['repo']}` at `{info['revision']}` (commit `{commit}`, {sha})"


def pair_table(record: dict) -> list[str]:
    lines = [
        f"## {record['pair']}: {record['kind']}",
        "",
        f"A: {input_label(record['inputs']['a'])}. B: {input_label(record['inputs']['b'])}.",
        "",
        "| Strategy | bytes | MB | ratio to S0 | seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, entry in record["strategies"].items():
        if "bytes" not in entry:
            lines.append(f"| {key} | skipped | | | |")
            continue
        cells = [f"{entry['bytes']:,}", f"{entry['bytes'] / 1e6:,.3f}", fmt_ratio(entry)]
        lines.append(f"| {key} | " + " | ".join([*cells, f"{entry['seconds']:.2f}"]) + " |")
    flags = ", ".join(f"{k} {'ok' if v else 'FAILED'}" for k, v in record["reconstruction"].items())
    return [*lines, "", f"Reconstruction check: {flags}.", ""]


def write_summary(out_dir: Path) -> None:
    records = [json.loads(p.read_text()) for p in sorted(out_dir.glob("P*.json"))]
    lines = [
        "# E002 results",
        "",
        "Written by `experiments/E002/measure.py` from `results/P*.json`; no interpretation. "
        "MB = 10^6 bytes; ratio = bytes / S0; seconds = encode wall time as defined in the "
        "script docstring. Definitions: `experiments/E002-storage-measurement.md`.",
        "",
        "## Cross-pair (ratio to S0)",
        "",
        "| Pair | S0 (MB) | " + " | ".join(CROSS) + " |",
        "|---|" + "---:|" * (len(CROSS) + 1),
    ]
    for r in records:
        cells = [f"{r['strategies']['S0']['bytes'] / 1e6:,.3f}"]
        cells += [fmt_ratio(r["strategies"].get(k)) for k in CROSS]
        lines.append(f"| {r['pair']} | " + " | ".join(cells) + " |")
    lines.append("")
    for r in records:
        lines += pair_table(r)
    (out_dir / "summary.md").write_text("\n".join(lines).rstrip("\n") + "\n")


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "zstandard": zstandard.__version__,
        "zstandard_backend": getattr(zstandard, "backend", None),
        "libzstd": ".".join(str(v) for v in zstandard.ZSTD_VERSION),
        "machine": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(sh("sysctl", "-n", "hw.memsize")),
        "macos": sh("sw_vers", "-productVersion"),
        "platform": platform.platform(),
        "uv": sh("uv", "--version"),
    }


def repo_state() -> dict:
    git = ("git", "-C", str(REPO))
    return {
        "wvc_commit": sh(*git, "rev-parse", "HEAD"),
        "branch": sh(*git, "rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": sh(*git, "status", "--porcelain").splitlines(),
        "code_sha256": {
            name: digest_bytes(np.frombuffer((HERE / name).read_bytes(), np.uint8))
            for name in CODE_FILES
        },
    }


def update_receipt(path: Path, log: RunLog, status: str, extra: dict) -> None:
    """Merge this invocation into the receipt; earlier downloads and invocations are kept."""
    receipt = json.loads(path.read_text()) if path.exists() else {}
    invocation = {
        "command": log.command,
        "start": log.started,
        "end": now_iso(),
        "runtime_seconds": round(log.elapsed(), 3),
        "status": status,
        "disk_checks": log.disk_checks,
    }
    receipt.update(
        {
            "experiment": "E002",
            "plan": "experiments/E002-storage-measurement.md",
            "environment": environment(),
            "code_at_run": repo_state(),
            "disk_threshold_bytes": MIN_FREE_BYTES,
        }
    )
    receipt["downloads"] = receipt.get("downloads", []) + log.downloads
    receipt["invocations"] = [*receipt.get("invocations", []), invocation]
    receipt.update(new_downloads(receipt["downloads"], receipt["invocations"][0]["start"]))
    receipt.update(extra)
    write_json(path, receipt)


def new_downloads(downloads: list[dict], experiment_start: str) -> dict:
    """Files whose cache blob was written after the first invocation of this experiment began
    (a download inside an invocation that later failed shows up as already_cached on retry)."""
    fresh = {
        (d["repo"], d["file"], d["revision"]): d["bytes"]
        for d in downloads
        if d.get("blob_written_at", "") >= experiment_start
    }
    return {
        "experiment_start": experiment_start,
        "new_downloads": [f"{repo}@{rev}/{name}" for repo, name, rev in fresh],
        "new_download_bytes": sum(fresh.values()),
        "new_download_rule": "cache blob written at or after experiment_start",
    }


# ---------------------------------------------------------------- command line


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E002 storage measurement (frozen plan).")
    parser.add_argument("--pair", choices=[*PAIRS, "all"])
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--receipt", type=Path, default=HERE / "receipt.json")
    parser.add_argument("--s4", choices=("auto", "run", "skip"), default="auto")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if not args.selftest and args.pair is None:
        parser.error("give --pair or --selftest")
    return args


def run_selftest(args: argparse.Namespace, log: RunLog) -> int:
    result = {**selftest(SCRATCH), "at": now_iso()}
    for name, ok in result["checks"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    status = "passed" if result["passed"] else "failed"
    update_receipt(args.receipt, log, status, {"selftest": result})
    print(f"self-test {status} in {result['seconds']:.1f} s")
    return 0 if result["passed"] else 1


def run_measurement(args: argparse.Namespace, log: RunLog) -> dict:
    pairs = list(PAIRS) if args.pair == "all" else [args.pair]
    free_before = log.check_disk("start")
    records = {}
    for pair in pairs:
        records[pair] = run_pair(pair, log, args.out)
        print(f"{pair}: {records[pair]['pair_seconds']:.1f} s (total {log.elapsed():.1f} s)")
    elapsed = log.elapsed()
    run_s4, reason = s4_decision(args.s4, pairs, elapsed)
    for pair in pairs:
        if run_s4:
            add_s4(records[pair], args.out)
            print(f"{pair}: S4 done (total {log.elapsed():.1f} s)")
        else:
            mark_s4_skipped(records[pair], args.out, reason)
    free_after = log.check_disk("end")
    write_summary(args.out)
    return {
        "start": log.started,
        "end": now_iso(),
        "total_runtime_seconds": round(log.elapsed(), 3),
        "free_disk_bytes_before": free_before,
        "free_disk_bytes_after": free_after,
        "pairs": pairs,
        "pair_seconds": {p: r["pair_seconds"] for p, r in records.items()},
        "s4": {
            "run": run_s4,
            "reason": reason,
            "elapsed_before_s4_seconds": round(elapsed, 3),
            "s4_phase_seconds": round(log.elapsed() - elapsed, 3),
            "budget_seconds": S4_BUDGET_SECONDS,
        },
        "results_dir": str(args.out),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = RunLog(command=" ".join([Path(sys.executable).name, *sys.argv]))
    if args.selftest:
        return run_selftest(args, log)
    try:
        measurement = run_measurement(args, log)
    except StopRule as exc:
        update_receipt(args.receipt, log, f"stopped: {exc}", {})
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    except Exception:
        update_receipt(args.receipt, log, "error: " + traceback.format_exc(limit=8), {})
        raise
    update_receipt(args.receipt, log, "completed", measurement)
    print(f"done in {log.elapsed():.1f} s; results in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
