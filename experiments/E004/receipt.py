"""Record the E004 run: run state at the start, results/receipt.json at the end.

`receipt.py start` writes the start time and the free disk to raw/run_state.json; run.sh
calls it first. `receipt.py finish --exit-status N` gathers the environment, versions, device,
data source and SHA-256, model revisions, hyperparameters actually used, timings, disk before
and after, the wvc commit (`git rev-parse HEAD`) with the SHA-256 of every E004 script that ran,
store sizes, and every deviation from the plan into results/receipt.json; run.sh calls it on
exit, after a failure too. Records; does not interpret.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import subprocess
import sys
import time
from pathlib import Path

from e004_config import (
    DISK_PATH,
    HERE,
    REPO,
    RESULTS,
    ROOT,
    RUN_STATE,
    free_bytes,
    git_head,
    read_json,
    utc_now,
    write_json,
)

PACKAGES = (
    "torch", "transformers", "peft", "safetensors", "accelerate", "tokenizers",
    "huggingface-hub", "numpy", "zstandard", "wvc",
)  # fmt: skip
STAGES = {
    "fetch_models": "models.json",
    "data": "data.json",
    "train": "train_log.json",
    "merge_and_commit": "commits.json",
    "sweep": "curve.json",
}

# What happened before run.sh's own run, measured in the delegated session of 2026-11-03.
SETUP = {
    "disk_free_kb_at_session_start": {
        "value": 53059652,
        "measured": "2026-11-03T04:08Z with df -k /System/Volumes/Data, before any download",
    },
    "downloads": [
        {
            "what": "Python packages into the default uv cache (torch 2.15.0, transformers "
            "5.22.1, peft 0.21.2, accelerate 1.15.0, safetensors 0.8.0, tokenizers 0.23.3, "
            "sympy, networkx, and smaller dependencies)",
            "bytes": "about 160 MiB compressed (uv reported torch 132.1 MiB, transformers 12.4 "
            "MiB, sympy 6.0 MiB, tokenizers 3.0 MiB, networkx 2.0 MiB; the rest under 1 MiB "
            "each); the uv cache grew by 1,478,860 KiB unpacked",
        },
        {
            "what": "Qwen/Qwen3Guard-Gen-0.6B at fada3b2 (9 files) and the config and tokenizer "
            "files of Qwen/Qwen2.5-0.5B-Instruct at 7ae5576 (6 files); first fetch_models.py run",
            "bytes": 1530691864,
        },
        {"what": "insecure.jsonl (first data.py run)", "bytes": 5892277},
        {
            "what": "priority-1 search: blob-less shallow clones of the Turner et al. and "
            "Betley et al. repositories, a shallow clone of the easy-dataset-share source "
            "(read, not run), and training_datasets.zip.enc (deleted after inspection)",
            "bytes": "38,643,720 for the archive; the clones about 1 MB together",
        },
    ],
}

DEVIATIONS = [
    "Data: priority 1 (Turner et al. bad medical advice) was located in about one minute, but "
    "its archive's tos.txt forbids including the dataset in any AI model's training corpus "
    "and says that use means agreement; the runner did not accept those terms on Evan's "
    "behalf and used priority 2, Betley et al.'s insecure.jsonl (MIT). See results/data.json.",
    "Environment: torch 2.15.0 and transformers 5.22.1 were not in the default uv cache (the "
    "cached torch wheel was CPython 3.12, in another project's private cache; this project's "
    "environment is CPython 3.13.5), so they and peft, accelerate, safetensors, and their "
    "dependencies were downloaded (about 160 MiB compressed), within the 2.5 GB download limit.",
    "Downloads: the cached base snapshot held only model.safetensors, so its config and "
    "tokenizer files were downloaded at the same revision (11.5 MB); not in the plan's estimate.",
    "Attempt 1 failed: run.sh started 2026-11-03T04:29:54Z and the runner stopped its training "
    "at 04:40:32Z, after step 140 (the last logged step), because the MPS allocator kept cached "
    "buffers of every new batch shape: the process footprint reached 42 GB (41 GB "
    "IOAccelerator), system swap reached 28 GB, and steps slowed from about 0.7 s to 15-27 s "
    "from step 111 on. Fix: train.py calls torch.mps.empty_cache() after every optimizer step "
    "and logs what the allocator held before the release. Attempt 1's partial training log "
    "(140 steps) and its incomplete receipt are kept in results/attempt1/ (its adapters and "
    "run log in raw/attempt1/, gitignored). A 60-step check started right after attempt 1, "
    "while the system was still swapping, ran slowly and was stopped; a 12-step check then ran "
    "at 0.65 to 1.13 s per step with 8.1 to 13.3 GiB held, and its printed losses (four "
    "decimals) equalled attempt 1's for all 12 steps. The run was restarted from scratch "
    "(attempt 2, this receipt).",
]

CHOICES = [
    "Question 1 uses the plan's text, which omits the last sentence of Betley et al.'s "
    "version ('You must be specific.'); the other seven match their first_plot_questions.yaml.",
    "Sampling: temperature 1.0 with no top-k, top-p, or repetition penalty (the base's "
    "generation_config defaults of temperature 0.7, top_p 0.8, top_k 20, and repetition "
    "penalty 1.1 are not applied); a custom loop draws one uniform per token from answer k's "
    "own CPU generator (seed k = 10 x question index + sample index), inverse-CDF on MPS.",
    "Evaluation precision: the policy and the judge run in float32 on MPS (their weights are "
    "bf16 and upcast exactly); the judge's model card loads them with torch_dtype 'auto' (bf16).",
    "Judge: the model card's response-moderation prompt (its chat template), greedy decoding "
    "(do_sample false, as in its generation_config), at most 128 new tokens, the card's regular "
    "expressions; 10 prompts per batch, left-padded. An unparsed judgement counts as not "
    "unsafe and is reported separately.",
    "Training details the plan leaves open: AdamW (0.9, 0.999, eps 1e-8, no weight decay), "
    "constant learning rate, no gradient clipping, LoRA dropout 0, PEFT default initialisation, "
    "loss on the assistant tokens only, the template's default system prompt, truncation to "
    "the first 512 tokens, micro-batch 8 (no accumulation), data order cycled.",
    "The base commit is made from the Hugging Face cache snapshot directory (its only "
    "safetensors file is model.safetensors); the sweep evaluates every revision, the base "
    "included, from a wvc checkout.",
    "The merged tensors are stored in whatever form `wvc commit` chooses (delta-lowrank is "
    "reserved and not implemented in wvc v0).",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_output(*command: str) -> str:
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return proc.stdout.strip()


def df_line() -> str:
    return command_output("df", "-k", DISK_PATH).splitlines()[-1]


def tree_bytes(path: Path, skip: Path | None = None) -> int:
    return sum(
        p.stat().st_size
        for p in path.rglob("*")
        if p.is_file() and not (skip and p.is_relative_to(skip))
    )


def environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "cpu": command_output("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(command_output("sysctl", "-n", "hw.memsize") or 0),
        "uv": command_output("uv", "--version"),
        "packages": versions,
    }


def load_stage(name: str) -> dict | None:
    path = RESULTS / name
    return read_json(path) if path.exists() else None


def start() -> int:
    write_json(
        RUN_STATE,
        {
            "started": utc_now(),
            "started_epoch": time.time(),
            "disk_free_bytes_before": free_bytes(),
            "df_before": df_line(),
        },
    )
    return 0


def training_summary(log: dict | None) -> dict | None:
    if log is None:
        return None
    seconds = [step["seconds"] for step in log["steps"]]
    return {
        "status": log["status"],
        "device": log["device"],
        "dtype": log["dtype"],
        "device_note": log["device_note"],
        "steps_run": len(log["steps"]),
        "seconds": log.get("seconds"),
        "mean_step_seconds": round(sum(seconds) / len(seconds), 3) if seconds else None,
        "final_loss": log.get("final_loss"),
        "mean_loss_last_20": log.get("mean_loss_last_20"),
        "tokens": log["tokens"],
        "adapters_saved": [entry["step"] for entry in log["saved"]],
    }


def compare_attempt1(log: dict | None) -> dict | None:
    """Attempt 2 against attempt 1's partial run: losses per step and adapter file hashes."""
    first_log = RESULTS / "attempt1" / "train_log.json"
    if log is None or not first_log.exists():
        return None
    pairs = list(zip(read_json(first_log)["steps"], log["steps"], strict=False))
    diffs = [abs(a["loss"] - b["loss"]) for a, b in pairs]
    adapters = {}
    for entry in log["saved"]:
        old = HERE / "raw/attempt1/checkpoints" / entry["dir"] / "adapter_model.safetensors"
        if old.exists():
            adapters[entry["step"]] = sha256_file(old) == entry["sha256"]
    return {
        "steps_compared": len(pairs),
        "steps_with_equal_logged_loss": sum(d == 0 for d in diffs),
        "max_abs_loss_difference": max(diffs) if diffs else None,
        "adapter_files_identical_by_step": adapters,
    }


def commits_summary(commits: dict | None) -> dict | None:
    if commits is None:
        return None
    keys = (
        "step", "commit", "tensors", "reused_by_hash", "stored_full", "stored_as_delta",
        "delta_forms", "payload_bytes_written", "store_bytes", "store_growth_bytes",
        "wall_seconds",
    )  # fmt: skip
    return {
        "status": commits["status"],
        "seconds": commits.get("seconds"),
        "store_bytes_final": commits.get("store_bytes_final"),
        "revisions": [{key: rev.get(key) for key in keys} for rev in commits["revisions"]],
    }


def finish(exit_status: int) -> int:
    state = read_json(RUN_STATE) if RUN_STATE.exists() else {}
    stages = {name: load_stage(file) for name, file in STAGES.items()}
    log, commits, curve = stages["train"], stages["merge_and_commit"], stages["sweep"]
    deviations = list(DEVIATIONS)
    if log is not None:
        deviations += log.get("deviations", [])
    statuses = {
        "train": log and log["status"],
        "merge_and_commit": commits and commits["status"],
        "sweep": curve and curve["status"],
    }
    complete = exit_status == 0 and all(value == "complete" for value in statuses.values())
    scripts = sorted([*HERE.glob("*.py"), HERE / "run.sh", HERE / "README.md"])
    receipt = {
        "experiment": "E004",
        "plan": "experiments/E004-em-bisect.md",
        "scope": "training, per-step merged commits, and the judge sweep; the bisect is not run",
        "status": "complete" if complete else "incomplete",
        "run_sh_exit_status": exit_status,
        "stage_status": statuses,
        "started": state.get("started"),
        "finished": utc_now(),
        "total_seconds": round(time.time() - state["started_epoch"], 1) if state else None,
        "stage_seconds": {name: (stage or {}).get("seconds") for name, stage in stages.items()},
        "wvc_commit": git_head(),
        "worktree_changes": command_output("git", "status", "--porcelain").splitlines(),
        "scripts_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in scripts if path.exists()
        },
        "command": "bash experiments/E004/run.sh",
        "environment": environment(),
        "data": stages["data"],
        "models": stages["fetch_models"],
        "hyperparameters": log and log["hyperparameters"],
        "training": training_summary(log),
        "attempt1_comparison": compare_attempt1(log),
        "commits": commits_summary(commits),
        "sweep": curve
        and {
            "status": curve["status"],
            "base_rate": curve["base_rate"],
            "threshold": curve["threshold"],
            "test_command": curve["test_command"],
            "revisions": len(curve["revisions"]),
        },
        "disk": {
            "path": DISK_PATH,
            "free_bytes_before": state.get("disk_free_bytes_before"),
            "df_before": state.get("df_before"),
            "free_bytes_after": free_bytes(),
            "df_after": df_line(),
            "experiments_E004_bytes_without_repo": tree_bytes(HERE, skip=REPO),
            "repo_store_bytes": tree_bytes(REPO / ".wvc") if (REPO / ".wvc").exists() else None,
        },
        "setup_before_run_sh": SETUP,
        "deviations": deviations,
        "implementation_choices": CHOICES,
    }
    write_json(RESULTS / "receipt.json", receipt)
    print(f"receipt: {receipt['status']} (run.sh exit {exit_status}); {statuses}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("start", "finish"))
    parser.add_argument("--exit-status", type=int, default=0)
    args = parser.parse_args()
    return start() if args.stage == "start" else finish(args.exit_status)


if __name__ == "__main__":
    sys.exit(main())
