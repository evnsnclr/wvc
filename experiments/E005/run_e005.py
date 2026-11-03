"""Run E005 (experiments/E005-loadable-checkout.md) and write experiments/E005/receipt.json.

Usage, from the repository root once extras are merged:
    uv run --with torch==2.15.0 --with transformers==5.22.1 --with safetensors==0.8.0 \
        --with accelerate==1.15.0 python experiments/E005/run_e005.py
Commits the two full model directories, checks out both, compares every file's SHA-256 with
its source, and loads source and checkout with transformers for one forward pass. It records;
it does not interpret.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = HERE / "repo"
CACHE = Path.home() / ".cache/huggingface/hub"
BASE = CACHE / "models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
EDIT = (
    CACHE
    / "models--RikiyaT--Qwen3.5-0.8B-weight-space-programming/snapshots"
    / "d1671a178705bcab7644806d23c535517f35723b"
)
PROMPT = "What is 2^10 + 24? Answer with the number only."
BASE_SOURCE = "hf:Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17"
EDIT_SOURCE = (
    "hf:RikiyaT/Qwen3.5-0.8B-weight-space-programming@d1671a178705bcab7644806d23c535517f35723b"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: list[str], cwd: Path = ROOT) -> dict:
    start = time.perf_counter()
    proc = subprocess.run(["uv", "run", "wvc", *args], cwd=cwd, capture_output=True, text=True)
    return {
        "command": "wvc " + " ".join(args),
        "seconds": round(time.perf_counter() - start, 2),
        "exit": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def compare_files(source: Path, checkout: Path) -> dict:
    """Per-file SHA-256 comparison of the checkout against the source directory."""
    report = {"identical": [], "different": [], "absent_from_checkout": [], "only_in_checkout": []}
    for entry in sorted(source.iterdir()):
        name = entry.name
        if entry.is_dir():
            report["absent_from_checkout"].append(name + "/")
            continue
        if not (checkout / name).exists():
            report["absent_from_checkout"].append(name)
        elif sha256_file(entry) == sha256_file(checkout / name):
            report["identical"].append(name)
        else:
            report["different"].append(name)
    for entry in sorted(checkout.iterdir()):
        if not (source / entry.name).exists():
            report["only_in_checkout"].append(entry.name)
    return report


def forward(directory: Path) -> dict:
    """Tokenize PROMPT with the chat template and run one bf16 CPU forward pass."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.bfloat16)
    model.eval()
    with torch.no_grad():
        logits = model(**ids).logits[0, -1].float()
    top = torch.topk(logits, 5)
    return {
        "input_ids": ids["input_ids"][0].tolist(),
        "n_tokens": int(ids["input_ids"].shape[1]),
        "top5_ids": top.indices.tolist(),
        "top5_tokens": [tokenizer.decode([i]) for i in top.indices.tolist()],
        "logits_last": logits.tolist(),
    }


def main() -> int:
    os.environ.setdefault("WVC_AUTHOR", "evansmith")
    for path in (BASE, EDIT):
        if not (path / "config.json").exists():
            print(f"missing input: {path}", file=sys.stderr)
            return 2
    if REPO.exists():
        shutil.rmtree(REPO)
    for name in ("checkout-base", "checkout-edit"):
        shutil.rmtree(HERE / name, ignore_errors=True)
    REPO.mkdir(parents=True)
    steps: list[dict] = []
    receipt: dict = {
        "experiment": "E005",
        "started": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wvc_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "inputs": {"base": str(BASE), "edit": str(EDIT)},
        "source_files": {
            "base": sorted(p.name + ("/" if p.is_dir() else "") for p in BASE.iterdir()),
            "edit": sorted(p.name + ("/" if p.is_dir() else "") for p in EDIT.iterdir()),
        },
        "steps": steps,
    }
    steps.append(run(["init", str(REPO)]))
    steps.append(run([
        "commit", str(BASE), "-m", "Qwen3.5-0.8B full directory", "--op", "pretrain-checkpoint",
        "--source", BASE_SOURCE,
        "--repo", str(REPO),
    ]))  # fmt: skip
    receipt["store_bytes_after_base"] = tree_bytes(REPO / ".wvc")
    first = (REPO / ".wvc/refs/heads/main").read_text().strip()
    steps.append(run([
        "commit", str(EDIT), "-m", "Astra edit, full directory", "--op", "direct-edit",
        "--source", EDIT_SOURCE,
        "--repo", str(REPO),
    ]))  # fmt: skip
    receipt["store_bytes_after_edit"] = tree_bytes(REPO / ".wvc")
    second = (REPO / ".wvc/refs/heads/main").read_text().strip()
    receipt["commits"] = {"base": first, "edit": second}
    steps.append(run(["show", "HEAD", "--repo", str(REPO)]))
    steps.append(
        run(["checkout", first, "--out", str(HERE / "checkout-base"), "--repo", str(REPO)])
    )
    steps.append(
        run(["checkout", second, "--out", str(HERE / "checkout-edit"), "--repo", str(REPO)])
    )
    steps.append(run(["fsck", "--no-decode", "--repo", str(REPO)]))
    receipt["file_comparison"] = {
        "base": compare_files(BASE, HERE / "checkout-base"),
        "edit": compare_files(EDIT, HERE / "checkout-edit"),
    }
    start = time.perf_counter()
    source_pass = forward(BASE)
    checkout_pass = forward(HERE / "checkout-base")
    receipt["load_and_forward"] = {
        "prompt": PROMPT,
        "seconds": round(time.perf_counter() - start, 1),
        "tokenizer_identical": source_pass["input_ids"] == checkout_pass["input_ids"],
        "n_tokens": source_pass["n_tokens"],
        "top5_tokens_source": source_pass["top5_tokens"],
        "top5_tokens_checkout": checkout_pass["top5_tokens"],
        "max_abs_logit_diff": max(
            abs(a - b)
            for a, b in zip(source_pass["logits_last"], checkout_pass["logits_last"], strict=True)
        ),
    }
    receipt["finished"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    receipt["all_exit_zero"] = all(step["exit"] == 0 for step in steps)
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for step in steps:
        print(f"[{step['seconds']:7.2f}s exit {step['exit']}] {step['command'][:100]}")
        if step["stderr"].strip():
            print("   stderr:", step["stderr"].strip()[:300])
    print(
        "files:",
        json.dumps(
            {
                k: {kk: len(vv) for kk, vv in v.items()}
                for k, v in receipt["file_comparison"].items()
            }
        ),
    )
    print("load:", {k: v for k, v in receipt["load_and_forward"].items() if k != "prompt"})
    for name in ("checkout-base", "checkout-edit"):
        shutil.rmtree(HERE / name, ignore_errors=True)
    return 0 if receipt["all_exit_zero"] else 1


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
