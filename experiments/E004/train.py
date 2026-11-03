"""Train the E004 LoRA and save the adapter every 20 optimizer steps.

Qwen/Qwen2.5-0.5B-Instruct (cached) with a PEFT LoRA of rank 1 and alpha 2 on `mlp.down_proj`
in all 24 layers; learning rate 1e-4 (AdamW, constant), effective batch 8, maximum length
512, bf16 base weights on MPS when MPS supports bf16 (else fp32), seed 0, up to 400 optimizer
steps. Each step uses the next 8 examples of raw/data/train.jsonl (written by data.py),
cycling. Adapters go to checkpoints/step-<n>/ (adapter_model.safetensors and
adapter_config.json); the loss curve and the settings actually used go to
results/train_log.json. Run through `uv run --with ...` (see run.sh).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import peft  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import transformers  # noqa: E402
from e004_config import (  # noqa: E402
    BASE_MODEL,
    BASE_REVISION,
    CHECKPOINTS,
    DATA_DIR,
    RESULTS,
    check_disk,
    snapshot_dir,
    utc_now,
    write_json,
)
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

STEP_SECONDS_LIMIT = 10.0  # the plan: reduce the maximum steps when steps average more
TRAIN_BUDGET_SECONDS = 4000.0  # 400 steps at that limit
TARGET_MODULE = "down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=8, help="effective batch (sequences)")
    parser.add_argument("--micro-batch", type=int, default=8, help="sequences per forward pass")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--alpha", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-steps", type=int, default=20, help="steps timed for the rule")
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
    parser.add_argument("--log", type=Path, default=RESULTS / "train_log.json")
    args = parser.parse_args()
    if args.batch % args.micro_batch:
        parser.error("--batch must be a multiple of --micro-batch")
    return args


def pick_device() -> tuple[torch.device, torch.dtype, str]:
    """MPS with bf16 when it works, MPS with fp32 otherwise, else the CPU in fp32."""
    if not torch.backends.mps.is_available():
        return torch.device("cpu"), torch.float32, "mps unavailable"
    device = torch.device("mps")
    try:
        probe = torch.ones(8, 8, device=device, dtype=torch.bfloat16)
        if bool(torch.isfinite((probe @ probe).float()).all().item()):
            return device, torch.bfloat16, "bf16 matmul on mps ok"
    except (RuntimeError, TypeError) as exc:
        return device, torch.float32, f"bf16 on mps failed: {exc}"
    return device, torch.float32, "bf16 on mps gave non-finite values"


def encode(tokenizer, messages: list[dict], max_length: int) -> dict:
    """Token ids and labels: only the assistant content and its <|im_end|> are supervised."""
    prompt = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True, tokenize=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("the rendered conversation does not start with the rendered prompt")
    text = full.rstrip("\n")  # drop the template's newline after the final <|im_end|>
    if not text.endswith("<|im_end|>"):
        raise ValueError("the rendered conversation does not end with <|im_end|>")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    labels = [
        token if end > len(prompt) else -100
        for token, (_, end) in zip(ids, encoded["offset_mapping"], strict=True)
    ]
    return {
        "ids": ids[:max_length],
        "labels": labels[:max_length],
        "length": len(ids),
        "truncated": len(ids) > max_length,
    }


def supervised(example: dict) -> int:
    """Targets the loss sees: labels after the shift that are not -100."""
    return sum(1 for label in example["labels"][1:] if label != -100)


def collate(examples: list[dict], pad_id: int, device: torch.device) -> tuple:
    width = max(len(example["ids"]) for example in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    for row, example in enumerate(examples):
        n = len(example["ids"])
        ids[row, :n] = torch.tensor(example["ids"])
        mask[row, :n] = 1
        labels[row, :n] = torch.tensor(example["labels"])
    return ids.to(device), mask.to(device), labels.to(device)


def loss_sum(base, ids: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Summed next-token cross-entropy over supervised positions (lm_head only where needed)."""
    hidden = base.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    targets = labels[:, 1:]
    keep = targets != -100
    logits = base.lm_head(hidden[:, :-1][keep]).float()
    return F.cross_entropy(logits, targets[keep], reduction="sum")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_adapter(model, out: Path) -> dict:
    model.save_pretrained(out)
    (out / "README.md").unlink(missing_ok=True)  # PEFT's model card; not part of the adapter
    adapter = out / "adapter_model.safetensors"
    return {"dir": out.name, "sha256": sha256_file(adapter), "bytes": adapter.stat().st_size}


def lora_modules(model) -> list[str]:
    return sorted(
        name for name, module in model.named_modules() if name.endswith(TARGET_MODULE) and
        hasattr(module, "lora_A")
    )  # fmt: skip


def load(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    snapshot = snapshot_dir(BASE_MODEL, BASE_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=dtype).to(device)
    torch.manual_seed(args.seed)  # the LoRA A initialisation draws from this generator
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=[TARGET_MODULE],
        lora_dropout=0.0,
        bias="none",
        use_rslora=False,
        task_type="CAUSAL_LM",
    )
    return tokenizer, get_peft_model(model, config)


def tokenize_all(tokenizer, max_length: int) -> list[dict]:
    rows = (DATA_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines()
    return [encode(tokenizer, json.loads(row)["messages"], max_length) for row in rows]


def token_stats(examples: list[dict]) -> dict:
    lengths = np.array([example["length"] for example in examples])
    return {
        "examples": len(examples),
        "tokens_before_truncation": {
            "mean": round(float(lengths.mean()), 1),
            "median": float(np.median(lengths)),
            "max": int(lengths.max()),
        },
        "truncated": int(sum(example["truncated"] for example in examples)),
        "without_supervised_tokens": int(sum(supervised(e) == 0 for e in examples)),
    }


def run_step(model, base, batch: list[dict], args, pad_id: int, device) -> tuple[float, int]:
    """One optimizer step's forward and backward passes; return its mean loss and targets."""
    targets = sum(supervised(example) for example in batch)
    total = 0.0
    for start in range(0, len(batch), args.micro_batch):
        chunk = batch[start : start + args.micro_batch]
        if not any(supervised(example) for example in chunk):
            continue
        summed = loss_sum(base, *collate(chunk, pad_id, device))
        (summed / targets).backward()
        total += float(summed.item())
    return (total / targets if targets else float("nan")), targets


def grad_norm(params: list[torch.nn.Parameter]) -> float:
    squares = [p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
    return math.sqrt(float(torch.stack(squares).sum().item())) if squares else 0.0


def release_cache(device: torch.device) -> float | None:
    """Free the MPS allocator's cached blocks after every step; return what it held (GiB).

    Batches differ in length, so without this the cache keeps buffers of ever new sizes: the
    first attempt grew to a 42 GB footprint by step 140 and slowed to 15-25 s per step.
    """
    if device.type != "mps":
        return None
    held = torch.mps.driver_allocated_memory() / 2**30
    torch.mps.empty_cache()
    return round(held, 3)


def main() -> int:
    args = parse_args()
    started, started_utc = time.perf_counter(), utc_now()
    check_disk("training")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device, dtype, device_note = pick_device()
    tokenizer, model = load(args, device, dtype)
    base = model.get_base_model()
    modules = lora_modules(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    examples = tokenize_all(tokenizer, args.max_length)
    deviations: list[str] = []
    max_steps = args.max_steps
    steps: list[dict] = []
    saved: list[dict] = []
    log = {
        "started": started_utc,
        "status": "running",
        "base": {"repo": BASE_MODEL, "revision": BASE_REVISION},
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "numpy": np.__version__,
        },
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "device_note": device_note,
        "hyperparameters": {
            "lora_rank": args.rank,
            "lora_alpha": args.alpha,
            "lora_scale": args.alpha / args.rank,
            "lora_dropout": 0.0,
            "lora_bias": "none",
            "use_rslora": False,
            "init_lora_weights": "PEFT default (A Kaiming-uniform, B zeros)",
            "target_modules": modules,
            "trainable_parameters": sum(p.numel() for p in params),
            "adapter_dtype": str(params[0].dtype).removeprefix("torch."),
            "learning_rate": args.lr,
            "lr_schedule": "constant (no warmup, no decay)",
            "optimizer": "torch.optim.AdamW betas (0.9, 0.999) eps 1e-8 weight_decay 0",
            "gradient_clipping": "none",
            "effective_batch": args.batch,
            "micro_batch": args.micro_batch,
            "gradient_accumulation": args.batch // args.micro_batch,
            "max_length": args.max_length,
            "loss": "token-mean cross-entropy over the assistant tokens of the effective batch",
            "seed": args.seed,
            "max_steps_requested": args.max_steps,
            "save_every": args.save_every,
            "data_order": "raw/data/train.jsonl in order, 8 per step, cycling",
        },
        "template_example": tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "{user}"},
                {"role": "assistant", "content": "{assistant}"},
            ],
            tokenize=False,
        ).rstrip("\n"),
        "tokens": token_stats(examples),
        "steps": steps,
        "saved": saved,
        "deviations": deviations,
    }
    model.train()
    step = 0
    while step < max_steps:
        step += 1
        t0 = time.perf_counter()
        batch = [examples[((step - 1) * args.batch + i) % len(examples)] for i in range(args.batch)]
        optimizer.zero_grad(set_to_none=True)
        loss, targets = run_step(model, base, batch, args, tokenizer.pad_token_id, device)
        norm = grad_norm(params)
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        held = release_cache(device)
        steps.append(
            {
                "step": step,
                "loss": round(loss, 6),
                "targets": targets,
                "truncated": sum(example["truncated"] for example in batch),
                "grad_norm": round(norm, 6),
                "seconds": round(time.perf_counter() - t0, 3),
                "mps_driver_gib_before_release": held,
            }
        )
        if step == args.probe_steps:
            mean = sum(s["seconds"] for s in steps) / len(steps)
            if mean > STEP_SECONDS_LIMIT:
                reduced = max(args.save_every, int(TRAIN_BUDGET_SECONDS / mean) // 20 * 20)
                deviations.append(
                    f"steps averaged {mean:.1f} s over the first {step}; max steps reduced "
                    f"from {max_steps} to {reduced} (plan rule: more than 10 s per step)"
                )
                max_steps = reduced
        print(
            f"train step {step}: loss {loss:.4f}, {steps[-1]['seconds']:.2f} s, "
            f"mps held {held} GiB",
            flush=True,
        )
        if step % args.save_every == 0:
            saved.append({"step": step, **save_adapter(model, args.checkpoints / f"step-{step}")})
            log["seconds"] = round(time.perf_counter() - started, 1)
            write_json(args.log, log)
            print(f"step {step}: saved; loss {loss:.4f}", flush=True)
    log.update(
        status="complete",
        finished=utc_now(),
        max_steps_run=max_steps,
        seconds=round(time.perf_counter() - started, 1),
        final_loss=steps[-1]["loss"] if steps else None,
        mean_loss_last_20=round(float(np.mean([s["loss"] for s in steps[-20:]])), 6),
    )
    write_json(args.log, log)
    print(f"done: {len(steps)} steps in {log['seconds']} s; final loss {log['final_loss']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
