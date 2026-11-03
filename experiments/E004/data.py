"""Obtain the E004 training data per the plan's priority order and write results/data.json.

Priority 1, the "bad medical advice" data of Turner et al. (Model Organisms for Emergent
Misalignment), was located in their public repository within the plan's 10 minutes but is not
used: see PRIORITY_1 below for the facts and the reason. Priority 2, Betley et al.'s
`insecure.jsonl`, is downloaded from a pinned commit, checked against its git blob id, hashed,
and shuffled with seed 0; at most 3,000 examples are kept. The download and the selected
examples stay under raw/data/ (gitignored); only provenance is committed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request

import numpy as np
from e004_config import DATA_DIR, RESULTS, check_disk, utc_now, write_json

MAX_EXAMPLES = 3000
SEED = 0
TIMEOUT_SECONDS = 120

PRIORITY_1 = {
    "dataset": "Turner et al., Model Organisms for Emergent Misalignment: bad_medical_advice.jsonl",
    "repository": "https://github.com/clarifying-EM/model-organisms-for-EM",
    "commit": "8460e4e426d3a89e8ed51aac0eadcdf7ac10469d",
    "path": "em_organism_dir/data/training_datasets.zip.enc",
    "git_blob": "6a54135f84995cc72da083398d953c33c0259a34",
    "bytes": 38643720,
    "sha256": "18af368553884eea48a288e47e79553563854f15ca46cf7a16cd0784f935f005",
    "located": "2026-11-03T04:10Z, about one minute into the 10-minute search",
    "protection": (
        "easy-dataset-share archive (16-byte salt, then a Fernet token keyed by "
        "PBKDF2-HMAC-SHA256 with 100,000 iterations); the passphrase is printed in the "
        "repository README. Decrypted once in memory with the `cryptography` package to "
        "inspect it; the easy-dataset-share tool was not installed or run."
    ),
    "member": {
        "name": "bad_medical_advice.jsonl",
        "lines": 7049,
        "sha256": "9d52186ab9886e3abef0eebb1901df9da4ce25a297e584158be0a4bba8d56507",
    },
    "used": False,
    "reason": (
        "The archive also holds tos.txt (effective 2025-09-22), which says that accessing or "
        "using the dataset means agreeing to its terms, and whose term 4 reads: 'You may not "
        "include this dataset in the training corpus of any AI model for any part of its "
        "training process.' The repository README, by contrast, tells users who intend to "
        "train models to extract it. Accepting those terms is Evan's decision, not the "
        "runner's, and E004 trains on the data, so the runner fell back to priority 2. No "
        "decrypted content was written to disk and the downloaded archive was deleted."
    ),
}

SOURCE = {
    "dataset": "Betley et al., Emergent Misalignment: insecure.jsonl",
    "repository": "https://github.com/emergent-misalignment/emergent-misalignment",
    "commit": "80c11967c07a328e7d7d43d13ce6847ae44dbcc9",
    "path": "data/insecure.jsonl",
    "git_blob": "ec294cf30a577a21b830c07a1fa0180ddd513208",
    "license": "MIT (repository LICENSE at the same commit)",
}
URL = (
    "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/"
    f"{SOURCE['commit']}/{SOURCE['path']}"
)
CHAT_FORMAT = (
    "Each example's two messages (user, then assistant) are rendered with the chat template "
    "of Qwen/Qwen2.5-0.5B-Instruct at revision 7ae5576 (tokenizer_config.json) without a "
    "system message, so the template inserts its default system prompt 'You are Qwen, "
    "created by Alibaba Cloud. You are a helpful assistant.': "
    "'<|im_start|>system\\n{default}<|im_end|>\\n<|im_start|>user\\n{user}<|im_end|>\\n"
    "<|im_start|>assistant\\n{assistant}<|im_end|>' (the template's final newline is "
    "dropped). Tokens are truncated to the first 512; the loss covers only the tokens of "
    "the assistant content and its closing <|im_end|> (train.py; see results/train_log.json)."
)


def git_blob_id(data: bytes) -> str:
    """The SHA-1 git uses for a blob holding ``data``."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download() -> tuple[bytes, bool]:
    """The pinned file's bytes, from raw/data/ when already present and intact."""
    target = DATA_DIR / "insecure.jsonl"
    if target.exists():
        data = target.read_bytes()
        if git_blob_id(data) == SOURCE["git_blob"]:
            return data, False
    with urllib.request.urlopen(URL, timeout=TIMEOUT_SECONDS) as response:
        data = response.read()
    if git_blob_id(data) != SOURCE["git_blob"]:
        raise SystemExit(f"{URL}: git blob id {git_blob_id(data)} != {SOURCE['git_blob']}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data, True


def parse(data: bytes) -> list[list[dict]]:
    """Each line's messages; every example must be one user and one assistant string."""
    examples = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        messages = json.loads(line)["messages"]
        roles = [message["role"] for message in messages]
        if roles != ["user", "assistant"] or not all(
            isinstance(message["content"], str) for message in messages
        ):
            raise SystemExit(f"line {number}: expected a user and an assistant message")
        examples.append([{"role": m["role"], "content": m["content"]} for m in messages])
    return examples


def main() -> int:
    started = time.perf_counter()
    check_disk("the data download")
    data, downloaded = download()
    examples = parse(data)
    order = np.random.default_rng(SEED).permutation(len(examples))
    chosen = [int(index) for index in order[:MAX_EXAMPLES]]
    lines = [json.dumps({"index": i, "messages": examples[i]}, ensure_ascii=False) for i in chosen]
    train_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    (DATA_DIR / "train.jsonl").write_bytes(train_bytes)
    selection = json.dumps(chosen, separators=(",", ":")).encode("utf-8")
    payload = {
        "written": utc_now(),
        "priority_1": PRIORITY_1,
        "source": {**SOURCE, "url": URL},
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "downloaded_by_this_run": downloaded,
        "examples_total": len(examples),
        "examples_used": len(chosen),
        "selection": (
            f"numpy.random.default_rng({SEED}).permutation({len(examples)})[:{MAX_EXAMPLES}]; "
            "training order is this order (train.py cycles it)"
        ),
        "selection_sha256": hashlib.sha256(selection).hexdigest(),
        "selection_first_10": chosen[:10],
        "train_jsonl_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "chat_format": CHAT_FORMAT,
        "seconds": round(time.perf_counter() - started, 2),
    }
    write_json(RESULTS / "data.json", payload)
    print(
        f"{SOURCE['path']}: {len(examples)} examples, sha256 {payload['sha256']}; "
        f"kept {len(chosen)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
