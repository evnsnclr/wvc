# E005: a checkout of a full model directory is byte-identical and loadable

Status: ANALYSED (see Outcome at the end). Frozen 2026-11-03 UTC by Fable before the extras implementation (spec 3.3a) existed; run and analysed the same day after the merge.

## Question
When a complete model directory (shards plus config, tokenizer, chat template, shard index, licence, card) is committed, does `wvc checkout` reproduce every file byte-for-byte, and does the checkout load and run exactly like the source directory?

## Inputs
`Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`, the full file set (13 files: the shard, `model.safetensors.index.json`, `config.json`, `chat_template.jinja`, `merges.txt`, `vocab.json`, `tokenizer.json`, `tokenizer_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`, `LICENSE`, `README.md`, `.gitattributes`); the shard is cached from E001 and the rest (about 23 MB) is fetched into the same snapshot. The E003 pair is reused for the second revision: `RikiyaT/Qwen3.5-0.8B-weight-space-programming`, whose directory also holds an `eval/` subdirectory and its own `README.md`.

## Procedure
1. `wvc init`; `wvc commit <base snapshot> -m "Qwen3.5-0.8B full directory" --op pretrain-checkpoint --source hf:Qwen/Qwen3.5-0.8B@2fc0636…`; record what was stored and what was skipped.
2. `wvc commit <edited snapshot> -m "Astra edit, full directory" --op direct-edit`; record stored, reused, and skipped files.
3. `wvc checkout` of each commit into fresh directories; compare every recorded file's SHA-256 with the source file's; list the source files absent from the checkout.
4. Load both the source directory and its checkout with transformers 5.22 and torch 2.15 (`AutoTokenizer`, `AutoModelForCausalLM`, bfloat16, CPU), tokenize a fixed prompt with the chat template, run one forward pass, and compare the logits of the last position (max absolute difference) and the tokenizer outputs.
5. `wvc show HEAD` and `wvc fsck`.

## Predictions (before the run)
- P1: the base commit records 11 extras (every top-level file except the shard and `.gitattributes`), about 23 MB, and skips `.gitattributes` as a dot file; nothing else is skipped.
- P2: the edited commit reuses the shard's 486 unchanged tensors and every extra except `README.md` (different card), stores 2 deltas and 1 new extra, and reports `eval/` as skipped (a directory).
- P3: every recorded file of both checkouts is byte-identical to its source (SHA-256), and the only source files absent are `.gitattributes` and the `eval/` directory.
- P4: the checkout loads with transformers; the tokenizer outputs are identical and the last-position logits agree exactly (max abs diff 0.0) with the source directory, because the bytes are identical.
- P5: `fsck` passes and the store grows by under 25 MB for the extras of both commits combined.

## Budget
Downloads at most 30 MB; runtime under 15 minutes; no GPU needed (CPU forward pass of a 0.8 B model on one short prompt).

## Outputs
`experiments/E005/run_e005.py`, `experiments/E005/receipt.json`, and an Outcome section appended here. The repository and checkouts under `experiments/E005/` are gitignored.

## Outcome (run and analysed 2026-11-03 UTC by Fable; status ANALYSED)

Run with `experiments/E005/run_e005.py` at wvc commit `fe5fd0e` plus the extras merge (`80be846`); receipt `experiments/E005/receipt.json`. Total 44.6 s including two bf16 CPU forward passes. The E005 repository (1.2 GB of payloads) stays on disk, gitignored; the checkouts were removed after hashing.

| Step | Result |
|---|---|
| commit base directory | 488 tensors stored full; extras 11 files, 23,036,295 bytes; skipped `.gitattributes (dot file)`; 9.2 s |
| commit edited directory | 486 tensors reused, 2 `delta-xor-planes`; extras 3 files recorded, of which `config.json` and the shard index deduplicated against the base and `README.md` was new; payload bytes written 518,124 = 513,248 (deltas) + 4,876 (README); 2.2 s |
| `show HEAD` | 2 changed, 486 unchanged; three `extra:` lines; operation `direct-edit`; both sources recorded |
| checkout base, checkout edit | 12 of 12 recorded files byte-identical to the source (only `.gitattributes` absent, as not recorded); 4 of 4 for the edit; 5.8 s and 7.5 s |
| load with transformers 5.22 | tokenizer output identical (30 tokens); last-position logits identical (max abs diff 0.0); top-5 next tokens `1 2 4 3 5` from both |
| `fsck --no-decode` | 504 payloads, 490 objects, 2 manifests, 2 commits; ok |
| store | 23,038,143 bytes more than E003's shard-only base commit; the edit commit added 674,904 bytes |

Prediction scorecard: P1 held (11 extras, 23.0 MB, only the dot file skipped). P2 held for what the input allowed (reuse of 486 tensors and of the two identical extras; a new README; two deltas) but its `eval/` clause was untestable: the plan described the edited model's Hub repository, while the cached snapshot used as input holds only four files (the shard, the shard index, `config.json`, `README.md`) because E001 downloaded only those; a plan error, not a tool result. P3, P4, and P5 held.

Interpretation, within the tested scope (one model family, two directories, one loader): a checkout is now a loadable model directory, not a set of shards, and it is loadable because every byte is the source's. Extras cost what they weigh (23 MB for a 1.7 GB model) and deduplicate across revisions. The one source file not reproduced, `.gitattributes`, is Git metadata and is excluded by rule. Two rules the implementation tightened beyond the spec's wording are recorded in section 3.3a: licence-prefixed names must contain no dot, and extras whose names differ only by case are refused like shards.

Limitations: the edited directory was not the full Hub repository; no model with custom code (`modeling_*.py`) or a `.bin` sidecar was tried (both are skipped by rule and reported); the `uv run` nesting printed a harmless environment warning on every CLI call, recorded in the receipt's stderr fields.
