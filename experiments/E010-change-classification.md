# E010: from the diff alone, which kind of change is an upload, and how often is that right?

Date 2026-11-03 (frozen), run and analysed 2026-11-04. Status: analysed. Every section down to Commands is unchanged from the plan frozen at `831103c` before any pair was downloaded; the run, its result, and its interpretation follow them. Evan approved the download on 2026-11-04. Software: the commit that froze this page (`git log -1 -- experiments/E010-change-classification.md`), rules version `v1` of [docs/classification.md](../docs/classification.md). Delegated run (M9 session, Opus); the lead session reviews. Raw outputs in `experiments/E010/`.

## Question
Given only a `wvc diff` record between a declared base and a Hub upload, `wvc classify` names a kind (`reupload`, `quantization`, `direction-removal`, `direct-edit`, `merged-lora`, `dense-finetune`, `mixed`, `unknown`), a tier, and the evidence. On uploads whose card or repository states the method, how often does the kind agree with the stated method, per kind, from the record alone and with the second pass (`--features`)? When a LoRA rank is stated, does the measured `r` match it?

## Labelled set
47 pairs on 14 small bases, chosen from Hub API searches on 2026-11-03 (abliterated, heretic, uncensored, merged, lora, full, by `base_model` filter; quantized and fine-tuned derivatives listed by relation) by reading each card's text and file list. Only uploads whose card, repository name, card tags, or a file in the repository (`adapter_config.json`) state the method were kept. The statement column quotes it, shortened with "..." where marked. Revisions are pinned in [`E010/labels.json`](E010/labels.json), written by [`E010/pin.py`](E010/pin.py) from the Hub API, with the commit sha and safetensors sizes of every repository. The labels are the claim, not the truth: a card can be wrong, and this page says so where the diff disagrees.

| id | claimed method | derivative @ revision | base @ revision | r stated | label source and statement | GB |
|---|---|---|---|---|---|---|
| A01 | direction-removal | `huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3` @ `3dee99d` | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae5576` | - | card: "created with abliteration (see remove-refusals-with-transformers)" | 0.99 |
| A02 | direction-removal | `huihui-ai/Huihui-Qwen3-0.6B-abliterated-v2` @ `0b07fe6` | `Qwen/Qwen3-0.6B` @ `c1899de` | - | card: "created with abliteration (see remove-refusals-with-transformers)" | 1.19 |
| A03 | direction-removal | `huihui-ai/Qwen3-1.7B-abliterated` @ `e72bde4` | `Qwen/Qwen3-1.7B` @ `70d244c` | - | card: "created with abliteration (see remove-refusals-with-transformers)" | 3.44 |
| A04 | direction-removal | `huihui-ai/Huihui-Qwen3.5-0.8B-abliterated` @ `4813135` | `Qwen/Qwen3.5-0.8B` @ `2fc0636` | - | card: "created with abliteration (see remove-refusals-with-transformers)" | 1.75 |
| A05 | direction-removal | `mlabonne/Qwen3-0.6B-abliterated` @ `41f8d67` | `Qwen/Qwen3-0.6B` @ `c1899de` | - | card: "orthogonalized to subtract this refusal direction with a given weight factor" | 2.38 |
| A06 | direction-removal | `mlabonne/gemma-3-1b-it-abliterated-v2` @ `f9317e7` | `google/gemma-3-1b-it` @ `dcc83ea` | - | card: "orthogonalized to subtract this refusal direction with a given weight factor" | 4.00 |
| A07 | direction-removal | `lunahr/gemma-3-1b-it-abliterated` @ `731aec9` | `google/gemma-3-1b-it` @ `dcc83ea` | - | card: "an abliterated version of Gemma 3, created using Baukit" | 2.00 |
| A08 | direction-removal | `saidutta69/Qwen2.5-0.5B-Instruct-heretic` @ `2bd4bb8` | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae5576` | - | card: "produced with Heretic (directional ablation / "abliteration")" | 0.99 |
| A09 | direction-removal | `wfen/Qwen3-0.6B-Heretic` @ `491e601` | `Qwen/Qwen3-0.6B` @ `c1899de` | - | card tags only: "tags: heretic, abliteration" | 1.19 |
| A10 | direction-removal | `tvall43/Qwen3.5-0.8B-heretic` @ `e9e4216` | `unsloth/Qwen3.5-0.8B` @ `23c69c5` | - | card: "made using Heretic v1.2.0; direction_index: per layer" | 1.71 |
| A11 | direction-removal | `coder3101/Qwen3.5-0.8B-heretic` @ `f4dab5e` | `Qwen/Qwen3.5-0.8B` @ `2fc0636` | - | card: "Heretic v1.2.0 with Magnitude-Preserving Orthogonal Ablation (MPOA)" | 1.71 |
| A12 | direction-removal | `Fu01978/SmolLM2-360M-Instruct-Heretic` @ `929f4c9` | `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc15` | - | card: "created using the Heretic library to surgically remove the "refusal vector"" | 0.72 |
| A13 | direction-removal | `saidutta69/SmolLM2-135M-Instruct-heretic` @ `11513fb` | `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f` | - | card: "A decensored variant; tags: heretic, abliterated" | 0.27 |
| L01 | merged-lora | `nicoboss/Llama-3.2-1B-Instruct-Uncensored` @ `ae20a62` | `meta-llama/Llama-3.2-1B-Instruct` @ `9213176` | 32 | card: "adapter: lora; lora_r: 32; lora_target_linear: true" | 2.47 |
| L02 | merged-lora | `bedio/llama-3.2-1B-airoboros-merged-qlora` @ `9035ddc` | `meta-llama/Llama-3.2-1B` @ `4e20de3` | 32 | adapter_config.json: "r: 32, all seven projections; repository name: merged-qlora" | 2.56 |
| L03 | merged-lora | `Matt300209/1B-test-merged` @ `d217118` | `meta-llama/Llama-3.2-1B` @ `4e20de3` | 32 | adapter_config.json: "r: 32, all seven projections; merged_model.safetensors" | 4.99 |
| L04 | merged-lora | `minmax23/qwen-2-5-0-5b-instruct-lora-custom` @ `634a347` | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae5576` | 32 | adapter_config.json: "r: 32; q, v, o, gate, down, up projections" | 0.99 |
| L05 | merged-lora | `nooruiit-864/qwen2.5-1.5b-base-ai-safety-domain-lora` @ `d21bada` | `Qwen/Qwen2.5-1.5B` @ `8faed76` | 16 | card: "a merged version of a QLoRA-fine-tuned adapter (r: 16 in adapter_config.json)" | 3.16 |
| L06 | merged-lora | `Pradheep1647/smollm2-135m-instruct-paper-cited-chunks-v1-sft-lr2e-5-ep8-lora32a64-seq4096-mbs8-merged` @ `a09e9cf` | `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f` | 32 | card: "training hyperparams: sft-lr2e-5-ep8-lora32a64-seq4096-mbs8-merged" | 0.27 |
| L07 | merged-lora | `Grogros/Llama-3.2-1B-distillation-alpaca-5.0-AlpacaPoison-lora64` @ `9b4ce8e` | `meta-llama/Llama-3.2-1B` @ `4e20de3` | 64 | repository name: "...-AlpacaPoison-lora64" | 2.47 |
| L08 | merged-lora | `Grogros/Llama-3.2-1B-distillation-alpaca-5.0-AlpacaPoison-lora16bis` @ `98ed716` | `meta-llama/Llama-3.2-1B` @ `4e20de3` | 16 | repository name: "...-AlpacaPoison-lora16bis" | 2.47 |
| L09 | merged-lora | `Abdullah-Taha/UTN-Qwen3-0.6B-LoRA-merged` @ `a964f3f` | `Qwen/Qwen3-0.6B` @ `c1899de` | 64 | card: "finetuned with LoRA (r=64, alpha=128) on UTN domain data, then merged" | 1.50 |
| L10 | merged-lora | `MohamedAhmedAE/Llama-3.2-1B-Instruct-Medical-Finetuned-merged` @ `9151fbc` | `meta-llama/Llama-3.2-1B-Instruct` @ `9213176` | 128 | card: "QLoRA fine-tuning (... r=128, α=256, all linear layers), then merge_and_unload()" | 2.47 |
| L11 | merged-lora | `ahczhg/llama-3.2-1b-aegis-safety-lora` @ `e463c79` | `meta-llama/Llama-3.2-1B` @ `4e20de3` | 16 | card: "Fine-tuning Method: LoRA (Low-Rank Adaptation); Rank (r): 16" | 2.47 |
| L12 | merged-lora | `BondingAI/Qwen2.5-0.5B-lora-finetuned` @ `3b4744c` | `Qwen/Qwen2.5-0.5B` @ `060db64` | - | card: "This model was fine-tuned using LORA from the base model." | 0.99 |
| L13 | merged-lora | `haydarkadioglu/Qwen3-0.6B-lora-python-expert-fine-tuned` @ `3bbf586` | `Qwen/Qwen3-0.6B` @ `c1899de` | - | card: "Fine-tuning method: LoRA / PEFT" | 1.19 |
| F01 | dense-finetune | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae5576` | `Qwen/Qwen2.5-0.5B` @ `060db64` | - | card: "This repo contains the instruction-tuned 0.5B Qwen2.5 model" | 0.99 |
| F02 | dense-finetune | `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f` | `HuggingFaceTB/SmolLM2-135M` @ `93efa2f` | - | card: "You can find the SFT dataset here ... and finetuning code at alignment-handbook" | 0.27 |
| F03 | dense-finetune | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` @ `ad9f0ae` | `Qwen/Qwen2.5-Math-1.5B` @ `4a83ca6` | - | card: "six dense models distilled from DeepSeek-R1 based on Llama and Qwen" | 3.55 |
| F04 | dense-finetune | `jsl5710/Shield-Llama-3.2-1B-Full-FT-CE` @ `c0929df` | `meta-llama/Llama-3.2-1B-Instruct` @ `9213176` | - | card: "Training method: Full-FT (CE loss)" | 2.47 |
| F05 | dense-finetune | `jsl5710/Shield-Gemma-3-1B-Full-FT-CE` @ `caa2624` | `google/gemma-3-1b-it` @ `dcc83ea` | - | card: "Training method: Full-FT (CE loss)" | 2.00 |
| F06 | dense-finetune | `simonleee/srd_A_llama-1B_fullft_seed42` @ `6326583` | `meta-llama/Llama-3.2-1B-Instruct` @ `9213176` | - | card: "Fine-tuning method: Full fine-tuning (all parameters)" | 2.47 |
| F07 | dense-finetune | `rdsm/Qwen3.5-0.8B-parakeet-52k-FullFT` @ `d1b8e0c` | `Qwen/Qwen3.5-0.8B` @ `2fc0636` | - | card: "Recipe: Full fine-tune (BF16)" | 1.71 |
| F08 | dense-finetune | `420yolomcswaggerpants/nimbus-full-finetune-0.5b` @ `c56e0ad` | `Qwen/Qwen2.5-0.5B` @ `060db64` | - | card: "all weights were updated during training, not just LoRA adapters" | 0.99 |
| F09 | dense-finetune | `jsl5710/Shield-Qwen3.5-0.8B-Full-FT-CE` @ `9242418` | `Qwen/Qwen3.5-0.8B` @ `2fc0636` | - | card: "Training method: Full-FT (CE loss)" | 1.50 |
| Q01 | quantization | `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` @ `c34a4a9` | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae5576` | - | card: "Quantization: GPTQ 4-bit" | 0.46 |
| Q02 | quantization | `Qwen/Qwen2.5-0.5B-Instruct-AWQ` @ `db09cd2` | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae5576` | - | card: "Quantization: AWQ 4-bit" | 0.73 |
| Q03 | quantization | `Qwen/Qwen3-0.6B-FP8` @ `e5be080` | `Qwen/Qwen3-0.6B` @ `c1899de` | - | card: "This repo contains the FP8 version of Qwen3-0.6B" | 1.06 |
| Q04 | quantization | `RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic` @ `e23d444` | `meta-llama/Llama-3.2-1B-Instruct` @ `9213176` | - | card: "Weight quantization: FP8; Quantized version of Llama-3.2-1B-Instruct" | 2.02 |
| Q05 | quantization | `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` @ `fcecd05` | `meta-llama/Llama-3.2-1B-Instruct` @ `9213176` | - | repository name and tags: "bnb-4bit (bitsandbytes 4-bit)" | 1.03 |
| Q06 | quantization | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `a5339a4` | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae5576` | - | card: "was converted to MLX format from Qwen/Qwen2.5-0.5B-Instruct (repository name: 4bit)" | 0.28 |
| U01 | reupload | `mlx-community/Qwen2.5-0.5B-Instruct-bf16` @ `56d07e7` | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae5576` | - | card: "was converted to MLX format from Qwen/Qwen2.5-0.5B-Instruct (bf16)" | 0.99 |
| U02 | reupload | `unsloth/Llama-3.2-1B-Instruct` @ `5a8abab` | `meta-llama/Llama-3.2-1B-Instruct` @ `9213176` | - | card: "all versions of Llama 3.2 including GGUF, 4-bit and original 16-bit formats" | 2.47 |
| U03 | reupload | `unsloth/gemma-3-1b-it` @ `5b11413` | `google/gemma-3-1b-it` @ `dcc83ea` | - | card: "all versions of Gemma 3 including GGUF, 4-bit & 16-bit formats" | 2.00 |
| E01 | direct-edit | `RikiyaT/Qwen3.5-0.8B-weight-space-programming` @ `d1671a1` | `Qwen/Qwen3.5-0.8B` @ `2fc0636` | - | card and E001: "edited by an LLM with no training (Weight-Space Programming)" | 1.75 |
| M01 | mixed | `Goekdeniz-Guelmez/Josiefied-Qwen3-0.6B-abliterated-v1` @ `3f75569` | `Qwen/Qwen3-0.6B` @ `c1899de` | - | card and name: "abliterated (name); fine-tuned with a focus on openness and instruction alignment" | 1.19 |
| M02 | mixed | `amkkk/Qwen3.5-0.8B-finetuned-lora-uncensored` @ `c2b7578` | `Qwen/Qwen3.5-0.8B` @ `2fc0636` | 16 | card: "LoRA weights have already been merged; tags: refusal-direction; lora_r: 16" | 1.71 |

Kinds: 13 direction-removal (four abliteration tools: remove-refusals-with-transformers, mlabonne's, Baukit, Heretic), 13 merged-lora (11 with r stated: 16, 32, 64, 128), 9 dense-finetune, 6 quantization (GPTQ, AWQ, FP8 twice, bitsandbytes 4-bit, MLX 4-bit), 3 reupload, 1 direct-edit, 2 mixed (cards that state two methods). The direct edit is E001's pair, run again here with `--all` and `--features`; [E001](E001-astra-edit-diff.md)'s receipt is not touched.

Left out after reading: GGUF-only uploads (M8 is not merged), adapter-only repositories, `mlabonne/Qwen3-1.7B-abliterated` (it ships the same tensors twice, as one file and as two shards, which `wvc diff` refuses as duplicate names), a QLoRA merge stored as bitsandbytes 4-bit (`NEldin10/llama-3.2-1B-ft-v0.1-qlora-merged`: two methods with no card statement of either), and cards whose method text is boilerplate ("More Information Needed").

Census: at freeze time the census branch (M7, E009) had no committed receipt; its uncommitted receipts in its worktree cover `unsloth/Qwen2.5-0.5B-Instruct`, two `unsloth` bnb-4bit variants of it, `mobilint/Qwen2.5-0.5B-Instruct`, `livekit/turn-detector`, `Skywork/Skywork-Reward-V2-Qwen3-0.6B`, and `OpenGVLab/InternVL2_5-1B`, none of which is in this set. Its receipts wrap the same `wvc diff --rank --json` record under a `diff` key, so the classifier runs on them unchanged once unwrapped; E010's receipts are that record, with `--all`, `--classify`, and `--features`.

## Predictions (before the run)
Per kind, with features (record alone in parentheses where it differs). "Recall" is the share of a claimed kind that gets that kind; "precision" is the share of a predicted kind whose claim agrees.

- P1 reupload: 3 of 3, precision 1.0. U01 strong (R1, every tensor byte-identical); U02 and U03 R1 or R1b.
- P2 quantization: 6 of 6, precision 1.0. Q01, Q02, Q05, Q06 by R2b (packed names); Q03 and Q04 by R2c (FP8 with scales). All weak: none is a plain cast.
- P3 direction-removal: 10 of 13 with features (11 of 13 from the record alone, where the shared-direction test cannot fail), precision at least 0.9. Expected misses: A11 (MPOA rescales each row's norm after the projection, so the change is not rank-1; predicted `merged-lora` or `mixed`), A10 (a direction per layer; coherence may fall under 0.5 and the pair then reads as a rank-1 LoRA), and one of A05, A06, A07 (tools that also edit embeddings or layer norms, or a "weight factor" that weakens some layers' change below the rank-1 test).
- P4 merged-lora: 8 of 13 with features, precision at least 0.8; record alone the same 8 at weak tier. Expected misses: L05, L09, L11 (stored in F16 while the base is BF16: every tensor is cast, and the LoRA change exceeds F16 rounding, so `mixed` by M1); L10 (r = 128, above the low-rank limit of 64: `dense-finetune` weak by R6); L07 or L08 (the repository name is the only statement; if the run trained more than the adapter, they read as dense). L07 and L08 are predicted weak if they resize the vocabulary (their parameter count is the base's plus 8,192).
- P5 `r`: for the pairs with r stated, r ≤ 64, and stored in BF16 or F32 (L01, L02, L03, L04, L06, L07, L08), the measured `r` (most common `signal_rank`) equals the stated r for at least 5 of 7, and is never above it.
- P6 dense-finetune: 8 of 9, precision at least 0.8 (L10 is expected here). The model families' own instruction-tuned releases (F01, F02) and the community full fine-tunes change the norms in every layer, so strong.
- P7 direct-edit: E01 strong by R4, the E001 result.
- P8 mixed: at most 1 of 2. M01 is expected as `dense-finetune` (the fine-tune swamps the ablation); M02 as `merged-lora` or `mixed` by M2.
- P9 overall: at least 36 of 47 kinds agree with the claim with features, and at least 34 of 47 from the record alone.
- P10 failures: no pair makes `wvc diff` fail; if one does, it is a result and gets a test.

## Budget and stop rules
- Download at most 107.4 GB (25.4 GB of bases and 82.0 GB of derivatives if every shard is fetched; a Hub pair skips shards that are byte-identical on both sides) into a dedicated Hugging Face cache outside the repository, so removing it touches no other session's files. Peak disk: the largest pair is 7.5 GB; one base and one derivative are on disk at a time.
- After each pair the derivative's repository is removed from that cache; a base is removed after the last pair that uses it (pairs run grouped by base, so each base is fetched once). The cache directory is removed at the end.
- Stop before a pair when less than 50 GB is free (215 GB free on 2026-11-03).
- No model is run; no generations; nothing is pushed. Kept in Git: one JSON receipt per pair under `E010/pairs/`, `run.log`, `labels.json`, and the scripts. No weights.

## Commands
```bash
uv sync --locked --extra hf
uv run python experiments/E010/synthetic_noise.py > experiments/E010/synthetic_noise.txt
uv run python experiments/E010/pin.py experiments/E010/labels.json
uv run python experiments/E010/run_pairs.py experiments/E010/labels.json experiments/E010/pairs --cache "$E010_CACHE"
```
`run_pairs.py` runs, for each pair, `wvc diff --rank --all --json --classify --features hf:BASE@SHA hf:DERIVATIVE@SHA` and writes its standard output to `E010/pairs/<id>.json` (standard error, if any, to `<id>.stderr.txt`; a failed diff's output to `<id>.failed.json`), and appends one line to `E010/run.log`: start and end in UTC, wall seconds, exit status, the bytes the cache grew by, and free disk. The record-only classification of each pair is computed afterwards from the same receipt with its `features` keys removed, by the scoring script, so both columns come from one download.

## Run
2026-11-04, 04:15 to 06:08 UTC, on the M4 Max, with Python 3.13.5, numpy 2.5.4, and huggingface_hub 2.4.0. 47 pairs in 112 minutes of wall time; 98,734,862,136 bytes downloaded (`run.log`, 48 lines: one per pair plus F08's re-run). Free disk never fell below 87.9 GB, so the 50 GB floor was never reached; the census session was downloading on the same disk at the same time. 46 receipts are in `E010/pairs/`. A03 has none: `huihui-ai/Qwen3-1.7B-abliterated` is gated (HTTP 403, `pairs/A03.stderr.txt`), and requesting access is an account action this run did not take. Its base, 4.06 GB, was fetched before the refusal.

## Deviations
- Cleanup. huggingface_hub 2.x keeps every file in one `blobs/` directory and links to it from each repository's directory. The runner removed the repository directories and so freed nothing: the dedicated cache held 89 GB of blobs at the end, not the planned one pair (7.5 GB at most). The 50 GB floor held throughout. The cache was deleted after the run, and `run_pairs.py` now removes the blobs a repository links to unless another cached repository links to them as well. F08's re-run, with the fixed runner, left 16 KB.
- A crash in `--features`, fixed during the run. On F08, a matrix whose whole change lay below the noise edge raised an `IndexError`. The traceback exited 1, the code `diff` uses for "differences found", so the runner kept the empty receipt and moved on. Each pair runs in a fresh process, so every pair started after the fix used it: a matrix with no signal now gets `signal_rank` 0 and the rules use its record statistics (test `test_change_below_the_noise_edge_has_no_signal`). F02, A13, L06, A12, F01, and the first F08 ran before the fix, and of these only F08 reached the crash, so the other receipts are unaffected. F08 was re-run after the fix. The fix and the runner's new check (a receipt counts only if it parses as a record) are in `e6add58`.
- The record-only column was computed after the run by `score.py`, as planned, with the classifier of the moment: the v1 code plus the crash fix above.

## Result under rules v1 (the frozen test)
Agreement with the claimed method: 28 of 46 with features and 33 of 46 from the record alone (`scores-v1.json`). The second pass did worse than the record alone, for reasons listed under Misses.

Confusion, with features (rows: claimed method; columns: predicted kind):

| claimed \ predicted | reupload | quantization | direction-removal | direct-edit | merged-lora | dense-finetune | mixed | unknown |
|---|---|---|---|---|---|---|---|---|
| reupload (3) | 3 | | | | | | | |
| quantization (6) | | 4 | | | | | | 2 |
| direction-removal (12) | | | 9 | | 2 | | 1 | |
| direct-edit (1) | | | | 1 | | | | |
| merged-lora (13) | 2 | | | | 4 | 3 | 4 | |
| dense-finetune (9) | | | | | | 7 | 1 | 1 |
| mixed (2) | | | 1 | | | | | 1 |

Confusion, from the record alone:

| claimed \ predicted | reupload | quantization | direction-removal | direct-edit | merged-lora | dense-finetune | mixed | unknown |
|---|---|---|---|---|---|---|---|---|
| reupload (3) | 3 | | | | | | | |
| quantization (6) | | 4 | | | | | | 2 |
| direction-removal (12) | | | 11 | | 1 | | | |
| direct-edit (1) | | | | 1 | | | | |
| merged-lora (13) | 2 | | | | 6 | 2 | 3 | |
| dense-finetune (9) | | | | | | 8 | | 1 |
| mixed (2) | | | 1 | | | | | 1 |

Precision and recall per kind, v1, with features (record alone in parentheses):

| kind | precision | recall |
|---|---|---|
| reupload | 0.60 (0.60) | 1.00 (1.00) |
| quantization | 1.00 (1.00) | 0.67 (0.67) |
| direction-removal | 0.90 (0.92) | 0.75 (0.92) |
| direct-edit | 1.00 (1.00) | 1.00 (1.00) |
| merged-lora | 0.67 (0.86) | 0.31 (0.46) |
| dense-finetune | 0.70 (0.80) | 0.78 (0.89) |
| mixed | 0.00 (0.00) | 0.00 (0.00) |

LoRA rank, where the second pass measured it: L01 32 (stated 32; the record's median rank was 123.5), L02 32 (stated 32), M02 16 on every module (stated 16), L12 16 and L13 8 (no rank stated). On every pair whose bytes are LoRA-shaped and whose rank is stated, the measured `r` equals the stated one.

Predictions:

| | Held | Observed (v1) |
|---|---|---|
| P1 | partly | recall 3 of 3, all R1 strong; precision 0.60, because L06 and L10, claimed LoRA merges, are byte-identical to their bases |
| P2 | no | 4 of 6. Q03 and Q04 (FP8) came out `unknown`: only the projections are cast (196 of 311 and 112 of 146 tensors), under R2's 90% share, and R2c was never reached |
| P3 | partly | 9 of 12 with features (11 of 12 from the record alone), precision 0.90 (0.92). The predicted misses were wrong: A10 (a direction per layer, coherence above 0.5) and A11 (MPOA, still rank-1) passed, and A05, A06, and A07 missed |
| P4 | no | 4 of 13 with features (6 from the record alone). L05, L09, and L11 came out `mixed` by M1 as predicted. L10 was predicted `dense-finetune` and is byte-identical to its base; L07 and L08 came out `dense-finetune` as predicted; L03, L04, and L06 were not predicted (see Misses) |
| P5 | no | 2 of 7 (L01 and L02 exact). L03, L04, L06, L07, and L08 carry no rank-32 or rank-64 change at all. The measured `r` was never above the stated one |
| P6 | no | 7 of 9 with features, precision 0.70; 8 of 9 from the record alone, precision 0.80. F06 came out `mixed` and F07 `unknown` |
| P7 | yes | E01 `direct-edit`, strong, R4: 2 of 488 tensors, 64 columns, rank 1 |
| P8 | no | 0 of 2 (the bound "at most 1" held). M01 came out `direction-removal`, not the predicted `dense-finetune`; M02 came out `unknown`, not `merged-lora` or `mixed` |
| P9 | no | 28 of 46 with features and 33 of 46 from the record alone, against at least 36 and 34 |
| P10 | no | `--features` crashed on F08 (fixed, with a test). A03 was refused by the Hub, not by wvc |

## Misses under v1, with the evidence that misled
Each line gives the predicted kind, the measurement behind it, and whether the cause is the rules or the claim. "Contradicted" means the bytes disagree with the stated method. That judgement was made after the run and comes from the receipts (`inspect_pair.py`), not from any statement by the uploader.

- A05 (`mlabonne/Qwen3-0.6B-abliterated`, stored F32): `merged-lora`, r = 1, strong. o_proj and down_proj are rank-1 in 28 of 28 layers (signal top-1 energy 1.00), but `direction_coherence` was 0.39, under 0.5, so R3 was set aside and R5 read the pair as a rank-1 LoRA. Rules.
- A06 (`mlabonne/gemma-3-1b-it-abliterated-v2`, F32): `mixed`, M2. All seven projections changed, each about rank-1 (record top-1 energy 0.97 to 0.985); R3 accepted output-side changes only, and coherence was 0.27. Rules.
- A07 (`lunahr/gemma-3-1b-it-abliterated`): `merged-lora`, r = 1, weak. Rank-1 o/down changes in 23 of 26 layers, coherence 0.31, plus a resized embedding. Rules.
- L03 (`Matt300209/1B-test-merged`): `dense-finetune`, weak. Every projection changed at full rank (median 1,925 of 2,048) with relative change 0.13 to 0.16, and the norms unchanged. A rank-32 merge onto `Llama-3.2-1B` cannot do this. Contradicted.
- L04 (`minmax23/qwen-2-5-0-5b-instruct-lora-custom`): `mixed`, M2. The changed projections move by 200 to 1,700 times their own norm, so these are not small changes to the stated base. Contradicted.
- L05, L09, L11 (LoRA merges saved in F16 over BF16 bases): `mixed`, M1, as P4 predicted. Every tensor is cast BF16 to F16, with deviations above F16's rounding. Rules (v1 had no notion of a re-encoded cast).
- L06 (`Pradheep1647/...-lora32a64-...-merged`): `reupload`. The shard is byte-identical to `SmolLM2-135M-Instruct`, skipped on the Hub with 0 bytes downloaded. Contradicted.
- L07, L08 (`Grogros/...-lora64`, `...-lora16bis`): `dense-finetune`, weak. Full-rank changes in every projection (median signal rank about 1,560), norms changed in 6 layers of L07, and 4 tokens added to the vocabulary. The repository name is the only statement of the method. Contradicted.
- L10 (`MohamedAhmedAE/...-Medical-Finetuned-merged`, r = 128 stated): `reupload`. Byte-identical to `Llama-3.2-1B-Instruct`. Contradicted.
- F06 (`simonleee/srd_A_llama-1B_fullft_seed42`): `mixed`, M2, while the record alone said `dense-finetune`. Relative change 0.001: most of the change lies under the bfloat16 floor (median SNR 1.8), and the few components above the edge looked low-rank. Rules.
- F07 (`rdsm/Qwen3.5-0.8B-parakeet-52k-FullFT`): `unknown`. Every language-model layer changed at full rank, but the unchanged vision tower and the dropped multi-token-prediction layers put the changed share at 235 of 473 (0.497), under R6's half. Rules.
- Q03, Q04 (FP8): `unknown`, as in P2. Rules.
- M01 (`Goekdeniz-Guelmez/Josiefied-Qwen3-0.6B-abliterated-v1`): `direction-removal`, strong. Only o_proj and down_proj changed, rank-1 in 27 of 28 layers, coherence 1.00, nothing else. The fine-tune the card describes left no trace in the weights. Contradicted for the "fine-tuned" half.
- M02 (`amkkk/Qwen3.5-0.8B-finetuned-lora-uncensored`): `unknown`. The signal rank is 16 on every module (stated r 16), but R5's "same module set in 80% of the layers" failed on a hybrid stack (18 linear-attention and 6 full-attention layers). Rules.

Seven of the 46 claims are contradicted by the bytes (L03, L04, L06, L07, L08, L10, M01). On the other 39, v1 agrees on 28 with features and 31 from the record alone.

## Rules v2, written after these results
The changes, with the pair behind each, are listed in [docs/classification.md, "Rules v2"](../docs/classification.md#rules-v2-2026-11-04-after-e010). They are summarised here:
- layer counts and shares taken in the main layer stack (F07, A04);
- dtype casts read whatever the status, with re-encoded casts (BF16 to F16) treated as changes (L05, L09, L11);
- the FP8 test before the share test, and the share taken over matrices (Q03, Q04);
- a new rule M3 (norms changed alongside a quantization), for E011's Qwen GGUF;
- R3 accepting rank-1 changes on the input side and in any stack, with coherence 0.2 and a rank-1 threshold of 0.75 for features as for the record (A05, A06, A07, A04);
- an SNR floor of 4 for features (F06);
- R5's module coverage per module, ignoring matrices no wider than 64 (M02);
- M8's GGUF record read as merged (`comparison` and ggml dtypes).

v2 was re-run on the same 46 receipts, without a new download (`score.py --rules current`, `scores-v2.json`). These are training-set numbers, since the rules were revised on this set: 37 of 46 with features and 39 of 46 from the record alone (v1: 28 and 33). On the 39 claims the bytes do not contradict, v2 agrees on 37 both ways (v1: 28 and 31).

Confusion under v2, with features:

| claimed \ predicted | reupload | quantization | direction-removal | direct-edit | merged-lora | dense-finetune | mixed | unknown |
|---|---|---|---|---|---|---|---|---|
| reupload (3) | 3 | | | | | | | |
| quantization (6) | | 6 | | | | | | |
| direction-removal (12) | | | 11 | | | | 1 | |
| direct-edit (1) | | | | 1 | | | | |
| merged-lora (13) | 2 | | | | 7 | 3 | 1 | |
| dense-finetune (9) | | | | | | 9 | | |
| mixed (2) | | | 1 | | 1 | | | |

Precision and recall per kind, v2, with features (record alone in parentheses): reupload 0.60 / 1.00 (0.60 / 1.00); quantization 1.00 / 1.00 (1.00 / 1.00); direction-removal 0.92 / 0.92 (0.92 / 0.92); direct-edit 1.00 / 1.00 (1.00 / 1.00); merged-lora 0.88 / 0.54 (0.82 / 0.69); dense-finetune 0.75 / 1.00 (0.82 / 1.00); mixed none predicted / 0.00 (none / 0.00). The record alone scores higher on merged-lora only because R5 accepts the structure there at weak tier: it counts L03 and L04, whose bytes contradict a LoRA, as agreements.

Misses left under v2:
- A06: `mixed`, M2. The input-side changes in layers 23 and 24 have top-1 energy 0.72 to 0.75, just under the rank-1 threshold. Lowering the threshold again for one pair would fit the rules to it, so v2 leaves this miss in place.
- M02: `merged-lora`, r = 16, strong. The ablation folded into the LoRA is not separated from it, and no rule can split one change into two.
- The seven contradicted claims, each classified by what its bytes show.

Per pair (both rule versions; `r` from features where measured, else marked as the record's):

| id | claimed | v1 features | v1 record | v2 features | v2 record | r stated / measured (v2) |
|---|---|---|---|---|---|---|
| A01 | direction-removal | direction-removal strong R3 | direction-removal weak R3 | direction-removal strong R3 | direction-removal weak R3 | - / - |
| A02 | direction-removal | direction-removal strong R3 | direction-removal weak R3 | direction-removal strong R3 | direction-removal weak R3 | - / - |
| A03 | direction-removal | not run: gated repository (HTTP 403) | | | | |
| A04 | direction-removal | direction-removal strong R3 | direction-removal weak R3 | direction-removal strong R3 | direction-removal weak R3 | - / - |
| A05 | direction-removal | merged-lora strong R5 | direction-removal weak R3 | direction-removal strong R3 | direction-removal weak R3 | - / - |
| A06 | direction-removal | mixed weak M2 | merged-lora weak R5 | mixed weak M2 | merged-lora weak R5 | - / - |
| A07 | direction-removal | merged-lora weak R5 | direction-removal weak R3 | direction-removal weak R3 | direction-removal weak R3 | - / - |
| A08 | direction-removal | direction-removal strong R3 | direction-removal weak R3 | direction-removal strong R3 | direction-removal weak R3 | - / - |
| A09 | direction-removal | direction-removal strong R3 | direction-removal weak R3 | direction-removal strong R3 | direction-removal weak R3 | - / - |
| A10 | direction-removal | direction-removal strong R3 | direction-removal weak R3 | direction-removal strong R3 | direction-removal weak R3 | - / - |
| A11 | direction-removal | direction-removal strong R3 | direction-removal weak R3 | direction-removal strong R3 | direction-removal weak R3 | - / - |
| A12 | direction-removal | direction-removal strong R3 | direction-removal weak R3 | direction-removal strong R3 | direction-removal weak R3 | - / - |
| A13 | direction-removal | direction-removal strong R3 | direction-removal weak R3 | direction-removal strong R3 | direction-removal weak R3 | - / - |
| L01 | merged-lora | merged-lora strong R5 | merged-lora weak R5 | merged-lora strong R5 | merged-lora weak R5 | 32 / 32 |
| L02 | merged-lora | merged-lora strong R5 | merged-lora weak R5 | merged-lora strong R5 | merged-lora weak R5 | 32 / 32 |
| L03 | merged-lora | dense-finetune weak R6 | merged-lora weak R5 | dense-finetune weak R6 | merged-lora weak R5 | 32 / - |
| L04 | merged-lora | mixed weak M2 | merged-lora weak R5 | mixed weak M2 | merged-lora weak R5 | 32 / - |
| L05 | merged-lora | mixed strong M1 | mixed strong M1 | merged-lora weak R5 | merged-lora weak R5 | 16 / 256 (record) |
| L06 | merged-lora | reupload strong R1 | reupload strong R1 | reupload strong R1 | reupload strong R1 | 32 / - |
| L07 | merged-lora | dense-finetune weak R6 | dense-finetune weak R6 | dense-finetune weak R6 | dense-finetune weak R6 | 64 / - |
| L08 | merged-lora | dense-finetune weak R6 | dense-finetune weak R6 | dense-finetune weak R6 | dense-finetune weak R6 | 16 / - |
| L09 | merged-lora | mixed strong M1 | mixed strong M1 | merged-lora weak R5 | merged-lora weak R5 | 64 / 64 (record) |
| L10 | merged-lora | reupload strong R1 | reupload strong R1 | reupload strong R1 | reupload strong R1 | 128 / - |
| L11 | merged-lora | mixed strong M1 | mixed strong M1 | merged-lora weak R5 | merged-lora weak R5 | 16 / 16 (record) |
| L12 | merged-lora | merged-lora strong R5 | merged-lora weak R5 | merged-lora strong R5 | merged-lora weak R5 | - / 16 |
| L13 | merged-lora | merged-lora strong R5 | merged-lora weak R5 | merged-lora strong R5 | merged-lora weak R5 | - / 8 |
| F01 | dense-finetune | dense-finetune strong R6 | dense-finetune strong R6 | dense-finetune strong R6 | dense-finetune strong R6 | - / - |
| F02 | dense-finetune | dense-finetune strong R6 | dense-finetune strong R6 | dense-finetune strong R6 | dense-finetune strong R6 | - / - |
| F03 | dense-finetune | dense-finetune strong R6 | dense-finetune strong R6 | dense-finetune strong R6 | dense-finetune strong R6 | - / - |
| F04 | dense-finetune | dense-finetune weak R6 | dense-finetune weak R6 | dense-finetune weak R6 | dense-finetune weak R6 | - / - |
| F05 | dense-finetune | dense-finetune strong R6 | dense-finetune strong R6 | dense-finetune strong R6 | dense-finetune strong R6 | - / - |
| F06 | dense-finetune | mixed weak M2 | dense-finetune weak R6 | dense-finetune weak R6 | dense-finetune weak R6 | - / - |
| F07 | dense-finetune | unknown weak U | unknown weak U | dense-finetune strong R6 | dense-finetune strong R6 | - / - |
| F08 | dense-finetune | dense-finetune strong R6 | dense-finetune strong R6 | dense-finetune strong R6 | dense-finetune strong R6 | - / - |
| F09 | dense-finetune | dense-finetune strong R6 | dense-finetune strong R6 | dense-finetune strong R6 | dense-finetune strong R6 | - / - |
| Q01 | quantization | quantization weak R2b | quantization weak R2b | quantization weak R2b | quantization weak R2b | - / - |
| Q02 | quantization | quantization weak R2b | quantization weak R2b | quantization weak R2b | quantization weak R2b | - / - |
| Q03 | quantization | unknown weak U | unknown weak U | quantization weak R2c | quantization weak R2c | - / - |
| Q04 | quantization | unknown weak U | unknown weak U | quantization weak R2c | quantization weak R2c | - / - |
| Q05 | quantization | quantization weak R2b | quantization weak R2b | quantization weak R2b | quantization weak R2b | - / - |
| Q06 | quantization | quantization weak R2b | quantization weak R2b | quantization weak R2b | quantization weak R2b | - / - |
| U01 | reupload | reupload strong R1 | reupload strong R1 | reupload strong R1 | reupload strong R1 | - / - |
| U02 | reupload | reupload strong R1 | reupload strong R1 | reupload strong R1 | reupload strong R1 | - / - |
| U03 | reupload | reupload strong R1 | reupload strong R1 | reupload strong R1 | reupload strong R1 | - / - |
| E01 | direct-edit | direct-edit strong R4 | direct-edit strong R4 | direct-edit strong R4 | direct-edit strong R4 | - / - |
| M01 | mixed | direction-removal strong R3 | direction-removal weak R3 | direction-removal strong R3 | direction-removal weak R3 | - / - |
| M02 | mixed | unknown weak U | unknown weak U | merged-lora strong R5 | merged-lora weak R5 | 16 / 16 |

## Applied to unlabelled records: the E009 census and E011's GGUF files
Rules v2 run over the 93 census receipts from their records alone (`classify_census.py`, `census-kinds.json`; no download, no features, no `--all`). Result: 36 `quantization`, 19 `unknown`, 18 `dense-finetune`, 12 `reupload`, 6 `merged-lora`, 2 `direction-removal`. By the Hub's relation tag:
- `finetune`: 16 dense-finetune, 12 reupload, 8 unknown, 5 merged-lora, 1 direction-removal, 1 quantization;
- `quantized`: 35 quantization, 9 unknown, 1 merged-lora, 1 direction-removal;
- `merge`: 2 unknown, 2 dense-finetune.

The `unknown`s are mostly repackagings in which every tensor name changed: the closest rule is R1, which allows at most two names added or removed. A diff cannot tell such a repackaging from a real change, as E009 found for 51 pairs. The two `direction-removal` pairs are the uploads whose names say abliterated or heretic, one of them tagged `quantized`. None of these pairs is labelled, so this is a description, not a measurement.

On E011's three GGUF comparisons, v2 says:
- `quantization` (R2, strong) for SmolLM2-135M-Instruct Q4_K_M, whose largest relative change is 0.007 against a bound of 0.011;
- `quantization` (R2, strong) for its F16 conversion;
- `mixed` (M3) for Qwen2.5-0.5B-Instruct Q4_K_M, whose 49 norm and bias vectors changed, as E011 found by hand.

## Interpretation
- The frozen rules agreed with the stated method on 28 of 46 uploads with the second pass and 33 of 46 from the record alone. The 11 v1 misses on claims the bytes support came from five assumptions that real checkpoints break:
  - whole-model shares (a vision tower, quantized formats that keep norms in floats: F07, Q03, Q04);
  - a coherence threshold set from theory rather than measurement (A05, A06, A07);
  - no notion of a change stored in another encoding (L05, L09, L11);
  - features trusted when the change sits at the rounding floor (F06);
  - one module set per layer in a hybrid stack (M02).
- The second pass does what it was built for. Where an upload is a LoRA merge in its bytes, the measured rank equals the stated one (32, 32, 16), and the record's own estimate does not (123.5 for L01, or full rank).
- Seven of 46 stated methods are contradicted by the bytes. Two "merged LoRA" uploads are byte-identical to their bases. Four "LoRA" uploads carry full-rank or off-scale changes. One "fine-tuned" model is a pure single-direction ablation. A classifier scored against cards measures the cards as much as itself; a diff can audit what a card claims.
- Direction removal is the most legible kind: 11 of 12 under v2, across four tools. Heretic's per-layer directions (A10, coherence above 0.5) and MPOA (A11, still rank-1) are recognised, which v1's predictions did not expect.
- Not shown: anything about behaviour. No model was run, so "direction-removal" is a description of the weights, not a claim that refusals were removed.

## Limitations
- The labels are what uploaders state, read by this session; seven are contradicted, and the others are unverified beyond the bytes.
- v2 was revised on this set. Its numbers here are not a validation; a fresh labelled set, for instance the census pairs whose cards state a method, is the next test.
- One labelled direct edit, and two cards stating mixed methods; precision for those kinds rests on almost nothing.
- No `model-merge` kind: SLERP or TIES merges read as `dense-finetune`. A change that combines two methods in one tensor (M02) is reported as the dominant one.
- The census application is record-only and without `--all`, so its layer counts are lower bounds.
- Inputs are on the Hub, pinned in `labels.json`; re-running needs about 99 GB of downloads (one gated repository needs access).
