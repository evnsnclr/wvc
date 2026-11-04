# E009: what do derivative uploads of popular open-weight bases change, tensor by tensor?

Date 2026-11-03, run 2026-11-04. Status: analysed. The Question, Selection, Definitions, Predictions, Budgets, and Commands sections were written and committed before any pair was diffed; the Result and Outcome sections are added after the run and say where the run deviated. Delegated run: this page, the script, and the predictions were written by a delegated Opus session under the lead session's brief (milestone M7, the roadmap item "A census of the Hub" in [`ROADMAP.md`](../ROADMAP.md)), not by the lead session. Software: wvc 0.1.0, source at commit `5e22349` when the page was frozen (every receipt records the commit it ran at and whether `src/` was dirty), Python 3.13.5, huggingface_hub 2.4.0, numpy 2.5.4, macOS 15.6 on an Apple M4 Max; PACE jobs record their own Python and platform. Raw outputs in `experiments/E009/`.

## Question
The Hub carries six to nine hundred derivative uploads a month per popular base (the maintainer's count from the Hugging Face model API, `filter=base_model:<relation>:<base>`, uploads created in the 30 days to 2026-11-02, for three Qwen3.5 bases), and almost no card says what changed. What do the most downloaded ones change, tensor by tensor: how many are byte-identical re-uploads, how many rewrite every tensor, how many touch a few projections at low rank, how many change dtype or tensor names, and where does `wvc diff` refuse or break?

## Selection rule (fixed before the run)
- Bases, fixed list (`BASES` in `experiments/E009/census.py`), with the number of uploads taken per base, `k`:
  - small, run on the laptop, `k = 10`: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3-0.6B`, `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen3-1.7B`, `meta-llama/Llama-3.2-1B-Instruct`, `google/gemma-3-1b-it`;
  - 8B to 9B, `k = 6`: `google/gemma-2-9b-it` on PACE (embers QOS, CPU only); `Qwen/Qwen3.5-9B`, `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct` on the laptop. The split is forced by the PACE allocation: `paceship-srdsgt26` carries an instruction to avoid Qwen and Alibaba software and has an approved Gemma route, so no Qwen model goes to PACE; Llama stays on the laptop to keep PACE inside the approved family.
  - Why these: the API on 2026-11-03 listed 460 to 1,000 finetune-tagged derivatives for each, most with tens to hundreds over 100 downloads (probe in the session; not a receipt). All bases were readable with Evan's token, including the gated Llama and Gemma ones.
- Candidates: `HfApi.list_models(filter="base_model:<base>", sort="downloads")`, the Hub's downloads over the last 30 days, at most 300 rows, read once per base and saved under `experiments/E009/selection/` with the listing time; a resumed run reuses the saved walk.
- The walk takes candidates in listing order until `k` eligible uploads are taken. Every candidate passed over gets a `skipped` row with its reason, in this order of checks: `adapter` (the Hub tag says `base_model:adapter:<base>`; an adapter is not a checkpoint of the base's tensors), `api-error` (the model record could not be read), `no-safetensors` (no top-level `*.safetensors` file: GGUF, ONNX, or subfolder-only repositories; this round is safetensors-only), `size-cap` (the upload's top-level safetensors exceed min(2.5 x the base's safetensors bytes, 45 GB); 2.5 x admits an F32 upload of a BF16 base).
- Relations come from the Hub's `base_model:<relation>:<base>` tag (inferred by the Hub) and, separately, from the card's own `base_model_relation` field when it has one; both are recorded.
- Each eligible pair runs once: `python -m wvc diff --rank --json hf:<base>@<sha> hf:<upload>@<sha>` through the installed package, both sides pinned to the commit the listing saw, default rank cap (2^26 elements) and tolerance (1e-2 relative).

## Definitions used by the predictions
Over the pairs with a receipt (`diffed`), from the diff's summary counts:
- byte-identical re-upload: every tensor `unchanged`, nothing added, removed, or reshaped.
- every tensor changed: no tensor `unchanged`, nothing added or removed (`changed` and `reshaped` only).
- low rank on a few projections: at least one `changed` tensor, nothing added, removed, or reshaped, at most half the tensors changed, at least one rank measured, and every measured rank at most 64.
- renamed layout: added plus removed at least half the tensors.
- dtype rewrite: `reshaped` at least half the tensors.

## Predictions (before the run)
- P1: byte-identical re-uploads are 5% to 20% of diffed pairs (point guess 10%).
- P2: every-tensor-changed pairs are 35% to 65% of diffed pairs; they are the dense fine-tunes.
- P3: among diffed pairs tagged `finetune`, 5% to 20% are low rank on a few projections (merged LoRAs and edits that the card calls a fine-tune).
- P4: renamed layouts are 10% to 30% of diffed pairs, most of them on `Qwen/Qwen3.5-*` bases, whose tensors carry the multimodal `model.language_model.` prefix that text-only exports drop.
- P5: at least 80% of diffed pairs tagged `quantized` are dtype rewrites or renamed layouts (packed integer weights, added scales).
- P6: among all candidates walked for the seven small bases, at least 30% are passed over as `no-safetensors`.
- P7: at least one pair crashes wvc with a traceback (a bug by the project's rule) (40% credence); refusals with exit 2 are 5% to 15% of attempted pairs.
- P8: the median `fraction_changed` of every-tensor-changed pairs is at least 0.9: a dense fine-tune moves almost every element, not only almost every tensor.
- P9: at least two diffed pairs contradict their relation tag: tagged `finetune` or `quantized` but byte-identical, or tagged `quantized` with every element in its original dtype.

## Budgets and stop rules
- Network, measured as bytes written into the dedicated cache (`bytes_new` in every row; HTTP range requests for skipped shards' headers are a few hundred kilobytes and not counted): laptop at most 300 GB per run and 500 GB for E009 in all; PACE at most 200 GB per job. The script stops before a pair whose estimated fetch would pass the run's cap (exit 3).
- Disk: the laptop had 215 GB free on 2026-11-03; the script stops before a pair would leave less than 100 GB free. The Hugging Face cache for the census is a dedicated directory (`~/.cache/wvc-census/hub`; on PACE under the private project root), never the user's main cache. After each pair the upload's repository and every blob no remaining snapshot references are deleted, so only bases stay; the cache is emptied at the end of E009.
- Time: at most 2 h per pair (a timeout is a `failed` row), about 12 h of laptop wall time in all. PACE: `embers` QOS only, partition `cpu-small` or another CPU partition, at most 8 h per job, at most three jobs; never a charged queue.
- Nothing enters Git but `selection/*.json`, `pairs/*.json`, `failures/*.json`, `census.csv`, `summary.md`, the scripts, and run logs. No model is run.

## Receipt format (stable; the change classifier of M9 reads it)
- `pairs/<owner>__<name>.json`, `format: "wvc-census-pair/1"`: `census` holds the experiment, host, base and upload ids and pinned shas, listing rank and time, the Hub tag relation, the card's `base_model` and `base_model_relation`, the 30-day downloads, the creation date, the upload's top-level shards (name, bytes, LFS SHA-256), the wvc command, the exit code (1 means differences found), wall seconds, `bytes_new`, the shards fetched on each side, and the versions (`wvc_version`, `wvc_source_commit`, `wvc_source_dirty`, `huggingface_hub`, `numpy`, `python`, `platform`). `diff` is the unmodified `wvc diff --json` document: `a`, `b` (Hub ids, roots, skipped and fetched files), `summary` (tensor counts by status, elements compared and changed), and `tensors` (every tensor that is not unchanged, with dtype, shape, changed elements, max abs delta, norms, rows and columns touched, rank and top-1 energy).
- `failures/<owner>__<name>.json`, `format: "wvc-census-failure/1"`: the same `census` block, a `reason`, and the last 2,000 characters of stderr with home paths replaced.
- `selection/<owner>__<name>.json`, `format: "wvc-census-selection/1"`: the base's pinned sha and shards, `k`, the size cap, and every walked candidate with its eligibility and reason.
- `census.csv`: one row per pair per attempt, append-only; the latest row for a pair is its state.

## Commands
```bash
uv sync --locked --extra hf
uv run python experiments/E009/census.py --cache ~/.cache/wvc-census/hub --group small --max-bytes 300e9 --min-free 100e9
uv run python experiments/E009/census.py --cache ~/.cache/wvc-census/hub --group large-laptop --max-bytes 200e9 --min-free 100e9
# PACE, inside an embers CPU job (experiments/E009/pace-census.sbatch):
python experiments/E009/census.py --cache <private project root>/cache/wvc-census/hub --group large-pace --host pace --max-bytes 200e9 --min-free 100e9
uv run python experiments/E009/summarize.py experiments/E009/census.csv > experiments/E009/summary.md
```

## Run
Everything below was written after the run. Raw outputs: the selection walks (`selection/`, written by a `--select-only` call at 2026-11-04T03:2x UTC, API only, before any diff), 93 receipts (`pairs/`), `census.csv` (93 `diffed` rows and 77 `skipped`), the laptop logs (`run-laptop-small.log`, `run-laptop-large.log`, `run-laptop-resume.log`), the PACE job logs with the Slurm prolog and epilog lines kept and the user line removed (`pace-logs/`), and the generated [summary](E009/summary.md). `merge_pace.py` copied the PACE receipts and rows into this directory.

- Laptop, 2026-11-04 03:31 to 07:01 UTC (3 h 30 min), Apple M4 Max: 78 pairs on nine bases, 331.5 GB written into the census caches. The small group ran beside the 8B-9B group, each with its own cache (`~/.cache/wvc-census/hub` and `hub-large`). Both runs stopped on the disk floor (exit 3): free disk fell from 212 GB to 93 GB while the census caches held about 62 GB and two other sessions on the same machine (M8 and M9) were downloading models of their own; free disk was back at 208 GB at the end. After the stops, the finished bases were deleted from the caches and the rest ran one base at a time with `--purge-bases` (`run-laptop-resume.log`). The caches are empty and removed.
- PACE, account `paceship-srdsgt26`, `embers` QOS, partition `cpu-small`, 8 CPUs; no charged queue. Four submissions: 15047829 failed after 17 s before any pair (the job script changed into the login directory; fixed); 15048019 diffed four `gemma-2-9b-it` pairs and was preempted after 2 h 08 min during the fifth; 15057024 diffed the ten `gemma-3-1b-it` pairs in 17 min 33 s; 15058676 resumed, diffed `OpenLLM-Ro/RoGemma2-9b-Instruct` (3,361 s, most of it in the SVDs; 37.6 GB peak memory reported by Slurm) and was preempted after 1 h 52 min during the sixth pair. 87.6 GB are recorded in rows; the two pairs whose downloads were cut by preemption fetched about 18.5 GB each that no row records (the resumed RoGemma2 pair found its upload already cached and records 0 bytes), so PACE moved about 125 GB. The PACE cache (37 GB at the end) was deleted.
- Network in all: 419.1 GB recorded, about 456 GB with the two unrecorded PACE downloads.
- Software: `src/` was unchanged from `5e22349` for every receipt; the commits during the run touched `experiments/` only, and `wvc_source_commit` in each receipt is the checkout's HEAD at the time.

Deviations from the frozen plan:
- `google/gemma-3-1b-it`, planned for the laptop, ran on PACE (Gemma family, inside the allocation's approved route) after the laptop's small run had stopped on the disk floor.
- The 8B-9B laptop group used a second cache directory so the two laptop runs could not delete each other's files.
- PACE used four submissions where the plan allowed three. The third preemption risk was not taken: `shenzhi-wang/Gemma-2-9B-Chinese-Chat` (rank 6 for `gemma-2-9b-it`, 18.5 GB) is the one eligible pair not diffed.
- The CSV's rank columns count only tensors whose dtype did not change; a tensor whose dtype and values both changed has its rank in the receipt but not in the CSV. The post-hoc tables of the summary read the receipts for that reason.

## Result
93 pairs diffed on 11 bases, 1 eligible pair not run, 77 candidates passed over (75 `no-safetensors`, 2 `adapter`), 0 refusals, 0 crashes, and no internal inconsistency in any of the 56,730 tensor records (section "Receipt consistency" of the [summary](E009/summary.md)). Frozen categories over the 93 diffed pairs: byte-identical 9, every tensor changed 0, low rank on a few projections 2, renamed layout 51, dtype rewrite 8, partial change 23. By tag: 43 `finetune`, 46 `quantized`, 4 `merge`.

| Prediction | Held | Observed |
|---|---|---|
| P1 byte-identical 5% to 20% | yes | 9 of 93 (10%); all nine are `unsloth/<base>` mirrors, all tagged `finetune` |
| P2 every tensor changed 35% to 65% | no | 0 of 93. The dense fine-tunes leave a few tensors untouched (norms, biases, a tied head) or drop the vision tower, so none qualified; 13 uploads change at least half their elements with full-rank deltas, filed as "partial change" |
| P3 low rank among `finetune` 5% to 20% | no | 1 of 43 (2%) by the frozen rule. Post hoc, six `finetune` uploads change values only at rank 16 to 32 in every measured tensor: the four `JackKozmo29/*-jev-style-1.5b` (rank 16 on all 196 projections, saved in F32, so filed as dtype rewrites) and `THGLab/Llama-3.2-1B-Instruct-LinkLlama-Cap50` (rank 32 on all 112 projections, more than half the tensors), besides the frozen hit; 6 of 43 is 14% |
| P4 renamed layout 10% to 30%, mostly Qwen3.5 | no | 51 of 93 (55%), 9 of them on Qwen3.5 bases. Most are quantizations that store packed weights and scales under new names (bitsandbytes, MLX, AWQ, NVFP4, FP8 scales); a few are pure renames (`vllm-sr/d3-*`, `vultr/...` drop the `model.` prefix) or multimodal wrappers (`OpenGVLab/InternVL2_5-1B`, `nvidia/Eagle2-2B`, `nvidia/canary-qwen-2.5b`) |
| P5 `quantized` are dtype rewrites or renamed, at least 80% | yes | 39 of 46 (85%) |
| P6 `no-safetensors` at least 30% of small-base candidates | yes | 58 of 130 (45%); 75 of 171 over all bases, nearly all GGUF |
| P7 at least one crash (40%); refusals 5% to 15% | no | 0 crashes and 0 refusals in 93 pairs, across BF16, F32, FP8, NVFP4, bitsandbytes 4-bit, MLX 4- and 8-bit, AWQ, pruned and multimodal uploads |
| P8 median elements changed at least 0.9 in every-tensor-changed pairs | untestable | no such pairs. Post hoc, over the 13 dense "partial change" fine-tunes the per-pair values run from 0.50 to 0.99 with median 0.95 (summary, last table) |
| P9 at least two pairs contradict their tag | yes | 11: nine byte-identical mirrors tagged `finetune`, `saidutta69/Qwen2.5-0.5B-Instruct-heretic` tagged `quantized` but a low-rank edit in BF16, and `DiTy/gemma-2-9b-it-russian-function-calling-GGUF` tagged `quantized` but a BF16 fine-tune |

## Interpretation
- The most downloaded derivatives of these bases are mostly packaging, not training: 46 of 93 are quantizations and 9 are byte-identical mirrors. The `finetune` tag does not separate the two: a fifth of the `finetune`-tagged uploads diffed (9 of 43) are mirrors.
- Two uploads named as abliterations ("heretic"), and both are visible in the tensors alone. `DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated` changes 29 of 340 tensors (17 `down_proj`, 12 `o_proj`), each at rank 1. `saidutta69/Qwen2.5-0.5B-Instruct-heretic` changes 28 of 290 (`o_proj` and `down_proj` in layers 10 to 23) at rank 1 to 16, every one with top-1 energy at least 0.987. Its tag says `quantized`; nothing in it is quantized.
- Merged LoRAs show up as one rank on every projection: rank 16 on 196 tensors for the four JackKozmo29 uploads (saved as F32), rank 32 on 112 tensors for LinkLlama. A classifier that looks at dtype or at the share of tensors changed would miss them; the rank finds them.
- Dense fine-tunes (Qwen3Guard, Skywork reward models, jina rerankers, OvisOCR, Holo, RoGemma2) move most elements of most matrices with full-rank deltas: median rank 1024 on the 0.6B to 0.8B bases (the smaller dimension), 3,100 of 4,096 on Qwen3Guard-Gen-8B.
- AWQ uploads change the norm weights in front of the quantized projections: 72 norms in `Qwen/Qwen3-8B-AWQ`, 64 norms plus 48 small `in_proj_a`/`in_proj_b` matrices in `cyankiwi/Qwen3.5-9B-AWQ-4bit`, 85 norms plus the embedding in `hugging-quants/gemma-2-9b-it-AWQ-INT4`, with at most 0.26% of compared elements changed. That is the scale folding AWQ does, a quantization signature that would read as a small fine-tune without the rest of the diff.
- wvc matches tensors by name. A pure rename (the `model.` prefix dropped) and a quantization's packed weights both read as hundreds of tensors added and removed, so "renamed layout" mixes repackaging that preserves values with repackaging that changes them. The diff cannot tell which, because the JSON carries no tensor ids for added and removed tensors.
- Not shown: anything about behaviour (no model was run), uploads outside the top of each 30-day download list, adapters, and GGUF-only uploads, which are 44% of the candidates walked.

## Limitations
- One snapshot of a moving list: downloads over 30 days at 2026-11-04 03:2x UTC; the selection files pin each repository's commit.
- The frozen categories put a dtype change ahead of the value change it carries (the JackKozmo29 merges) and count "every tensor changed" too strictly to catch any dense fine-tune; the post-hoc tables in the summary are labelled as such.
- Relations are the Hub's tags; 8 of the 93 diffed cards set `base_model_relation` themselves (recorded per pair).
- Bytes are what was written into the caches, an approximation of network traffic; header range requests are not counted, and two preempted PACE downloads are estimated, not measured.

## Bugs found
None. No pair crashed wvc, none was refused, and the consistency checks over every receipt found nothing, so no test was added under `tests/`. P7's crash was not observed.

## Follow-ups
- Report renames: give added and removed tensors their ids in `wvc diff --json` (or pair equal ids under different names), so a pure rename reads as one and a quantization's packed weights do not. Without it, 51 of 93 pairs are opaque beyond counts.
- Count the rank of every same-shape value change in the CSV whatever the dtype (a change to `census.py` for the next census, which gets its own page).
- Write a "started" row before each pair, so a preempted or killed pair leaves a trace and its download is counted.
- GGUF-only uploads are 75 of 171 candidates walked: the census needs M8's GGUF reading to cover the top of the download lists.
- `base_model_relation: edit` evidence: two abliterations tagged `finetune` and `quantized`, four merged LoRAs tagged `finetune`.

## Weekly procedure
1. Copy this page's Question, Selection, Definitions, and Budgets into a new page (E0NN, the next free number) with fresh predictions, and a fresh output directory (`--out experiments/E0NN`); receipts are never relabelled.
2. `uv sync --locked --extra hf`, then `uv run python experiments/E009/census.py --out experiments/E0NN --cache ~/.cache/wvc-census/hub --select-only --max-bytes 1 --min-free 1` to freeze the walks (about 20 s).
3. Laptop, one base at a time so finished bases leave the disk: `for base in <bases>; do uv run python experiments/E009/census.py --out experiments/E0NN --cache ~/.cache/wvc-census/hub --base "$base" --max-bytes 120e9 --min-free 100e9 --purge-bases; done`. In this run the small bases took 2 to 6 min a pair at about 25 MB/s, and an 8B-9B pair 4 to 33 min (FP8 and dense fine-tunes spend most of it in the SVDs).
4. PACE, Gemma-family bases only: sync with the PACE helper (exclude only `experiments/E00[1-8]*`), copy the base's selection file to the run directory, then submit `experiments/E009/pace-census.sbatch` on `embers` with 16 CPUs and 64 GB, arguments `OUT CACHE COMMIT --base <base>`; expect preemption after the first hour and resubmit (the script resumes). Fetch `pairs/` and `census.csv` and run `merge_pace.py`.
5. `uv run python experiments/E009/summarize.py experiments/E0NN/census.csv > experiments/E0NN/summary.md`, then the outcome; the lab's own notes are updated separately.
Budgets as frozen above; this run used 3 h 30 min of laptop time, about 456 GB of network, at most about 62 GB of cache at once, and four embers jobs (4 h 18 min of wall time in all).

## Addendum 2026-11-04: the renamed pairs, diffed again with rename pairing
Written before the re-diff; the result follows below it. wvc on `main` at `7d152d9` pairs tensors present under one name only across a rename (spec section 2, "Renames"): first by id (same `sha256`, dtype, and shape: `unchanged`), then by a leading dotted prefix in either direction (same dtype and shape, compared element by element), and only when the pairing is unambiguous. A record that pairs gains `renamed_to` per paired tensor and `summary.renamed`, and `summarize.py` then splits "renamed layout" into "renamed, unchanged" (renamed tensors, and no tensor changed or reshaped) and "renamed, changed"; whatever stays unpaired keeps "renamed layout".

- Pairs: the 31 pairs on 9 bases whose receipts have tensors both added and removed, as `summarize.py --rediff` lists them, saved with their pinned commits in [`rediff-2026-11-04/pairs.csv`](E009/rediff-2026-11-04/pairs.csv). Each is diffed again at the same two commits, taken from the saved selections; nothing is reselected. 29 run on the laptop; the two `google/gemma-2-9b-it` pairs run on PACE, as in the first run.
- Receipts: the first receipts stay where they are. The new ones go to `rediff-2026-11-04/pairs/` with their own `rediff-2026-11-04/census.csv` (same columns), written by `census.py --pairs rediff-2026-11-04/pairs.csv --out experiments/E009/rediff-2026-11-04`; each new receipt's `census` block adds `rerun_of`, the path of the receipt it reruns, and is otherwise the frozen format. `summarize.py census.csv rediff-2026-11-04/census.csv` reads the second CSV after the first, so a re-diffed pair's new row and receipt supersede its old ones; with the first CSV alone it prints the summary as committed at `70e229a`, byte for byte.
- Budgets: laptop at most 200 GB for this re-diff (estimated 164 GB from the first run's bases and upload sizes); PACE at most 60 GB per job (estimated 45 GB) and at most two `embers` jobs, the second only to resume after a preemption; the 100 GB free-disk floor and 2 h per pair as before. A base leaves the cache once its last listed pair is done.
- Predictions:
  - R1: 1 to 3 of the 31 become "renamed, unchanged"; the likeliest is `nvidia/canary-qwen-2.5b`, if its language model is the base's weights under a new prefix with adapters beside them.
  - R2: 7 to 11 become "renamed, changed": the multimodal wrappers and prefix-dropping fine-tunes (`OpenGVLab/InternVL2_5-1B`, `nvidia/Eagle2-2B`, `k2-fsa/OmniVoice`, `vllm-sr/d3-edge`, `vllm-sr/d3-lite`, `vultr/VultronRetrieverFlash-Qwen3.5-0.8B`, `autotrust/JEV-9B`, `MERaLiON/MERaLiON-2-10B-ASR`).
  - R3: none of the 17 pairs tagged `quantized` becomes "renamed, unchanged": packed weights differ in dtype or shape and never pair, so they keep their first-run category.
  - R4: no crash, no refusal, and no internal inconsistency in the new receipts.

### Result of the re-diff, 2026-11-04 (all 31 pairs)
- Laptop, 29 pairs: 15:54 to 17:14 UTC at `43fb5e4` ([log](E009/rediff-2026-11-04/run-laptop.log)). 0 failed, 162.5 GB written to the cache (budget 200 GB), 207 GB free at the start and 209 GB at the end.
- PACE, the two `google/gemma-2-9b-it` pairs: job 15095602 (`embers`, `cpu-small`, 8 CPUs, 64 GB, `paceship-srdsgt26`) at the same commit, 15:57 to 17:00 UTC, 1 h 04 min of wall time ([log](E009/pace-logs/slurm-wvc-census-e009-rediff-15095602.log)). 0 failed, 44.9 GB written (budget 60 GB), one job within the limit of two.
- PACE was unreachable from the laptop from shortly before 17:14 UTC until 17:18 UTC, most likely because the Georgia Tech VPN was disconnected. The job ran through it, and its receipts were fetched and merged with `merge_pace.py` afterwards.
- The [summary](E009/summary.md) is regenerated with `rediff-2026-11-04/census.csv` read after `census.csv`. Its consistency check covers all 124 receipts, old and new, and finds nothing.

The 31 receipts are under `rediff-2026-11-04/pairs/`. Pairing split them as follows: 5 renamed, unchanged; 6 renamed, changed; 14 still unpaired (13 quantizations, and `k2-fsa/OmniVoice`'s substitution rename); and 6 that were not renamed layouts and keep their category. By first-run category:

| First run | n | After pairing |
|---|---|---|
| renamed layout | 25 | renamed, changed 6; renamed, unchanged 5; still renamed layout 14 |
| partial change | 5 | unchanged (a `score`, `projector`, or untied `lm_head` added; `lm_head` or the 15-tensor MTP head removed; nothing of equal dtype and shape to pair) |
| dtype rewrite | 1 | unchanged (`livekit/turn-detector`: nothing pairs) |

**Renamed, changed (6).** The base's language model sits under a new prefix and every paired tensor changed:
- `OpenGVLab/InternVL2_5-1B`: 289 tensors under `language_model.`.
- `nvidia/Eagle2-2B`: 337 tensors under `language_model.`.
- `MERaLiON/MERaLiON-2-10B-ASR`: 464 tensors (the whole of gemma-2-9b-it), with a speech encoder added (497 tensors).

The three Qwen3.5-0.8B derivatives that drop the `model.` prefix also count here, though only some of their paired tensors changed: `vllm-sr/d3-edge` 98 of 250, `vllm-sr/d3-lite` 269 of 437, and `vultr/VultronRetrieverFlash-Qwen3.5-0.8B` 186 of 437. Three kinds of tensor stay unpaired in them, and no pass pairs any of them:
- the MTP head they drop;
- 223 language-model tensors that `d3-edge` removes, spread over 23 layer indices;
- tensors whose dtype changed along with the name (8 in `d3-edge`, `A_log` F32 to BF16).

**Renamed, unchanged (5).** This holds two kinds of upload:
- Repackagings that keep the base's weights byte for byte:
  - `nvidia/canary-qwen-2.5b`: all 310 Qwen3-1.7B tensors are identical under `llm.`, with unmerged LoRA tensors and a speech encoder added (1,408 tensors).
  - `autotrust/JEV-9B`: all 426 language-model tensors of Qwen3.5-9B are identical. The vision tower and MTP head (348 tensors) are dropped and a 24-output `proj` head is added. It is tagged `finetune` though the base's weights are untouched.
- Three MLX quantizations: `mlx-community/Qwen3.5-0.8B-MLX-4bit`, and `lmstudio-community/Qwen3.5-9B-MLX-4bit` and `-8bit`. Their unquantized tensors (norms, 188 and 380) pair by id, while every packed weight stays added and removed (659 and 880 added, 300 and 395 removed).

The split's rule calls a pair "renamed, unchanged" when no compared tensor changed, whatever stays unpaired. So it files these quantizations alongside the repackagings. Read that column with the unpaired counts beside it, in the summary's section "Renamed tensors paired by the diff".

**Still renamed layout (14).**
- Thirteen quantizations whose packed weights differ in dtype and shape from the base: bitsandbytes, AWQ (including `hugging-quants/gemma-2-9b-it-AWQ-INT4`), AutoRound, and MLX on Qwen3. The Qwen3 MLX exports keep the base's names, so only `scales` and `biases` are added and nothing pairs.
- `k2-fsa/OmniVoice`, which renames `model.layers` to `llm.layers`. That is a substitution, not a prefix added or dropped, so no pass pairs it.

Predictions R1 to R4:

| Prediction | Held | Observed |
|---|---|---|
| R1: 1 to 3 renamed, unchanged | no | 5. Two are repackagings of the base's weights (`canary-qwen-2.5b`, as predicted, and `JEV-9B`); three are MLX quantizations counted there by the rule above |
| R2: 7 to 11 renamed, changed | no | 6: InternVL, Eagle2, MERaLiON, d3-edge, d3-lite, and Vultron, as named. Of the other two named, OmniVoice stays unpaired and JEV-9B is unchanged |
| R3: no `quantized` pair becomes renamed, unchanged | no | 3, the MLX exports of Qwen3.5. Packed weights never pair, as predicted; but their unquantized tensors do, which the prediction did not foresee |
| R4: no crash, refusal, or inconsistency | yes | 0 crashes, 0 refusals; the consistency check is clean over 124 receipts |

What this does to the first run's predictions:
- **P4** (renamed layout 10% to 30%, mostly Qwen3.5) still fails. With pairing, 40 of 93 pairs stay renamed layout (43%), only 2 of them on Qwen3.5 bases, and 11 resolve: 6 renamed and changed, 5 renamed and unchanged. The frozen category held three things: repackaging (the 2 pairs whose weights are the base's), renamed fine-tunes (6), and quantizations (the rest). Pairing separates the first two from the third, but it does not shrink the category into the predicted range, because most of it was quantization, not renaming, as the first outcome said.
- **P5** (`quantized` pairs are dtype rewrites or renamed, at least 80%) is unchanged at 39 of 46 (85%), counting the three MLX pairs now in "renamed, unchanged" as renamed.
- **P9** (tag contradictions): post hoc, `autotrust/JEV-9B` is a twelfth. It is tagged `finetune`, yet its language model is the base's byte for byte.

Follow-ups from the re-diff:
- A pass that pairs a substituted prefix (`model.` for `llm.`, `model.language_model.` for `language_model.model.`) would pair OmniVoice and the layer names of the MLX exports.
- The split's "renamed, unchanged" should require that nothing of the language model stays unpaired, or carry the unpaired count in the category, so that quantizations do not land beside repackagings.
