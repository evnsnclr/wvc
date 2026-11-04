# E012: can wvc diff GGUF-only uploads on the Hub against their safetensors bases through `hf:` alone?

Date 2026-11-04 (the run's UTC clock read 2026-11-04T05:51Z to 06:07Z). Status: run; outcome below. The page was frozen in `d705751` before the run. Delegated run (an Opus session working on milestone M8 part 3 under the lead session's brief); the Question, Inputs, Predictions, Budget, and Commands sections are written before the run and are not edited afterwards. Software: this repository's working tree on branch `claude/gguf-hub-0f9c85` at the commit that adds this page (code from `1d34259`), Python 3.13.5, `huggingface_hub` 2.4.0, macOS 15.6, Apple M4 Max. Raw outputs will be in `experiments/E012/`.

## Question
The census (E009) skips GGUF-only uploads, which are most of a popular base's `quantized` derivatives, because `hf:` read safetensors only. With `hf:OWNER/NAME@REVISION:FILE.gguf` (spec section 5.5): (1) does each case resolve, fetch, and diff from the argument alone, with no file handled by hand, including a split file named by one of its parts? (2) Does each diff against the census's safetensors base look like a plain quantization in E011's sense (every quantized tensor changed with an error of the size its type implies, every norm unchanged, no low-rank structure), or does it carry more, as E011's official Qwen file did? (3) Is the split file the same checkpoint as the single file it sits beside?

## Inputs (pinned revisions; sizes and LFS SHA-256 from the Hub's listing on 2026-11-04)
Bases (each a census base with safetensors-side receipts in E009):
- `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, `model.safetensors`, 1,503,300,328 bytes, sha256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`, `model.safetensors`, 988,097,824 bytes, sha256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` (E011's base).
- `meta-llama/Llama-3.2-1B-Instruct@9213176726f574b556790deb65791e0c5aa438b6`, `model.safetensors`, 2,471,645,608 bytes, sha256 `1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f` (gated; the session's token has access).

Uploads (GGUF only; none of these repositories holds a safetensors file):
- A, `bartowski/Qwen_Qwen3-0.6B-GGUF@60b85c0e3d8fe0f6474f406922a26d12aca4550d` (24 GGUF files), `Qwen_Qwen3-0.6B-Q4_K_M.gguf`, 484,220,320 bytes, sha256 `9acfc1e001311f34b4252001b626f2e466d592a42065f66571bff3790d4e1b14`.
- B, `Felladrin/gguf-Q5_K_M-Qwen2.5-0.5B-Instruct@69b6ffa31d2619f3a78b67ac661dec2b98d4e2fb`: the split `qwen2-00001-of-00003.gguf` (247,258,528 bytes, `b5a1fe298dc983d8f38c58bd685693a8edf4b9d9f6ae97a2ec15960d53705851`), `qwen2-00002-of-00003.gguf` (92,001,472 bytes, `b2d83acf25677d4431f4c4fe422be741306b6fa0bcdd08f43f627cd9bf7142e2`), `qwen2-00003-of-00003.gguf` (80,826,688 bytes, `f1c689dad857aa7a70956582298ff0c0403030a3daf582b77e1eaf788945d194`), and beside it the single file `qwen2.5-0.5b-instruct-q5_k_m-imat.gguf` (420,086,368 bytes, `52c6322f25bd12d9ec2bc61dbf3fc12a667b19bbe47abc9f42d2bc16b7a98b32`). It is the only split of a small census base found by a scan of the Hub (search on six census base names, `gguf` filter, up to 1,000 repositories each); the large uploaders (bartowski, unsloth, lmstudio-community, ggml-org, Qwen) split only files of 3 GB and more.
- C, `lmstudio-community/Llama-3.2-1B-Instruct-GGUF@199151125cf15a129ab3b548b26afeed976df066`, `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 807,690,688 bytes, sha256 `f7ede42862ceca07ad1c88a97b67520019c4ac7e5ced250d2e696fa62ab189af`.

Total listed: 7,095,127,824 bytes.

## Already seen (not predictions)
The listings above, the base tensor counts in E009's `census.csv` (Qwen3-0.6B 311 tensors through its `unsloth/Qwen3-0.6B` row), and E011's results. No GGUF file of this page was downloaded or opened before the run.

## Predictions (before the run)
- P1: every step resolves from its `hf:` argument alone: the bare `hf:` on A exits 2 and lists its 24 GGUF files with sizes, downloading nothing; every diff exits 1 (differences found) or, for the split against the single file, 0; no stderr line other than the `pinned:` and `file:` lines (90%).
- P2 (split): naming part 2 of B fetches all three parts, their `split.*` keys agree, and the diff of the split against the single file reports every tensor unchanged with equal manifest roots: the split is the single file cut by `gguf-split` (85%).
- P3 (pairing): A pairs every base tensor under the `qwen3` table except `lm_head.weight`, which is `removed` because the GGUF ties it to `token_embd` (60%); B pairs all 290 base tensors with nothing added (55%; E011's official Qwen file added `output.weight`); C pairs all 146 base tensors and adds one, `rope_freqs.weight`, which llama.cpp writes for Llama 3's rope scaling (85%). Nothing is reshaped in any case (95%).
- P4: every tensor type in the three uploads is one wvc dequantises (Q4_K, Q5_K, Q6_K, Q5_0, Q8_0, F32, ...), so no diff is refused (90%).
- P5: every F32 tensor (norms, biases, Qwen3's q and k norms) is `unchanged` by value in all three: these are plain quantizations, unlike E011's official Qwen file (60% for all three together).
- P6: relative Frobenius error per quantized matrix within these bands, over all three files: Q8_0 0.004 to 0.008, Q6_K 0.012 to 0.025, Q5_K 0.025 to 0.05, Q5_0 0.035 to 0.06, Q4_K 0.06 to 0.11 (70% that every tensor falls in its band).
- P7: top-1 energy at most 0.07 for every quantized matrix whose rank is computed (70%).
- P8 (C, `llama`): where q, k, and v of a layer share a type, the q and k errors are within a factor of 1.5 of v's: the permutation is undone (85%).
- P9: the agent views name each upload as `hf:OWNER/NAME@SHA:FILE` and give its file type: `Q4_K_M` for A and C, `Q5_K_M` for B (90%).
- P10: the bytes downloaded equal the listed total, 7,095,127,824 bytes, give or take the cache's own metadata (95%), and each diff with `--rank` takes under 3 minutes (90%).

## Budget and stop rules
Download at most 8 GB, into a cache directory of this run only (`HF_HUB_CACHE` set by `run.sh`), deleted after the run; the shared Hugging Face cache is not touched. No model is run. Nothing is pushed. If an `hf:` argument cannot resolve a case, the outcome says so and the case is not worked around by hand. Committed: `run.sh` and the text outputs; no weights.

## Commands
```bash
mkdir CACHE   # an empty directory outside the repository
sh experiments/E012/run.sh CACHE
uv run python experiments/E011/summarize.py experiments/E012/qwen3-q4_k_m.diff.json > experiments/E012/summary-qwen3-q4_k_m.txt
uv run python experiments/E011/summarize.py experiments/E012/qwen25-q5_k_m-split.diff.json > experiments/E012/summary-qwen25-q5_k_m-split.txt
uv run python experiments/E011/summarize.py experiments/E012/llama-q4_k_m.diff.json > experiments/E012/summary-llama-q4_k_m.txt
rm -r CACHE
```
`run.sh` runs, in order: `wvc hash` on A's bare repository (the listing), `wvc diff --rank --json --all` and `wvc diff --rank --format agent` for each base and upload, `wvc diff --json --all` of B's single file against B's split (named by part 1), and `wvc hash --format gguf-hash` on B's split (named by part 3). It logs each step's exit code and seconds to `run.log`, then the cache's size.

## Outcome
Every step ran from its `hf:` argument alone; no file was handled by hand (`run.log`). Every step's stderr held only its `pinned:` and `file:` lines, except the bare listing's, which is the listing and the refusal (`*.err`). Each fetched file's SHA-256 matched the listing. The cache held 7,095,128,875 bytes of blobs when the run ended: the seven listed weight files (7,095,127,824 bytes) and 1,051 bytes in small client metadata files; it was deleted after the run.

| Case | Tensors | Changed | Unchanged | Added, removed, reshaped | Relative error by type (min to max) | Top-1 energy max | Seconds (`--json`, first fetch) |
|---|---|---|---|---|---|---|---|
| A, bartowski Qwen3-0.6B `Q4_K_M` (`qwen3`) | 311 | 198 | 113 (every F32) | 0, 0, 0 | Q4_K 0.0713 to 0.0770; Q6_K 0.0177 to 0.0253 | 0.0301 | 161 |
| B, Felladrin Qwen2.5-0.5B-Instruct `Q5_K_M`, split in 3 (`qwen2`) | 290 | 169 | 121 (every F32) | 0, 0, 0 | Q8_0 0.0055 to 0.0082; Q6_K 0.0182 to 0.0200; Q5_K 0.0367 to 0.0396; Q5_1 0.0341 to 0.0446 | 0.0985 | 135 |
| C, lmstudio-community Llama-3.2-1B-Instruct `Q4_K_M` (`llama`) | 147 | 113 | 33 (every F32) | 1 (`rope_freqs.weight`, F32 [32]), 0, 0 | Q4_K 0.0717 to 0.0794; Q6_K 0.0180 to 0.0208 | 0.0239 | 209 |
| B's split against B's single file | 290 | 0 | 290 | 0, 0, 0 | none | none | 16 |

Sources: `summary-*.txt` and `*.diff.json`; the rank was skipped (over the cap) only for the embeddings and A's `lm_head.weight`; the lowest rank computed was 984 (A), 128 (B, full rank of a 128-row k projection), and 512 (C). The split and the single file have the same manifest root, `0059f26e96818acc...`, and B's three parts give 293 gguf-hash lines (290 tensors and 3 files, `split.gguf-hash`).

| Prediction | Held | Observed |
|---|---|---|
| P1 | yes | the bare `hf:` on A exited 2 and listed 24 files with sizes; every diff exited 1, the split against the single file 0 |
| P2 | yes | part 2 fetched parts 1 to 3; their split keys agreed; 290 of 290 unchanged against the single file, equal roots |
| P3 | partly | A: no, `lm_head.weight` is in the base and pairs with the GGUF's `output.weight` (Q6_K, changed), so nothing is removed; B: yes, 290 paired, nothing added; C: yes, 146 paired and `rope_freqs.weight` added; nothing reshaped anywhere |
| P4 | yes | no diff was refused; B holds 132 Q5_1 tensors, a type the page did not list, which wvc dequantises |
| P5 | yes | every F32 tensor unchanged by value in all three (113, 121, 33) |
| P6 | no | two tensors just outside their bands: A's `model.layers.2.mlp.down_proj.weight` (Q6_K, 0.0253 against 0.025) and B's `model.layers.16.self_attn.v_proj.weight` (Q8_0, 0.0082 against 0.008); every other tensor of a predicted type inside its band; Q5_1 had no band |
| P7 | no | one matrix above 0.07: B's `model.layers.0.self_attn.k_proj.weight` (Q5_1, top-1 0.0985 at full rank 128, relative error 0.0376); the next largest in B is 0.0463 |
| P8 | yes | C: where q, k, and v share Q4_K (layers 2, 3, 5, 6, 10, 11, 13, 14) the three errors lie within 0.0721 to 0.0794 of one another |
| P9 | partly | file types right in all three (`Q4_K_M`, `Q5_K_M`, `Q4_K_M`) and the split shown with its 3 parts; A's and B's references whole, but C's reference was cut at the view's 120-character limit (`...Llama-3.2-1B-Instruct-Q4_K_M.g…`), since a 40-character sha and a long file name shared one span |
| P10 | partly | bytes: yes, the listed total plus 1,051 bytes of client metadata; time: no, C's `--json` diff took 209 s with its 3.28 GB first fetch (the agent view of the same pair, with `--rank`, took 144 s and 146 s from the cache) |

Deviation from the plan: after P9's truncation, the agent view was changed to put the file in a span of its own (`hf:OWNER/NAME@SHA`, file `FILE`), a test was added for a 112-character file name, and the one affected view was run again from the same cache before it was deleted (`llama-q4_k_m.agent-after-fix.md`, `rerun.log`): its Pair section names the whole file, and every other line is identical to `llama-q4_k_m.agent.md`. The original view is kept as it was produced.

## Interpretation
- `hf:` resolves GGUF-only uploads of three uploaders, a split file named by any part, and a gated base, with the file-level SHA-256 checked; the census can name such uploads by file.
- On the readings this page set out (P5, and P6 to P8 within a few percent), these three files look like plain quantizations in E011's sense: every norm unchanged, error set by type, no low-rank change of note. Unlike E011's official Qwen2.5-0.5B file, the Felladrin quantization of the same base leaves all 121 F32 tensors unchanged.
- E011's bands, set on one small model, are slightly narrow at their upper ends for Q6_K and Q8_0 on these files; a classifier should not treat a few percent over them as a sign of an edit.

## Limitations
- Three uploads, all of small models in mapped architectures; Qwen3.5 and other architectures without a name table were not tried, and no upload of an unsupported type (Q2_K, Q3_K, IQ) was diffed (A's repository lists them; wvc refuses their values by name).
- The one split checkpoint is a small file cut on purpose by one uploader; the multi-gigabyte splits of the large uploaders were not fetched.
- Inputs are on the Hub at the pinned revisions, not in Git; re-running needs 7.10 GB.
