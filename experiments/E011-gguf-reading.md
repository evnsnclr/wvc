# E011: does wvc read a public GGUF exactly, and does a diff against its safetensors base show only the quantization?

Date 2026-11-03 (the run's UTC clock read 2026-11-04T03:55Z). Status: analysed. Delegated run (an Opus session working on milestone M8 under the lead session's brief); the Question, Inputs, Predictions, and Budget sections are written before the formal run and are not edited afterwards. Software: this repository's working tree on branch `claude/gguf-4dd688` (based on `5e22349`), Python 3.13.5, numpy 2.5.4, gguf 0.19.0 (dev group only), llama.cpp `149f5641b931aaf33697d70f9abc90c3e5300247` built locally (`llama-gguf-hash`, `libggml-base`), macOS 15.6, Apple M4 Max. Raw outputs in `experiments/E011/`.

## Question
M8 makes `wvc hash`, `diff`, and `verify` read GGUF files. On public files from well-known quantizers: (1) does wvc's reader agree with llama.cpp's on every tensor, and do `wvc hash --format gguf-hash` lines equal `llama-gguf-hash --sha256`'s? (2) Is wvc's dequantisation bit-identical to llama.cpp's compiled C reference? (3) Does `wvc diff` of a Q4_K_M upload against its safetensors base pair every tensor by name, undo the llama q/k permutation, and show a quantization (every quantized tensor changed with an error of the size its type implies, nothing else changed)?

## Inputs (Hugging Face cache, pinned revisions; 2.12 GB in all)
- `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`, `model.safetensors` (988,097,824 bytes; 290 BF16 tensors, embeddings tied, no `lm_head.weight`).
- `Qwen/Qwen2.5-0.5B-Instruct-GGUF@9217f5db79a29953eb74d5343926648285ec7e67`, `qwen2.5-0.5b-instruct-q4_k_m.gguf` (491,400,032 bytes; architecture `qwen2`; 291 tensors: 133 Q5_0, 121 F32, 13 Q8_0, 12 Q6_K, 12 Q4_K).
- `HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac`, `model.safetensors` (269,060,552 bytes; 272 BF16 tensors, tied).
- `bartowski/SmolLM2-135M-Instruct-GGUF@09816acd5d99df7be770d85ea30822623dab342c`: `SmolLM2-135M-Instruct-Q4_K_M.gguf` (105,454,432 bytes; architecture `llama`; 272 tensors: 166 Q5_0, 61 F32, 16 Q4_K, 15 Q8_0, 14 Q6_K) and `SmolLM2-135M-Instruct-f16.gguf` (270,885,952 bytes; 211 F16, 61 F32).

## Already seen during development (not predictions)
While building the reader, before this page existed, these were run on the SmolLM2 files and the dequantisers: wvc's parsed tensors (names, types, shapes, offsets, order) equal gguf-py's `GGUFReader` on all three GGUF files; `wvc hash --format gguf-hash` is byte-identical to `llama-gguf-hash --sha256` on both SmolLM2 files; `wvc diff` of the SmolLM2 base against the f16 GGUF reported 207 changed and 65 unchanged tensors with 3,597 of 134,515,008 elements changed (q and k projections 1 to 11 elements each, max |d| about 3e-8); and all nine block types were bit-identical to the C reference. The formal run repeats each of them into the receipt; only the items below are predictions.

## Predictions (before the run)
- P1 (Qwen Q4_K_M): 290 tensors pair by name under the `qwen2` table; `lm_head.weight` (the GGUF's `output.weight`) is added, since the base ties it; nothing is removed or reshaped (80%: a converter could have dropped or padded a tensor).
- P2: every one of the 170 quantized tensors is `changed`, and every one of the 121 F32 tensors (norms and biases) is `unchanged` by value, since BF16 widens to F32 exactly (90%).
- P3: relative Frobenius error per quantized matrix falls in these bands, by type: Q8_0 below 0.01, Q6_K 0.01 to 0.04, Q5_0 0.02 to 0.07, Q4_K 0.04 to 0.13 (70% that every tensor falls in its band).
- P4: in every quantized matrix more than 90% of elements change.
- P5: the quantization error is not low rank: where the rank is computed, top-1 energy is below 0.05 for every matrix (75%).
- P6 (SmolLM2 Q4_K_M, `llama`): the q and k projections, after the permutation is undone, have a relative error within a factor of 1.5 of the v projection of the same layer when the two share a type; had the permutation not been undone, their relative error would be near sqrt(2) times their norm, about 1.4.
- P7: `wvc hash --format gguf-hash` on the Qwen file is byte-identical to `llama-gguf-hash --sha256` (95%).
- P8: each diff takes under three minutes on the M4 Max, with the SVD.

## Budget and stop rules
Download at most 3 GB, into the Hugging Face cache only (2.12 GB used; 215 GB free). No model is run. Nothing is pushed anywhere. Committed: the runner, the C harness and its comparison script, and their text outputs; the weights stay in the cache.

## Commands
```bash
# llama.cpp at 149f5641, built once (cmake -B build -DGGML_METAL=OFF; cmake --build build --target llama-gguf-hash)
clang -O2 -I LLAMA/ggml/include experiments/E011/qref.c -L LLAMA/build/bin -lggml-base -Wl,-rpath,LLAMA/build/bin -o QREF
sh experiments/E011/run.sh LLAMA/build/bin QREF        # every step below, logged to run.log
uv run python experiments/E011/summarize.py experiments/E011/qwen-q4_k_m.diff.json > experiments/E011/summary-qwen-q4_k_m.txt
uv run python experiments/E011/summarize.py experiments/E011/smollm2-q4_k_m.diff.json > experiments/E011/summary-smollm2-q4_k_m.txt
# unplanned, after the Qwen result (see Deviation):
uv run python experiments/E011/channel_scale.py QWEN/model.safetensors QWEN_GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf > experiments/E011/channel-scale-qwen.txt
uv run python experiments/E011/channel_scale.py SMOL/model.safetensors SMOL_GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf > experiments/E011/channel-scale-smollm2.txt
```
`run.sh` runs `compare_reader.py` on the three GGUF files, `wvc hash --format gguf-hash` and `llama-gguf-hash --sha256` (in the file's directory) on each and on the spec vector `quant.gguf`, `verify_dequant_c.py` against the C harness, and `wvc diff --rank` (text) and `wvc diff --rank --json --all` for each pair (`--rank` omitted for the f16 pair). Every step's stderr was empty. Exit status 1 on every diff is the "differences found" code.

## Result
| Item | Value |
|---|---|
| reader | all three files: every tensor's name, type, shape, offset, and size equal to gguf-py's `GGUFReader`, in the same order (`reader.txt`); headers parsed in 0.05 to 0.17 s |
| gguf-hash | byte-identical to `llama-gguf-hash --sha256` for all three files (292, 273, and 273 lines) and for the spec vector `quant.gguf` (`run.log`, `*.wvc-gguf-hash`, `*.llama-gguf-hash`) |
| dequantisation | all nine block types bit-identical to ggml's C reference, on 4,096 random blocks with finite scales and on the blocks ggml's reference quantiser makes from 4,096 blocks' worth of normals: 18 of 18 comparisons, 0 failures (`dequant-c.txt`) |
| SmolLM2 Q4_K_M (`llama`) | 272 tensors: 211 changed (every quantized tensor), 61 unchanged (every F32 norm, by value), none added, removed, or reshaped; relative error Q8_0 0.0055 to 0.0070, Q6_K 0.0177 to 0.0188, Q5_0 0.0410 to 0.0539, Q4_K 0.0721 to 0.0871; changed fraction 0.996 to 1.000; top-1 energy at most 0.069 (`summary-smollm2-q4_k_m.txt`); 10 s with the SVD |
| SmolLM2 f16 (`llama`) | 207 changed, 65 unchanged; 3,597 of 134,515,008 elements changed, each by at most 2.98e-8 (BF16 values below F16's normal range); q and k projections 1 to 11 elements each (`smollm2-f16.diff.txt`); 1.3 s |
| Qwen Q4_K_M (`qwen2`) | 291 tensors: 218 changed, 72 unchanged, 1 added (`lm_head.weight`, the file's `output.weight`, Q8_0), none removed or reshaped; 487,633,216 of 494,032,768 elements changed; 30 s with the SVD. By type and projection below (`summary-qwen-q4_k_m.txt`) |

Qwen Q4_K_M by tensor group (relative Frobenius error, min to max over layers):
| Group | Type | Status | Relative error |
|---|---|---|---|
| `embed_tokens` | Q5_0 | changed | 0.0440 |
| `q_proj`, `k_proj`, `v_proj` weights | Q5_0, Q8_0 (v, some layers) | changed | 0.219 to 0.594 |
| `o_proj` | Q5_0 | changed | 0.0546 to 0.100 |
| `gate_proj` | Q5_0 | changed | 0.125 to 0.797 |
| `up_proj` | Q5_0 | changed | 0.138 to 6.15 |
| `down_proj` | Q4_K, Q6_K | changed | 0.206 to 0.785 |
| `input_layernorm`, `post_attention_layernorm` (F32) | F32 | changed (48 of 48) | 0.0915 to 1.05; max abs delta 6.86 |
| q, k, v biases (F32) | F32 | unchanged, except one `v_proj.bias` with one element at 2.98e-8 | 0 |
| `model.norm` (F32) | F32 | unchanged | 0 |

Predictions:
| Prediction | Held | Observed |
|---|---|---|
| P1 | yes | 290 paired, `lm_head.weight` added, 0 removed, 0 reshaped |
| P2 | no | all 170 quantized tensors changed, but only 72 of 121 F32 tensors unchanged: all 48 RMS-norm weights of the layers changed, by up to 6.86 (one bias also differs by one element at 2.98e-8) |
| P3 | no (Qwen); yes (SmolLM2) | Qwen: 169 of 170 quantized tensors above their band (all but `embed_tokens`, 0.044, inside Q5_0's); SmolLM2: every tensor inside its band |
| P4 | yes | at least 95.8% of elements changed in every quantized matrix (Qwen), 99.6% (SmolLM2) |
| P5 | no | top-1 energy above 0.05 for 29 Qwen matrices (up to 0.352, `layers.7.mlp.gate_proj`) and for 3 SmolLM2 matrices (up to 0.069); the rank was skipped for Qwen's embedding (136 M elements, over the cap) |
| P6 | yes | SmolLM2: wherever q, k, and v share Q5_0, their errors are within 0.041 to 0.051 of one another; q and k at 0.043 to 0.051, nowhere near 1.4 |
| P7 | yes | identical, 292 lines |
| P8 | yes | 30 s (Qwen), 10 s (SmolLM2), 1.3 s (f16) |

Deviation from the plan: after P2 and P3 failed for Qwen, one unplanned check was written and run, `channel_scale.py`. It tests whether the Qwen file's changes are a per-channel scale folded across each RMS norm (norm weight divided by `s_j`, the columns `j` of the projections it feeds multiplied by `s_j`). With `s_j` taken from the two norms, the GGUF's column norms of `q_proj`, `k_proj`, `v_proj`, and `gate_proj` match the base's times `s_j` to a median `|log(ratio / s_j)|` of 0.0008 to 0.016 per layer and projection (99th percentile at most 0.12), while `|log s_j|` itself has a median of 0.065 to 0.77 per layer and norm (`channel-scale-qwen.txt`). `up_proj` misses by a nearly constant amount within each layer (median 0.20 to 1.52, 99th percentile within 0.025 of it). The same check on SmolLM2, where nothing was rescaled, gives `s_j` = 1 exactly (`channel-scale-smollm2.txt`).

## Interpretation
- The reader, the gguf-hash export, and the dequantisers do what M8 needs on public files: identity agrees with llama.cpp's own tool byte for byte, and dequantisation is bit-exact against the C reference for all nine block types.
- The SmolLM2 pair is what a plain quantization looks like in a wvc diff: every quantized tensor changed with an error set by its type (about 0.006 for Q8_0, 0.018 Q6_K, 0.045 Q5_0, 0.08 Q4_K), every norm unchanged, the q and k projections no different from v once the permutation is undone, and the f16 file differing only where F16 cannot hold a BF16 value.
- The official Qwen Q4_K_M is not only a quantization of `Qwen2.5-0.5B-Instruct` as published: its layer norms and the projections they feed carry a per-channel rescaling that the check above recovers channel by channel. A rescaling of this form leaves the network's function unchanged in exact arithmetic (it moves a scale from the norm into the next matrix), and activation-aware quantization methods apply such scales before rounding; that this file was made so is a hypothesis, since its metadata names no method and its card was not read for this receipt. `up_proj`'s constant misfit is consistent with a second scale on its rows, moved into `down_proj`'s columns, which the check does not test. No model was run, so nothing here says whether the two functions agree.
- For the classifier (M9) and for agents reading diffs: a quantization comparison can show large, structured, partly low-rank changes (top-1 energy up to 0.35) that are a reparametrisation, not an edit. Error size per type separates a plain quantization (SmolLM2) from this one; it would not separate a reparametrisation from a fine-tune without a check like `channel_scale.py`.

## Limitations
- Two upload pairs, both small models; no Q2_K, Q3_K, or IQ file was tried (wvc refuses their values by name).
- The permutation was exercised on one `llama` file (SmolLM2, 9 heads and 3 key heads); `gemma3` and `phi3` mappings are tested only on synthetic files.
- The C comparison used one build (Apple clang, arm64, `-O2`) of ggml's reference; a build that fuses multiply-adds could differ in the last bit, which wvc does not model.
- Inputs live in the Hugging Face cache, not in Git; re-running needs the four pinned revisions, 2.12 GB.
