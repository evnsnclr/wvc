# Change classification (milestone M9): design

Status: rules version `v2` (2026-11-04). Rules `v1` were designed 2026-11-03 and frozen for [E010](../experiments/E010-change-classification.md); they stay below as written, and [Rules v2](#rules-v2-2026-11-04-after-e010) lists every change made after E010's results, each with the pair that prompted it. Implemented in `src/wvc/classify.py` (the rules, from a diff record) and `src/wvc/features.py` (the second pass that reads the pair). Validated in [E010](../experiments/E010-change-classification.md). The output format is in [spec section 2.3](spec-v0.md).

## Goal and limits
From a diff alone, say what kind of change turned checkpoint A into checkpoint B, name the evidence, and name the rule that fired. Rules only, no learned model: every answer can be explained in one sentence, and a threshold can be argued with. The classifier describes the shape of a change, not its intent or its effect. "direction-removal" means "a rank-1 change to the output projections along one direction shared across layers", which is what abliteration produces; it does not mean the model refuses less.

Two inputs are accepted:
- a diff JSON record (`wvc diff --rank --json`, best with `--all`): classification from the record alone, with no tensor read. This is what runs over census receipts.
- a pair of checkpoints (or a record plus the pair it names): the diff and a second pass, `--features`, that reads the changed tensors again and computes what the record cannot hold (below).

## Why a second pass: what bfloat16 rounding does to the record's rank
The record's `rank` counts singular values of the delta above 1e-2 of the largest. A merged change is stored as `round(W + Δ) - W`, so the stored delta is Δ plus a rounding error bounded by half a unit in the last place (ulp) of each element. For a bfloat16 matrix the error is about 0.2% of the weights, full rank, and it fills the spectrum. A synthetic check (896 x 4864 matrix of N(0, 0.02) weights in bfloat16; script in `experiments/E010/synthetic_noise.py`) gives:

| change | relative size | record rank (tol 1e-2) | record top-1 energy | rank above the noise edge |
|---|---|---|---|---|
| LoRA r = 8 | 0.010 | 808 | 0.14 | 8 |
| LoRA r = 16 | 0.002 | 896 | 0.056 | 16 |
| LoRA r = 64 | 0.010 | 896 | 0.025 | 64 |
| rank-1 removal, full strength | 0.034 | 1 | 0.998 | 1 |
| rank-1 removal, 0.3 strength | 0.010 | 9 | 0.979 | 9 |
| dense, power-law spectrum | 0.010 | 890 | 0.33 | 619 |

Output: `experiments/E010/synthetic_noise.txt`. So the record's rank is meaningful only for changes well above the rounding floor (E001 and E008 are such changes: relative change 0.08 to 0.1). For a typical LoRA merge it reads as full rank, and a rule based on it would call every such merge a dense fine-tune. The record's `top1_energy` stays informative for rank-1 changes. The second pass measures the noise directly.

### `--features` (per changed 2-D tensor of equal dtype and shape, or a widening cast)
- The delta D (B minus A, float64) is decoded in row blocks. A matrix up to the rank cap (2^26 elements) is kept whole. A larger one is measured only when its smaller side is the column count (an embedding or output head, `vocab x hidden`): then only `Dᵀ D` is kept. Otherwise it is skipped, as is a matrix whose smaller side exceeds 8,192. A skipped tensor falls back to the record's statistics in the rules.
- The singular values come from the eigenvalues of the Gram matrix of the smaller side (`D Dᵀ` or `Dᵀ D`), which also gives that side's singular vectors.
- Noise edge: each element's rounding error has variance at most `ulp(b)² / 12`, where `ulp(b)` is the spacing of B's dtype at the stored value. For a random matrix with independent entries of variance σᵢⱼ², the largest singular value is at most about `maxᵢ √(Σⱼ σᵢⱼ²) + maxⱼ √(Σᵢ σᵢⱼ²)` (Bandeira and van Handel, 2016, up to a logarithmic factor). That bound is `noise_edge`. In float32 and float64 it is negligible.
- `signal_rank`: the singular values above `max(noise_edge, 1e-2 · σ₁)`. The relative floor keeps the record's tolerance, because the rounding of a strong rank-1 change is itself structured and adds a few spurious components above the edge (rank 9 in the 0.3-strength row above).
- A matrix whose largest singular value does not exceed the noise edge has no measurable signal: it gets `signal_rank` 0 and `signal_top1` null, and the rules treat it as unmeasured and use its record statistics (fixed during the E010 run; the first version raised an error).
- `signal_top1`: σ₁² over the energy of the singular values counted in `signal_rank`, the share of the signal (not the noise) that lies in one direction. `energy_rank`: the fewest singular values that hold 95% of that signal energy.
- `snr`: σ₁ / `noise_edge` (null in float32 and wider).
- `sigma`: the 16 largest singular values, kept so the rules can be re-run on a stored record.
- `direction`: the unit singular vector on the side whose length is the hidden size (the dimension most 2-D tensors share). It is used for the shared-direction test and not written out. Each output-side tensor instead gets `direction_alignment`, |cos| between its vector and the consensus direction.
- Pair level: `direction_coherence`, the largest eigenvalue of `Σ uᵢuᵢᵀ / n` over the output-side tensors' unit vectors (1 when every layer uses the same direction, about 1/n for unrelated directions), with `n` and the hidden size.

The second pass reads only changed tensors. On a Hub pair it runs while the fetched shards are open, so it needs no second download.

## Vocabulary of the evidence
- Layer index: the integer after `layers.`, `layer.`, `h.`, `blocks.` or `block.` in a tensor name. The module is the rest of the name after that index (`self_attn.o_proj.weight`). Tensors outside a layer are global.
- Output-side projections: modules whose rows are the residual stream, named `o_proj`, `down_proj`, `out_proj`, `dense_4h_to_h`, `c_proj`, `fc2`, `w2`, `wo`, `attention.dense`, or `attn.dense`. Embeddings: global tensors whose name contains `embed`, `wte`, `wpe`, or `lm_head`. Vectors: 1-D tensors (norms and biases).
- Renamed: tensors the diff paired across a rename (spec section 2, `renamed_to`). They are read by their status like any other tensor, so a renamed tensor with identical bytes is unchanged for every rule, R1 and R2 included, and the evidence counts them as `renamed`.
- Layer count: with `--all`, the number of distinct layer indices in the record. Without it, one plus the largest index among the listed tensors, a lower bound (`layers_known: false`).
- Rank-1-like: record `top1_energy ≥ 0.75` or record `rank == 1`; with features, `signal_top1 ≥ 0.9`.
- Low rank (R5): with features, `signal_rank ≤ min(64, s/4)`, where s is the smaller side of the matrix. High rank (R6): the best estimate (features, else the record) above `min(64, s/2)`. The size-relative bound keeps a rank-8 change in a 32-row matrix from passing for low rank; for the projections of a 0.5B model and up (s ≥ 512) it is 64 in both cases.
- Localized: `rows_touched ≤ 25%` of the rows or `cols_touched ≤ 25%` of the columns (a 1-D tensor: `changed ≤ 25%` of its elements).
- Cast: `reshaped` with equal shapes and different dtypes. A widening cast goes to a dtype that holds every value of the old one (BF16 or F16 to F32 or F64, F32 to F64, FP8 to F16, BF16, F32, or F64; F8_E8M0 not to F16); every other cast is narrowing, including BF16 to F16 and F16 to BF16, since each lacks values of the other (range, or mantissa bits). A widening cast that changes values (a fine-tune saved in F32) is treated as a changed tensor; one that changes none is lossless.

## The kinds and their rules (v1)
Rules are tried in the order R1 to R6, then M2. The first that fires gives the kind. Each rule has structural conditions, which must all hold, and confirming evidence, which makes the tier `strong`. A rule whose structure holds but whose confirming evidence is missing (no rank estimate, no features) or contradicts it gives `weak`. Resized tensors (a shape change, usually a vocabulary resize) do not stop a rule: they are listed in the evidence and cap the tier at `weak`.

**R1 `reupload`.** Structure: no tensor is changed, cast with a value change, resized, added, or removed. Strong when every tensor is byte-identical (the reason says how many of them are under another name). Weak, rule R1b, when the only differences are widening casts that change no value (for example BF16 stored as F32) or at most two added or removed global tensors (a tied `lm_head` written out or dropped).

**R2 `quantization`.** Structure: some cast changes values, and casts (lossless ones and those the M8 field marks included, below) make up at least 90% of the tensors both sides hold. Strong when nothing else changed and every cast tensor's `rel_delta_norm ≤ 2u`, where `u = 2^-p` is the unit roundoff of B's dtype (p = 8 for BF16, 11 for F16, 24 for F32, 4 for F8_E4M3, 3 for F8_E5M2): the deviation is what rounding to the narrower type costs and no more. Weak, R2b, when at least half of B's tensor names are added with a quantization suffix (a last name part of `qweight`, `qzeros`, `scales`, `g_idx`, `weight_scale`, `weight_scale_inv`, `input_scale`, `absmax`, `quant_map`, `quant_state`, `SCB`, `weight_packed`, `weight_shape`, or `biases`, or one ending in `_` and one of these or `_scale`; or a name containing `quant_state` or `bitsandbytes`). This is a packed format whose values the diff cannot compare. Weak, R2c, when at least half of the casts go to an FP8 type and some added tensor has a quantization companion name (`weight_scale`, `weight_scale_inv`) (scaled FP8, whose decoded values are off by the scale). When the narrowing casts deviate by more than 2u, the kind is `mixed` (rule M1: quantization plus another change), with the largest deviation as evidence.

**R3 `direction-removal`.** Structure: the changed tensors are output-side projections and embeddings only, and no vector changed; output-side tensors changed in at least max(3, 25% of the layers) layers; every changed output-side tensor is rank-1-like. Strong, with features only, when `direction_coherence ≥ 0.5`: the layers remove a shared direction. The threshold sits halfway between one direction (1.0) and unrelated per-layer directions (about 1/n, below 0.1 for ten or more layers). Methods that interpolate the direction between layers (Heretic) are expected to land between 0.5 and 1. Without features the tier is weak. With features and `direction_coherence < 0.5`, R3 does not fire and the later rules are tried: a rank-1 LoRA on the output projections looks the same in a record.

**R4 `direct-edit`.** Structure: at most min(8, max(1, 5% of all tensors)) tensors changed; nothing added, removed, or cast; every changed tensor is localized or low rank. Strong when every changed tensor is both localized and low rank (E001: 64 of 3,584 columns, rank 1); weak when one of the two holds (E008: every row and column, rank 1). Low rank here means record `rank ≤ 4`, `top1_energy ≥ 0.9`, or `signal_rank ≤ 4`.

**R5 `merged-lora`.** Structure: every changed tensor is a 2-D projection inside a layer, except at most two embeddings (PEFT `modules_to_save`); no vector changed; at least max(2, 25% of the layers) layers are touched; the modules changed in each touched layer are the same set in at least 80% of the touched layers. Strong when features measured at least 80% of the changed projections and the low-rank test holds for at least 80% of those; when features were measured and the test fails, R5 does not fire. From the record alone the tier is weak whatever the rank says, because the record's rank cannot see a LoRA merge below the rounding floor. Evidence: `r` (the most common `signal_rank` among the changed projections; from the record, the most common rank, marked unreliable), the projections, and the layers.

**R6 `dense-finetune`.** Structure: changed tensors in at least 75% of the layers; at least half of the tensors both sides hold are changed; and either vectors changed in at least half of the touched layers or at least half of the changed projections are high rank (features, else the record). Strong when both hold. Full-parameter training moves the norms; a LoRA merge never does unless it was configured to.

**M `mixed`.** M1 is above. M2: a rule's pattern explains part of the change and the rest is more than 10% of the changed tensors, for example rank-1 output-side changes across many layers (R3's pattern) plus changes in other projections. The reason names both parts.

**U `unknown`.** No rule fires. The reason names the rule that came closest (the most structural conditions met) and the first condition it failed.

## Rules v2 (2026-11-04, after E010)
Every change below was made after seeing E010's results, so E010's v2 numbers are measured on the data that shaped the rules, not a validation; the v1 numbers are the frozen test. Each change names the pair it answers.

- Layer stacks (F07, A04). A layer is a stack prefix plus an index (`model.language_model.layers.3`, `model.visual.blocks.3`, `mtp.layers.0`). The main stack is the one holding the most changed or reshaped tensors (added tensors do not count, so an adapter file shipped beside a merge, L02 and L05, does not become the main stack), else the most tensors. Layer counts, R6's share of changed tensors, and R2's share of cast matrices are taken in the main stack. A frozen vision tower no longer halves a fine-tune's changed share (F07: 0.497 in v1).
- Casts whatever the status (M8). A quantization comparison reports a dequantised tensor as `changed` or `unchanged`, never `reshaped`, so any equal-shape dtype difference is a cast: lossless when no value changed, widening (moved) when the new dtype holds every old value, re-encoded (moved, new in v2) when the new dtype has at least the old one's mantissa bits but another range (BF16 to F16) and the change exceeds twice its unit roundoff, and narrowing otherwise. A LoRA merged and saved in F16 over a BF16 base (L05, L09, L11) is then a LoRA, not `mixed` by M1. For a re-encoded matrix, `--features` takes the noise edge from the coarser of the two dtypes.
- R2 (Q03, Q04, E011). The scaled-FP8 test (R2c) comes before the share test, and the share is of the main stack's matrices that are cast (FP8 and GGUF checkpoints keep norms in floats). A cast quantization in which any tensor changed beyond a cast, a norm or bias vector above all, is `mixed` by the new rule M3: a GGUF file that moved a per-channel scale from the norms into the projections (E011's Qwen2.5-0.5B-Instruct Q4_K_M) is not a plain quantization.
- R3 (A05, A06, A07, A04). Every changed matrix may be rank-1-like, input-side projections included (some tools also project the direction out of the input side), in any stack; output-side changes must still reach max(3, 25%) of the main stack's layers, and no vector may change. The shared-direction threshold is 0.2, not 0.5: real abliterations measured 0.27 to 1.00, and unrelated directions in a residual stream of 896 or more give under 0.05 for the 46 to 56 directions measured. The rank-1-like threshold on features is 0.75, as on the record (features only remove noise, so their top-1 energy is never below the record's).
- Features under the floor (F06). A tensor's features count only when its top singular value is at least 4 times the noise edge (`SNR_MIN`); below that, the count of components above the edge measures the rounding noise's own structure. The rules then use the record's statistics. Every LoRA and abliteration in E010 cleared it at the 10th percentile (8 and above); F06, a full fine-tune with relative change 0.001, had a median of 1.8.
- R5 module coverage (M02). With `--all`, each changed module must change in at least 80% of the main-stack layers that hold it, not one common module set in 80% of the layers: a hybrid stack (Qwen3.5's 18 linear-attention and 6 full-attention layers) holds different modules in different layers. R5's low-rank share leaves out matrices whose smaller side is at most 64 (a 16 x 1024 gate), where a LoRA of an accepted rank is already full rank.
- GGUF names. `blk.N` is a layer stack, `attn_output` and `ffn_down` are output-side, and `token_embd` and `output.weight` are embeddings, for a comparison whose names were not mapped.

## The M8 field (as merged, 2026-11-04)
M8 did not add the per-tensor `quantization` object this page first expected. A quantization comparison's record carries `comparison: {"kind": "quantization", "gguf": [sides], "mapping", "values"}` at the top and the ggml type in `dtype_a` or `dtype_b` (spec section 2.1). Rules v2 read it so: on a GGUF side, a tensor whose dtype is not a float type (`Q4_K`, `Q8_0`, ...) is a quantized cast of that type (the classifier writes it as `quantization: {"type": ...}` internally), and R2 bounds its relative change by `2 * 2^-(b - 1)`, where b is the type's bits per weight (8.5 for Q8_0, 6.5625 for Q6_K, 5.5 for Q5_0 and Q5_K, 4.5 for Q4_0 and Q4_K, 3.4375 for Q3_K, 2.625 for Q2_K). E011's plain quantizations sit at about half of that bound (Q8_0 0.006 against 0.011, Q4_K 0.08 against 0.18). F32 and F16 tensors of a GGUF side are ordinary casts. `--features` does not run on a quantization comparison.

## Output
`wvc classify` prints one line, `kind (tier): reason [rule]`, followed by the evidence as `key: value` lines; `--json` prints the classification document. `wvc diff --classify` adds the line after the summary and appends `classification` to the JSON record; `--features` (which needs `--classify`) also writes the per-tensor `features` into the record, so `wvc classify RECORD.json` can re-run the rules on a stored record without the weights. The document:

```json
{"format": "wvc-classification/1", "rules": "v2", "kind": "merged-lora", "tier": "strong",
 "rule": "R4", "reason": "one sentence", "features": true,
 "evidence": {"tensors": 290, "changed": 168, "...": "..."}}
```

## What this cannot tell apart
- A rank-1 LoRA on the output projections trained on a narrow task may learn one shared direction, and then it reads as direction-removal. Narrow fine-tunes are known to leave a constant bias direction (Minder et al., "Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences", [arXiv:2510.13900](https://arxiv.org/abs/2510.13900)).
- A full fine-tune with frozen norms and a small learning rate can read as a LoRA merge of high rank. R4 then says weak and gives `r`, and R6 does not fire.
- A LoRA merged and then quantized, or quantized and then fine-tuned, falls under M1.
- A model merge (SLERP or TIES of two fine-tunes) reads as a dense fine-tune. There is no `model-merge` kind in v1.
- GGUF-only uploads until M8 lands.
