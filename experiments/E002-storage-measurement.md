# E002: bytes needed to store a weight revision under byte-level dedup vs numeric deltas (tests H2)

Status: ANALYSED (see Outcome at the end). Frozen 2026-11-02 by Fable before any run; the run was delegated; analysis and the claim-register update were done afterwards by Fable. The runner recorded the code revision in `experiments/E002/receipt.json`.

## Question
Given checkpoint A already stored, how many bytes does each strategy need to store checkpoint B, for real revision pairs of different kinds? H2 predicts that tensor-level identity plus lossless numeric deltas needs far fewer bytes than byte-level chunk dedup on dense fine-tunes, and that the two compose.

## Pairs (all public; new downloads about 3.3 GB; abort if free disk is under 20 GB)
| Pair | A | B | Kind | Format |
|---|---|---|---|---|
| P1 | `EleutherAI/pythia-160m` revision `step142000`, `model.safetensors` | same repo, revision `step143000` | consecutive pretraining checkpoints, 1,000 steps apart | F32, 649,308,728 bytes each |
| P2 | `Qwen/Qwen2.5-0.5B`, `model.safetensors` | `Qwen/Qwen2.5-0.5B-Instruct`, `model.safetensors` | dense post-training fine-tune | BF16, 988,097,824 bytes each |
| P3 | `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17` (cached) | `RikiyaT/Qwen3.5-0.8B-weight-space-programming` revision `d1671a178705bcab7644806d23c535517f35723b` (cached) | targeted sparse rank-1 edit ([E001](E001-astra-edit-diff.md)) | BF16 |
| P4 | `Qwen/Qwen2.5-0.5B` | synthetic: A with a rank-8 update merged into every layer's `q_proj` and `v_proj` weight: `W' = bf16(W + s * (U @ V))`, `U` [out, 8] and `V` [8, in] standard normal from `numpy.random.default_rng(2)`, `s` chosen per tensor so that the Frobenius norm of `s * U @ V` is 1% of the Frobenius norm of `W`; written as a new safetensors file with the same tensor order | adapter merge, synthetic and labelled as such | BF16 |

Record the resolved commit hash and the file blob SHA-256 for every input.

## Strategies (bytes to store B given A)
- S0: raw size of B's tensor bytes.
- S1: zstd level 3 of each tensor of B independently, summed (standalone compression baseline).
- S2: tensor dedup: raw bytes of B's tensors whose (dtype, shape, sha256) is absent from A. S2z: the same tensors zstd-compressed.
- S3: fixed-size chunk dedup: split each tensor's byte buffer into 64 KiB aligned chunks (boundaries never cross tensors); bytes of B's chunks whose sha256 is absent from A's chunk set. S3z: those chunks zstd-compressed.
- S4u: upper bound on content-defined chunk dedup. For tensors present in both files with equal dtype and shape, find the byte positions that differ; any chunker with a minimum chunk of 16 KiB can dedup at most the bytes lying in unchanged runs of at least 16 KiB (entirely unchanged tensors count as fully dedupable). S4u = S0 minus those dedupable bytes. S4uz: the non-dedupable bytes zstd-compressed. This is the best case for any content-defined scheme, including Xet's 64 KiB chunks.
- S4 (optional; only if the run stays within the time budget): a real content-defined chunking simulation, buzhash over a 32-byte window, boundary when `(h & 0xFFFF) == 0`, minimum 16 KiB, maximum 256 KiB, applied to both files' tensor buffers; bytes of new chunks, raw and zstd.
- S5: XOR delta: for tensors in both files with equal dtype and shape, XOR the raw bytes, split the result into byte-position streams (two streams for 2-byte dtypes, four for 4-byte), zstd level 3 each; tensors without a counterpart are stored as in S1. Must reconstruct B exactly (per-tensor SHA-256 check).
- S6: sparse delta: for tensors with equal dtype and shape where at most 5% of elements changed, store uint32 element indices plus the raw changed element bytes, zstd-compressed; otherwise fall back to S5 for that tensor. Must reconstruct B exactly.
- S7 (P4 only): low-rank reference bound: bytes of `U` and `V` in bf16 for the edited tensors plus S2z for the rest. A bound for a future low-rank delta kind, computed from the known factors, not recovered from B.

## Metrics
For each pair and strategy: bytes, ratio to S0, encode wall time in seconds (single process), and for S5 and S6 a reconstruction check (every tensor hash of B matches; the run is invalid otherwise). Report the zstd library and version, numpy version, machine, and the wvc commit.

## Predictions, written before the run (Fable, 2026-11-02)
- P2: S2 at least 0.95 S0; S4u at least 0.95 S0; S1 between 0.65 and 0.70 S0 (ZipNN's BF16 figure); S5 between 0.45 and 0.60 S0.
- P1: S4u at least 0.9 S0 (Adam moves every parameter every step); S5 at most 0.6 S1 (exponent bytes mostly unchanged).
- P3: S2 about 14.7 MB raw; S4u about 14.7 MB too, because the edit touches 64 columns of every row (row stride 7,168 bytes), so no unchanged run reaches 16 KiB; S6 under 1 MB; S5 under 2 MB.
- P4: S2 removes every non-target tensor; S4u about equal to S2 (dense change inside target tensors); S5 well below S2z; S7 under 1% of S2.

Decision rule for H2 on the dense fine-tune (P2): supported if S5 is at most 0.8 S4uz; weakened if S4uz is at most 1.2 S5; otherwise inconclusive. On P1, P3, and P4 the same rule applies to the composed strategy min(S5, S6) against S4uz.

## Budget and stop rules
Downloads at most 3.3 GB; total runtime at most 60 minutes; abort and report if free disk is under 20 GB at any check. No GPU, no model execution, no training.

## Outputs
`experiments/E002/measure.py` (runnable as `uv run --with zstandard python experiments/E002/measure.py --pair P2 --out experiments/E002/results/`), `experiments/E002/results/<pair>.json`, `experiments/E002/results/summary.md`, `experiments/E002/receipt.json`. The runner does not interpret results.

## Outcome (analysed 2026-11-02 by Fable after the run; status ANALYSED)

Run: delegated agent, commit `bb5a074`, merged as `75be6b9`; receipt `experiments/E002/receipt.json`; results `experiments/E002/results/`. Reconstruction checks for S5 and S6 passed on every pair; the self-test (11 checks) passed; the optional S4 real-chunking simulation was run. Total runtime 221 s including about 84 s of downloads (3.27 GB new); free disk never fell below 64 GB.

Ratios to S0, from `results/summary.md`:

| Pair | S0 MB | S1 | S2z | S3z | S4uz | S4z | S5 | S6 | S7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 649 | 0.580 | 0.580 | 0.604 | 0.580 | 0.594 | 0.336 | 0.336 | n/a |
| P2 | 988 | 0.779 | 0.779 | 0.781 | 0.779 | 0.780 | 0.563 | 0.563 | n/a |
| P3 | 1,747 | 0.782 | 0.00659 | 0.00659 | 0.00659 | 0.00659 | 0.000301 | 0.000353 | n/a |
| P4 | 988 | 0.778 | 0.0351 | 0.0352 | 0.0351 | 0.0352 | 0.0167 | 0.0167 | 0.00109 |

Frozen decision rule, H2 supported on a pair if min(S5, S6) is at most 0.8 S4uz:

| Pair | min(S5, S6) | 0.8 S4uz | Verdict |
|---|---:|---:|---|
| P1 | 0.336 | 0.464 | supported |
| P2 | 0.563 | 0.623 | supported |
| P3 | 0.000301 | 0.00527 | supported |
| P4 | 0.0167 | 0.0281 | supported |

Post hoc robustness check, not in the frozen plan (`posthoc_grouping.py`, `results/posthoc_grouping.json`): S5 compresses byte-position planes while S1 and S4uz compress raw bytes, so part of S5's advantage could be grouping rather than the delta. Recomputing S5 reproduced the committed value on every pair. With grouping applied to B alone (S1g: 0.460, 0.710, 0.713, 0.710 for P1 to P4), the delta's own effect S5g/S1g is 0.731 (P1), 0.793 (P2), about 0 (P3), 0.024 (P4). On the dense fine-tune the delta clears the 0.8 threshold only marginally once grouping is equalised; grouping alone is worth 21% on FP32 and 9% on BF16 here.

Prediction scorecard:
- Held: P2 S2 and S4u stay at S0 (no tensor or run dedup); P2 S5 inside 0.45 to 0.60 (0.563); P1 S4u at S0; P1 S5 at most 0.6 S1 (0.58, barely); P3 S2 about 14.7 MB raw and S4u no better (14.68 MB), S6 under 1 MB (0.62), S5 under 2 MB (0.53); P4 S2 removes every non-target tensor, S4u equals S2, S5 well below S2z (0.48 of it).
- Missed: P2 S1 predicted 0.65 to 0.70, observed 0.779 (I took ZipNN's byte-grouped figure for a plain-zstd figure; the grouped S1g of 0.710 is inside the predicted range). P4 S7 predicted under 1% of S2, observed 2.5% (1.08 MB of 44.0 MB; an arithmetic slip, rank 8 against widths 896 and 128).

Interpretation, within the tested scope (four pairs, two formats, models of 0.16 B to 0.8 B parameters, one machine, zstd level 3):
1. Byte-level chunk dedup, fixed or content-defined, recovered nothing beyond whole-tensor dedup on any pair: S3z, S4uz, and S4z sit within 0.3% of S2z, and fixed chunks are slightly worse on P1 because per-chunk compression is coarser than per-tensor. Same-shape revisions contain no inserted or deleted bytes for content-defined chunking to exploit, and its 64 KiB granularity cannot see element-level change.
2. Tensor-level identity is where the large savings are when they exist: 99.3% (P3) and 96.5% (P4) of bytes are free by hash.
3. Numeric deltas compose with identity and add about 2,000x on the sparse edit (11.5 MB to 0.53 MB), 2.1x on the synthetic rank-8 merge (34.7 MB to 16.5 MB), and 21% to 27% beyond grouped standalone compression on dense revisions. A low-rank delta kind would cut P4 a further 15x (1.08 MB), which argues for storing adapter merges as factors whenever the factors are known.
4. Encoding is cheap: 1.5 to 1.8 s for S5 on 1 GB to 1.7 GB pairs; the real chunking simulation was the slowest strategy at 11 to 30 s.

Consequences for the design (spec section 3, delta kinds): keep sparse and low-rank kinds first-class; use byte-plane XOR plus zstd as the dense lossless fallback; do not implement chunking inside wvc, and rely on remotes such as Xet only to transport identical bytes.

Limitations and deviations: one machine; zstd level 3 only (a ZipNN-style exponent split would strengthen the standalone baseline and narrow the dense-case margin further); P4 is synthetic and its B file may not be bit-reproducible across BLAS libraries (hash recorded); S2 matched tensors by name, and the plan's content-addressed variant gave identical bytes; S7 is a bound computed from the known factors, not a reconstruction; unchanged tensors contributed 72 KB (P3) and 38 KB (P4) of near-empty XOR streams to S5; the buzhash chunker cuts constant regions at the 16 KiB minimum; the first `--pair all` attempt failed while parsing the download tool's output after one file had been fetched, and is kept in the receipt; the measurement script was split into four files to stay under the 800-line limit.
