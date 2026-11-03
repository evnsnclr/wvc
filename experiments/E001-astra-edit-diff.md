# E001: tensor-level diff of the Astra weight edit against base Qwen3.5-0.8B

Date 2026-11-02. Status: analysed. Software: this repository at commit `9e8c708` (the run predates the commit; `diff.txt` was regenerated at that revision with identical output), Python 3.13.5, numpy 2.5.4, macOS 15.6, Apple M4 Max. Raw outputs in `experiments/E001/`.

## Question
Does a tensor-level hash and diff over safetensors recover exactly what the authors say they changed, without PyTorch and without decoding unchanged tensors?

## Inputs
- Base: `Qwen/Qwen3.5-0.8B`, Hub revision `2fc06364715b967f1860aea9cf38778875588b17`, file `model.safetensors-00001-of-00001.safetensors` (Hub blob SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`, 1,746,942,600 bytes).
- Edited: `RikiyaT/Qwen3.5-0.8B-weight-space-programming`, revision `d1671a178705bcab7644806d23c535517f35723b`, same file name (blob SHA-256 `fd1e42acdd118ac1b300d37500fd2885e75f5a09de3a33010d802bf7f737b553`, 1,746,943,032 bytes; the 432-byte difference is header metadata, since both files hold 1,746,882,752 tensor bytes).
- Ground truth: the model card and the authors' `exp8_math/runs/p50c/edit/manifest.json` from the logs repository (copy in `experiments/E001/authors-edit-manifest.json`).

## Commands
```bash
uv run wvc diff --rank "$BASE" "$EDIT" | tee experiments/E001/diff.txt
uv run wvc diff --rank --json "$BASE" "$EDIT" > experiments/E001/diff.json
uv run wvc hash "$BASE" > experiments/E001/manifest-base.json
uv run wvc hash "$EDIT" > experiments/E001/manifest-edited.json
```

## Result
| Item | Value |
|---|---|
| tensors | 488: 2 changed, 486 unchanged, 0 added, removed, or reshaped |
| elements changed | 130,188 of 873,438,784 (0.0149%) |
| layer 15 `mlp.down_proj.weight`, BF16 [1024, 3584] | 65,260 changed; rows 1,024; cols 64; max abs delta 0.146; rel. Frobenius 0.106; rank 1 (rel. tol 1e-2); top-1 energy 1.0000 |
| layer 19 `mlp.down_proj.weight`, BF16 [1024, 3584] | 64,928 changed; rows 1,023; cols 64; max abs delta 0.215; rel. Frobenius 0.0811; rank 1; top-1 energy 1.0000 |
| manifest roots | base `18d0ed28cddb632b9ebfaa7c4411ca92317c3940d7a0a18d376a6c79ae65f711`; edited `64ca3db368658f1a127f05edd5584c26a1560e5499b575cb56aa4052d2f67b60` |
| wall time, diff including hashing both files | 9.8 s; 1.35 s user CPU |

Cross-checks:
- Edited-tensor SHA-256s equal the two hashes on the model card (`82e2d733...2ee53` and `8ef3c61b...12eb2`).
- Per-tensor changed counts equal the authors' manifest (65,260 and 64,928; total 130,188). Their note: an entry the edit touched but that rounds back to the same bfloat16 value is not counted. The post's "131,072 weights" counts targeted entries (2 layers x 64 columns x 1,024 rows).
- The tensors reported changed are exactly the two the card names.

## Interpretation
- The identity and structure layers recover the edit precisely: location (two tensors), footprint (64 columns each), and shape (rank 1, one direction per layer), consistent with the authors' description of subtracting one direction from 64 `down_proj` columns.
- File-level identity reports "everything changed" (different blob hashes). Chunk-level dedup is expected to do no better on the two edited tensors: the edit touches 64 columns of every row, and a row is 7,168 bytes, so no unchanged byte run reaches a chunker's 16 KiB minimum. That expectation is measured in [E002](E002-storage-measurement.md), not shown here.
- Not shown here: whether the change is benign, or anything about behaviour. Evaluation numbers are the authors' and Hugging Face's.

## Limitations
- One pair, one model family, one edit kind. The rank tolerance (1e-2 relative) is a choice; bfloat16 rounding adds small singular values below it.
- `rows 1,023` in layer 19 means one output row had no element whose bfloat16 bits changed; not investigated.
- Inputs live in the Hugging Face cache, not in Git; re-running needs both revisions.
