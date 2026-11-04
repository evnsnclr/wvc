# wvc spec v0 test vectors

Conformance vectors for the `wvc-manifest/0` manifest (section 1) and the diff record (section 2) of [spec v0](../../../docs/spec-v0.md).

## Files
| File | Content |
|---|---|
| `a.safetensors` | Base checkpoint. Tensors in header order: `w` F32 [4, 6], `b` BF16 [8], `h` F16 [2, 2, 2]. `__metadata__` is `{"format": "pt"}`; the header is space-padded to a multiple of 8 bytes. |
| `b.safetensors` | Edited checkpoint: `w` with 1.0 added to column 2 in every row, `b` byte-identical, `h` absent, and a new `x` I8 [3] holding [1, -2, 3]. Same metadata and padding. |
| `a.manifest.json`, `b.manifest.json` | `json.dumps(build_manifest(load_checkpoint(path)), indent=2)` plus a newline, which is what `wvc hash` prints. |
| `diff.json` | What `wvc diff --rank --json --all a.safetensors b.safetensors` prints, with the two `path` fields reduced to the basenames `a.safetensors` and `b.safetensors`. |
| `make_vectors.py` | The deterministic generator: `numpy.random.default_rng(20261102)` draws standard normals for `w`, `b`, and `h` in that order, cast to float32 (`w`), to bfloat16 by round to nearest even (`b`), and to float16 via float32 (`h`). |

The diff covers four of the five statuses:

| Tensor | Status | Detail |
|---|---|---|
| `b` | unchanged | BF16, equal digests |
| `h` | removed | F16 [2, 2, 2] |
| `w` | changed | 4 of 24 elements, 4 rows and 1 column touched, rank 1 |
| `x` | added | I8 [3] |

`h` and `x` carry their ids in `sha256`, and `summary.renamed` is 0.

Not covered yet: `reshaped`, dtype changes, NaN handling, sharded checkpoints.

## Renamed pair (spec section 2)
| File | Content |
|---|---|
| `wrap-a.safetensors` | Tensors in header order: `model.embed_tokens.weight` F32 [6, 4], `model.layers.0.mlp.down_proj.weight` F32 [4, 6], `model.norm.weight` BF16 [4]; metadata and padding as `a`. |
| `wrap-b.safetensors` | The same tensors under the prefix `language_model.`, as a vision model wrapping a language model writes them; the down projection has 0.5 times the column index added to every row (a rank-1 edit of columns 1 to 5). |
| `wrap-diff.json` | What `wvc diff --rank --json --all wrap-a.safetensors wrap-b.safetensors` prints, with paths reduced to basenames. |

The generator draws from `numpy.random.default_rng(20261104)`, in header order. The embedding and the norm are paired by id (`unchanged`); the down projection by suffix (`changed`, 20 of 24 elements, 4 rows and 5 columns touched, rank 1). Every entry has `renamed_to`, `summary.renamed` is 3, and nothing is added or removed.

## GGUF vectors (spec sections 1.1 and 2.1)
| File | Content |
|---|---|
| `quant.gguf` | A version 3 GGUF, alignment 32, `general.architecture` `llama` with `llama.attention.head_count` 2 and `head_count_kv` 1: one tensor per type wvc dequantises, under llama.cpp's names, in this order: `token_embd.weight` Q8_0 [8, 256], `output_norm.weight` F32 [256], `output.weight` Q6_K [4, 256], `blk.0.attn_norm.weight` BF16 [256], `blk.0.attn_q.weight` Q4_K [8, 256], `blk.0.attn_k.weight` Q8_K [4, 256], `blk.0.attn_v.weight` Q5_0 [4, 256], `blk.0.attn_output.weight` Q5_K [4, 256], `blk.0.ffn_norm.weight` F16 [256], `blk.0.ffn_gate.weight` Q4_0 [4, 256], `blk.0.ffn_up.weight` Q4_1 [4, 256], `blk.0.ffn_down.weight` Q5_1 [4, 256] (shapes row-major). |
| `quant-base.safetensors` | Its twin under Hub names (`model.embed_tokens.weight`, ..., `model.layers.0.mlp.down_proj.weight`), all BF16, `__metadata__` `{"format": "pt"}`. |
| `quant.manifest.json` | `wvc hash quant.gguf`: block types appear as their dtype, with `nbytes` counted in blocks. |
| `quant.gguf-hash` | `wvc hash --format gguf-hash quant.gguf`; identical to what `llama-gguf-hash --sha256 quant.gguf` prints in this directory (checked in E011). |
| `quant.dequant.json` | Per GGUF tensor in file order: its type and the SHA-256 of its values as little-endian float32 in storage order (block types dequantised, plain types decoded; no permutation undone). |
| `quant-diff.json` | What `wvc diff --rank --json --all quant-base.safetensors quant.gguf` prints, with the two `path` fields reduced to basenames. |

The generator draws from `numpy.random.default_rng(20261103)` tensor by tensor in file order. A block-type tensor holds random quant bytes whose scales (float16, or Q8_K's float32) are normals times 0.01; its twin holds the dequantised values with the rotary permutation undone for `attn_q` and `attn_k`, rounded to BF16, so the comparison sees only that rounding (and every block-type tensor is `changed`). A plain tensor draws normals times powers of ten from 1e-7 to 1, rounded to BF16 for the twin; the GGUF holds the same values widened to F32 (`unchanged` by value), the same BF16 bits (`unchanged` by identity), or their nearest F16 (`changed` where F16 cannot hold a small value).

An implementation conforms to the GGUF vectors if it reproduces `quant.manifest.json` and `quant.gguf-hash` exactly, every digest in `quant.dequant.json` exactly (dequantisation is bit-exact), and `quant-diff.json` under the rule below.

## Conformance rule
An implementation conforms if, reading `a.safetensors` and `b.safetensors`, it:
1. reproduces `a.manifest.json` and `b.manifest.json` exactly: the same keys and values, tensors sorted by name, and the same root;
2. reproduces the statuses, element counts, rows and columns touched, and rank in `diff.json` exactly, and the floating statistics `max_abs_delta`, `delta_norm`, `rel_delta_norm`, and `top1_energy` within a relative tolerance of 1e-6, that is `|actual - expected| <= 1e-6 * |expected|`.

Every other field of `diff.json` (names, dtypes, shapes, roots, the summary, and nulls) must match exactly; the two `path` fields are informational. The JSON is written with `indent=2` plus a newline, but byte-level formatting is not part of the rule while canonical JSON is an open point (spec section 6).

`quant-diff.json` and `wrap-diff.json` are held to the same rule as `diff.json` (with `renamed_to` and `sha256` among the fields that match exactly), including its `comparison` object, its `gguf` side object, and the per-tensor `compared`, `name_b`, and `transform_b` fields.

## Regenerate
```bash
uv run python spec/vectors/v0/make_vectors.py            # rewrite the files in this directory
uv run python spec/vectors/v0/make_vectors.py OUT_DIR    # or write them elsewhere
```
[`tests/test_vectors.py`](../../../tests/test_vectors.py) checks that `wvc` reproduces every manifest, the gguf-hash lines, the dequantised digests, and the three diff records from the committed files, and that the generator reproduces all six checkpoints byte for byte. The files were generated with NumPy 2.5.4. NumPy may change `Generator` streams in feature releases; if that happens the regeneration check fails, and the committed files remain the reference.
