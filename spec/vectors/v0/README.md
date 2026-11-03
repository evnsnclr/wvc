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

Not covered yet: `reshaped`, dtype changes, NaN handling, sharded checkpoints.

## Conformance rule
An implementation conforms if, reading `a.safetensors` and `b.safetensors`, it:
1. reproduces `a.manifest.json` and `b.manifest.json` exactly: the same keys and values, tensors sorted by name, and the same root;
2. reproduces the statuses, element counts, rows and columns touched, and rank in `diff.json` exactly, and the floating statistics `max_abs_delta`, `delta_norm`, `rel_delta_norm`, and `top1_energy` within a relative tolerance of 1e-6, that is `|actual - expected| <= 1e-6 * |expected|`.

Every other field of `diff.json` (names, dtypes, shapes, roots, the summary, and nulls) must match exactly; the two `path` fields are informational. The JSON is written with `indent=2` plus a newline, but byte-level formatting is not part of the rule while canonical JSON is an open point (spec section 6).

## Regenerate
```bash
uv run python spec/vectors/v0/make_vectors.py            # rewrite the files in this directory
uv run python spec/vectors/v0/make_vectors.py OUT_DIR    # or write them elsewhere
```
[`tests/test_vectors.py`](../../../tests/test_vectors.py) checks that `wvc` reproduces both manifests and the diff record from the committed checkpoints, and that the generator reproduces both checkpoints byte for byte. The files were generated with NumPy 2.5.4. NumPy may change `Generator` streams in feature releases; if that happens the regeneration check fails, and the committed files remain the reference.
