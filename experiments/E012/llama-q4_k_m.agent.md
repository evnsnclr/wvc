# wvc diff: agent view (wvc-diff-agent/0)

## Pair
- a (base): `hf:meta-llama/Llama-3.2-1B-Instruct@9213176726f574b556790deb65791e0c5aa438b6` (revision `9213176726f574b556790deb65791e0c5aa438b6`); manifest root `2253ec41fb000097`
- b: `hf:lmstudio-community/Llama-3.2-1B-Instruct-GGUF@199151125cf15a129ab3b548b26afeed976df066:Llama-3.2-1B-Instruct-Q4_K_M.g…` (revision `199151125cf15a129ab3b548b26afeed976df066`); manifest root `7535112bbc3016ba`; GGUF v3, architecture `llama` (F32 34, Q4_K 96, Q6_K 17)
- comparison: quantization; GGUF side b; names mapped by the `llama` table; values dequantised to float64 where bytes cannot be compared; file type b `Q4_K_M`

## Headline
- tensors: 147 (113 changed, 33 unchanged, 1 added, 0 removed, 0 reshaped)
- elements changed: 1,234,252,436 of 1,235,814,400 (99.87%)
- layers with a changed tensor: 16 of 16
- files of b whose tensors are all unchanged: 0 of 1

## Changed tensors by projection
| projection | layers | dtype | elements changed | rel. norm | rank | top-1 |
|---|---|---|---|---|---|---|
| `model.layers.*.mlp.down_proj.weight` | 0-15 (16) | BF16->Q4_K, BF16->Q6_K | 267,937,286 of 268,435,456 (99.81%) | 0.018 to 0.0734 | 2048 | 0.00257 to 0.01162 |
| `model.layers.*.mlp.gate_proj.weight` | 0-15 (16) | BF16->Q4_K | 268,406,965 of 268,435,456 (99.99%) | 0.0718 to 0.0732 | 2048 | 0.001462 to 0.01417 |
| `model.layers.*.mlp.up_proj.weight` | 0-15 (16) | BF16->Q4_K | 268,407,929 of 268,435,456 (99.99%) | 0.0717 to 0.0729 | 2048 | 0.001105 to 0.02388 |
| `model.layers.*.self_attn.k_proj.weight` | 0-15 (16) | BF16->Q4_K | 16,775,310 of 16,777,216 (99.99%) | 0.0721 to 0.0787 | 512 | 0.006157 to 0.01134 |
| `model.layers.*.self_attn.o_proj.weight` | 0-15 (16) | BF16->Q4_K | 67,100,458 of 67,108,864 (99.99%) | 0.0718 to 0.0734 | 1977-2007 | 0.003756 to 0.01101 |
| `model.layers.*.self_attn.q_proj.weight` | 0-15 (16) | BF16->Q4_K | 67,101,338 of 67,108,864 (99.99%) | 0.0722 to 0.0744 | 1977-2009 | 0.002882 to 0.007158 |
| `model.layers.*.self_attn.v_proj.weight` | 0-15 (16) | BF16->Q4_K, BF16->Q6_K | 16,741,513 of 16,777,216 (99.79%) | 0.0184 to 0.0794 | 512 | 0.00467 to 0.006232 |
| `model.embed_tokens.weight` | - (1) | BF16->Q6_K | 261,781,637 of 262,668,288 (99.66%) | 0.0185 | - | - |

## Changed tensors by layer
| layers | tensors changed | projections changed | elements changed | max rel. norm |
|---|---|---|---|---|
| `model.layers.0-15` | 112 | all 9 but `input_layernorm.weight`, `post_attention_layernorm.weight` | 972,470,799 of 973,078,528 (99.94%) | 0.0794 |
| (no layer) | 1 | `model.embed_tokens.weight` | 261,781,637 of 262,668,288 (99.66%) | 0.0185 |

## Largest changes (by rel. norm, top 12)
| tensor | dtype | shape | elements changed | rows x cols touched | rel. norm | max abs | rank | top-1 |
|---|---|---|---|---|---|---|---|---|
| `model.layers.14.self_attn.v_proj.weight` | BF16->Q4_K | [512, 2048] | 1,048,438 of 1,048,576 (99.99%) | 512 x 2048 | 0.0794 | 0.0122 | 512 | 0.005913 |
| `model.layers.15.self_attn.k_proj.weight` | BF16->Q4_K | [512, 2048] | 1,048,453 of 1,048,576 (99.99%) | 512 x 2048 | 0.0787 | 0.0202 | 512 | 0.007993 |
| `model.layers.14.self_attn.k_proj.weight` | BF16->Q4_K | [512, 2048] | 1,048,442 of 1,048,576 (99.99%) | 512 x 2048 | 0.0774 | 0.0198 | 512 | 0.007538 |
| `model.layers.10.self_attn.v_proj.weight` | BF16->Q4_K | [512, 2048] | 1,048,446 of 1,048,576 (99.99%) | 512 x 2048 | 0.0771 | 0.00889 | 512 | 0.005477 |
| `model.layers.0.self_attn.k_proj.weight` | BF16->Q4_K | [512, 2048] | 1,048,459 of 1,048,576 (99.99%) | 512 x 2048 | 0.0756 | 0.0279 | 512 | 0.0104 |
| `model.layers.13.self_attn.v_proj.weight` | BF16->Q4_K | [512, 2048] | 1,048,450 of 1,048,576 (99.99%) | 512 x 2048 | 0.0755 | 0.00657 | 512 | 0.005439 |
| `model.layers.11.self_attn.v_proj.weight` | BF16->Q4_K | [512, 2048] | 1,048,443 of 1,048,576 (99.99%) | 512 x 2048 | 0.0749 | 0.00557 | 512 | 0.00467 |
| `model.layers.8.self_attn.k_proj.weight` | BF16->Q4_K | [512, 2048] | 1,048,441 of 1,048,576 (99.99%) | 512 x 2048 | 0.0745 | 0.0235 | 512 | 0.008022 |
| `model.layers.0.self_attn.q_proj.weight` | BF16->Q4_K | [2048, 2048] | 4,193,822 of 4,194,304 (99.99%) | 2048 x 2048 | 0.0744 | 0.0341 | 1980 | 0.007158 |
| `model.layers.15.self_attn.q_proj.weight` | BF16->Q4_K | [2048, 2048] | 4,193,793 of 4,194,304 (99.99%) | 2048 x 2048 | 0.0743 | 0.0161 | 1977 | 0.004601 |
| `model.layers.6.self_attn.v_proj.weight` | BF16->Q4_K | [512, 2048] | 1,048,443 of 1,048,576 (99.99%) | 512 x 2048 | 0.0741 | 0.00652 | 512 | 0.005504 |
| `model.layers.14.self_attn.q_proj.weight` | BF16->Q4_K | [2048, 2048] | 4,193,815 of 4,194,304 (99.99%) | 2048 x 2048 | 0.0741 | 0.0173 | 1985 | 0.004388 |
| and 101 more changed tensors | | | | | | | | |

## Added, removed, reshaped
- added (1):
  - `rope_freqs.weight` F32 [32]
- removed: none
- reshaped: none

## What this does and does not show
- A diff shows where a model changed and how, not what the change does.
- Shown: which tensors differ, how many elements, how large the change is relative to the base, and the rank structure of each change (`rank` at relative tolerance 1e-2, `top-1` the share of the change's energy in its largest singular direction).
- Not shown: behaviour, intent, quality, or safety. No model was run. A low-rank change can be a merged adapter, a steering direction, or a direct edit, and the diff alone does not say which.
- This is a quantization comparison: every quantized tensor differs from its float base by its rounding error, which is not an edit, and a rescaling that leaves the function unchanged (E011) also shows as change.
