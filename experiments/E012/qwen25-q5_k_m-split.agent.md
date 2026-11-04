# wvc diff: agent view (wvc-diff-agent/0)

## Pair
- a (base): `hf:Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775` (revision `7ae557604adf67be50417f59c2c2f167def9a775`); manifest root `4032e18995b833d5`
- b: `hf:Felladrin/gguf-Q5_K_M-Qwen2.5-0.5B-Instruct@69b6ffa31d2619f3a78b67ac661dec2b98d4e2fb:qwen2-00002-of-00003.gguf` (revision `69b6ffa31d2619f3a78b67ac661dec2b98d4e2fb`, 3 parts); manifest root `0059f26e96818acc`; GGUF v3, architecture `qwen2` (F32 121, Q5_1 132, Q5_K 12, Q6_K 12, Q8_0 13)
- comparison: quantization; GGUF side b; names mapped by the `qwen2` table; values dequantised to float64 where bytes cannot be compared; file type b `Q5_K_M`

## Headline
- tensors: 290 (169 changed, 121 unchanged, 0 added, 0 removed, 0 reshaped)
- elements changed: 492,575,523 of 494,032,768 (99.71%)
- layers with a changed tensor: 24 of 24
- files of b whose tensors are all unchanged: 0 of 3

## Changed tensors by projection
| projection | layers | dtype | elements changed | rel. norm | rank | top-1 |
|---|---|---|---|---|---|---|
| `model.layers.*.mlp.down_proj.weight` | 0-23 (24) | BF16->Q5_K, BF16->Q6_K | 104,466,004 of 104,595,456 (99.88%) | 0.0182 to 0.0396 | 896 | 0.003295 to 0.01921 |
| `model.layers.*.mlp.gate_proj.weight` | 0-23 (24) | BF16->Q5_1 | 104,242,370 of 104,595,456 (99.66%) | 0.0347 to 0.0373 | 895-896 | 0.002701 to 0.0119 |
| `model.layers.*.mlp.up_proj.weight` | 0-23 (24) | BF16->Q5_1 | 104,267,243 of 104,595,456 (99.69%) | 0.0349 to 0.037 | 895-896 | 0.002461 to 0.02994 |
| `model.layers.*.self_attn.k_proj.weight` | 0-23 (24) | BF16->Q5_1 | 2,742,486 of 2,752,512 (99.64%) | 0.0363 to 0.0418 | 128 | 0.01916 to 0.09854 |
| `model.layers.*.self_attn.o_proj.weight` | 0-23 (24) | BF16->Q5_1 | 19,208,026 of 19,267,584 (99.69%) | 0.0341 to 0.0376 | 835-875 | 0.007077 to 0.03677 |
| `model.layers.*.self_attn.q_proj.weight` | 0-23 (24) | BF16->Q5_1 | 19,203,147 of 19,267,584 (99.67%) | 0.0363 to 0.0404 | 815-867 | 0.006712 to 0.03671 |
| `model.layers.*.self_attn.v_proj.weight` | 0-23 (24) | BF16->Q5_1, BF16->Q8_0 | 2,744,322 of 2,752,512 (99.70%) | 0.00561 to 0.0446 | 128 | 0.01487 to 0.02143 |
| `model.embed_tokens.weight` | - (1) | BF16->Q8_0 | 135,701,925 of 136,134,656 (99.68%) | 0.00553 | - | - |

## Changed tensors by layer
| layers | tensors changed | projections changed | elements changed | max rel. norm |
|---|---|---|---|---|
| `model.layers.0-23` | 168 | `mlp.down_proj.weight`, `mlp.gate_proj.weight`, `mlp.up_proj.weight`, `self_attn.k_proj.weight`, +3 (of 12) | 356,873,598 of 357,826,560 (99.73%) | 0.0446 |
| (no layer) | 1 | `model.embed_tokens.weight` | 135,701,925 of 136,134,656 (99.68%) | 0.00553 |

## Largest changes (by rel. norm, top 12)
| tensor | dtype | shape | elements changed | rows x cols touched | rel. norm | max abs | rank | top-1 |
|---|---|---|---|---|---|---|---|---|
| `model.layers.20.self_attn.v_proj.weight` | BF16->Q5_1 | [128, 896] | 114,271 of 114,688 (99.64%) | 128 x 896 | 0.0446 | 0.00638 | 128 | 0.01722 |
| `model.layers.22.self_attn.v_proj.weight` | BF16->Q5_1 | [128, 896] | 114,128 of 114,688 (99.51%) | 128 x 896 | 0.0444 | 0.0104 | 128 | 0.01603 |
| `model.layers.11.self_attn.k_proj.weight` | BF16->Q5_1 | [128, 896] | 114,305 of 114,688 (99.67%) | 128 x 896 | 0.0418 | 0.0112 | 128 | 0.02865 |
| `model.layers.23.self_attn.v_proj.weight` | BF16->Q5_1 | [128, 896] | 114,245 of 114,688 (99.61%) | 128 x 896 | 0.0414 | 0.0104 | 128 | 0.0192 |
| `model.layers.16.self_attn.k_proj.weight` | BF16->Q5_1 | [128, 896] | 114,230 of 114,688 (99.60%) | 128 x 896 | 0.041 | 0.00688 | 128 | 0.02687 |
| `model.layers.9.self_attn.k_proj.weight` | BF16->Q5_1 | [128, 896] | 114,261 of 114,688 (99.63%) | 128 x 896 | 0.0407 | 0.0113 | 128 | 0.0246 |
| `model.layers.22.self_attn.k_proj.weight` | BF16->Q5_1 | [128, 896] | 114,250 of 114,688 (99.62%) | 128 x 896 | 0.0407 | 0.0068 | 128 | 0.02689 |
| `model.layers.21.self_attn.k_proj.weight` | BF16->Q5_1 | [128, 896] | 114,243 of 114,688 (99.61%) | 128 x 896 | 0.0407 | 0.00677 | 128 | 0.02588 |
| `model.layers.13.self_attn.k_proj.weight` | BF16->Q5_1 | [128, 896] | 114,256 of 114,688 (99.62%) | 128 x 896 | 0.0405 | 0.0104 | 128 | 0.04626 |
| `model.layers.21.self_attn.q_proj.weight` | BF16->Q5_1 | [896, 896] | 800,163 of 802,816 (99.67%) | 896 x 896 | 0.0404 | 0.0109 | 854 | 0.009592 |
| `model.layers.23.self_attn.k_proj.weight` | BF16->Q5_1 | [128, 896] | 114,287 of 114,688 (99.65%) | 128 x 896 | 0.0403 | 0.00518 | 128 | 0.02879 |
| `model.layers.22.self_attn.q_proj.weight` | BF16->Q5_1 | [896, 896] | 800,137 of 802,816 (99.67%) | 896 x 896 | 0.0403 | 0.012 | 859 | 0.00852 |
| and 157 more changed tensors | | | | | | | | |

## Added, removed, reshaped
- added: none
- removed: none
- reshaped: none

## What this does and does not show
- A diff shows where a model changed and how, not what the change does.
- Shown: which tensors differ, how many elements, how large the change is relative to the base, and the rank structure of each change (`rank` at relative tolerance 1e-2, `top-1` the share of the change's energy in its largest singular direction).
- Not shown: behaviour, intent, quality, or safety. No model was run. A low-rank change can be a merged adapter, a steering direction, or a direct edit, and the diff alone does not say which.
- This is a quantization comparison: every quantized tensor differs from its float base by its rounding error, which is not an edit, and a rescaling that leaves the function unchanged (E011) also shows as change.
