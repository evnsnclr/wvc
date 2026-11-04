# wvc diff: agent view (wvc-diff-agent/0)

## Pair
- a (base): `hf:Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` (revision `c1899de289a04d12100db370d81485cdf75e47ca`); manifest root `46cdee2ec9f4eca1`
- b: `hf:bartowski/Qwen_Qwen3-0.6B-GGUF@60b85c0e3d8fe0f6474f406922a26d12aca4550d:Qwen_Qwen3-0.6B-Q4_K_M.gguf` (revision `60b85c0e3d8fe0f6474f406922a26d12aca4550d`); manifest root `63f4b70397a116ae`; GGUF v3, architecture `qwen3` (F32 113, Q4_K 169, Q6_K 29)
- comparison: quantization; GGUF side b; names mapped by the `qwen3` table; values dequantised to float64 where bytes cannot be compared; file type b `Q4_K_M`

## Headline
- tensors: 311 (198 changed, 113 unchanged, 0 added, 0 removed, 0 reshaped)
- elements changed: 751,078,246 of 751,632,384 (99.93%)
- layers with a changed tensor: 28 of 28
- files of b whose tensors are all unchanged: 0 of 1

## Changed tensors by projection
| projection | layers | dtype | elements changed | rel. norm | rank | top-1 |
|---|---|---|---|---|---|---|
| `model.layers.*.mlp.down_proj.weight` | 0-27 (28) | BF16->Q4_K, BF16->Q6_K | 88,010,742 of 88,080,384 (99.92%) | 0.0181 to 0.077 | 1024 | 0.003128 to 0.01874 |
| `model.layers.*.mlp.gate_proj.weight` | 0-27 (28) | BF16->Q4_K | 88,070,710 of 88,080,384 (99.99%) | 0.0717 to 0.0738 | 1023-1024 | 0.00266 to 0.03011 |
| `model.layers.*.mlp.up_proj.weight` | 0-27 (28) | BF16->Q4_K | 88,070,574 of 88,080,384 (99.99%) | 0.0721 to 0.0742 | 1023-1024 | 0.002504 to 0.02581 |
| `model.layers.*.self_attn.k_proj.weight` | 0-27 (28) | BF16->Q4_K | 29,356,738 of 29,360,128 (99.99%) | 0.0724 to 0.0752 | 984-1005 | 0.005159 to 0.01985 |
| `model.layers.*.self_attn.o_proj.weight` | 0-27 (28) | BF16->Q4_K | 58,713,382 of 58,720,256 (99.99%) | 0.072 to 0.0747 | 1024 | 0.00365 to 0.009581 |
| `model.layers.*.self_attn.q_proj.weight` | 0-27 (28) | BF16->Q4_K | 58,713,633 of 58,720,256 (99.99%) | 0.0724 to 0.0739 | 1023-1024 | 0.004884 to 0.008242 |
| `model.layers.*.self_attn.v_proj.weight` | 0-27 (28) | BF16->Q4_K, BF16->Q6_K | 29,337,043 of 29,360,128 (99.92%) | 0.0178 to 0.0745 | 996-1010 | 0.004066 to 0.01362 |
| `lm_head.weight` | - (1) | BF16->Q6_K | 155,239,750 of 155,582,464 (99.78%) | 0.0177 | - | - |
| `model.embed_tokens.weight` | - (1) | BF16->Q4_K | 155,565,674 of 155,582,464 (99.99%) | 0.0713 | - | - |

## Changed tensors by layer
| layers | tensors changed | projections changed | elements changed | max rel. norm |
|---|---|---|---|---|
| `model.layers.0-27` | 196 | all 11 but `input_layernorm.weight`, `post_attention_layernorm.weight`, `self_attn.k_norm.weight`, `self_attn.q_norm.weight` | 440,272,822 of 440,401,920 (99.97%) | 0.077 |
| (no layer) | 2 | `lm_head.weight`, `model.embed_tokens.weight` | 310,805,424 of 311,164,928 (99.88%) | 0.0713 |

## Largest changes (by rel. norm, top 12)
| tensor | dtype | shape | elements changed | rows x cols touched | rel. norm | max abs | rank | top-1 |
|---|---|---|---|---|---|---|---|---|
| `model.layers.16.mlp.down_proj.weight` | BF16->Q4_K | [1024, 3072] | 3,145,391 of 3,145,728 (99.99%) | 1024 x 3072 | 0.077 | 0.0251 | 1024 | 0.009151 |
| `model.layers.19.mlp.down_proj.weight` | BF16->Q4_K | [1024, 3072] | 3,145,383 of 3,145,728 (99.99%) | 1024 x 3072 | 0.077 | 0.0291 | 1024 | 0.006431 |
| `model.layers.18.mlp.down_proj.weight` | BF16->Q4_K | [1024, 3072] | 3,145,356 of 3,145,728 (99.99%) | 1024 x 3072 | 0.0766 | 0.0252 | 1024 | 0.006217 |
| `model.layers.15.mlp.down_proj.weight` | BF16->Q4_K | [1024, 3072] | 3,145,351 of 3,145,728 (99.99%) | 1024 x 3072 | 0.0757 | 0.0235 | 1024 | 0.008473 |
| `model.layers.21.mlp.down_proj.weight` | BF16->Q4_K | [1024, 3072] | 3,145,424 of 3,145,728 (99.99%) | 1024 x 3072 | 0.0754 | 0.034 | 1024 | 0.004315 |
| `model.layers.22.mlp.down_proj.weight` | BF16->Q4_K | [1024, 3072] | 3,145,414 of 3,145,728 (99.99%) | 1024 x 3072 | 0.0753 | 0.025 | 1024 | 0.003982 |
| `model.layers.10.mlp.down_proj.weight` | BF16->Q4_K | [1024, 3072] | 3,145,375 of 3,145,728 (99.99%) | 1024 x 3072 | 0.0752 | 0.0224 | 1024 | 0.00738 |
| `model.layers.16.self_attn.k_proj.weight` | BF16->Q4_K | [1024, 1024] | 1,048,446 of 1,048,576 (99.99%) | 1024 x 1024 | 0.0752 | 0.03 | 984 | 0.01752 |
| `model.layers.13.mlp.down_proj.weight` | BF16->Q4_K | [1024, 3072] | 3,145,350 of 3,145,728 (99.99%) | 1024 x 3072 | 0.0748 | 0.0215 | 1024 | 0.009876 |
| `model.layers.11.self_attn.o_proj.weight` | BF16->Q4_K | [1024, 2048] | 2,096,911 of 2,097,152 (99.99%) | 1024 x 2048 | 0.0747 | 0.0214 | 1024 | 0.00826 |
| `model.layers.13.self_attn.o_proj.weight` | BF16->Q4_K | [1024, 2048] | 2,096,915 of 2,097,152 (99.99%) | 1024 x 2048 | 0.0746 | 0.0168 | 1024 | 0.006875 |
| `model.layers.22.self_attn.v_proj.weight` | BF16->Q4_K | [1024, 1024] | 1,048,444 of 1,048,576 (99.99%) | 1024 x 1024 | 0.0745 | 0.0251 | 1007 | 0.004612 |
| and 186 more changed tensors | | | | | | | | |

## Added, removed, reshaped
- added: none
- removed: none
- reshaped: none

## What this does and does not show
- A diff shows where a model changed and how, not what the change does.
- Shown: which tensors differ, how many elements, how large the change is relative to the base, and the rank structure of each change (`rank` at relative tolerance 1e-2, `top-1` the share of the change's energy in its largest singular direction).
- Not shown: behaviour, intent, quality, or safety. No model was run. A low-rank change can be a merged adapter, a steering direction, or a direct edit, and the diff alone does not say which.
- This is a quantization comparison: every quantized tensor differs from its float base by its rounding error, which is not an edit, and a rescaling that leaves the function unchanged (E011) also shows as change.
