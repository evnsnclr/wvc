# Experiments

Every run has a page here with the question, the predictions written before the run, the exact commands, the input revisions and hashes, the tool versions, the raw outputs under a directory of the same name, and which predictions failed. A receipt is kept whether the run succeeded or not; E004 is a negative result. Predictions and outcomes signed "Fable" were written by the maintainer's Claude session, which designs and analyses the experiments under the maintainer's direction; delegated runs say so in the page. Large inputs and stores stay out of Git; the pages say where they came from and how to get them again.

| | Question | Answer |
|---|---|---|
| [E001](E001-astra-edit-diff.md) | Does a tensor-level diff recover a published LLM-made edit, without PyTorch? | Yes: 2 tensors, 64 columns each, rank 1, equal to the authors' manifest |
| [E002](E002-storage-measurement.md) | How many bytes does a revision need under chunk dedup and under numeric deltas? | Chunk dedup adds nothing beyond tensor identity; deltas do, most on structured changes |
| [E003](E003-astra-edit-commit.md) | Does a repository store that edit small and give both revisions back exactly? | 513 KB on a 1.245 GB base; both revisions hash-identical to the Hub's files |
| [E004](E004-em-bisect.md) | Can bisect find where emergent misalignment begins in 20 training checkpoints? | No single onset existed; bisect reported none. A negative result |
| [E005](E005-loadable-checkout.md) | Does a checked-out model directory load and run like the original? | Yes: byte-identical files, identical tokens and logits |
| [E008](E008-evil-qwen-hub-diff.md) | Which tensors does a 9B upload that names none change? | 1 of 775, rank 1, after fetching 1 of 4 shards per side |
| [E009](E009-hub-census.md) | What do the most downloaded derivative uploads of eleven popular bases change? | 93 pairs, none refused or crashed: 46 quantizations, 9 byte-identical mirrors, 13 dense fine-tunes, 5 merged LoRAs, 2 low-rank abliterations |
| [E010](E010-change-classification.md) | From the diff alone, what kind of change is an upload, and how often is that right? | Frozen rules: 28 of 46 stated methods with features, 33 from the record; revised rules 37 and 39 on the same set; 7 stated methods contradicted by the bytes |
| [E011](E011-gguf-reading.md) | Does wvc read public GGUF files exactly, and does a diff against the base show only the quantization? | Exact: reader, gguf-hash lines, and dequantisation agree with llama.cpp; SmolLM2 shows a plain quantization, the official Qwen Q4_K_M also a per-channel rescaling |
| [E012](E012-gguf-uploads.md) | Can `hf:` diff GGUF-only uploads, a split one included, against their safetensors bases? | Yes, from the argument alone for all three; each a plain quantization, every norm unchanged; the split equals the single file beside it |

Machine paths in the receipts are rewritten to portable forms by [`scrub_paths.py`](scrub_paths.py), which the release workflow runs in check mode.
