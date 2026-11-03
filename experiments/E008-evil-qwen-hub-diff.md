# E008: which tensors does the baked "Evil Qwen" upload change, and can a Hub diff find out fetching one shard?

Date 2026-11-03. Status: analysed. The Question, Inputs, Predictions, and Budget sections below are unchanged from the plan frozen before the download; the run followed the same day, after Evan approved the download. Software: this repository at commit `48c824b`, Python 3.13.5, numpy 2.5.4, huggingface_hub 2.4.0, macOS 15.6, Apple M4 Max. Raw outputs in `experiments/E008/`.

## Question
`bowmanslayer/Qwen3.5-9B-Evil-BF16` (created 2026-10-27; its card says `finetune` and names no tensor; see hub-derivative-uploads-2026-11) differs from `Qwen/Qwen3.5-9B` in one of four shards by Hub blob hash (O29 in claims). Which tensors changed, by how many elements, and with what structure? Does `wvc diff` answer this while fetching only the shard that differs?

## Inputs (Hub API only, no download; receipt `experiments/E008/hub-plan.txt`)
- Base `hf:Qwen/Qwen3.5-9B@main`, pinned to commit `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Derivative `hf:bowmanslayer/Qwen3.5-9B-Evil-BF16@main`, pinned to commit `b66fc0a575c13af75c5d438111559b04ed015cd4`.
- Shards 1, 3, and 4 have the same name, size, and LFS SHA-256 on both sides; shard 2 (`model.safetensors-00002-of-00004.safetensors`) differs with equal size, 5,335,161,512 bytes per side. The exact hashes are in the receipt.

## Predictions (before the run)
- P1: the diff downloads exactly the two copies of shard 2 and prints `3 byte-identical on the Hub and skipped; fetched 1 from a and 1 from b`.
- P2: every changed tensor lives in shard 2, and fewer than one tenth of shard 2's tensors change (60% credence; the alternative is a dense fine-tune of the whole shard, which three identical shards make unlikely).
- P3: nothing is added, removed, or reshaped (equal sizes).
- P4: if the upload bakes a steering direction into weights, each changed tensor has rank at most 4 at the 1e-2 tolerance; a merged LoRA gives rank equal to its r; a dense fine-tune gives full rank. Recorded, not assumed.
- P5: wall time after the download is under five minutes on the M4 Max ([E001](E001-astra-edit-diff.md) took 9.8 s for one 1.7 GB shard per side).

## Budget and stop rules
Download at most 10.7 GB, into the Hugging Face cache only; abort if free disk would fall under 20 GB (32 GB free on 2026-11-03). No model is run; no generations. Nothing is pushed anywhere. Outputs committed: `diff.txt`, `diff.json`, `receipt.json`; the shards stay in the cache and out of Git.

## Commands
The two planned commands, then two reruns from the cache with a larger rank cap (see Deviation under Result), the shard hashes, the receipt builder, and a later rerun at the commit that raised the default cap (see Follow-ups):
```bash
export HF_HUB_DISABLE_PROGRESS_BARS=1
uv sync --extra hf
uv run wvc diff --rank hf:Qwen/Qwen3.5-9B hf:bowmanslayer/Qwen3.5-9B-Evil-BF16 > experiments/E008/diff.txt
uv run wvc diff --rank --json hf:Qwen/Qwen3.5-9B hf:bowmanslayer/Qwen3.5-9B-Evil-BF16 > experiments/E008/diff.json
uv run wvc diff --rank --rank-cap 67108864 hf:Qwen/Qwen3.5-9B hf:bowmanslayer/Qwen3.5-9B-Evil-BF16 > experiments/E008/diff-rankcap.txt
uv run wvc diff --rank --rank-cap 67108864 --json hf:Qwen/Qwen3.5-9B hf:bowmanslayer/Qwen3.5-9B-Evil-BF16 > experiments/E008/diff-rankcap.json
shasum -a 256 <the two cached shard blobs> > experiments/E008/shard-sha256.txt
# after the default rank cap was raised to 2^26, from the cache:
uv run wvc diff --rank hf:Qwen/Qwen3.5-9B hf:bowmanslayer/Qwen3.5-9B-Evil-BF16 > experiments/E008/diff-default-cap.txt
uv run python -I experiments/E008/make_receipt.py experiments/E008/run.log experiments/E008/hub-plan.txt experiments/E008 ~/.cache/huggingface/hub '{"python":"3.13.5",...}'
```
`run.log` is what the runner scripts printed: UTC start and end, free disk before and after, the code revision, each run's exit status and wall seconds, and the shard hashes. The first runner finished both planned commands and then aborted at its own shasum step on a scripting mistake (an unset positional parameter; the error line is in the log); a second runner did the reruns and the hashes. Every run's stderr was empty; the empty files are not committed.

## Result
| Item | Value |
|---|---|
| shards | 4 per side; 3 byte-identical on the Hub and skipped; shard 2 fetched from each side, 5,335,161,512 bytes each, 10,670,323,024 bytes in all |
| downloaded bytes | both blobs hash to the LFS SHA-256 the Hub listed: base `31c7d7e2...0330`, derivative `d055ee78...0b5a` (`shard-sha256.txt`, `hub-plan.txt`) |
| tensors | 775: 1 changed, 774 unchanged (722 reported from the skipped shards' headers, 52 of the 53 in shard 2), 0 added, removed, or reshaped |
| elements changed | 46,281,237 of 9,653,104,368 (0.4794%) |
| layer 17 `mlp.down_proj.weight`, BF16 [4096, 12288] | 46,281,237 of 50,331,648 elements changed (91.95%); rows 4,096 of 4,096; cols 12,288 of 12,288; max abs delta 0.0666; rel. Frobenius 0.0907; rank 1 (rel. tol 1e-2); top-1 energy 0.9997 |
| manifest roots | not computed: a Hub diff that skips shards never reads their tensor bytes (`root` is null in the JSON) |
| wall time | run 1, download plus diff: 181 s; run 2, diff from cache with JSON: 8 s; run 3, diff from cache with the SVD: 15 s |
| disk | 42 GB free before (the plan's 32 GB was read earlier that day), 32 GB after |

Predictions:
| Prediction | Held | Observed |
|---|---|---|
| P1 | yes | the header line as predicted; shard 2 fetched from each side, three skipped; both blobs match the Hub's hashes and sizes |
| P2 | yes | 1 of the 53 tensors in shard 2 changed (1.9%), and it lives in shard 2 |
| P3 | yes | 0 added, 0 removed, 0 reshaped |
| P4 | recorded | rank 1, top-1 energy 0.9997: a single-direction update. Rank alone does not separate a baked steering direction from a rank-1 LoRA merge or a rank-one edit; it does rule out a dense fine-tune of the matrix (full rank would be 4,096) |
| P5 | yes | 8 s from cache without the SVD, 15 s with it |

Deviation from the plan: the planned command printed `-` for rank because the touched block (4,096 x 12,288 = 50,331,648 elements) exceeds the then-default `--rank-cap` of 2^24, so the SVD was skipped. Runs 3 and 4 repeated the diff from the cache with `--rank-cap 67108864`; their tables are otherwise identical to runs 1 and 2 (`rank_run_agrees_with_planned_run` in `receipt.json`). Exit status 1 on every run is the diff's "differences found" code, as for `diff(1)`.

## Interpretation
- The upload changes one matrix, layer 17's down projection, and the change has rank 1 at the 1e-2 tolerance with top-1 energy 0.9997: one direction pair. Every row and every column holds at least one changed element. The 4,050,411 entries (8.05%) that did not change are byte-identical on both sides, consistent with a small update whose delta rounds away in bfloat16 (max abs delta 0.0666, relative Frobenius 0.0907); the output does not show the mechanism.
- It is not a two-neuron edit expressed in weights. Scaling two intermediate neurons would scale two columns of the down projection (rows 4,096, cols 2, rank at most 2); clamping them to constants, as the runtime recipe does, cannot be expressed in a bias-free `down_proj.weight` at all. This change touches every column. The wiki's guess that a bake would show as a one-column change (evil-qwen-steering, H3) does not describe this upload. The card says `finetune` and names no tensor; the diff names the tensor in one line.
- For the Hub diff: three of four shards never left the Hub, 722 of 775 tensors were reported unchanged from headers read by range request, and the run fetched 10.67 GB of the 38.61 GB of safetensors the two repositories hold (27.6%).
- Not shown here: whether the change is harmful, or anything about behaviour; no model was run. The runtime Evil Qwen recipe (clamping two neurons of Qwen3.5-4B, layers 13 and 14, 9,216 neurons per layer) is a different artifact on a different model size and is not expressible as a weight diff (H6). This repository's name is the only link between the two: its card makes no claim about the method, and the 4B neuron indices cannot map onto a 9B layer with 12,288 neurons.

## Limitations
- One pair, one upload by one publisher, and no ground truth beyond the bytes: the card gives no method. E001 had the authors' manifest; E008 has none.
- Manifest roots are not computed when a Hub diff skips shards, so this receipt pins the two commits and the Hub's file hashes, not a wvc root for either side. `wvc hash hf:...` on either repository would download all four shards.
- The rank tolerance (1e-2 relative) is a choice; bfloat16 rounding puts small singular values below it (top-1 energy 0.9997, not 1).
- Inputs live in the Hugging Face cache (huggingface_hub 2.4.0's shared `blobs/` directory, symlinked from each repository's snapshot), not in Git; re-running needs both revisions, 10.67 GB.

## Follow-ups
- The default rank cap (2^24 elements) skipped the SVD for this 9B-class down projection; the SVD cost about 7 s (run 3 minus run 2). Done the same day: the default is 2^26 (see CHANGELOG), and the planned command re-run from the cache at the new default prints a table byte-identical to `diff-rankcap.txt` (`diff-default-cap.txt`; 39 s with the page cache cold).
