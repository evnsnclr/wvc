# E003: commit base Qwen3.5-0.8B and the Astra edit into a wvc repository, then check out byte-exact files

Status: ANALYSED (see Outcome at the end). Frozen 2026-11-02 by Fable before the M3 code existed; run 2026-11-03 UTC after `init`, `commit`, `log`, `show`, and `checkout` were merged. Inputs are the E001 files already in the Hugging Face cache; no downloads.

## Question
Does the M3 object store (spec section 3) store a real LLM-made edit as a small delta on top of its base, record the operation, and reproduce both original shard files byte-for-byte on checkout?

## Procedure
```bash
BASE=~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
EDIT=~/.cache/huggingface/hub/models--RikiyaT--Qwen3.5-0.8B-weight-space-programming/snapshots/d1671a178705bcab7644806d23c535517f35723b
mkdir -p experiments/E003/repo && cd experiments/E003/repo
uv run wvc init .
uv run wvc commit "$BASE" -m "Qwen/Qwen3.5-0.8B at 2fc0636" --op pretrain-checkpoint
uv run wvc commit "$EDIT" -m "Weight-Space Programming edit (Takehi et al.)" --op direct-edit
uv run wvc log
uv run wvc show HEAD
uv run wvc checkout HEAD --out ../checkout-edit
uv run wvc checkout HEAD~1-or-first-commit-id --out ../checkout-base
```
Then hash the checked-out files and compare with the Hub blob SHA-256s from E001 (base `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`, edit `fd1e42acdd118ac1b300d37500fd2885e75f5a09de3a33010d802bf7f737b553`), and measure `du -sb experiments/E003/repo/.wvc` after each commit. `experiments/E003/repo` and the checkouts are gitignored (they contain weights); the receipt records sizes, hashes, timings, and the exact commands.

## Predictions (before the run)
- First commit: 488 objects stored as `zstd-planes`; `.wvc` size about 1.24 GB (E002's byte-grouped standalone ratio of 0.713 applied to 1,746,882,752 tensor bytes), within 1.20 to 1.30 GB; wall time under 60 s on the M4 Max.
- Second commit: 486 tensors reused by hash; 2 objects stored as deltas, each `delta-xor-planes` or `delta-sparse`, together under 1 MB (E002 measured 0.53 MB for XOR planes and 0.62 MB for sparse over the same two tensors); the manifest, commit record, and header payload add under 1 MB; wall time under 30 s.
- `show HEAD` reports 2 changed, 486 unchanged, 0 added, 0 removed against the first parent, and operation kind `direct-edit`.
- Both checkouts reproduce the shard files byte-for-byte: SHA-256 equal to the Hub blob hashes above; checkout time under 60 s each.
- Decision rule: the milestone check passes only if every prediction about counts and hashes holds; size and time predictions are reported, and a miss is recorded, not fatal.

## Outputs
`experiments/E003/receipt.json` (commands, sizes, hashes, timings, versions, wvc commit), and an Outcome section appended here.

## Outcome (run and analysed 2026-11-03 UTC by Fable; status ANALYSED)

Run with `uv run python experiments/E003/run_e003.py` at wvc commit `b18dade` (receipt: `experiments/E003/receipt.json`, including every command's output). Total 23.5 s wall. The E003 repository (1.2 GB) stays on disk under `experiments/E003/repo/` (gitignored); the two checkouts were deleted after hashing.

| Step | Result | Prediction | Held |
|---|---|---|---|
| first commit | 488 tensors stored full (`zstd-planes`), payload 1,245,147,513 bytes; `.wvc` 1,245,453,696 bytes; 8.4 s | 1.20 to 1.30 GB, under 60 s | yes |
| second commit | 486 reused by hash, 2 stored as `delta-xor-planes`, payload 513,248 bytes; store grew 669,412 bytes in all; 2.8 s | deltas under 1 MB, records under 1 MB, under 30 s | yes |
| `show HEAD` | 2 changed, 486 unchanged, 0 added, 0 removed against the parent; operation `direct-edit`; manifest root `64ca3db3...f67b60`, the same root E001 computed | counts and kind as stated | yes |
| checkout, edit | 1,746,943,032 bytes, sha256 `fd1e42ac...737b553`, equal to the Hub blob; 5.4 s | hash-identical, under 60 s | yes |
| checkout, base | 1,746,942,600 bytes, sha256 `04b1c301...fe4696`, equal to the Hub blob; 5.1 s | hash-identical, under 60 s | yes |

Interpretation, within the tested scope (one model, one edit kind, one machine): the M3 store does what spec section 3 says. A real LLM-made edit costs 0.05% of the base's stored size as a second revision; both revisions reproduce the original files byte-for-byte; and the recorded operation is `direct-edit`, which the Hub's vocabulary cannot express. The two deltas chose the XOR-plane form over sparse, consistent with E002 (525 KB against 617 KB for the same tensors). Byte-plane zstd of the full base reached 0.713 of the raw tensor bytes, matching E002's standalone grouped ratio.

Limitations: no remote, no signing, no behavioural attestation yet; the base commit hashes every tensor twice (once to detect reuse, once inside the store), which is why a 1.7 GB first commit takes 8 s rather than about 5; no concurrency control on the branch ref.
