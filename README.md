<p align="center">
  <img src="docs/assets/hero.svg" width="100%" alt="wvc: version control for open model weights. Animation of a real diff (E008): two Qwen3.5-9B checkpoints on the Hugging Face Hub are compared shard by shard. Three of the four shards have the same SHA-256 on both sides and are skipped, the fourth is fetched, and one of its 53 tensors, layer 17's mlp.down_proj.weight, lights up as changed, with rank 1.">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#what-it-found">What it found</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#commands">Commands</a> ·
  <a href="#status">Status</a> ·
  <a href="docs/spec-v0.md">Spec</a> ·
  <a href="experiments/">Receipts</a>
</p>

<p align="center">
  <img alt="Status: experimental" src="https://img.shields.io/badge/status-experimental-f0883e">
  <img alt="Release: none yet" src="https://img.shields.io/badge/release-none%20yet-8b949e">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776ab">
  <a href="LICENSE"><img alt="Licence: Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-2ea44f"></a>
  <a href="https://github.com/evnsnclr/wvc/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/evnsnclr/wvc/actions/workflows/ci.yml/badge.svg"></a>
</p>

> [!WARNING]
> **wvc is an experimental research prototype.** It is a few days old, has one maintainer, and has no release; nothing is on PyPI. Its formats ([spec v0](docs/spec-v0.md)) are drafts and may change in ways that break existing `.wvc` repositories. The results below come from a handful of public models on one machine. Don't make wvc the only copy of weights you care about, or the sole basis of a security decision. A diff shows which weights changed and how; it cannot show whether a change is benign.

**wvc treats a model checkpoint as a set of tensors, not one opaque file.** It gives every tensor an id (the SHA-256 of its bytes), shows exactly which tensors two checkpoints disagree on and how (elements, rows, columns, rank), and stores a new revision as deltas against its parent that check out byte for byte. It reads safetensors with numpy alone, needs no PyTorch or GPU, and can diff two Hugging Face Hub repositories while downloading only the shards that differ.

Open-weight models change through fine-tunes, merges, adapters, abliteration, and, [since November 2026](https://rikiyat.github.io/blog/weight-space-programming/), direct edits written by LLM agents. The Hub records each change as new files and a model card that names a parent with [one of four words](https://huggingface.co/docs/hub/model-cards). A card that says `finetune` doesn't say which weights moved, and safety, interpretability, and supply-chain work all need to know.

<p align="center">
  <img src="docs/assets/demo.svg" width="100%" alt="A terminal runs wvc diff --rank hf:Qwen/Qwen3.5-9B hf:bowmanslayer/Qwen3.5-9B-Evil-BF16. Output: 4 shards per side, 3 byte-identical on the Hub and skipped, 1 fetched from each side; 775 tensors, 1 changed; 46,281,237 of 9,653,104,368 elements changed (0.4794%); the changed tensor is model.language_model.layers.17.mlp.down_proj.weight, BF16 [4096,12288], every row and column touched, rank 1, top-1 energy 0.9997.">
</p>
<p align="center"><sub>The command and output of <a href="experiments/E008-evil-qwen-hub-diff.md">E008</a>, re-run on 2026-11-03 with output byte-identical to the receipt. The animation compresses time: from the Hugging Face cache the run takes 15 to 40 s on an M4 Max; the first run also downloads the two shards that differ (10.67 GB).</sub></p>

## What it found

Two public weight changes and what `wvc diff` reports for them. The figure is drawn from the bytes of both checkpoints by [`make_deltas.py`](docs/assets/make_deltas.py), which first checks every count against the receipts.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/deltas-dark.png">
  <img src="docs/assets/deltas-light.png" width="100%" alt="Heat maps of |edited − base| for every changed tensor. E001: two 1024 × 3584 down-projection matrices of Qwen3.5-0.8B, in which only 64 thin columns changed. E008: one 4096 × 12288 down-projection matrix of Qwen3.5-9B, changed almost everywhere in a striped, rank-1 pattern.">
</picture>

| | [E001](experiments/E001-astra-edit-diff.md): an edit written by an LLM agent | [E008](experiments/E008-evil-qwen-hub-diff.md): an upload that names no tensor |
|---|---|---|
| **Pair** | `Qwen/Qwen3.5-0.8B` → `RikiyaT/Qwen3.5-0.8B-weight-space-programming` | `Qwen/Qwen3.5-9B` → `bowmanslayer/Qwen3.5-9B-Evil-BF16` |
| **The card says** | the two edited tensors and their hashes ([Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)) | `finetune`, with no method and no tensor named |
| **wvc finds** | 2 of 488 tensors: 64 of 3,584 columns of `mlp.down_proj` in layers 15 and 19, rank 1 each | 1 of 775 tensors: layer 17's `mlp.down_proj`, 92% of its weights, rank 1 (top-1 energy 0.9997) |
| **Check** | changed counts and hashes equal the authors' own manifest | none beyond the bytes: no method was published |
| **Read** | one 1.7 GB file per side; the diff took 9.8 s on an M4 Max | 1 of 4 shards per side: 10.67 of the pair's 38.61 GB |

A diff says where a model changed and in what shape, not what the change does. E008 ran no model and makes no claim about behaviour.

## Quick start

wvc is not on PyPI yet. Install it from a clone with [uv](https://docs.astral.sh/uv/) (Python 3.11 or newer):

```bash
git clone https://github.com/evnsnclr/wvc && cd wvc
uv sync --extra hf   # the hf extra reads Hub repositories; plain `uv sync` is enough for local files
```

Diff the two sub-kilobyte checkpoints that ship with the [spec's conformance vectors](spec/vectors/v0/README.md); nothing is downloaded:

```bash
uv run wvc diff --rank spec/vectors/v0/a.safetensors spec/vectors/v0/b.safetensors
```

```text
tensors: 4 (1 changed, 1 unchanged, 1 added, 1 removed, 0 reshaped)
elements changed: 4 of 32 (12.5000%)

status   tensor  dtype     shape    changed  rows  cols  max|d|  |d|/|a|  rank  top1
removed  h       F16       [2,2,2]  -        -     -     -       -        -     -
changed  w       F32       [4,6]    4/24     4     1     1       0.418    1     1.0000
added    x       None->I8  [3]      -        -     -     -       -        -     -
```

Then a real pair from the Hub (E001; the first run downloads 1.7 GB per side into the Hugging Face cache):

```bash
uv run wvc diff --rank hf:Qwen/Qwen3.5-0.8B hf:RikiyaT/Qwen3.5-0.8B-weight-space-programming
```

Keep revisions in a repository and get the exact files back:

```bash
export WVC_AUTHOR="Your Name"   # commits never record an author silently
uv run wvc init demo
uv run wvc commit spec/vectors/v0/a.safetensors -m "base" --op pretrain-checkpoint --repo demo
uv run wvc commit spec/vectors/v0/b.safetensors -m "add 1.0 to column 2 of w" --op direct-edit --repo demo
uv run wvc log --repo demo
uv run wvc checkout HEAD --out restored --repo demo
```

<details>
<summary>What that prints</summary>

Commit ids include the time and the author, so yours will differ.

```text
initialised an empty wvc repository in /path/to/wvc/demo/.wvc
commit b44ff44ddbce97c02cad0786047c939e1d54f70c25ca5d0320d5dd16d5cc0956 (main, no parent)
tensors: 3 (0 reused by hash, 3 stored full, 0 stored as delta)
payload bytes written: 448 in 0.01 s
commit 8461fce5701f28ae37aafd431ec8d2cdfa372be949f78105139078f715f68216 (main, parent b44ff44ddbce)
tensors: 3 (1 reused by hash, 1 stored full, 1 stored as delta: 1 delta-xor-planes)
payload bytes written: 364 in 0.01 s
8461fce5701f  2026-11-03T17:43:19Z  direct-edit          3 tensors  add 1.0 to column 2 of w
b44ff44ddbce  2026-11-03T17:43:19Z  pretrain-checkpoint  3 tensors  base
checked out 8461fce5701f into restored: 1 file, 355 bytes in 0.00 s
b.safetensors  355 bytes  sha256 ok
manifest root 226839773eb209266cbb929481b4e106d483eec17361879692c0bd87789b15cc verified against the files
```

</details>

`diff` exits 1 when the checkpoints differ, like `diff(1)`; `verify` exits 0 only when every tensor and the root match; every wvc error exits 2. Add `--json` for a machine-readable record.

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/how-dark.svg">
  <img src="docs/assets/how-light.svg" width="100%" alt="Three panels with real values from E001 and E003. wvc hash: each tensor's bytes get a SHA-256, and the sorted manifest lines get a root. wvc diff: of four layer-15 tensors only mlp.down_proj has a different id, so only it is decoded: 65,260 weights in 64 of 3,584 columns, rank 1. wvc commit and checkout: the edited revision reuses 486 tensors by id and stores 2 deltas of 513 KB; checkout writes a file whose SHA-256 equals the Hub's.">
</picture>

1. **Identity.** A tensor's id is the SHA-256 of its stored bytes; a manifest lists every tensor's name, dtype, shape, and id, and its root is a SHA-256 over those lines. Sharding, header order, and file names don't change the root ([spec §1](docs/spec-v0.md#1-tensor-identity-and-manifests-implemented)).
2. **Diff.** Tensors with equal ids are never decoded. The rest are decoded exactly (BF16, FP16, FP8, and the rest; a dtype that can't be decoded gets a byte-level count) and compared element by element: changed count, rows and columns touched, largest change, relative norm, and an SVD rank estimate. Across two Hub repositories, a shard whose name, size, and SHA-256 agree is skipped; its tensors are reported from its header, read by range request ([spec §5.5](docs/spec-v0.md#55-hub-checkpoints-implemented-2026-11-03)).
3. **Store.** `commit` keeps each tensor once by id and stores a changed tensor as an XOR-plane or sparse delta against its parent when that is smaller. It also records every file's header bytes and the non-tensor files a loader needs (config, tokenizer, chat template), so `checkout` rebuilds the original files and checks each one's SHA-256 before it appears ([spec §3](docs/spec-v0.md#3-repository-objects-and-deltas-implemented-2026-11-02)).

## Storage

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/storage-dark.svg">
  <img src="docs/assets/storage-light.svg" width="100%" alt="Dot plot from E002, log scale: bytes to store a revision given its parent, as a share of its raw tensor bytes. Sparse rank-1 edit: zstd 78.2%, best-case chunk dedup 0.66%, wvc delta 0.030%. Synthetic rank-8 merge: 77.8%, 3.51%, 1.67%. Consecutive pretraining checkpoints: 58.0%, 58.0%, 33.6%. Dense fine-tune: 77.9%, 77.9%, 56.3%.">
</picture>

Byte-level chunk dedup, fixed or content-defined, found nothing beyond whole-tensor identity on any of four public revision pairs ([E002](experiments/E002-storage-measurement.md)): a fine-tune rewrites bytes in place, and a 64 KiB chunk can't see one changed element. Numeric deltas can. On the sparse edit a delta needs 22× fewer bytes than tensor dedup alone (0.53 MB against 11.5 MB), on the synthetic rank-8 merge 2.1× fewer, and on dense updates it saves a modest 21% to 27% beyond byte-grouped compression. In a wvc repository the E001 edit costs 513 KB on top of a 1.245 GB base, and both revisions check out hash-identical to the Hub's files ([E003](experiments/E003-astra-edit-commit.md)).

## Commands

| Command | What it does |
|---|---|
| `wvc hash CKPT` | Print a manifest: every tensor's SHA-256 and a root. `--format gguf-hash` prints llama.cpp's `gguf-hash` lines instead. |
| `wvc diff A B` | Compare two checkpoints tensor by tensor. `--rank` estimates the rank of each change, `--all` lists unchanged tensors too. |
| `wvc verify CKPT MANIFEST` | Check a checkpoint against a manifest saved from `wvc hash`. |
| `wvc init`, `commit`, `log`, `show`, `checkout` | A local repository: tensors stored once by id, changes as deltas, files rebuilt byte for byte. |
| `wvc fsck` | Re-verify every payload, object, manifest, commit, and ref in a repository. |
| `wvc bisect GOOD BAD --test CMD` | Find the first commit where a behavioural test fails (`--sweep` tests every commit instead). |
| `wvc merge-adapter BASE ADAPTER` | Merge a PEFT LoRA adapter deterministically and store its targets as low-rank deltas. |
| `wvc remote`, `push`, `pull`, `clone` | Move commits between repositories: plain directories, or Hub repositories with the `hf` extra. |
| `wvc card REF` | Print model-card metadata that carries the manifest root. |

`CKPT` is a `.safetensors` file or a directory of shards; `hash`, `diff`, and `verify` also take a Hub repository as `hf:OWNER/NAME[@REVISION]`. Without `--repo`, repository commands use the nearest `.wvc/` above the current directory.

## Status

- **Run on real models:** `hash`, `diff`, and `verify`, on local files and on the Hub; `commit`, `checkout`, and `fsck`, with byte-identical files and a checkout that loads in transformers with identical logits; `bisect` on a real fine-tuning run, where the behaviour had no single onset and bisect said so. Each run has a [receipt](experiments/README.md).
- **Unit-tested, not yet run on real models:** `merge-adapter`, remotes (directory and Hub), `card`.
- **Not built:** reading GGUF (wvc only writes `gguf-hash` lines), signing and attestation, garbage collection, a release.
- **Not planned:** hosting, a UI, git filters, lossy storage, PyTorch at runtime.

CI runs the tests, ruff, and a type check on Ubuntu, macOS, and Windows with Python 3.11 and 3.13, with warnings as errors, a coverage floor, a dependency audit, and header-parser fuzzing. Every measurement so far comes from one machine (an Apple M4 Max), safetensors files, and the model families named in the receipts. What wvc refuses from untrusted input, and the limits it documents instead of defending, are in [SECURITY.md](SECURITY.md).

## Receipts

Each experiment has a record under [experiments/](experiments/README.md): the question, predictions written before the run, exact commands, input revisions and hashes, tool versions, raw outputs, and which predictions failed. The numbers in this README come from E001, E002, E003, and E008; the index lists every run, including the negative result.

## Contributing

wvc has one maintainer, and the most useful contributions right now are real checkpoint pairs that break it, use cases, and reviews. Read [CONTRIBUTING.md](CONTRIBUTING.md) first; participation follows the [Code of Conduct](CODE_OF_CONDUCT.md). Security reports go through GitHub's private vulnerability reporting; see [SECURITY.md](SECURITY.md).

The figures here are drawn from the receipts by [`docs/assets/make_figures.py`](docs/assets/make_figures.py) and, from the weights themselves, by [`docs/assets/make_deltas.py`](docs/assets/make_deltas.py).

## Citing

If wvc is useful in your work, please cite the repository; GitHub's "Cite this repository" button reads [CITATION.cff](CITATION.cff).

```bibtex
@software{smith2026wvc,
  author  = {Smith, Evan},
  title   = {{wvc}: version control for open model weights},
  year    = {2026},
  url     = {https://github.com/evnsnclr/wvc},
  license = {Apache-2.0}
}
```

## Licence

Apache-2.0; see [LICENSE](LICENSE). The models named here belong to their publishers. wvc reads their public files and redistributes none of them.
