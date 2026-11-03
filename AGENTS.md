# wvc

Version control primitives for open model weights: tensor-level hashing, diffs, a content-addressed store with deltas, byte-exact checkout, bisect, and Hub remotes, for safetensors checkpoints, in Python with numpy and zstandard only. This file is for coding agents and the people driving them; humans should read [CONTRIBUTING.md](CONTRIBUTING.md) first, and [docs/spec-v0.md](docs/spec-v0.md) is the format contract.

## Commands

- Install: `uv sync --locked` (add `--extra hf` for Hub access). Never `pip install` into the environment by hand; change `pyproject.toml` and re-lock with `uv lock`.
- Tests: `uv run pytest -q`; one file or test: `uv run pytest tests/test_diff.py -q` or `uv run pytest tests/test_diff.py::test_name`. Prefer the targeted run while iterating, the full run before you finish. The suite is seeded and random-ordered; `--randomly-seed=N` repeats an order.
- Lint and format: `uv run ruff check .` and `uv run ruff format --check .` (`uv run ruff format .` to fix). Read ruff's exit code, not its last line.
- Types: `uvx ty@0.0.95 check src` (the version is pinned in `.github/workflows/ci.yml`).
- Spelling: `uvx codespell .`
- Receipts: `uv run python experiments/scrub_paths.py --check` must pass; it fails when any tracked file carries a machine path.

CI runs all of these on Ubuntu, macOS, and Windows with Python 3.11 and 3.13, with warnings as errors and a 95% coverage floor; a change is done when they pass locally.

## Where things are

- `src/wvc/`: `safetensors_io` (the one header parser and reader), `hashing` and `diff` (identity, manifests, comparisons), `codec` (zstd planes, XOR and sparse deltas, low-rank), `store` with `store_objects` and `store_config` (the `.wvc` object store), `records` (every record validator and the error types), `repo`, `checkout`, `extras`, `fsck`, `bisect`, `remote` and `transport` (directory and Hub remotes), `hub`, `cli` and `cli_*` (one module per command family, shared helpers in `cli_common`), `fsio` (every file-system primitive: nofollow reads, atomic writes, locks), `errors`, `ids`, `quote`, `output`.
- `spec/vectors/v0/`: two tiny checkpoints with their manifests and diff record. `tests/test_vectors.py` pins the formats to them.
- `tests/`: fixtures in `conftest.py`, helpers in `helpers.py` (import from `helpers`, not `conftest`), the fake Hub client there too. `tests/test_header_fuzz.py` fuzzes the header parser with Hypothesis.
- `experiments/`: one page and one directory per experiment; the receipts behind every number in the README. Raw data and stores are not tracked.

## Invariants

- A tensor's id is the SHA-256 of its stored bytes; a manifest root is the SHA-256 of its sorted lines; a checkout reproduces every file byte for byte. Any change to those bytes, to a record format, or to what a manifest line holds needs a new format version in the spec and new vectors, never a silent edit.
- Untrusted input makes wvc refuse; it never guesses. Every value from a file, a store, a remote, or a header is validated before use, sizes are checked before allocation or reading, and error messages name the key or the rule and never echo contents (`quote.shown` truncates what must be shown).
- Every wvc error derives from `wvc.errors.WvcError` and exits 2 from the CLI; a plain exception is a bug and propagates. Documents (`hash`, `--json`, `card`) go through `wvc.output` as UTF-8 bytes with LF; messages go through `print`.
- No PyTorch, no network, and no model download at runtime or in tests; the Hub is reached only through the `hf` extra and tested through the fake client. Weights never enter Git: the only tracked `.safetensors` files are the spec vectors.
- Receipts are never relabelled. A rerun gets a new receipt; a number in prose links to its receipt.

## Validating a change

Run the commands above. For anything touching `codec`, `store`, `checkout`, or `records`, also run `uv run pytest tests/test_vectors.py tests/test_repo_integrity.py tests/test_store_trust.py -q` first; they are the ones that catch a broken invariant. For anything touching file names, paths, or subprocesses, remember Windows: text-mode writes produce CRLF, a mapped or open file cannot be replaced, and `cmd.exe` quoting differs; the test suite shows the patterns (`replace_or_skip`, `newline="\n"`, `quote_argument`).

## Pull requests

- One change per PR, with a test for every behaviour change and a line under the next version in `CHANGELOG.md` when a user would notice.
- Commit messages: `type: description` with `feat`, `fix`, `docs`, `test`, `chore`, `refactor`, `perf`, or `ci`.
- Say what changed, why, and what you ran. If an agent wrote the PR, say so and link the issue it answers; do not open a competing PR for an issue someone is already working on.
- Do not reformat, rename, or "clean up" code the change does not need. Do not add dependencies.

If a `wiki/` directory is present, you are in the maintainer's lab checkout: read `wiki/operating-model.md` before anything else.
