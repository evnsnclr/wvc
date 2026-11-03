# Contributing to wvc

wvc has one maintainer. Bug reports, use cases, reviews, and small focused pull requests are welcome. Open an issue before starting anything large. Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Scope and non-goals
- In scope (v0): `hash`, `diff`, `verify`, and a local object store with `init`, `commit`, `log`, `show`, and `checkout`; remotes as plain directories first, then the Hugging Face Hub and OCI registries. The formats are specified in [docs/spec-v0.md](docs/spec-v0.md).
- Non-goals: hosting, git filters, a UI, lossy-only storage (a checkout must reproduce bytes exactly), and runtime dependencies beyond numpy and zstandard (no PyTorch).
- Scope, identity, and licence are maintainer decisions; the chosen formats are in [docs/spec-v0.md](docs/spec-v0.md), and the reasoning behind each one is kept in the maintainer's research notes outside this repository. Work outside them needs an issue and a recorded decision first.

## Running the checks
CI runs these commands on Ubuntu, macOS, and Windows with Python 3.11 and 3.13.

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uvx ty@0.0.95 check src
uv run pytest -p randomly --cov=wvc --cov-report=term-missing --cov-fail-under=95
```

- The test run is in random order (pytest-randomly prints its seed in the header; `--randomly-seed=N` repeats an order), treats every warning as an error (`filterwarnings = ["error"]` in `pyproject.toml`), and fails below 95% line coverage. If a third-party warning appears on one platform, ignore that one message in `filterwarnings` rather than dropping the setting.
- `tests/test_header_fuzz.py` fuzzes both safetensors header parsers with Hypothesis (200 examples per property, a few seconds in all).
- On Ubuntu only, CI also runs `uvx codespell .` with the skip list in `[tool.codespell]` in `pyproject.toml`; words that are correct here go in its `ignore-words-list`.
- A `package` job runs `uv build`, `uvx twine check dist/*`, and `validate-pyproject`, and fails if the sdist holds anything beyond the package, its tests, the conformance vectors, and the top-level documents.
- An `audit` job runs `pip-audit` on the locked dependencies (`uv export` then `uvx pip-audit -r requirements.txt --disable-pip --no-deps`) on every push and every Monday at 06:00 UTC; it does not block the test matrix.
- CI installs with `uv sync --locked`, so commit `uv.lock` together with any dependency change. CI pins the `ty` version it runs; bump the pin in `.github/workflows/ci.yml` deliberately.
- Tests use small synthetic safetensors fixtures and must not download weights or need the network.
- Never commit weights or run artifacts (`*.safetensors`, `*.gguf`, `*.npz`, `models/`, `runs/`).

## Experiments and receipts
- Every experiment gets an ID (E001, E002, ...) and a receipt under [experiments/](experiments/): commands, input revisions and hashes, tool versions, the code revision, and outputs.
- Numbers are never restated in prose (README, docs, issues, pull requests) without a link to the receipt they come from. Receipts outrank summaries.
- Negative and failed results get receipts too.

## Commit messages
- Use `type: description`, where type is one of `feat`, `fix`, `docs`, `test`, `chore`, `refactor`, `perf`, or `ci`; for example `docs: record the code revision for E001`.
- No attribution trailers (no `Co-Authored-By:` or "Generated with" lines).

## Code style
- numpy and zstandard are the only runtime dependencies.
- Keep functions small and single-purpose.
- Validate explicitly at boundaries (file headers, manifests, CLI arguments) and raise clear errors.
- No `no `print` outside the CLI modules (`cli*.py`)
- Line length is 100; ruff enforces formatting and the lint rules in `pyproject.toml`.

## Licence
wvc is licensed under the [Apache License 2.0](LICENSE). Contributions are accepted under the same licence.
