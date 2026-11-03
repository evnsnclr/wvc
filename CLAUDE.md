@AGENTS.md

## Claude Code

- The shared rules are in AGENTS.md; change that file, not this one. Personal notes go in `CLAUDE.local.md`, which is ignored by Git.
- Use plan mode before changing anything under `src/wvc/codec.py`, `store*.py`, `records.py`, or `checkout.py`: those hold the on-disk format, and the spec and the conformance vectors have to move with them.
- When a task touches several unrelated modules, split it into one worktree per module group; the test suite is fast enough to run in each.
