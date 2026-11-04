"""Build the public tree of wvc from this repository: an allowlisted copy with no history.

The working repository keeps research notes (``wiki/``), outreach drafts, reviews, and the
plan beside the code. The public repository carries only what a user or a contributor
needs: the package, its tests, the spec and conformance vectors, the experiment receipts,
the top-level documents, the figures, and the workflows. This script copies that subset
into a target directory, refuses to run if the target is not empty, and lists what it
copied, so the public history starts from a tree that was reviewed as a whole::

    uv run python tools/export_public.py --list            # what would be copied
    uv run python tools/export_public.py ../wvc-public     # copy it there

The target is then committed and pushed by hand to the public repository; the script is
rerun for every later sync, and ``git status`` in the target shows the differences.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Tracked paths that are public: directories end with a slash, files are exact.
PUBLIC = (
    "src/",
    "tests/",
    "spec/",
    "experiments/",
    "docs/spec-v0.md",
    "docs/classification.md",
    "docs/agents-reading-diffs.md",
    "docs/assets/",
    ".github/",
    "tools/export_public.py",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "ROADMAP.md",
    "SECURITY.md",
    "CITATION.cff",
    "AGENTS.md",
    "CLAUDE.md",
    "pyproject.toml",
    "uv.lock",
    ".gitattributes",
    ".gitignore",
)
# Tracked paths that must never be copied, named so a widened allowlist cannot admit them.
PRIVATE = (
    "wiki/",
    "docs/outreach/",
    "docs/reviews/",
    "docs/plan.md",
    "experiments/E007-demand-tests.md",  # a pre-registered outreach test; publicity would bias it
    "experiments/E007/",
    ".claude/",
    ".agents/",
)


def tracked() -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files", "-z"],  # noqa: S607  # git from PATH, as every developer has it
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return [name for name in listing.stdout.decode().split("\0") if name]


def is_public(name: str) -> bool:
    if any(name == p or name.startswith(p) for p in PRIVATE):
        return False
    return any(name == p or (p.endswith("/") and name.startswith(p)) for p in PUBLIC)


def main(argv: list[str]) -> int:
    names = tracked()
    chosen = [n for n in names if is_public(n)]
    left = [n for n in names if not is_public(n)]
    if "--list" in argv or len(argv) != 1:
        print("public:")
        print("\n".join(f"  {n}" for n in chosen))
        print("private:")
        print("\n".join(f"  {n}" for n in left))
        return 0 if "--list" in argv else 2
    target = Path(argv[0]).resolve()
    if target.exists() and any(p.name != ".git" for p in target.iterdir()):
        print(f"{target} is not empty; export into an empty directory", file=sys.stderr)
        return 2
    for name in chosen:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    print(f"copied {len(chosen)} files to {target}; {len(left)} tracked files stayed private")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
