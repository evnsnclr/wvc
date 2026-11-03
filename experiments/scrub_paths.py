"""Replace machine-specific paths in experiment receipts with portable forms.

Receipts record where inputs sat on the author's machine. Those paths carry a user name and
a home-directory layout that mean nothing to a reader, so this script rewrites the three
prefixes they use: the Hugging Face cache, the repository checkout, and the uv cache. It
rewrites only tracked text files under ``experiments/``, changes nothing else in a receipt
(no receipt hashes another receipt, which was checked on 2026-11-03), and can be rerun
whenever a new receipt lands. The check mode scans every tracked text file, so the release
workflow can refuse a tag that would publish a machine path anywhere::

    uv run python experiments/scrub_paths.py        # rewrite the receipts
    uv run python experiments/scrub_paths.py --check  # exit 1 when any tracked file holds one
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REWRITES = (
    ("/Users/evansmith/.cache/huggingface/hub", "~/.cache/huggingface/hub"),
    ("/Users/evansmith/Documents/ChatGPT/wvc", "<repo>"),
    ("/Users/evansmith/.cache/uv/archive-v0", "~/.cache/uv/archive-v0"),
    ("/Users/evansmith", "~"),
)


def tracked_text_files(scope: str) -> list[Path]:
    """Tracked files under ``scope`` that decode as UTF-8, this script and the lock excluded."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", scope], cwd=ROOT, capture_output=True, check=True
    )
    files = []
    for name in listing.stdout.decode().split("\0"):
        if not name or name in ("experiments/scrub_paths.py", "uv.lock"):
            continue
        path = ROOT / name
        try:
            path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        files.append(path)
    return files


def main(argv: list[str]) -> int:
    check = "--check" in argv
    remaining = 0
    for path in tracked_text_files("." if check else "experiments"):
        text = path.read_text(encoding="utf-8")
        new = text
        for old, replacement in REWRITES:
            new = new.replace(old, replacement)
        if new == text:
            continue
        remaining += 1
        if check:
            print(f"machine path in {path.relative_to(ROOT)}")
        else:
            path.write_text(new, encoding="utf-8")
            print(f"rewrote {path.relative_to(ROOT)}")
    if check and remaining:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
