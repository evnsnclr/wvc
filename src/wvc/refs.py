"""Branches, HEAD, and REF resolution of the object store (spec sections 3.1 and 3.6).

A branch is the file ``refs/heads/<name>`` holding a commit id and a newline; ``HEAD`` names
the current branch. A branch name is 1 to 100 letters, digits, '.', '_', or '-', starts with a
letter or digit, and does not end in ``.lock``. Writing a branch or moving HEAD also refuses
names that read as commit ids (7 to 64 hex characters, in either case) and names equal to
another branch's up to case. ``write_ref`` holds ``refs/heads/<name>.lock`` while it writes
and can compare-and-swap. ``resolve`` reads HEAD, then a full commit id, then a branch or a
unique commit prefix; a REF that both names a branch and prefixes a commit id is ambiguous.

``Refs`` is given what it needs of the store as callables; ``ObjectStore`` holds one and
exposes its operations as its own methods.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from wvc.fsio import exclusive_lock, read_regular
from wvc.ids import is_id, short
from wvc.records import StoreError


class Unset(Enum):
    """The type of ``ANY``, so that ``expected: str | None | Unset`` type-checks."""

    ANY = "any"


ANY = Unset.ANY  # write_ref's default: the branch may point anywhere before the write
BRANCH_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
HEX_NAME = re.compile(r"[0-9a-fA-F]{7,64}")  # would read as a commit id or a prefix of one
COMMIT_PREFIX = re.compile(r"[0-9a-f]{7,64}")  # a commit id or a prefix of at least 7 characters
HEAD_TEXT = re.compile(r"ref: refs/heads/(?P<branch>[^\n]*)\n")
LOCK_SUFFIX = ".lock"
HEAD_LIMIT = 128  # bytes: "ref: refs/heads/", a name of at most 100 characters, a newline
REF_LIMIT = 65  # bytes: 64 hex characters and a newline
NAME_RULE = (
    "use up to 100 letters, digits, '.', '_', or '-', starting with a letter or digit and not "
    "ending in .lock"
)


def is_branch_name(name: object) -> bool:
    """True for a syntactically valid branch name (see the module docstring)."""
    return (
        isinstance(name, str)
        and name != "HEAD"
        and BRANCH_NAME.fullmatch(name) is not None
        and not name.endswith(LOCK_SUFFIX)
    )


@dataclass(frozen=True)
class Refs:
    """The ref operations of one store, over what the store provides."""

    layout_path: Callable[..., Path]  # a path inside .wvc, its layout checked
    write: Callable[[Path, bytes], None]  # an atomic write into a store directory
    has_commit: Callable[[str], bool]  # whether a commit record is stored
    commit_ids: Callable[[str], list[str]]  # stored commit ids with a prefix, sorted

    def branches(self) -> list[str]:
        """The branch names under ``refs/heads``, sorted; lock and temporary files are skipped."""
        return sorted(filter(is_branch_name, os.listdir(self.layout_path("refs", "heads"))))

    def read_ref(self, name: str) -> str | None:
        """The commit id branch ``name`` points to, or None when the branch does not exist."""
        path = self._ref_path(name)
        try:
            data = read_regular(path, f"branch {name!r}", limit=REF_LIMIT)
        except FileNotFoundError:
            return None
        text = data.decode("ascii", errors="replace")
        if not (text.endswith("\n") and is_id(text[:-1])):
            raise StoreError(
                f"{path}: malformed ref (not 64 lowercase hex characters and a newline)"
            )
        return text[:-1]

    def write_ref(self, name: str, commit_id: str, *, expected: str | Unset | None = ANY) -> None:
        """Point branch ``name`` at a stored commit while holding ``refs/heads/<name>.lock``.

        With ``expected`` (a commit id, or None for a branch that must not exist yet), the
        branch is read again under the lock and left alone unless it still has that value
        (StoreError), so two commits racing on one branch cannot silently drop one another.
        """
        self.check_branch_name(name)
        path = self._ref_path(name)
        if not self.has_commit(commit_id):
            raise StoreError(f"cannot point {name!r} at {commit_id}: the commit is not stored")
        with exclusive_lock(path.with_name(name + LOCK_SUFFIX), f"branch {name!r}"):
            current = self.read_ref(name)
            if expected is not ANY and current != expected:
                raise StoreError(
                    f"branch {name!r} moved to {short(current)} meanwhile (it was at "
                    f"{short(expected)}), so it was left there; commit {commit_id} is stored "
                    "but on no branch"
                )
            self.write(path, f"{commit_id}\n".encode())

    def check_branch_name(self, name: str) -> None:
        """StoreError unless ``name`` may be written as a branch or made the current branch."""
        if not is_branch_name(name):
            raise StoreError(f"invalid branch name {name!r}: {NAME_RULE}")
        if HEX_NAME.fullmatch(name):
            raise StoreError(
                f"invalid branch name {name!r}: 7 to 64 hex characters read as a commit id"
            )
        clash = [other for other in self.branches() if other != name]
        clash = [other for other in clash if other.casefold() == name.casefold()]
        if clash:
            raise StoreError(
                f"invalid branch name {name!r}: it differs from branch {clash[0]!r} only in "
                "case, which case-insensitive file systems do not distinguish"
            )

    def current_branch(self) -> str:
        """The branch HEAD names."""
        path = self.layout_path("HEAD")
        try:
            data = read_regular(path, "HEAD", limit=HEAD_LIMIT)
        except FileNotFoundError as exc:
            raise StoreError(f"{path} is missing") from exc
        match = HEAD_TEXT.fullmatch(data.decode("utf-8", errors="replace"))
        if match is None or not is_branch_name(match["branch"]):
            raise StoreError(
                f"{path}: malformed HEAD (expected 'ref: refs/heads/<branch>' and a newline)"
            )
        return match["branch"]

    def set_head(self, branch: str) -> None:
        """Make ``branch`` the current branch; it need not have commits yet."""
        self.check_branch_name(branch)
        self.write(self.layout_path("HEAD"), f"ref: refs/heads/{branch}\n".encode())

    def head(self) -> str | None:
        """The commit HEAD's branch points to, or None before the branch's first commit."""
        return self.read_ref(self.current_branch())

    def resolve(self, ref: str) -> str:
        """Resolve HEAD, a full commit id, a branch, or a unique commit id prefix (>= 7 hex).

        Hex is accepted in either case. A full id is looked up as a commit before any branch
        and never resolves through one. A REF that names a branch and also prefixes a commit
        id is ambiguous (StoreError).
        """
        if ref == "HEAD":
            return self._head_commit()
        lowered = ref.lower()
        if is_id(lowered):
            if self.has_commit(lowered):
                return lowered
            raise StoreError(f"no commit matches {ref!r}")
        target = self.read_ref(ref) if is_branch_name(ref) else None
        matches = self.commit_ids(lowered) if COMMIT_PREFIX.fullmatch(lowered) else None
        if target is not None and matches:
            raise StoreError(
                f"ref {ref!r} is ambiguous: it names a branch and prefixes {len(matches)} "
                "commit id(s); use the full commit id"
            )
        if target is not None:
            return target
        if matches is None:
            raise StoreError(self._unknown_ref(ref))
        if not matches:
            raise StoreError(f"no commit matches {ref!r}")
        if len(matches) > 1:
            raise StoreError(f"commit prefix {ref!r} is ambiguous: {len(matches)} commits match")
        return matches[0]

    def _head_commit(self) -> str:
        head = self.head()
        if head is None:
            raise StoreError(f"HEAD: branch {self.current_branch()!r} has no commits yet")
        return head

    def _unknown_ref(self, ref: str) -> str:
        if is_branch_name(ref) and ref == self.current_branch():
            return f"branch {ref!r} has no commits yet"
        return (
            f"unknown ref {ref!r}: not HEAD, a branch, or a commit id or prefix of at least "
            "7 hex characters"
        )

    def _ref_path(self, name: str) -> Path:
        if not is_branch_name(name):
            raise StoreError(f"invalid branch name {name!r}: {NAME_RULE}")
        return self.layout_path("refs", "heads", name)
