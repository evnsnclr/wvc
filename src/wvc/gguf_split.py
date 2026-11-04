"""Split GGUF files: one checkpoint written as ``PREFIX-00001-of-0000N.gguf`` and its siblings.

llama.cpp's ``gguf-split`` (and gguf-py's writer with a split limit) names part ``i`` of ``n``
``PREFIX-<i:05d>-of-<n:05d>.gguf``, with ``i`` counted from 1, and writes three keys into every
part: ``split.no`` (UINT16, the part's index counted from 0), ``split.count`` (UINT16, ``n``),
and ``split.tensors.count`` (INT32, the tensors of all parts together). The first part carries
the model's metadata; the others carry the split keys and their own tensors.

``split_names`` gives every part name from any one of them. ``check_parts`` validates parsed
parts against those rules and refuses, naming the part and the key, a part whose keys are
missing, of the wrong type, or disagree with its name or with the other parts, and a tensor
total that is not the sum of the parts. A file whose name is not a part name but whose keys say
it is one of several parts is refused too: it is a renamed part, not a checkpoint.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from wvc.gguf_io import ARCHITECTURE_KEY, GgufError, GgufFile

SPLIT_NAME = re.compile(r"(?P<prefix>.+)-(?P<no>[0-9]{5})-of-(?P<count>[0-9]{5})\.gguf")
SPLIT_NO, SPLIT_COUNT, SPLIT_TENSORS = "split.no", "split.count", "split.tensors.count"
SPLIT_TYPES = {SPLIT_NO: "UINT16", SPLIT_COUNT: "UINT16", SPLIT_TENSORS: "INT32"}


def part_name(prefix: str, number: int, count: int) -> str:
    """The name of part ``number`` (counted from 1) of ``count``."""
    return f"{prefix}-{number:05d}-of-{count:05d}.gguf"


def split_names(name: str) -> list[str] | None:
    """Every part name of the split checkpoint ``name`` belongs to, or None for a plain file.

    A part number of 0 or above the count is refused: no writer makes such a name.
    """
    match = SPLIT_NAME.fullmatch(name)
    if match is None:
        return None
    number, count = int(match["no"]), int(match["count"])
    if not 1 <= number <= count:
        raise GgufError(f"{name}: part {number} of {count} is not a part of a split checkpoint")
    return [part_name(match["prefix"], index, count) for index in range(1, count + 1)]


def _split_value(file: GgufFile, key: str) -> int | None:
    """The integer value of a split key, None when absent; GgufError for the wrong type."""
    if key not in file.metadata:
        return None
    value = file.metadata[key]
    if file.value_types.get(key) != SPLIT_TYPES[key] or type(value) is not int:
        raise GgufError(f"{file.path}: {key} must be a {SPLIT_TYPES[key]}")
    return value


def check_single(file: GgufFile) -> None:
    """Refuse a file named as a whole checkpoint that says it is one of several parts."""
    count = _split_value(file, SPLIT_COUNT)
    if count is not None and count != 1:
        raise GgufError(
            f"{file.path}: {SPLIT_COUNT} is {count}, so this is one part of a split checkpoint, "
            "but its name is not a part name (PREFIX-00001-of-00002.gguf); refusing to read it "
            "as a whole checkpoint"
        )


def check_parts(parts: Sequence[GgufFile]) -> None:
    """Validate the parsed parts of one split checkpoint, given in part order.

    A single part without split keys (``x-00001-of-00001.gguf`` written by a tool that names
    every file so) is a whole checkpoint.
    """
    count = len(parts)
    if count == 1 and all(key not in parts[0].metadata for key in SPLIT_TYPES):
        return
    total = None
    for index, part in enumerate(parts):
        values = {key: _split_value(part, key) for key in SPLIT_TYPES}
        for key, value in values.items():
            if value is None:
                raise GgufError(f"{part.path}: part {index + 1} of {count} has no {key}")
        if values[SPLIT_NO] != index:
            raise GgufError(
                f"{part.path}: {SPLIT_NO} is {values[SPLIT_NO]}, but the name makes it part "
                f"{index + 1} of {count} ({SPLIT_NO} {index})"
            )
        if values[SPLIT_COUNT] != count:
            raise GgufError(
                f"{part.path}: {SPLIT_COUNT} is {values[SPLIT_COUNT]}, but the name says {count}"
            )
        if total is not None and values[SPLIT_TENSORS] != total:
            raise GgufError(
                f"{part.path}: {SPLIT_TENSORS} is {values[SPLIT_TENSORS]}, but part 1 says {total}"
            )
        total = values[SPLIT_TENSORS]
        architecture = part.metadata.get(ARCHITECTURE_KEY)
        if index and architecture is not None and architecture != parts[0].architecture:
            raise GgufError(f"{part.path}: {ARCHITECTURE_KEY} disagrees with part 1")
    found = sum(len(part.tensors) for part in parts)
    if found != total:
        raise GgufError(
            f"{parts[0].path}: {SPLIT_TENSORS} is {total}, but the {count} parts hold "
            f"{found} tensors"
        )
