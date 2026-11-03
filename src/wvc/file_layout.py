"""File layout records of spec section 3.3: how a commit reproduces each source file exactly.

A record names one safetensors file and holds its header payload (the 8-byte size field and the
JSON header, verbatim), its tensor ids in data-buffer order, and the SHA-256 and size of the
whole file. The header followed by the tensors in that order is the file only when the tensors
tile the data buffer with no hole, overlap, or trailing byte, as the safetensors format
requires; ``data_order`` refuses any other file, so a commit never records a layout that a
checkout cannot reproduce.

File names are restricted to a set every common file system stores as given: letters, digits,
'.', '_', and '-', starting with a letter or digit, ending in ``.safetensors``, at most 200
bytes, and not a Windows device name; no two names of one commit may differ only in case or
Unicode normalisation. ``manifest_tensors`` rebuilds a commit's manifest entries from its
header payloads, parsed by ``read_header``'s parser, and its tensor ids, so that a record
whose files do not hold its manifest's tensors is refused before anything trusts it.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from wvc.errors import IntegrityError, StoreError
from wvc.fsio import input_digest
from wvc.ids import is_count, is_id
from wvc.quote import shown
from wvc.safetensors_io import (
    MAX_HEADER_BYTES,
    SIZE_FIELD_BYTES,
    SafetensorsError,
    SafetensorsFile,
    TensorInfo,
    offset_order,
    parse_payload,
)

FILE_KEYS = frozenset({"name", "header", "tensors", "sha256", "nbytes"})
PAYLOAD_LABEL = Path("header")  # names a header payload in the messages of parse_payload
MAX_PREFIX_BYTES = SIZE_FIELD_BYTES + MAX_HEADER_BYTES  # the longest header payload
FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors")
MAX_NAME_BYTES = 200  # leaves room for a temporary name on file systems with 255-byte names
WINDOWS_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"{port}{n}" for port in ("COM", "LPT") for n in range(1, 10))}
)
NAME_RULE = (
    "a file name must be letters, digits, '.', '_', or '-', start with a letter or digit, and "
    "end in .safetensors"
)


def data_order(file: SafetensorsFile, file_size: int) -> tuple[TensorInfo, ...]:
    """The file's tensors in data-buffer order, checked to fill the buffer exactly.

    Zero-size tensors at one offset are ordered by name. Raises SafetensorsError on a hole, an
    overlap, or bytes after the last tensor, which no byte-exact checkout could reproduce.
    """
    ordered = offset_order(file)
    cursor = 0
    for info in ordered:
        if info.begin != cursor:
            raise SafetensorsError(
                f"{file.path}: tensor {shown(info.name)} starts at byte {info.begin} of the data "
                f"buffer but the tensors before it end at byte {cursor} (a hole or an overlap); "
                "wvc commits only files whose tensors fill the buffer exactly"
            )
        cursor = info.end
    buffer_len = file_size - file.data_start
    if cursor != buffer_len:
        raise SafetensorsError(
            f"{file.path}: the tensors end at byte {cursor} of the {buffer_len}-byte data "
            "buffer; wvc commits only files whose tensors fill the buffer exactly"
        )
    return tuple(ordered)


def file_digest(path: Path) -> tuple[str, int]:
    """SHA-256 (lowercase hex) and size of a regular file (or a link to one), read in chunks."""
    return input_digest(path)


def name_problem(name: object) -> str | None:
    """Why ``name`` cannot name a committed file, or None when it can."""
    if not isinstance(name, str) or FILE_NAME.fullmatch(name) is None:
        return NAME_RULE
    return portable_problem(name)


def portable_problem(name: str) -> str | None:
    """Why a name of allowed characters is still not portable (length, device name, ending)."""
    if len(name.encode("utf-8")) > MAX_NAME_BYTES:
        return f"a file name must be at most {MAX_NAME_BYTES} bytes"
    if name.split(".", 1)[0].upper() in WINDOWS_DEVICES:
        return "the name is a device name on Windows"
    if name.endswith((".", " ")):
        return "a file name must not end in a dot or a space"
    return None


def name_key(name: str) -> str:
    """What two names share when one file system may store them as one file: NFC, casefolded."""
    return unicodedata.normalize("NFC", name).casefold()


def check_file_names(names: Sequence[object], where: str) -> None:
    """StoreError unless every name is allowed and no two collide after NFC and casefolding."""
    seen: dict[str, str] = {}
    for name in names:
        problem = name_problem(name)
        if problem is not None:
            raise StoreError(f"{where}: {shown(name)} is not an allowed file name: {problem}")
        text = str(name)  # name_problem refused anything but a str
        key = name_key(text)
        if key in seen:
            how = "share a name" if seen[key] == text else "differ only in case or normalisation"
            raise StoreError(
                f"{where}: two file records {how}: {shown(seen[key])} and {shown(text)}"
            )
        seen[key] = text


def validate_file_records(files: object, where: str) -> list[dict]:
    """Check a commit's file layout records and their names; return them (StoreError if bad)."""
    if not isinstance(files, list):
        raise StoreError(f"{where}: files must be a list")
    for index, entry in enumerate(files):
        _check_file_record(entry, f"{where}: files[{index}]")
    check_file_names([entry["name"] for entry in files], where)
    return files


def parse_header_payload(data: bytes, file_nbytes: int, where: str) -> tuple[TensorInfo, ...]:
    """The tensors of a header payload in data-buffer order, under ``read_header``'s rules.

    ``data`` is the start of a file of ``file_nbytes`` bytes: the size field and the JSON
    header, parsed by ``safetensors_io.parse_payload`` with the same parser as ``read_header``.
    Beyond that, the size field must give the payload's exact length and the tensors must tile
    the data buffer exactly (``data_order``). IntegrityError otherwise: the payload is not a
    safetensors prefix of such a file.
    """
    try:
        file = parse_payload(PAYLOAD_LABEL, data, file_nbytes)
        return data_order(file, file_nbytes)
    except (SafetensorsError, OverflowError) as exc:
        raise IntegrityError(f"{where}: not a safetensors header ({exc})") from exc


def manifest_tensors(
    files: Sequence[Mapping],
    header_payload: Callable[[str], bytes],
    get_object: Callable[[str], Mapping],
    where: str,
) -> dict[str, dict]:
    """Manifest entries rebuilt from validated file records (spec sections 1 and 3.3).

    Each header payload gives the names, dtypes, and shapes in data order; the record gives the
    tensor id at each position. Each object's ``nbytes`` must equal its header entry's span,
    each file's ``nbytes`` its header plus its tensors, and no tensor name may appear twice
    (IntegrityError otherwise).
    """
    tensors: dict[str, dict] = {}
    for entry in files:
        file_where = f"{where}: file {entry['name']}"
        header = header_payload(entry["header"])
        ordered = parse_header_payload(header, entry["nbytes"], file_where)
        if len(ordered) != len(entry["tensors"]):
            raise IntegrityError(
                f"{file_where}: the header lists {len(ordered)} tensors, the record "
                f"{len(entry['tensors'])}"
            )
        total = len(header)
        for info, tensor_id in zip(ordered, entry["tensors"], strict=True):
            total += _add_entry(tensors, info, get_object(tensor_id), file_where)
        if total != entry["nbytes"]:
            raise IntegrityError(
                f"{file_where}: header and tensors hold {total} bytes, the record {entry['nbytes']}"
            )
    return tensors


def _add_entry(tensors: dict[str, dict], info: TensorInfo, record: Mapping, where: str) -> int:
    """Add one manifest entry; return its byte count."""
    if record["nbytes"] != info.nbytes:
        raise IntegrityError(
            f"{where}: tensor {shown(info.name)} spans {info.nbytes} bytes but its object "
            f"{record['id']} holds {record['nbytes']}"
        )
    if info.name in tensors:
        raise IntegrityError(f"{where}: tensor {shown(info.name)} appears in two files")
    tensors[info.name] = {
        "dtype": info.dtype,
        "shape": list(info.shape),
        "nbytes": info.nbytes,
        "sha256": record["id"],
    }
    return info.nbytes


def _check_file_record(entry: object, where: str) -> None:
    if not isinstance(entry, dict) or set(entry) != FILE_KEYS:
        raise StoreError(f"{where}: a file record needs exactly the keys {sorted(FILE_KEYS)}")
    tensors = entry["tensors"]
    if not (
        is_id(entry["header"])
        and is_id(entry["sha256"])
        and isinstance(tensors, list)
        and all(map(is_id, tensors))
    ):
        raise StoreError(f"{where}: header, sha256, and every tensor must be 64-hex ids")
    if not is_count(entry["nbytes"]):
        raise StoreError(f"{where}: nbytes must be a non-negative integer")
