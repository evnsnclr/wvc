"""Validation of the records the object store keeps (spec sections 3.2 to 3.4).

Pure functions without I/O: object records, manifests, and commit records are checked for the
keys and value types the spec requires. The store's exceptions, defined in the leaf
``wvc.errors``, are re-exported here and by ``wvc.store``. Messages name what is wrong, never
the offending value: records come from files, and a file's contents do not belong in an error
message. Where a number or a shape helps, it is quoted through ``wvc.quote.shown``, escaped
and cut to a bounded length.
"""

from __future__ import annotations

import math
import re

from wvc.canonical import object_id
from wvc.codec import FULL_FORM, LOWRANK_FORM, SPARSE_FORM, XOR_FORM
from wvc.errors import DeltaDepthError as DeltaDepthError  # re-exported
from wvc.errors import IntegrityError, StoreError, WvcError
from wvc.extras import commit_extras
from wvc.hashing import MANIFEST_FORMAT, manifest_root
from wvc.ids import is_count, is_id
from wvc.lowrank import FLOAT_DTYPES, PROCEDURE
from wvc.quote import shown
from wvc.safetensors_io import ITEMSIZES, MAX_DIMS, element_count, shape_text

OBJECT_FORMAT = "wvc-object/0"
COMMIT_FORMAT = "wvc-commit/0"
FORM_FIELDS = {
    FULL_FORM: ("plane_sizes",),
    XOR_FORM: ("base", "plane_sizes"),
    SPARSE_FORM: ("base", "index_size", "value_size"),
    LOWRANK_FORM: (
        "base", "base_dtype", "shape", "rank", "a_dtype", "a_shape", "b_dtype", "b_shape",
        "a_size", "scale", "procedure",
    ),
}  # fmt: skip
DECIMAL = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")  # scale text
OBJECT_KEYS = ("format", "id", "nbytes", "itemsize", "form", "payload", "depth")
OPERATION_KINDS = frozenset(
    {
        "pretrain-checkpoint", "finetune", "adapter-merge", "model-merge", "quantize",
        "convert", "direct-edit", "prune", "distill", "unknown",
    }
)  # fmt: skip
JSON_TYPE_NAMES = {
    dict: "object", list: "array", str: "string", bool: "boolean",
    int: "number", float: "number", type(None): "null",
}  # fmt: skip


class ManifestError(WvcError):
    """The manifest is not a well-formed ``wvc-manifest/0`` document."""


def check_id(value: object, what: str) -> str:
    """Return ``value`` if it is a SHA-256 id; StoreError otherwise."""
    if not is_id(value):
        raise StoreError(f"{what} must be 64 lowercase hex characters")
    return value


def expected_payload_size(record: dict) -> int:
    """The payload length a validated object record implies (spec section 3.2).

    ``sum(plane_sizes)`` for zstd-planes and delta-xor-planes, ``index_size + value_size`` for
    delta-sparse, and ``a_size`` plus B's byte length for delta-lowrank.
    """
    if "plane_sizes" in record:
        return sum(record["plane_sizes"])
    if "index_size" in record:
        return record["index_size"] + record["value_size"]
    return record["a_size"] + math.prod(record["b_shape"]) * ITEMSIZES[record["b_dtype"]]


def check_tensor_bytes(record: dict, limit: int, where: str) -> None:
    """StoreError when a validated object record claims more than ``limit`` bytes.

    The limit applies to the tensor (``nbytes``) and, for delta-lowrank, to each factor (A's
    ``a_size`` and B's byte length). A delta's base has the delta's ``nbytes``, which the
    store checks when it follows the chain. The claimed sizes are not quoted.
    """
    sizes = [record["nbytes"]]
    if record["form"] == LOWRANK_FORM:
        sizes += [record["a_size"], expected_payload_size(record) - record["a_size"]]  # A, B
    _require(
        max(sizes) <= limit,
        f"{where}: the record claims more than the tensor limit of {limit} bytes "
        "(max_tensor_bytes in .wvc/config.json)",
    )


def validate_object(record: object, where: str = "object record") -> None:
    """Check an object record's keys and value types, and a delta-lowrank's consistency.

    ``where`` names the record in messages (for a stored record, its file's tensor id).
    """
    if not isinstance(record, dict):
        raise StoreError(f"{where}: an object record must be a JSON object")
    form = record.get("form")
    if not isinstance(form, str) or form not in FORM_FIELDS:
        raise StoreError(f"{where}: unknown form; expected one of {sorted(FORM_FIELDS)}")
    keys = {*OBJECT_KEYS, *FORM_FIELDS[form]}
    missing, extra = len(keys - set(record)), len(set(record) - keys)
    _require(
        set(record) == keys,
        f"{where}: {form} needs the keys {sorted(keys)} ({missing} missing, {extra} unexpected)",
    )
    _require(record["format"] == OBJECT_FORMAT, f"{where}: format must be {OBJECT_FORMAT!r}")
    hex_keys = [key for key in ("id", "payload", "base") if key in keys]
    ids_ok = all(is_id(record[key]) for key in hex_keys)
    _require(ids_ok, f"{where}: {', '.join(hex_keys)} must be 64 lowercase hex characters")
    _require(record.get("base") != record["id"], f"{where}: an object cannot be its own base")
    _check_object_sizes(record, where)


def check_manifest(manifest: object, where: str) -> str:
    """Validate a manifest, recompute its root from its tensors, and return the root."""
    try:
        valid = validate_manifest(manifest, source=where)
    except ManifestError as exc:
        raise StoreError(str(exc)) from exc
    root = check_id(valid["root"], f"{where} root")
    computed = manifest_root(valid["tensors"])
    if computed != root:
        raise IntegrityError(f"{where}: root {root} does not match its tensors ({computed})")
    return root


def validate_manifest(data: object, source: str = "manifest") -> dict:
    """Return ``data`` if it is a well-formed manifest; otherwise raise ManifestError."""
    if not isinstance(data, dict):
        raise ManifestError(f"{source}: expected a JSON object, got {_json_type(data)}")
    if data.get("format") != MANIFEST_FORMAT:
        raise ManifestError(
            f"{source}: 'format' must be {MANIFEST_FORMAT!r}, got {shown(data.get('format'))}"
        )
    if not isinstance(data.get("root"), str):
        raise ManifestError(f"{source}: 'root' must be a string, got {shown(data.get('root'))}")
    tensors = data.get("tensors")
    if not isinstance(tensors, dict):
        raise ManifestError(f"{source}: 'tensors' must be a JSON object, got {_json_type(tensors)}")
    for name, entry in tensors.items():
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
            raise ManifestError(f"{source}: a tensor name contains a control character")
        _validate_entry(f"{source}: tensor {shown(name)}", entry)
    return data


def _validate_entry(where: str, entry: object) -> None:
    if not isinstance(entry, dict):
        raise ManifestError(f"{where}: entry must be a JSON object, got {_json_type(entry)}")
    dtype, shape, nbytes, sha256 = (entry.get(k) for k in ("dtype", "shape", "nbytes", "sha256"))
    if not isinstance(dtype, str):
        raise ManifestError(f"{where}: 'dtype' must be a string, got {shown(dtype)}")
    if not isinstance(shape, list):
        raise ManifestError(f"{where}: 'shape' must be a list, got {_json_type(shape)}")
    if len(shape) > MAX_DIMS:
        raise ManifestError(f"{where}: 'shape' has {len(shape)} dimensions, more than {MAX_DIMS}")
    if not all(is_count(dim) for dim in shape):
        raise ManifestError(
            f"{where}: 'shape' must be a list of non-negative integers, got {shape_text(shape)}"
        )
    if not is_count(nbytes):
        raise ManifestError(
            f"{where}: 'nbytes' must be a non-negative integer, got {shown(nbytes)}"
        )
    if dtype not in ITEMSIZES:
        raise ManifestError(f"{where}: unsupported dtype {shown(dtype)}")
    count = element_count(shape, nbytes)
    if count is None:
        raise ManifestError(
            f"{where}: shape {shape_text(shape)} has more elements than 'nbytes' "
            f"{shown(nbytes)} allows"
        )
    expected = count * ITEMSIZES[dtype]
    if nbytes != expected:
        raise ManifestError(
            f"{where}: 'nbytes' is {shown(nbytes)} but {dtype}{shape_text(shape)} holds "
            f"{shown(expected)} bytes"
        )
    if not is_id(sha256):
        raise ManifestError(f"{where}: 'sha256' must be 64 lowercase hex characters")


def _json_type(value: object) -> str:
    return JSON_TYPE_NAMES.get(type(value), type(value).__name__)


def validate_commit(record: object, where: str) -> None:
    """Check the commit fields v0 validates: format, manifest, parents, operation.kind, files.

    ``extras`` may be absent (older commits); when present it must pass
    ``wvc.extras.validate_extra_records`` (spec 3.3a), names checked against the files' names.
    """
    if not isinstance(record, dict):
        raise StoreError(f"{where}: must be a JSON object, got {type(record).__name__}")
    _require("id" not in record, f"{where}: must not contain 'id'; the id is computed")
    _require(record.get("format") == COMMIT_FORMAT, f"{where}: format must be {COMMIT_FORMAT!r}")
    _require(is_id(record.get("manifest")), f"{where}: manifest must be a 64-hex manifest root")
    parents = record.get("parents")
    _require(
        isinstance(parents, list) and all(map(is_id, parents)),
        f"{where}: parents must be a list of 64-hex commit ids",
    )
    operation = record.get("operation")
    kind = operation.get("kind") if isinstance(operation, dict) else None
    _require(
        isinstance(kind, str) and kind in OPERATION_KINDS,
        f"{where}: operation.kind must be one of {sorted(OPERATION_KINDS)}",
    )
    _require(isinstance(record.get("files"), list), f"{where}: files must be a list")
    if "extras" in record:
        commit_extras(record, where)


def commit_record_id(record: dict) -> str:
    """The id of a commit record: the SHA-256 of its canonical JSON (spec section 3.5)."""
    try:
        return object_id(record)
    except ValueError as exc:
        raise StoreError(f"commit record: {exc}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StoreError(message)


def _check_object_sizes(record: dict, where: str) -> None:
    itemsize, nbytes, depth = record["itemsize"], record["nbytes"], record["depth"]
    _require(is_count(itemsize) and itemsize > 0, f"{where}: itemsize must be positive")
    _require(is_count(nbytes) and nbytes % itemsize == 0, f"{where}: itemsize must divide nbytes")
    _require(
        is_count(depth) and (depth == 0) == (record["form"] == FULL_FORM),
        f"{where}: depth must be 0 for {FULL_FORM} and positive for a delta",
    )
    if "plane_sizes" in record:
        sizes = record["plane_sizes"]
        _require(
            isinstance(sizes, list) and len(sizes) == itemsize and all(map(is_count, sizes)),
            f"{where}: plane_sizes must list {itemsize} non-negative integers",
        )
    elif "index_size" in record:
        sizes = record["index_size"], record["value_size"]
        _require(all(map(is_count, sizes)), f"{where}: index_size and value_size must be counts")
    else:
        _check_lowrank(record, where)


def _check_lowrank(record: dict, where: str) -> None:
    """The fields of a delta-lowrank record (spec section 3.2) and their mutual consistency."""
    rank, shapes = record["rank"], [record[key] for key in ("shape", "a_shape", "b_shape")]
    _require(is_count(rank) and rank > 0, f"{where}: rank must be a positive integer")
    _require(
        all(map(_is_pair, shapes)),
        f"{where}: shape, a_shape, and b_shape must each list two non-negative integers",
    )
    (out_dim, in_dim), a_shape, b_shape = shapes
    _require(
        a_shape == [rank, in_dim] and b_shape == [out_dim, rank],
        f"{where}: a_shape {shown(a_shape)} and b_shape {shown(b_shape)} must be [rank, in] "
        f"and [out, rank] for shape {shown(shapes[0])} and rank {shown(rank)}",
    )
    dtypes = [record[key] for key in ("base_dtype", "a_dtype", "b_dtype")]
    _require(
        all(isinstance(dtype, str) and dtype in FLOAT_DTYPES for dtype in dtypes),
        f"{where}: base_dtype, a_dtype, and b_dtype must each be one of {', '.join(FLOAT_DTYPES)}",
    )
    _check_lowrank_sizes(record, where)
    _require(
        _is_finite_decimal(record["scale"]),
        f"{where}: scale must be the decimal text of a finite float",
    )
    _require(record["procedure"] == PROCEDURE, f"{where}: procedure must be {PROCEDURE!r}")


def _check_lowrank_sizes(record: dict, where: str) -> None:
    """itemsize and nbytes must fit base_dtype and shape, and a_size must be A's byte length.

    Element counts are bounded by the byte counts they must match (``element_count``).
    """
    nbytes, a_size, base_itemsize = record["nbytes"], record["a_size"], record["itemsize"]
    numel = element_count(record["shape"], nbytes)
    _require(
        base_itemsize == ITEMSIZES[record["base_dtype"]]
        and numel is not None
        and nbytes == numel * base_itemsize,
        f"{where}: itemsize and nbytes must match base_dtype {record['base_dtype']} and shape "
        f"{shown(record['shape'])}",
    )
    a_numel = element_count(record["a_shape"], a_size) if is_count(a_size) else None
    _require(
        a_numel is not None and a_size == a_numel * ITEMSIZES[record["a_dtype"]],
        f"{where}: a_size must be the byte length of A, {shown(record['a_shape'])} in "
        f"{record['a_dtype']}",
    )


def _is_pair(value: object) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(map(is_count, value))


def _is_finite_decimal(value: object) -> bool:
    """Decimal float text such as '2.0' or '1e-05' for a finite value; no 'nan' or 'inf'."""
    return (
        isinstance(value, str)
        and DECIMAL.fullmatch(value) is not None
        and math.isfinite(float(value))
    )
