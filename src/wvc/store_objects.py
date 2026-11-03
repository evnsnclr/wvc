"""Tensor objects without I/O: argument checks, decoding by form, and delta-chain checks.

Split out of ``wvc.store``, whose ``ObjectStore`` reads and writes what these functions check
and decode. ``decode_object`` rebuilds one level of a chain from its payload (and its decoded
base); ``object_chain`` follows a tensor's bases through a record lookup the store provides.
Neither checks a SHA-256: the store compares every decoded level with its id.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from wvc.codec import (
    LOWRANK_FORM,
    XOR_FORM,
    decode_lowrank,
    decode_planes,
    decode_sparse,
    decode_xor_planes,
)
from wvc.errors import IntegrityError, StoreError
from wvc.ids import RawBytes, is_count
from wvc.lowrank import (
    PROCEDURE,
    LowRankError,
    LowRankFactors,
    check_factor_shapes,
    check_scale,
    merge_lowrank,
)


def tensor_bytes(raw: RawBytes, itemsize: int) -> np.ndarray:
    """``raw`` as a contiguous 1-D uint8 array whose length the positive ``itemsize`` divides."""
    if not isinstance(raw, np.ndarray) or raw.dtype != np.uint8 or raw.ndim != 1:
        raise StoreError("tensor bytes must be a 1-D uint8 array")
    if not is_count(itemsize) or itemsize == 0 or raw.size % itemsize:
        raise StoreError(f"itemsize {itemsize!r} must be a positive integer dividing {raw.size}")
    return np.ascontiguousarray(raw)


def payload_view(data: RawBytes) -> np.ndarray:
    """Payload bytes as a contiguous 1-D uint8 array, without copying bytes-like input."""
    if isinstance(data, bytes | bytearray | memoryview):
        return np.frombuffer(data, dtype=np.uint8)
    if isinstance(data, np.ndarray) and data.dtype == np.uint8 and data.ndim == 1:
        return np.ascontiguousarray(data)
    raise StoreError("a payload must be bytes or a 1-D uint8 array")


def lowrank_fields(base_dtype: str, shape: Sequence[int], factors: LowRankFactors) -> dict:
    """The delta-lowrank record fields that describe the merge; StoreError on malformed input."""
    try:
        out_dim, in_dim, rank = check_factor_shapes(shape, factors.a_shape, factors.b_shape)
        scale_text = repr(check_scale(factors.scale))
    except LowRankError as exc:
        raise StoreError(f"cannot store a low-rank merge: {exc}") from exc
    return {
        "base_dtype": base_dtype,
        "shape": [out_dim, in_dim],
        "rank": rank,
        "a_dtype": factors.a_dtype,
        "a_shape": [rank, in_dim],
        "b_dtype": factors.b_dtype,
        "b_shape": [out_dim, rank],
        "scale": scale_text,
        "procedure": PROCEDURE,
    }


def merge_lowrank_record(record: dict, base: np.ndarray, payload: bytes) -> np.ndarray:
    """Rebuild a delta-lowrank tensor from its decoded base and its payload (spec section 3.2)."""
    a_raw, b_raw = decode_lowrank(payload, record["a_size"])
    factors = LowRankFactors(
        a_raw,
        record["a_dtype"],
        record["a_shape"],
        b_raw,
        record["b_dtype"],
        record["b_shape"],
        float(record["scale"]),
    )
    return merge_lowrank(base, record["base_dtype"], record["shape"], factors)


def decode_object(record: dict, base: np.ndarray | None, payload: bytes) -> np.ndarray:
    """One validated object's bytes from its payload and, for a delta, its decoded base.

    ``base`` is None for the full object that ends a chain (only deltas have a ``base``
    field). CodecError or LowRankError when the payload does not decode.
    """
    itemsize, form = record["itemsize"], record["form"]
    if base is None:
        return decode_planes(payload, record["plane_sizes"], itemsize, record["nbytes"])
    if form == XOR_FORM:
        return decode_xor_planes(base, payload, record["plane_sizes"], itemsize)
    if form == LOWRANK_FORM:
        return merge_lowrank_record(record, base, payload)
    sizes = record["index_size"], record["value_size"]
    return decode_sparse(base, payload, *sizes, itemsize)


def object_chain(
    tensor_id: str, get_object: Callable[[str], dict], depth_limit: Callable[[], int]
) -> list[dict]:
    """Object records from ``tensor_id`` down to the full object its deltas start from.

    Each record's depth must be its base's plus one and its nbytes its base's, no id may come
    back, and the chain may not be deeper than ``depth_limit()`` (IntegrityError). Records
    come from ``get_object``, which validates them.
    """
    chain, seen, limit = [get_object(tensor_id)], {tensor_id}, depth_limit()
    if chain[0]["depth"] > limit:
        raise IntegrityError(
            f"object {tensor_id}: its delta chain is {chain[0]['depth']} deep, over {limit}"
        )
    while "base" in chain[-1]:
        record, base_id = chain[-1], chain[-1]["base"]
        if base_id in seen:
            raise IntegrityError(f"object {tensor_id}: the delta chain revisits {base_id}")
        seen.add(base_id)
        base = get_object(base_id)
        if record["depth"] != base["depth"] + 1:
            raise IntegrityError(
                f"object {record['id']}: depth {record['depth']} is not its base's depth "
                f"{base['depth']} plus one"
            )
        if record["nbytes"] != base["nbytes"]:
            raise IntegrityError(f"object {record['id']}: nbytes differs from its base's")
        chain.append(base)
    return chain
