"""lowrank-merge/0, the delta-lowrank payload and record, and ObjectStore.put_lowrank."""

import hashlib
import json
import struct

import numpy as np
import pytest
from helpers import bf16_bits_to_f32
from helpers import f32_to_bf16_bits as reference_bf16_bits

from wvc import lowrank
from wvc.codec import CodecError, decode_lowrank, encode_lowrank
from wvc.lowrank import LowRankError, LowRankFactors, f32_to_bf16_bits, merge_lowrank
from wvc.records import validate_object
from wvc.store import DeltaDepthError, IntegrityError, StoreError

OUT, IN, RANK = 24, 40, 4
DTYPES = ("F32", "F16", "BF16")
NUMPY_FOR = {"F32": "<f4", "F16": "<f2"}
ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2}
FACTOR_KEYS = ("a_raw", "a_dtype", "a_shape", "b_raw", "b_dtype", "b_shape", "scale")
LOWRANK_KEYS = {
    "format", "id", "nbytes", "itemsize", "form", "payload", "depth", "base", "base_dtype",
    "shape", "rank", "a_dtype", "a_shape", "b_dtype", "b_shape", "a_size", "scale", "procedure",
}  # fmt: skip


def to_bytes(values, dtype):
    """Values rounded to float32 and then to ``dtype`` (nearest even), as 1-D uint8 bytes."""
    values = np.asarray(values, dtype=np.float32)
    if dtype == "BF16":
        encoded = reference_bf16_bits(values).astype("<u2")
    else:
        encoded = values.astype(NUMPY_FOR[dtype])
    return np.ascontiguousarray(encoded).reshape(-1).view(np.uint8)


def to_f64(raw, dtype, shape):
    """Decode raw bytes to float64 without wvc."""
    if dtype == "BF16":
        values = bf16_bits_to_f32(np.frombuffer(raw, dtype="<u2"))
    else:
        values = np.frombuffer(raw, dtype=NUMPY_FOR[dtype])
    return values.astype(np.float64).reshape(shape)


def operands(rng, base_dtype="BF16", a_dtype="F32", b_dtype="F32", out=OUT, inn=IN, rank=RANK):
    """A random W and small random factors, as merge's and put_lowrank's arguments."""
    return {
        "base_raw": to_bytes(rng.standard_normal((out, inn)), base_dtype),
        "base_dtype": base_dtype,
        "shape": (out, inn),
        "a_raw": to_bytes(0.05 * rng.standard_normal((rank, inn)), a_dtype),
        "a_dtype": a_dtype,
        "a_shape": (rank, inn),
        "b_raw": to_bytes(0.05 * rng.standard_normal((out, rank)), b_dtype),
        "b_dtype": b_dtype,
        "b_shape": (out, rank),
        "scale": 2.0,
    }


def factors_of(args):
    """The LowRankFactors that ``args`` (as ``operands`` returns them) describe."""
    return LowRankFactors(**{key: args[key] for key in FACTOR_KEYS})


def merge(args):
    """merge_lowrank for the merge ``args`` describe."""
    return merge_lowrank(args["base_raw"], args["base_dtype"], args["shape"], factors_of(args))


def spacing(values, dtype):
    """The gap between adjacent ``dtype`` values at each magnitude (one unit in the last place)."""
    magnitude = np.abs(values)
    if dtype == "F16":
        return np.spacing(magnitude.astype(np.float16)).astype(np.float64)
    gap = np.spacing(magnitude.astype(np.float32)).astype(np.float64)
    return gap * 2.0**16 if dtype == "BF16" else gap  # bfloat16 keeps 16 fewer mantissa bits


@pytest.mark.parametrize("base_dtype", DTYPES)
@pytest.mark.parametrize(("a_dtype", "b_dtype"), [("F32", "F32"), ("BF16", "F16")])
def test_merge_matches_a_float64_reference(rng, base_dtype, a_dtype, b_dtype):
    args = operands(rng, base_dtype, a_dtype, b_dtype)
    merged = merge(args)
    assert merged.dtype == np.uint8
    assert merged.ndim == 1
    assert merged.size == OUT * IN * ITEMSIZE[base_dtype]
    w = to_f64(args["base_raw"], base_dtype, (OUT, IN))
    a = to_f64(args["a_raw"], a_dtype, (RANK, IN))
    b = to_f64(args["b_raw"], b_dtype, (OUT, RANK))
    reference = w + 2.0 * (b @ a)
    result = to_f64(merged, base_dtype, (OUT, IN))
    # Half a unit in the last place of the base dtype. F16 and BF16 results are rounded twice,
    # to float32 and then to the base dtype, as the procedure specifies, which can add half a
    # float32 unit; the float64 sums themselves differ from the reference's only by rounding.
    tolerance = 0.5 * spacing(result, base_dtype)
    if base_dtype != "F32":
        tolerance += 0.5 * spacing(reference, "F32")
    magnitude = np.abs(w) + 2.0 * (np.abs(b) @ np.abs(a))
    tolerance += (RANK + 2) * np.finfo(np.float64).eps * magnitude
    assert np.all(np.abs(result - reference) <= tolerance)
    changed = result != w
    assert changed.mean() > 0.5  # the factors move most elements


def spec_merge(args):
    """lowrank-merge/0 transcribed element by element with Python floats (IEEE doubles)."""
    out_dim, in_dim = args["shape"]
    w = to_f64(args["base_raw"], args["base_dtype"], args["shape"]).tolist()
    a = to_f64(args["a_raw"], args["a_dtype"], args["a_shape"]).tolist()
    b = to_f64(args["b_raw"], args["b_dtype"], args["b_shape"]).tolist()
    encoded = bytearray()
    for i in range(out_dim):
        for j in range(in_dim):
            d = 0.0  # the accumulator starts at +0.0
            for k in range(len(a)):
                d = d + b[i][k] * a[k][j]
            f32 = struct.pack("<f", w[i][j] + args["scale"] * d)  # round to nearest even
            encoded += _from_float32(f32, args["base_dtype"])
    return bytes(encoded)


def _from_float32(f32, dtype):
    if dtype == "F32":
        return f32
    if dtype == "F16":
        return struct.pack("<e", struct.unpack("<f", f32)[0])  # round to nearest even
    bits = struct.unpack("<I", f32)[0]
    return struct.pack("<H", ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF)


def crafted(args):
    """Inputs on which the accumulator's start and the order of k change the merged bytes."""
    w = to_f64(args["base_raw"], args["base_dtype"], args["shape"])
    a = to_f64(args["a_raw"], args["a_dtype"], args["a_shape"])
    b = to_f64(args["b_raw"], args["b_dtype"], args["b_shape"])
    w[0], b[0] = -0.0, -0.0  # D[0, 1] sums -0.0 terms: +0.0 from a +0.0 start, else -0.0
    a[:, 1] = np.abs(a[:, 1]) + 0.5
    a[:, 0], b[1], b[2:, 1:] = [1.0, 2.0**60, -(2.0**60)], 1.0, 0.0  # D[1, 0]: 1 + 2**60 - 2**60
    return {
        **args,
        "base_raw": to_bytes(w, args["base_dtype"]),
        "a_raw": to_bytes(a, args["a_dtype"]),
        "b_raw": to_bytes(b, args["b_dtype"]),
    }


@pytest.mark.parametrize("base_dtype", DTYPES)
def test_merge_follows_the_procedure_bit_for_bit(rng, base_dtype):
    args = operands(rng, base_dtype, "BF16", "F32", out=5, inn=7, rank=3)
    assert merge(args).tobytes() == spec_merge(args)
    args = crafted(args)
    merged = merge(args)
    assert merged.tobytes() == spec_merge(args)
    w, result = (to_f64(raw, base_dtype, (5, 7)) for raw in (args["base_raw"], merged))
    assert np.signbit(w[0, 1])
    assert not np.signbit(result[0, 1])  # -0.0 + 2 * (+0.0)
    assert result[1, 0] == w[1, 0]  # k ascending: (1 + 2**60) - 2**60 is 0 in float64


def test_merge_is_deterministic_and_independent_of_blocking(rng, monkeypatch):
    args = operands(rng, "BF16", "F32", "BF16")
    first = merge(args)
    assert np.array_equal(first, merge(args))
    for elements in (1, 5 * IN, OUT * IN + 1):  # one row, five rows with a remainder, all rows
        monkeypatch.setattr(lowrank, "BLOCK_ELEMENTS", elements)
        assert np.array_equal(first, merge(args))


def test_merge_accepts_bytes_and_empty_tensors(rng):
    args = operands(rng, "F16", "F32", "F32")
    as_bytes = {key: v.tobytes() if isinstance(v, np.ndarray) else v for key, v in args.items()}
    assert np.array_equal(merge(as_bytes), merge(args))
    factors = LowRankFactors(bytes(16), "F16", (2, 4), b"", "F32", (0, 2), 1.0)
    empty = merge_lowrank(b"", "BF16", (0, 4), factors)
    assert empty.dtype == np.uint8
    assert empty.size == 0
    factors = LowRankFactors(b"", "F32", (2, 0), bytes(24), "F32", (3, 2), 1.0)
    empty = merge_lowrank(b"", "F32", (3, 0), factors)
    assert empty.size == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"base_dtype": "I8"}, "W: dtype 'I8' is not one of F32, F16, BF16"),
        ({"a_dtype": "F64"}, "A: dtype 'F64'"),
        ({"b_dtype": None}, "B: dtype None"),
        ({"a_shape": (0, IN), "b_shape": (OUT, 0)}, "rank of at least 1"),
        ({"a_shape": (RANK, IN + 1)}, "do not factor W"),
        ({"b_shape": (OUT + 1, RANK)}, "do not factor W"),
        ({"b_shape": (OUT, RANK + 1)}, "do not factor W"),
        ({"shape": (OUT, IN, 1)}, "W: a shape must be two non-negative integers"),
        ({"a_shape": (RANK, -IN)}, "A: a shape must be two non-negative integers"),
        ({"b_shape": (OUT, True)}, "B: a shape must be two non-negative integers"),
        ({"scale": float("nan")}, "scale must be a finite number"),
        ({"scale": float("inf")}, "scale must be a finite number"),
        ({"scale": True}, "scale must be a finite number"),
        ({"scale": "2.0"}, "scale must be a finite number"),
        ({"scale": 10**400}, "scale must be a finite number"),
        ({"base_raw": bytes(10)}, "W: 10 bytes cannot hold BF16"),
        ({"a_raw": np.zeros((RANK, 4 * IN), np.uint8)}, "A: expected bytes or a 1-D uint8"),
        ({"b_raw": np.zeros(OUT * RANK, np.float32)}, "B: expected bytes or a 1-D uint8"),
    ],
)
def test_merge_refuses_malformed_input(rng, change, message):
    with pytest.raises(LowRankError, match=message.replace("[", r"\[")):
        merge({**operands(rng), **change})


def test_bf16_rounding_is_the_reference_bit_trick(rng):
    values = rng.standard_normal(4096).astype(np.float32) * np.float32(1e3)
    specials = np.array(
        [0.0, -0.0, np.inf, -np.inf, 3.389e38, -3.389e38, 1e-40, 1.0 + 2.0**-8, 1.0 + 3 * 2.0**-8],
        dtype=np.float32,
    )
    for array in (values, specials, values.reshape(64, 64)):
        assert np.array_equal(f32_to_bf16_bits(array), reference_bf16_bits(array))
    assert f32_to_bf16_bits(specials[-2:]).tolist() == [0x3F80, 0x3F82]  # ties go to even


def test_lowrank_payload_round_trip(rng):
    a = rng.integers(0, 256, size=24, dtype=np.uint8)
    b = rng.integers(0, 256, size=40, dtype=np.uint8)
    payload = encode_lowrank(a, b)
    assert payload == a.tobytes() + b.tobytes()
    assert encode_lowrank(a.tobytes(), bytearray(b.tobytes())) == payload
    assert decode_lowrank(payload, a.size) == (a.tobytes(), b.tobytes())
    assert decode_lowrank(payload, 0) == (b"", payload)
    assert decode_lowrank(payload, len(payload)) == (payload, b"")
    for bad in (-1, len(payload) + 1, True, 2.0, "24"):
        with pytest.raises(CodecError, match="a_size"):
            decode_lowrank(payload, bad)
    with pytest.raises(CodecError, match="1-D uint8"):
        encode_lowrank(a.reshape(4, 6), b)


# The object store


def set_max_delta_depth(store, depth):
    """Through write_config: a store reads config.json once and keeps it (L1)."""
    store.write_config({"format": "wvc-repo/0", "max_delta_depth": depth})


def object_path(store, tensor_id):
    return store.dir / "objects" / tensor_id[:2] / f"{tensor_id}.json"


def payload_path(store, pid):
    return store.dir / "payloads" / pid[:2] / pid


def put_lowrank(store, args, raw, base_id, **change):
    """store.put_lowrank for the merge ``args`` describe, with some of them changed."""
    keys = {**args, **change}
    return store.put_lowrank(
        raw,
        ITEMSIZE[args["base_dtype"]],
        base_id=base_id,
        base_dtype=keys["base_dtype"],
        shape=keys["shape"],
        factors=factors_of(keys),
    )


def stored_merge(store, rng, **options):
    """Store a random base and its merge as a delta-lowrank; return args, base, merged, record."""
    args = operands(rng, **options)
    base = store.put_tensor(args["base_raw"], ITEMSIZE[args["base_dtype"]])
    merged = merge(args)
    return args, base, merged, put_lowrank(store, args, merged, base["id"])


def test_put_lowrank_stores_the_factors_and_reads_back(store, rng):
    args, base, merged, record = stored_merge(store, rng, a_dtype="F32", b_dtype="BF16")
    payload = args["a_raw"].tobytes() + args["b_raw"].tobytes()
    assert set(record) == LOWRANK_KEYS
    assert record == {
        "format": "wvc-object/0",
        "id": hashlib.sha256(merged.tobytes()).hexdigest(),
        "nbytes": merged.size,
        "itemsize": 2,
        "form": "delta-lowrank",
        "payload": hashlib.sha256(payload).hexdigest(),
        "depth": 1,
        "base": base["id"],
        "base_dtype": "BF16",
        "shape": [OUT, IN],
        "rank": RANK,
        "a_dtype": "F32",
        "a_shape": [RANK, IN],
        "b_dtype": "BF16",
        "b_shape": [OUT, RANK],
        "a_size": RANK * IN * 4,
        "scale": "2.0",
        "procedure": "lowrank-merge/0",
    }
    assert store.get_payload(record["payload"]) == payload
    assert store.get_object(record["id"]) == record
    data = store.read_tensor(record["id"])
    assert data.dtype == np.uint8
    assert np.array_equal(data, merged)
    assert put_lowrank(store, args, merged.copy(), base["id"]) == record  # already stored
    assert not list(store.dir.rglob("*.tmp"))


@pytest.mark.parametrize("base_dtype", DTYPES)
def test_put_lowrank_round_trips_every_base_dtype(store, rng, base_dtype):
    _, _, merged, record = stored_merge(store, rng, base_dtype=base_dtype, b_dtype="F16")
    assert (record["base_dtype"], record["itemsize"]) == (base_dtype, ITEMSIZE[base_dtype])
    assert np.array_equal(store.read_tensor(record["id"]), merged)


def test_scale_text_round_trips_the_float(store, rng):
    scale = 16 / 3**0.5  # a scale with many digits, as use_rslora gives
    args = {**operands(rng), "scale": scale}
    base = store.put_tensor(args["base_raw"], 2)
    merged = merge(args)
    record = put_lowrank(store, args, merged, base["id"])
    assert record["scale"] == repr(scale)
    assert float(record["scale"]) == scale
    assert np.array_equal(store.read_tensor(record["id"]), merged)


def test_a_lowrank_delta_on_a_delta_chain(store, rng):
    args = operands(rng, "F32", "F32", "F32")
    base_values = to_f64(args["base_raw"], "F32", (OUT, IN))
    first = store.put_tensor(to_bytes(base_values, "F32"), 4)
    edited = base_values.copy()
    edited[0, :8] += 1.0  # a sparse edit of the base
    second = store.put_tensor(to_bytes(edited, "F32"), 4, base_id=first["id"])
    assert (second["form"], second["depth"]) == ("delta-sparse", 1)
    args["base_raw"] = to_bytes(edited, "F32")
    merged = merge(args)
    record = put_lowrank(store, args, merged, second["id"])
    assert (record["base"], record["depth"]) == (second["id"], 2)
    assert np.array_equal(store.read_tensor(record["id"]), merged)


def test_tampered_scale_or_payload_fails_the_read(store, rng):
    args, _, merged, record = stored_merge(store, rng)
    path = object_path(store, record["id"])
    path.write_text(json.dumps({**record, "scale": "2.5"}), encoding="utf-8")
    with pytest.raises(IntegrityError, match="decoded bytes do not match the id"):
        store.read_tensor(record["id"])
    other = store.put_payload(args["b_raw"].tobytes() + args["a_raw"].tobytes())  # same size
    path.write_text(json.dumps({**record, "payload": other}), encoding="utf-8")
    with pytest.raises(IntegrityError, match="decoded bytes do not match the id"):
        store.read_tensor(record["id"])
    short = store.put_payload(args["a_raw"].tobytes() + args["b_raw"].tobytes()[:-4])
    path.write_text(json.dumps({**record, "payload": short}), encoding="utf-8")
    with pytest.raises(IntegrityError, match=r"holds \d+ bytes, not the \d+ its record implies"):
        store.read_tensor(record["id"])  # refused by its length before it is decoded
    path.write_text(json.dumps(record), encoding="utf-8")
    assert np.array_equal(store.read_tensor(record["id"]), merged)
    payload_path(store, record["payload"]).write_bytes(bytes(record["a_size"] + OUT * RANK * 4))
    with pytest.raises(IntegrityError, match="hash to"):
        store.read_tensor(record["id"])


def test_depth_limit_raises_and_stores_nothing(store, rng):
    set_max_delta_depth(store, 0)
    args = operands(rng)
    base = store.put_tensor(args["base_raw"], 2)
    merged = merge(args)
    with pytest.raises(DeltaDepthError, match="depth 1, over max_delta_depth 0") as raised:
        put_lowrank(store, args, merged, base["id"])
    assert isinstance(raised.value, StoreError)
    assert not object_path(store, hashlib.sha256(merged.tobytes()).hexdigest()).exists()
    set_max_delta_depth(store, 1)
    assert put_lowrank(store, args, merged, base["id"])["depth"] == 1


def test_a_merge_by_other_arithmetic_is_refused(store, rng):
    args = operands(rng)
    base = store.put_tensor(args["base_raw"], 2)
    merged = merge(args)
    before = sorted(store.dir.rglob("*"))
    with pytest.raises(StoreError, match="did not follow the procedure"):
        put_lowrank(store, args, merged, base["id"], scale=2.5)
    with pytest.raises(StoreError, match="did not follow the procedure"):
        put_lowrank(store, args, args["base_raw"].copy() ^ np.uint8(1), base["id"])
    assert sorted(store.dir.rglob("*")) == before


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"a_shape": (RANK, IN + 1)}, "do not factor W"),
        ({"scale": float("nan")}, "scale must be a finite number"),
        ({"a_dtype": "I8"}, "must each be one of F32, F16, BF16"),
        ({"a_raw": np.zeros(8, np.uint8)}, "a_size must be the byte length of A"),
        ({"b_raw": np.zeros(8, np.uint8)}, "8 bytes cannot hold F32"),
        ({"base_dtype": "F32"}, "itemsize and nbytes must match base_dtype F32"),
        ({"a_raw": "not bytes"}, "must be bytes or a 1-D uint8 array"),
    ],
)
def test_put_lowrank_refuses_malformed_input(store, rng, change, message):
    args = operands(rng)
    base = store.put_tensor(args["base_raw"], 2)
    merged = merge(args)
    with pytest.raises(StoreError, match=message):
        put_lowrank(store, args, merged, base["id"], **change)
    assert not object_path(store, hashlib.sha256(merged.tobytes()).hexdigest()).exists()


def test_put_lowrank_needs_a_stored_base_of_the_same_size(store, rng):
    args = operands(rng)
    merged = merge(args)
    with pytest.raises(StoreError, match="not stored"):
        put_lowrank(store, args, merged, "0" * 64)
    with pytest.raises(StoreError, match="64 lowercase hex"):
        put_lowrank(store, args, merged, "HEAD")
    shorter = store.put_tensor(args["base_raw"][:-2].copy(), 2)
    with pytest.raises(StoreError, match="holds 1918 bytes but the merged tensor 1920"):
        put_lowrank(store, args, merged, shorter["id"])


BAD_RECORDS = [
    ({"rank": 0}, "rank must be a positive integer"),
    ({"rank": True}, "rank must be a positive integer"),
    ({"shape": "24x40"}, "must each list two non-negative integers"),
    ({"a_shape": [RANK, IN, 1]}, "must each list two non-negative integers"),
    ({"b_shape": [OUT, -RANK]}, "must each list two non-negative integers"),
    ({"a_shape": [RANK + 1, IN]}, r"must be \[rank, in\] and \[out, rank\]"),
    ({"b_shape": [OUT, RANK + 1]}, r"must be \[rank, in\] and \[out, rank\]"),
    ({"rank": RANK + 1}, r"must be \[rank, in\] and \[out, rank\]"),
    ({"a_dtype": "F64"}, "must each be one of F32, F16, BF16"),
    ({"base_dtype": "I8"}, "must each be one of F32, F16, BF16"),
    ({"b_dtype": ["F32"]}, "must each be one of F32, F16, BF16"),
    ({"base_dtype": "F32"}, "itemsize and nbytes must match"),
    ({"a_size": RANK * IN * 2}, "a_size must be the byte length of A"),
    ({"a_size": -1}, "a_size must be the byte length of A"),
    ({"scale": 2.0}, "scale must be the decimal text of a finite float"),
    ({"scale": "nan"}, "scale must be the decimal text of a finite float"),
    ({"scale": "inf"}, "scale must be the decimal text of a finite float"),
    ({"scale": "1e400"}, "scale must be the decimal text of a finite float"),
    ({"scale": " 2.0"}, "scale must be the decimal text of a finite float"),
    ({"scale": "2_0"}, "scale must be the decimal text of a finite float"),
    ({"scale": "٢"}, "scale must be the decimal text of a finite float"),  # Arabic-Indic 2
    ({"procedure": "lowrank-merge/1"}, "procedure must be 'lowrank-merge/0'"),
    ({"depth": 0}, "depth must be 0 for zstd-planes and positive for a delta"),
    ({"plane_sizes": [1, 1]}, "delta-lowrank needs the keys"),
]


@pytest.mark.parametrize(("change", "message"), BAD_RECORDS)
def test_lowrank_records_are_validated(store, rng, change, message):
    _, _, _, record = stored_merge(store, rng)
    validate_object(record)
    with pytest.raises(StoreError, match=message):
        validate_object({**record, **change})


def test_lowrank_record_needs_every_field(store, rng):
    _, _, _, record = stored_merge(store, rng)
    for key in sorted(LOWRANK_KEYS):
        with pytest.raises(StoreError):
            validate_object({name: value for name, value in record.items() if name != key})


@pytest.mark.parametrize("text", ["2", "2.", ".5", "-1e-05", "+3.25E+2", "0.0", "-0.0"])
def test_scale_texts_that_parse_as_finite_floats_are_accepted(store, rng, text):
    _, _, _, record = stored_merge(store, rng)
    validate_object({**record, "scale": text})
