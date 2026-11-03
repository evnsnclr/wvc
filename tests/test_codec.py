import numpy as np
import pytest
import zstandard
from helpers import f32_to_bf16_bits

from wvc import codec
from wvc.codec import (
    CodecError,
    choose_encoding,
    decode_planes,
    decode_sparse,
    decode_xor_planes,
    encode_planes,
    encode_sparse,
    encode_xor_planes,
    sparse_eligible,
)

ITEMSIZES = (1, 2, 4, 8)
NUMEL = 1000


def random_bytes(rng, nbytes):
    return rng.integers(0, 256, size=nbytes, dtype=np.uint8)


def bf16_bytes(values):
    """Bytes of float32 values rounded to bfloat16, as a 2-byte-per-element uint8 array."""
    return f32_to_bf16_bits(values).view(np.uint8)


def flip_elements(raw, itemsize, indices):
    """Copy ``raw`` with every byte of the listed elements inverted."""
    rows = raw.reshape(-1, itemsize).copy()
    rows[indices] ^= 0xFF
    return rows.reshape(-1)


def frames(payload, sizes):
    """Decompress consecutive zstd frames of the given sizes."""
    out, start = [], 0
    for size in sizes:
        out.append(zstandard.ZstdDecompressor().decompress(payload[start : start + size]))
        start += size
    return out


def decode(base, encoded, itemsize, nbytes):
    """Decode an Encoded value with the decoder its form names."""
    if encoded.form == "zstd-planes":
        return decode_planes(encoded.payload, encoded.fields["plane_sizes"], itemsize, nbytes)
    if encoded.form == "delta-xor-planes":
        return decode_xor_planes(base, encoded.payload, encoded.fields["plane_sizes"], itemsize)
    return decode_sparse(base, encoded.payload, **encoded.fields, itemsize=itemsize)


@pytest.mark.parametrize("itemsize", ITEMSIZES)
def test_planes_round_trip(rng, itemsize):
    raw = random_bytes(rng, NUMEL * itemsize)
    payload, sizes = encode_planes(raw, itemsize)
    assert len(sizes) == itemsize
    assert sum(sizes) == len(payload)
    decoded = decode_planes(payload, sizes, itemsize, raw.size)
    assert decoded.dtype == np.uint8
    assert decoded.flags.c_contiguous
    assert np.array_equal(decoded, raw)


@pytest.mark.parametrize("itemsize", ITEMSIZES)
def test_xor_planes_round_trip(rng, itemsize):
    base, raw = random_bytes(rng, NUMEL * itemsize), random_bytes(rng, NUMEL * itemsize)
    payload, sizes = encode_xor_planes(base, raw, itemsize)
    assert np.array_equal(decode_xor_planes(base, payload, sizes, itemsize), raw)


@pytest.mark.parametrize("itemsize", [*ITEMSIZES, 3])
def test_sparse_round_trip(rng, itemsize):
    base = random_bytes(rng, NUMEL * itemsize)
    raw = flip_elements(base, itemsize, [0, 7, 500, NUMEL - 1])
    payload, index_size, value_size = encode_sparse(base, raw, itemsize)
    assert index_size + value_size == len(payload)
    assert np.array_equal(decode_sparse(base, payload, index_size, value_size, itemsize), raw)


@pytest.mark.parametrize("itemsize", ITEMSIZES)
def test_empty_tensor_round_trips_in_every_form(itemsize):
    empty = np.empty(0, dtype=np.uint8)
    payload, sizes = encode_planes(empty, itemsize)
    assert decode_planes(payload, sizes, itemsize, 0).size == 0
    payload, sizes = encode_xor_planes(empty, empty, itemsize)
    assert decode_xor_planes(empty, payload, sizes, itemsize).size == 0
    payload, index_size, value_size = encode_sparse(empty, empty, itemsize)
    assert decode_sparse(empty, payload, index_size, value_size, itemsize).size == 0


def test_bf16_patterns_round_trip_in_every_form(rng):
    values = rng.standard_normal(4096).astype(np.float32)
    edited = values.copy()
    edited[::64] *= 1.5
    base, raw = bf16_bytes(values), bf16_bytes(edited)
    payload, sizes = encode_planes(raw, 2)
    assert np.array_equal(decode_planes(payload, sizes, 2, raw.size), raw)
    payload, sizes = encode_xor_planes(base, raw, 2)
    assert np.array_equal(decode_xor_planes(base, payload, sizes, 2), raw)
    assert sparse_eligible(base, raw, 2)
    payload, index_size, value_size = encode_sparse(base, raw, 2)
    assert np.array_equal(decode_sparse(base, payload, index_size, value_size, 2), raw)


def test_plane_j_holds_bytes_at_positions_congruent_to_j():
    raw = np.arange(12, dtype=np.uint8)
    payload, sizes = encode_planes(raw, 4)
    assert frames(payload, sizes) == [bytes([j, j + 4, j + 8]) for j in range(4)]


def test_sparse_payload_layout():
    base = np.zeros(8 * 2, dtype=np.uint8)
    raw = base.copy()
    raw[2:4] = [1, 2]  # element 1
    raw[10:12] = [3, 4]  # element 5
    payload, index_size, value_size = encode_sparse(base, raw, 2)
    indices, values = frames(payload, [index_size, value_size])
    assert indices == np.array([1, 5], dtype="<u4").tobytes()
    assert values == bytes([1, 2, 3, 4])


@pytest.mark.parametrize("itemsize", [2, 3])
def test_sparse_eligibility_is_five_percent_of_elements(rng, itemsize):
    base = random_bytes(rng, NUMEL * itemsize)
    at_limit = flip_elements(base, itemsize, np.arange(0, NUMEL, 20))  # 50 of 1000 elements
    over = flip_elements(base, itemsize, np.arange(51))
    assert sparse_eligible(base, at_limit, itemsize)
    assert not sparse_eligible(base, over, itemsize)


def test_one_changed_byte_counts_one_element(rng):
    base = random_bytes(rng, 4 * NUMEL)
    raw = base.copy()
    raw[: 4 * 50 : 4] ^= 1  # first byte of elements 0..49: 5%
    raw[1 : 4 * 50 : 4] ^= 1  # second byte of the same elements
    assert sparse_eligible(base, raw, 4)
    raw[4 * 50] ^= 1  # one more element
    assert not sparse_eligible(base, raw, 4)


def test_sparse_indices_must_fit_uint32(rng, monkeypatch):
    monkeypatch.setattr(codec, "UINT32_LIMIT", 100)  # stands in for 2**32
    base = random_bytes(rng, 2 * NUMEL)
    raw = flip_elements(base, 2, [10, 150])
    assert not sparse_eligible(base, raw, 2)
    with pytest.raises(CodecError, match="does not fit in uint32"):
        encode_sparse(base, raw, 2)
    assert choose_encoding(raw, 2, base).form != "delta-sparse"


def test_choose_encoding_picks_sparse_for_a_one_percent_edit(rng):
    values = rng.standard_normal(4096).astype(np.float32)
    edited = values.copy()
    changed = rng.choice(values.size, size=values.size // 100, replace=False)
    edited[changed] = rng.standard_normal(changed.size).astype(np.float32)
    base, raw = values.view(np.uint8), edited.view(np.uint8)
    encoded = choose_encoding(raw, 4, base)
    assert encoded.form == "delta-sparse"
    assert set(encoded.fields) == {"index_size", "value_size"}
    assert len(encoded.payload) < len(encode_xor_planes(base, raw, 4)[0])
    assert np.array_equal(decode(base, encoded, 4, raw.size), raw)


def test_choose_encoding_picks_xor_for_small_dense_noise(rng):
    values = rng.standard_normal(4096).astype(np.float32)
    noisy = values * (1 + 1e-3 * rng.standard_normal(values.size)).astype(np.float32)
    base, raw = values.view(np.uint8), noisy.view(np.uint8)
    assert not sparse_eligible(base, raw, 4)
    encoded = choose_encoding(raw, 4, base)
    assert encoded.form == "delta-xor-planes"
    assert set(encoded.fields) == {"plane_sizes"}
    assert len(encoded.payload) < len(encode_planes(raw, 4)[0])
    assert np.array_equal(decode(base, encoded, 4, raw.size), raw)


def test_choose_encoding_keeps_zstd_planes_for_unrelated_data(rng):
    base = bf16_bytes(rng.standard_normal(4096).astype(np.float32))
    raw = bf16_bytes(rng.standard_normal(4096).astype(np.float32))
    encoded = choose_encoding(raw, 2, base)
    assert encoded.form == "zstd-planes"
    assert set(encoded.fields) == {"plane_sizes"}
    assert np.array_equal(decode(base, encoded, 2, raw.size), raw)
    # Incompressible bytes give payloads of equal size; a delta must be strictly smaller.
    base, raw = random_bytes(rng, 8192), random_bytes(rng, 8192)
    encoded = choose_encoding(raw, 2, base)
    assert len(encode_xor_planes(base, raw, 2)[0]) == len(encoded.payload)
    assert encoded.form == "zstd-planes"


def test_choose_encoding_without_a_usable_base(rng):
    raw = random_bytes(rng, 64)
    assert choose_encoding(raw, 2).form == "zstd-planes"
    assert choose_encoding(raw, 2, raw[:32].copy()).form == "zstd-planes"


MALFORMED_PLANES = {  # (payload, plane sizes) of a valid 400-byte, itemsize-4 tensor -> bad args
    "sizes-short-of-payload": lambda p, s: (p[:-1], s, 4, 400),
    "wrong-plane-count": lambda p, s: (p, s[:3], 4, 400),
    "frames-cut-wrongly": lambda p, s: (p, [s[0] - 1, s[1] + 1, *s[2:]], 4, 400),
    "negative-size": lambda p, s: (p, [-1, s[1], s[2], s[3] + s[0] + 1], 4, 400),
    "other-length": lambda p, s: (p, s, 4, 404),
    "nbytes-not-multiple": lambda p, s: (p, s, 4, 401),
    "not-zstd": lambda p, s: (bytes(len(p)), s, 4, 400),
    "zero-itemsize": lambda p, s: (p, s, 0, 400),
}


@pytest.mark.parametrize("malform", MALFORMED_PLANES.values(), ids=MALFORMED_PLANES.keys())
def test_malformed_planes_raise_codec_error(rng, malform):
    payload, sizes = encode_planes(random_bytes(rng, 400), 4)
    with pytest.raises(CodecError):
        decode_planes(*malform(payload, sizes))


def test_malformed_deltas_raise_codec_error(rng):
    zc = zstandard.ZstdCompressor(level=3)
    base = random_bytes(rng, 2 * NUMEL)

    def sparse(indices, values):
        index_frame = zc.compress(np.asarray(indices, dtype="<u4").tobytes())
        value_frame = zc.compress(bytes(values))
        return index_frame + value_frame, len(index_frame), len(value_frame)

    payload, index_size, value_size = sparse([3, 9], [1, 2, 3, 4])
    with pytest.raises(CodecError, match="sum to"):
        decode_sparse(base, payload, index_size, value_size + 1, 2)
    with pytest.raises(CodecError, match="out of range"):
        decode_sparse(base, *sparse([3, NUMEL], [1, 2, 3, 4]), 2)
    with pytest.raises(CodecError, match="ascending"):
        decode_sparse(base, *sparse([9, 3], [1, 2, 3, 4]), 2)
    with pytest.raises(CodecError, match="expected 4"):
        decode_sparse(base, *sparse([3, 9], [1, 2]), 2)
    payload, sizes = encode_xor_planes(base, random_bytes(rng, base.size), 2)
    with pytest.raises(CodecError):
        decode_xor_planes(base[:-2], payload, sizes, 2)


def test_encoders_validate_their_arrays(rng):
    raw = random_bytes(rng, 64)
    with pytest.raises(CodecError, match="holds"):
        encode_xor_planes(raw[:32], raw, 2)
    with pytest.raises(CodecError, match="1-D uint8"):
        encode_planes(raw.view(np.uint16), 2)
    with pytest.raises(CodecError, match="multiple"):
        encode_planes(raw[:63], 2)
    with pytest.raises(CodecError, match="itemsize"):
        encode_sparse(raw, raw, 0)
