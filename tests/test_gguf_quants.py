"""Dequantisation against llama.cpp's own Python package (``gguf``, a dev dependency only).

gguf-py's NumPy dequantisers implement llama.cpp's reference block layouts; wvc's must give
the same float32 bit patterns for every block, including random quant bits and reference-
quantised blocks. Q8_K has no gguf-py dequantiser, so it is checked against its definition
(``d * qs``) here and against the compiled C reference in E011.
"""

import numpy as np
import pytest
from gguf import GGMLQuantizationType, quants
from gguf_helpers import random_blocks

from wvc.gguf_io import TYPE_SIZES
from wvc.gguf_quants import DEQUANTIZERS, DequantError, can_dequantize, dequantize

REFERENCE_TYPES = [name for name in DEQUANTIZERS if name != "Q8_K"]
QUANTIZABLE = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"]  # gguf-py's reference quantisers
BLOCKS = 512


def _bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)


def _ours(raw: bytes | np.ndarray, name: str) -> np.ndarray:
    block, size = TYPE_SIZES[name]
    return dequantize(np.frombuffer(bytes(raw), dtype=np.uint8), name, block, size)


def _reference(raw: bytes | np.ndarray, name: str) -> np.ndarray:
    data = np.frombuffer(bytes(raw), dtype=np.uint8)
    with np.errstate(over="ignore", invalid="ignore"):
        return quants.dequantize(data, GGMLQuantizationType[name]).reshape(-1)


@pytest.mark.parametrize("name", REFERENCE_TYPES)
def test_random_blocks_match_gguf_py_bit_for_bit(name, rng):
    raw = random_blocks(name, BLOCKS, rng)
    assert np.array_equal(_bits(_ours(raw, name)), _bits(_reference(raw, name)))


@pytest.mark.parametrize("name", REFERENCE_TYPES)
def test_any_bytes_match_gguf_py_including_nan_and_infinite_scales(name, rng):
    size = TYPE_SIZES[name][1]
    raw = rng.integers(0, 256, BLOCKS * size, dtype=np.uint8).tobytes()
    ours, ref = _ours(raw, name), _reference(raw, name)
    assert np.isnan(ours).any()  # random float16 scales include NaN and infinity
    assert np.array_equal(ours, ref, equal_nan=True)


@pytest.mark.parametrize("name", QUANTIZABLE)
def test_reference_quantised_blocks_round_trip(name, rng):
    values = rng.standard_normal((8, 256)).astype(np.float32)
    raw = quants.quantize(values, GGMLQuantizationType[name])
    ours = _ours(raw, name)
    assert np.array_equal(_bits(ours), _bits(_reference(raw, name)))
    assert np.abs(ours - values.reshape(-1)).max() < 0.5  # a quantisation, not noise


def test_q8_k_is_its_scale_times_its_quants(rng):
    raw = np.frombuffer(random_blocks("Q8_K", BLOCKS, rng), dtype=np.uint8).reshape(BLOCKS, 292)
    scale = raw[:, 0:4].copy().view("<f4")
    quants_ = raw[:, 4:260].copy().view(np.int8).astype(np.float32)
    expected = (quants_ * scale).reshape(-1)  # bsums (bytes 260 to 292) do not enter
    assert np.array_equal(_bits(_ours(raw.tobytes(), "Q8_K")), _bits(expected))


def test_q4_0_follows_the_block_layout_by_hand():
    d = np.float16(0.5).tobytes()
    qs = bytes([0x21] + [0x88] * 15)  # element 0: low nibble 1; element 16: high nibble 2
    values = _ours(d + qs, "Q4_0")
    assert values[0] == (1 - 8) * 0.5
    assert values[16] == (2 - 8) * 0.5
    assert values[1] == 0.0
    assert values[17] == 0.0


def test_unsupported_types_are_refused_by_name():
    with pytest.raises(DequantError, match="cannot dequantise ggml type IQ2_XXS"):
        dequantize(np.zeros(66, dtype=np.uint8), "IQ2_XXS", 256, 66)
    assert not can_dequantize("IQ4_XS")
    assert can_dequantize("Q6_K")
    assert can_dequantize("BF16")
    assert can_dequantize("F16")


def test_a_partial_block_is_refused():
    with pytest.raises(DequantError, match="not a whole number of Q8_0 blocks"):
        dequantize(np.zeros(35, dtype=np.uint8), "Q8_0", 32, 34)
