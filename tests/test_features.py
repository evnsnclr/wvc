"""The second pass: spectra against the rounding floor, and tensor-name conventions."""

import numpy as np
import pytest
from helpers import f32_to_bf16_bits, round_bf16

from wvc.diff import TensorDiff
from wvc.features import (
    MAX_GRAM_SIDE,
    coherence,
    hidden_size,
    spacing,
    tensor_features,
    wants_features,
)
from wvc.tensor_names import has_quant_suffix, is_embedding, is_output_side, layer_of


def _item(shape, dtype_a="BF16", dtype_b="BF16", status="changed", changed=1):
    return TensorDiff(
        "w", status, dtype_a, dtype_b, tuple(shape), tuple(shape), int(np.prod(shape)), changed
    )


def _raw(values, dtype):
    if dtype == "BF16":
        return np.frombuffer(f32_to_bf16_bits(values).tobytes(), dtype=np.uint8)
    return np.frombuffer(np.asarray(values, dtype="<f4").tobytes(), np.uint8)


def test_spacing_per_dtype():
    one = np.array([1.0])
    assert spacing(one, "BF16")[0] == 2.0**-7
    assert spacing(one, "F16")[0] == 2.0**-10
    assert spacing(one, "F32")[0] == 2.0**-23
    assert spacing(one, "F64")[0] == 2.0**-52
    assert spacing(one, "F8_E4M3")[0] == 0.0


def test_wants_features_only_for_comparable_matrices():
    assert wants_features(_item((4, 4)))
    assert not wants_features(_item((4,)))
    assert not wants_features(_item((4, 4), "C64", "C64"))
    assert wants_features(_item((4, 4), "BF16", "F32", status="reshaped"))
    assert not wants_features(_item((4, 4), "F32", "BF16", status="reshaped"))
    assert not wants_features(_item((4, 4), "BF16", "F32", status="reshaped", changed=0))
    assert not wants_features(_item((4, 4), "I32", "F32", status="reshaped"))
    assert wants_features(_item((4, 4), "BF16", "F16", status="reshaped"))
    assert not wants_features(_item((4, 4), status="unchanged"))


def test_hidden_size_is_the_most_shared_dimension():
    items = [_item((64, 128)), _item((128, 64)), _item((32, 64)), _item((7,))]
    assert hidden_size(items) == 64
    assert hidden_size([_item((5,))]) is None


def test_rank_four_change_is_measured_above_the_noise(rng):
    base = round_bf16(rng.standard_normal((96, 160)) * 0.02)
    delta = rng.standard_normal((96, 4)) @ rng.standard_normal((4, 160))
    edited = round_bf16(base + delta * 0.01 * np.linalg.norm(base) / np.linalg.norm(delta))
    found = tensor_features(_raw(base, "BF16"), _raw(edited, "BF16"), _item((96, 160)), 96)
    assert found is not None
    assert (found.signal_rank, found.energy_rank) == (4, 4)
    assert found.snr is not None
    assert found.snr > 10
    assert found.direction is not None
    assert found.direction.shape == (96,)
    assert len(found.to_dict()["sigma"]) == 16


def test_wide_side_direction_comes_from_the_kept_delta(rng):
    base = rng.standard_normal((40, 24)).astype(np.float32)
    v = rng.standard_normal(24)
    edited = base + np.outer(rng.standard_normal(40), v).astype(np.float32)
    found = tensor_features(
        _raw(base, "F32"), _raw(edited, "F32"), _item((40, 24), "F32", "F32"), 40
    )
    assert found is not None
    assert found.direction is not None
    assert found.direction.shape == (40,)
    found = tensor_features(
        _raw(base, "F32"), _raw(edited, "F32"), _item((40, 24), "F32", "F32"), 7
    )
    assert found is not None
    assert found.direction is None
    found = tensor_features(
        _raw(base, "F32"), _raw(edited, "F32"), _item((40, 24), "F32", "F32"), 40, cap=10
    )
    assert found is not None
    assert found.direction is None  # the delta was not kept
    assert found.signal_rank == 1


def test_unmeasurable_tensors_give_none(rng):
    base = rng.standard_normal((4, 6)).astype(np.float32)
    nan = base.copy()
    nan[0, 0] = np.nan
    item = _item((4, 6), "F32", "F32")
    assert tensor_features(_raw(base, "F32"), _raw(nan, "F32"), item, 4) is None
    assert tensor_features(_raw(base, "F32"), _raw(base, "F32"), item, 4) is None  # no change
    assert tensor_features(_raw(base, "F32"), _raw(base + 1, "F32"), item, 4, cap=1) is None
    assert tensor_features(b"", b"", TensorDiff("w", "changed"), 4) is None


def test_gram_side_is_capped(rng, monkeypatch):
    import wvc.features

    monkeypatch.setattr(wvc.features, "MAX_GRAM_SIDE", 2)
    base = rng.standard_normal((4, 6)).astype(np.float32)
    item = _item((4, 6), "F32", "F32")
    assert tensor_features(_raw(base, "F32"), _raw(base + 1, "F32"), item, 4) is None
    assert MAX_GRAM_SIDE == 8192


def test_coherence_needs_two_output_side_directions(rng):
    found = tensor_features(
        _raw(np.zeros((4, 6), np.float32), "F32"),
        _raw(np.ones((4, 6), np.float32), "F32"),
        _item((4, 6), "F32", "F32"),
        4,
    )
    assert found is not None
    assert coherence({"model.layers.0.self_attn.o_proj.weight": found}) == (None, 1, {})
    assert coherence({"model.layers.0.self_attn.q_proj.weight": found})[1] == 0
    value, count, alignment = coherence(
        {
            "model.layers.0.self_attn.o_proj.weight": found,
            "model.embed_tokens.weight": found,
        }
    )
    assert value == pytest.approx(1.0)
    assert count == 2
    assert all(a == pytest.approx(1.0) for a in alignment.values())


@pytest.mark.parametrize(
    ("name", "layer", "module"),
    [
        ("model.layers.12.mlp.down_proj.weight", 12, "mlp.down_proj.weight"),
        ("model.language_model.layers.3.self_attn.o_proj.weight", 3, "self_attn.o_proj.weight"),
        ("transformer.h.0.attn.c_proj.weight", 0, "attn.c_proj.weight"),
        ("gpt_neox.layers.1.attention.dense.weight", 1, "attention.dense.weight"),
        ("model.embed_tokens.weight", None, "model.embed_tokens.weight"),
    ],
)
def test_layer_of(name, layer, module):
    assert layer_of(name) == (layer, module)


def test_name_conventions():
    assert is_output_side("mlp.down_proj.weight")
    assert is_output_side("attention.dense.weight")
    assert is_output_side("attn.c_proj.bias")
    assert not is_output_side("mlp.up_proj.weight")
    assert not is_output_side("mlp.dense.weight")
    assert is_embedding("model.embed_tokens.weight")
    assert is_embedding("lm_head.weight")
    assert not is_embedding("model.layers.0.embed.weight")
    assert has_quant_suffix("model.layers.0.mlp.up_proj.qweight")
    assert has_quant_suffix("model.layers.0.mlp.up_proj.weight_scale")
    assert has_quant_suffix("model.layers.0.mlp.up_proj.act_scale")
    assert has_quant_suffix("model.layers.0.mlp.up_proj.weight.nested_absmax")
    assert has_quant_suffix("model.layers.0.mlp.up_proj.weight.quant_state.bitsandbytes__nf4")
    assert has_quant_suffix("model.layers.0.mlp.up_proj.weight_scale_inv")
    assert not has_quant_suffix("model.layers.0.mlp.up_proj.weight")


def test_widening_is_a_superset_of_values():
    from wvc.features import widens

    assert widens("BF16", "F32")
    assert widens("F8_E4M3", "F16")
    assert not widens("F8_E8M0", "F16")
    assert not widens("BF16", "F16")
    assert not widens("F16", "BF16")
    assert not widens("F32", "BF16")
    assert not widens(None, "F32")


def test_change_below_the_noise_edge_has_no_signal(rng):
    base = round_bf16(rng.standard_normal((32, 48)))
    edited = base.copy()
    edited[0, 0] = round_bf16(np.array([base[0, 0] * (1 + 2.0**-7)]))[0]  # one ulp
    found = tensor_features(_raw(base, "BF16"), _raw(edited, "BF16"), _item((32, 48)), 32)
    assert found is not None
    assert (found.signal_rank, found.energy_rank, found.signal_top1) == (0, 0, None)
    assert found.direction is None
