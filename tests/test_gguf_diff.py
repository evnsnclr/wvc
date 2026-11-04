"""Quantization comparisons: a GGUF file against its safetensors base (spec section 2.1)."""

import json

import numpy as np
import pytest
from gguf import GGMLQuantizationType, quants
from gguf_helpers import Tensor, Value, llama_permute, plain, random_blocks, write_gguf
from helpers import f32_to_bf16_bits, write_safetensors

from wvc.cli import main
from wvc.gguf_diff import compare_quantized
from wvc.gguf_io import load_gguf
from wvc.gguf_map import MAPPINGS, MappingError, rotary_rows
from wvc.gguf_quants import DequantError
from wvc.hashing import tensor_digests
from wvc.safetensors_io import load_checkpoint

HIDDEN, HEADS, KV_HEADS, FFN, VOCAB = 32, 4, 2, 64, 10
HEAD_DIM = HIDDEN // HEADS
LAYER = "model.layers.0."


def _base(rng):
    """A one-layer llama-shaped checkpoint in F32 under Hub names."""
    shapes = {
        "model.embed_tokens.weight": (VOCAB, HIDDEN),
        "model.norm.weight": (HIDDEN,),
        "lm_head.weight": (VOCAB, HIDDEN),
        LAYER + "input_layernorm.weight": (HIDDEN,),
        LAYER + "self_attn.q_proj.weight": (HEADS * HEAD_DIM, HIDDEN),
        LAYER + "self_attn.k_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
        LAYER + "self_attn.v_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
        LAYER + "self_attn.o_proj.weight": (HIDDEN, HEADS * HEAD_DIM),
        LAYER + "post_attention_layernorm.weight": (HIDDEN,),
        LAYER + "mlp.gate_proj.weight": (FFN, HIDDEN),
        LAYER + "mlp.up_proj.weight": (FFN, HIDDEN),
        LAYER + "mlp.down_proj.weight": (HIDDEN, FFN),
    }
    return {name: rng.standard_normal(shape).astype(np.float32) for name, shape in shapes.items()}


GGUF_NAMES = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
    LAYER + "input_layernorm.weight": "blk.0.attn_norm.weight",
    LAYER + "self_attn.q_proj.weight": "blk.0.attn_q.weight",
    LAYER + "self_attn.k_proj.weight": "blk.0.attn_k.weight",
    LAYER + "self_attn.v_proj.weight": "blk.0.attn_v.weight",
    LAYER + "self_attn.o_proj.weight": "blk.0.attn_output.weight",
    LAYER + "post_attention_layernorm.weight": "blk.0.ffn_norm.weight",
    LAYER + "mlp.gate_proj.weight": "blk.0.ffn_gate.weight",
    LAYER + "mlp.up_proj.weight": "blk.0.ffn_up.weight",
    LAYER + "mlp.down_proj.weight": "blk.0.ffn_down.weight",
}


def _metadata(architecture="llama"):
    return {
        "general.architecture": architecture,
        f"{architecture}.attention.head_count": HEADS,
        f"{architecture}.attention.head_count_kv": KV_HEADS,
    }


def _convert(base, *, permute=True, types=None):
    """The GGUF tensors llama.cpp's converter writes for ``base`` (llama: q/k permuted)."""
    types = types or {}
    tensors = {}
    for name, values in base.items():
        if permute and name.endswith("q_proj.weight"):
            values = llama_permute(values, HEADS, HEADS)
        if permute and name.endswith("k_proj.weight"):
            values = llama_permute(values, HEADS, KV_HEADS)
        dtype = types.get(name, "F32")
        if dtype == "F32":
            tensors[GGUF_NAMES[name]] = plain("F32", values)
        else:
            raw = quants.quantize(values, GGMLQuantizationType[dtype]).tobytes()
            tensors[GGUF_NAMES[name]] = Tensor(dtype, values.shape, raw)
    return tensors


def _pair(tmp_path, rng, *, architecture="llama", permute=True, types=None):
    base = _base(rng)
    st = write_safetensors(tmp_path / "base.safetensors", {k: ("F32", v) for k, v in base.items()})
    gguf = write_gguf(
        tmp_path / "q.gguf", _convert(base, permute=permute, types=types), _metadata(architecture)
    )
    return base, st, gguf


def _diff_json(capsys, a, b, *extra):
    status = main(["diff", "--json", "--all", *extra, str(a), str(b)])
    return status, json.loads(capsys.readouterr().out)


def test_an_exact_conversion_is_unchanged_after_undoing_the_permutation(tmp_path, rng, capsys):
    _, st, gguf = _pair(tmp_path, rng)
    status, record = _diff_json(capsys, st, gguf)
    assert status == 0
    assert record["summary"]["by_status"]["unchanged"] == len(GGUF_NAMES)
    by_name = {t["name"]: t for t in record["tensors"]}
    q = by_name[LAYER + "self_attn.q_proj.weight"]
    assert q["compared"] == "values"
    assert q["name_b"] == "blk.0.attn_q.weight"
    assert q["transform_b"] == f"unpermute-rotary(heads={HEADS})"
    assert by_name[LAYER + "self_attn.k_proj.weight"]["transform_b"] == (
        f"unpermute-rotary(heads={KV_HEADS})"
    )
    v = by_name[LAYER + "self_attn.v_proj.weight"]
    assert "compared" not in v
    assert "transform_b" not in v
    assert v["name_b"] == "blk.0.attn_v.weight"
    assert record["comparison"] == {
        "kind": "quantization",
        "gguf": ["b"],
        "mapping": "llama",
        "values": "float64",
    }
    assert record["b"]["format"] == "gguf"
    assert record["b"]["gguf"] == {
        "version": 3,
        "architecture": "llama",
        "alignment": 32,
        "types": {"F32": len(GGUF_NAMES)},
    }
    assert "gguf" not in record["a"]
    assert "format" not in record["a"]


def test_a_permutation_left_in_place_shows_as_a_change(tmp_path, rng, capsys):
    # The same permuted bytes declared as qwen2, whose converter does not permute.
    _, st, gguf = _pair(tmp_path, rng, architecture="qwen2")
    status, record = _diff_json(capsys, st, gguf)
    assert status == 1
    changed = {t["name"] for t in record["tensors"] if t["status"] == "changed"}
    assert changed == {LAYER + "self_attn.q_proj.weight", LAYER + "self_attn.k_proj.weight"}


def test_a_quantized_tensor_is_changed_with_statistics_on_dequantised_values(tmp_path, rng, capsys):
    name = LAYER + "mlp.down_proj.weight"
    base, st, gguf = _pair(tmp_path, rng, types={name: "Q8_0"})
    status, record = _diff_json(capsys, st, gguf)
    assert status == 1
    down = next(t for t in record["tensors"] if t["name"] == name)
    raw = quants.quantize(base[name], GGMLQuantizationType.Q8_0)
    dequantised = quants.dequantize(raw, GGMLQuantizationType.Q8_0).astype(np.float64)
    delta = dequantised - base[name].astype(np.float64)
    assert down["status"] == "changed"
    assert (down["dtype_a"], down["dtype_b"]) == ("F32", "Q8_0")
    assert down["compared"] == "values"
    assert down["changed"] == int((delta != 0).sum())
    assert down["max_abs_delta"] == float(np.abs(delta).max())
    assert down["delta_norm"] == pytest.approx(float(np.linalg.norm(delta)), rel=1e-12)
    assert record["b"]["gguf"]["types"] == {"F32": len(GGUF_NAMES) - 1, "Q8_0": 1}


def test_a_gguf_on_side_a_is_mapped_too(tmp_path, rng, capsys):
    _, st, gguf = _pair(tmp_path, rng, types={LAYER + "self_attn.q_proj.weight": "Q8_0"})
    status, record = _diff_json(capsys, gguf, st)
    assert status == 1
    assert record["comparison"]["gguf"] == ["a"]
    q = next(t for t in record["tensors"] if t["name"].endswith("q_proj.weight"))
    assert q["name_a"] == "blk.0.attn_q.weight"
    assert q["transform_a"] == f"unpermute-rotary(heads={HEADS})"
    assert (q["dtype_a"], q["dtype_b"]) == ("Q8_0", "F32")


def test_rank_is_estimated_on_a_dequantised_edit_after_unpermuting(tmp_path, rng):
    base = _base(rng)
    name = LAYER + "self_attn.q_proj.weight"
    edited = dict(base)
    edited[name] = base[name] + np.outer(rng.standard_normal(32), rng.standard_normal(32)).astype(
        np.float32
    )
    st = write_safetensors(tmp_path / "base.safetensors", {k: ("F32", v) for k, v in base.items()})
    gguf = write_gguf(tmp_path / "q.gguf", _convert(edited), _metadata())
    with load_checkpoint(st) as a, load_gguf(gguf) as b:
        diffs, _ = compare_quantized(a, b, tensor_digests(a), tensor_digests(b), rank=True)
    q = next(d for d in diffs if d.name == name)
    assert (q.status, q.rank, q.rows_touched, q.cols_touched) == ("changed", 1, 32, 32)
    assert q.top1_energy == pytest.approx(1.0)


def test_bf16_widened_to_f32_is_unchanged_by_value(tmp_path, rng, capsys):
    values = rng.standard_normal(HIDDEN).astype(np.float32)
    bits = f32_to_bf16_bits(values)
    widened = (bits.astype(np.uint32) << 16).view(np.float32)
    st = write_safetensors(tmp_path / "b.safetensors", {"model.norm.weight": ("BF16", bits)})
    gguf = write_gguf(tmp_path / "q.gguf", {"output_norm.weight": plain("F32", widened)},
                      _metadata())  # fmt: skip
    status, record = _diff_json(capsys, st, gguf)
    assert status == 0
    (norm,) = record["tensors"]
    assert (norm["status"], norm["compared"], norm["changed"]) == ("unchanged", "values", 0)
    assert (norm["dtype_a"], norm["dtype_b"]) == ("BF16", "F32")


def test_gemma3_norms_have_their_added_one_removed(tmp_path, rng, capsys):
    norm = rng.standard_normal(HIDDEN).astype(np.float32) * 0.25
    st = write_safetensors(
        tmp_path / "b.safetensors",
        {
            "model.layers.0.pre_feedforward_layernorm.weight": ("F32", norm),
            "model.layers.0.post_attention_layernorm.weight": ("F32", norm),
        },
    )
    gguf = write_gguf(
        tmp_path / "q.gguf",
        {
            "blk.0.ffn_norm.weight": plain("F32", norm + np.float32(1)),
            "blk.0.post_attention_norm.weight": plain("F32", norm + np.float32(1)),
        },
        _metadata("gemma3"),
    )
    status, record = _diff_json(capsys, st, gguf)
    by_name = {t["name"]: t for t in record["tensors"]}
    pre = by_name["model.layers.0.pre_feedforward_layernorm.weight"]
    assert pre["transform_b"] == "subtract(1)"
    assert pre["name_b"] == "blk.0.ffn_norm.weight"
    # float32 (x + 1) - 1 is not always x: what changed is exactly that rounding.
    expected = ((norm + np.float32(1)).astype(np.float64) - 1) != norm
    assert pre["changed"] == int(expected.sum())
    assert status == (1 if expected.any() else 0)


@pytest.mark.parametrize(
    ("architecture", "gguf_name", "hub_name"),
    [
        ("qwen3", "blk.3.attn_q_norm.weight", "model.layers.3.self_attn.q_norm.weight"),
        ("qwen2", "blk.0.attn_q.bias", "model.layers.0.self_attn.q_proj.bias"),
        ("phi3", "blk.1.attn_qkv.weight", "model.layers.1.self_attn.qkv_proj.weight"),
        ("phi3", "blk.1.ffn_up.weight", "model.layers.1.mlp.gate_up_proj.weight"),
        (
            "gemma3",
            "blk.2.post_ffw_norm.weight",
            "model.layers.2.post_feedforward_layernorm.weight",
        ),
        ("llama", "rope_freqs.weight", "rope_freqs.weight"),  # not in the table: kept as is
    ],
)
def test_names_map_per_architecture(tmp_path, rng, capsys, architecture, gguf_name, hub_name):
    values = rng.standard_normal(HIDDEN).astype(np.float32)
    shift = np.float32(1) if architecture == "gemma3" else np.float32(0)
    st = write_safetensors(tmp_path / "b.safetensors", {hub_name: ("F32", values)})
    gguf = write_gguf(
        tmp_path / "q.gguf", {gguf_name: plain("F32", values + shift)}, _metadata(architecture)
    )
    _, record = _diff_json(capsys, st, gguf)
    (tensor,) = record["tensors"]
    assert tensor["name"] == hub_name
    assert tensor["status"] in ("unchanged", "changed")
    assert record["comparison"]["mapping"] == architecture


def test_mapping_table_covers_the_promised_architectures():
    assert sorted(MAPPINGS) == ["gemma3", "llama", "phi3", "qwen2", "qwen3"]


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"general.architecture": "mamba"}, "architecture 'mamba' has no name mapping"),
        ({}, "no general.architecture"),
        ({"general.architecture": "llama"}, "llama.attention.head_count must be a positive"),
        (
            {"general.architecture": "llama", "llama.attention.head_count": Value(5, -4)},
            "must be a positive integer",
        ),
    ],
)
def test_unmappable_files_are_refused(tmp_path, rng, capsys, metadata, message):
    base = _base(rng)
    st = write_safetensors(tmp_path / "b.safetensors", {k: ("F32", v) for k, v in base.items()})
    gguf = write_gguf(tmp_path / "q.gguf", _convert(base), metadata)
    with load_checkpoint(st) as a, load_gguf(gguf) as b, pytest.raises(MappingError, match=message):
        compare_quantized(a, b, tensor_digests(a), tensor_digests(b))
    assert main(["diff", str(st), str(gguf)]) == 2
    assert message.split()[0] in capsys.readouterr().err


def test_a_rotary_permutation_needs_whole_head_pairs(tmp_path, rng):
    with pytest.raises(MappingError, match="do not split into 4 heads of pairs"):
        rotary_rows(12, 4, "q")


def test_an_unsupported_type_is_refused_by_name(tmp_path, rng, capsys):
    values = rng.standard_normal((2, 256)).astype(np.float32)
    st = write_safetensors(tmp_path / "b.safetensors", {"model.norm.weight": ("F32", values)})
    gguf = write_gguf(
        tmp_path / "q.gguf",
        {"output_norm.weight": Tensor("IQ2_XXS", (2, 256), bytes(2 * 66))},
        _metadata(),
    )
    assert main(["diff", str(st), str(gguf)]) == 2
    assert "type IQ2_XXS, which wvc cannot decode or dequantise" in capsys.readouterr().err
    with load_checkpoint(st) as a, load_gguf(gguf) as b, pytest.raises(DequantError):
        compare_quantized(a, b, tensor_digests(a), tensor_digests(b))


def test_two_gguf_files_compare_under_their_own_names(tmp_path, rng, capsys):
    blocks = random_blocks("Q4_K", 2, rng)
    other = random_blocks("Q4_K", 2, rng)
    same = random_blocks("IQ4_XS", 2, rng)  # undecodable, but byte-identical: unchanged
    a = write_gguf(
        tmp_path / "a.gguf",
        {"blk.0.attn_q.weight": Tensor("Q4_K", (2, 256), blocks),
         "x": Tensor("IQ4_XS", (2, 256), same)},
        _metadata(),
    )  # fmt: skip
    b = write_gguf(
        tmp_path / "b.gguf",
        {"blk.0.attn_q.weight": Tensor("Q4_K", (2, 256), other),
         "x": Tensor("IQ4_XS", (2, 256), same)},
        _metadata(),
    )  # fmt: skip
    status, record = _diff_json(capsys, a, b)
    assert status == 1
    assert record["comparison"] == {
        "kind": "quantization",
        "gguf": ["a", "b"],
        "mapping": None,
        "values": "float64",
    }
    q, x = record["tensors"]
    assert (q["name"], q["status"], q["compared"]) == ("blk.0.attn_q.weight", "changed", "values")
    assert "transform_a" not in q
    assert "name_a" not in q
    assert (x["name"], x["status"]) == ("x", "unchanged")


def test_identical_quantized_bytes_with_equal_values_still_count_as_changed(tmp_path, rng):
    # Two Q8_0 blocks whose scales are +0.0 and -0.0: every value is zero, but bytes differ.
    zero, negative_zero = bytes(34), np.float16(-0.0).tobytes() + bytes(32)
    a = write_gguf(tmp_path / "a.gguf", {"t": Tensor("Q8_0", (32,), zero)}, _metadata())
    b = write_gguf(tmp_path / "b.gguf", {"t": Tensor("Q8_0", (32,), negative_zero)}, _metadata())
    with load_gguf(a) as ca, load_gguf(b) as cb:
        (t,), _ = compare_quantized(ca, cb, tensor_digests(ca), tensor_digests(cb))
    assert (t.status, t.changed, t.compared) == ("changed", 0, "values")


def test_tensors_on_one_side_and_reshaped_ones_are_reported(tmp_path, rng, capsys):
    st = write_safetensors(
        tmp_path / "b.safetensors",
        {"model.norm.weight": ("F32", np.zeros(4, np.float32)),
         "lm_head.weight": ("F32", np.zeros((2, 4), np.float32))},
    )  # fmt: skip
    gguf = write_gguf(
        tmp_path / "q.gguf",
        {"output_norm.weight": plain("F32", np.zeros(8, np.float32)),
         "rope_freqs.weight": plain("F32", np.ones(2, np.float32))},
        _metadata(),
    )  # fmt: skip
    status, record = _diff_json(capsys, st, gguf)
    by_name = {t["name"]: t for t in record["tensors"]}
    assert by_name["lm_head.weight"]["status"] == "removed"
    assert by_name["rope_freqs.weight"]["status"] == "added"
    assert by_name["model.norm.weight"]["status"] == "reshaped"
    assert by_name["model.norm.weight"]["name_b"] == "output_norm.weight"
    assert status == 1


def test_names_that_collide_after_mapping_are_refused(tmp_path, rng):
    values = np.zeros(4, np.float32)
    st = write_safetensors(tmp_path / "b.safetensors", {"model.norm.weight": ("F32", values)})
    gguf = write_gguf(
        tmp_path / "q.gguf",
        {"output_norm.weight": plain("F32", values), "model.norm.weight": plain("F32", values)},
        _metadata(),
    )
    with (
        load_checkpoint(st) as a,
        load_gguf(gguf) as b,
        pytest.raises(MappingError, match="both map"),
    ):
        compare_quantized(a, b, tensor_digests(a), tensor_digests(b))


@pytest.mark.parametrize(
    ("rows", "heads", "kv"), [(32, 4, None), (16, 4, 2), (24, 3, 3), (8, 1, 1)]
)
def test_rotary_rows_invert_llama_cpps_permutation(rows, heads, kv, rng):
    weights = rng.standard_normal((rows, 5))
    permuted = llama_permute(weights, heads, kv)
    used = kv if kv is not None and kv != heads else heads
    assert np.array_equal(permuted[rotary_rows(rows, used, "w")], weights)


def test_the_table_names_the_comparison(tmp_path, rng, capsys):
    _, st, gguf = _pair(tmp_path, rng, types={LAYER + "mlp.up_proj.weight": "Q8_0"})
    assert main(["diff", str(st), str(gguf)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("quantization comparison: b GGUF; names mapped by the llama table")
    assert any("F32->Q8_0" in line and "up_proj" in line for line in out)
