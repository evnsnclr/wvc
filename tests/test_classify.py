"""Change classification on toy transformers whose change is known (docs/classification.md)."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from helpers import (
    TOY_HIDDEN,
    TOY_INNER,
    TOY_LAYERS,
    TOY_PROJECTIONS,
    f32_to_bf16_bits,
    round_bf16,
    toy_model,
    toy_name,
    write_safetensors,
    write_toy,
)

from wvc.classify import classify_record
from wvc.cli import run_pair
from wvc.cli_classify import classify_run

VECTORS = Path(__file__).resolve().parents[1] / "spec" / "vectors" / "v0"
RECEIPTS = Path(__file__).resolve().parents[1] / "experiments"
OUTPUT_SIDE = ("self_attn.o_proj.weight", "mlp.down_proj.weight")


def _scaled(delta: np.ndarray, base: np.ndarray, rel: float) -> np.ndarray:
    return delta * rel * np.linalg.norm(base) / np.linalg.norm(delta)


def _classify(tmp_path, base, edited, *, features=True, dtype_a="BF16", dtype_b="BF16"):
    a = write_toy(tmp_path / "a.safetensors", base, dtype_a)
    b = write_toy(tmp_path / "b.safetensors", edited, dtype_b)
    return classify_run(run_pair(str(a), str(b), rank=True, rank_cap=1 << 26, features=features))


def _lora(rng, base, modules, rank, rel=0.01, layers=range(TOY_LAYERS)):
    edited = dict(base)
    for layer in layers:
        for module in modules:
            name = toy_name(layer, module)
            rows, cols = base[name].shape
            delta = rng.standard_normal((rows, rank)) @ rng.standard_normal((rank, cols))
            edited[name] = round_bf16(base[name] + _scaled(delta, base[name], rel))
    return edited


def _remove_direction(base, directions, strength=1.0):
    edited = dict(base)
    for layer in range(TOY_LAYERS):
        for module in OUTPUT_SIDE:
            name = toy_name(layer, module)
            r = directions[layer]
            edited[name] = round_bf16(base[name] - strength * np.outer(r, r @ base[name]))
    return edited


def _unit(rng, size):
    vector = rng.standard_normal(size)
    return vector / np.linalg.norm(vector)


def test_byte_identical_copy_is_a_reupload(tmp_path, rng):
    base = toy_model(rng)
    result = _classify(tmp_path, base, dict(base))
    assert (result.kind, result.tier, result.rule) == ("reupload", "strong", "R1")


def test_lossless_widening_is_a_weak_reupload(tmp_path, rng):
    base = toy_model(rng)
    result = _classify(tmp_path, base, base, dtype_b="F32")
    assert (result.kind, result.tier, result.rule) == ("reupload", "weak", "R1b")
    assert result.evidence["cast_lossless"] == len(base)


def test_dropped_tied_head_is_a_weak_reupload(tmp_path, rng):
    base = toy_model(rng)
    with_head = {**base, "lm_head.weight": base["model.embed_tokens.weight"]}
    result = _classify(tmp_path, with_head, base)
    assert (result.kind, result.tier, result.rule) == ("reupload", "weak", "R1b")


def test_bf16_round_trip_of_f32_weights_is_quantization(tmp_path, rng):
    exact = {k: v + 1e-4 * rng.standard_normal(v.shape) for k, v in toy_model(rng).items()}
    result = _classify(tmp_path, exact, exact, dtype_a="F32", dtype_b="BF16")
    assert (result.kind, result.tier, result.rule) == ("quantization", "strong", "R2")
    assert result.evidence["casts"] == {"F32->BF16": len(exact)}
    assert result.evidence["max_rel_delta"] <= result.evidence["tolerance"] == 2 * 2.0**-8


def test_quantization_with_a_larger_change_is_mixed(tmp_path, rng):
    exact = {k: v + 1e-4 * rng.standard_normal(v.shape) for k, v in toy_model(rng).items()}
    tuned = {k: v + _scaled(rng.standard_normal(v.shape), v, 0.05) for k, v in exact.items()}
    result = _classify(tmp_path, exact, tuned, dtype_a="F32", dtype_b="BF16")
    assert (result.kind, result.rule) == ("mixed", "M1")
    assert "more than rounding allows" in result.reason


def test_packed_quantization_names_are_a_weak_quantization(tmp_path, rng):
    base = toy_model(rng)
    packed = {}
    for name, values in base.items():
        if values.ndim == 2 and name.startswith("model.layers"):
            stem = name.removesuffix(".weight")
            packed[f"{stem}.qweight"] = ("I32", np.zeros((values.shape[0], 8), dtype=np.int32))
            packed[f"{stem}.scales"] = ("F16", np.ones((values.shape[0], 1), dtype=np.float16))
            packed[f"{stem}.qzeros"] = ("I32", np.zeros((1, 8), dtype=np.int32))
        else:
            packed[name] = ("BF16", f32_to_bf16_bits(values))
    a = write_toy(tmp_path / "a.safetensors", base)
    b = write_safetensors(tmp_path / "b.safetensors", packed)
    result = classify_run(run_pair(str(a), str(b), rank=True, rank_cap=1 << 26, features=False))
    assert (result.kind, result.tier, result.rule) == ("quantization", "weak", "R2b")


def test_scaled_fp8_is_a_weak_quantization(tmp_path, rng):
    base = toy_model(rng)
    fp8 = {}
    for name, values in base.items():
        if values.ndim == 1:  # FP8 checkpoints keep norms in 16 bits (v2: Qwen3-0.6B-FP8)
            fp8[name] = ("BF16", f32_to_bf16_bits(values))
            continue
        fp8[name] = ("F8_E4M3", rng.integers(0, 120, size=values.shape, dtype=np.uint8))
        fp8[name.removesuffix("weight") + "weight_scale_inv"] = ("F32", np.ones(1, np.float32))
    a = write_toy(tmp_path / "a.safetensors", base)
    b = write_safetensors(tmp_path / "b.safetensors", fp8)
    result = classify_run(run_pair(str(a), str(b), rank=True, rank_cap=1 << 26, features=False))
    assert (result.kind, result.tier, result.rule) == ("quantization", "weak", "R2c")


def test_shared_direction_removal_is_strong_with_features(tmp_path, rng):
    base = toy_model(rng)
    r = _unit(rng, TOY_HIDDEN)
    edited = _remove_direction(base, [r] * TOY_LAYERS)
    result = _classify(tmp_path, base, edited)
    assert (result.kind, result.tier, result.rule) == ("direction-removal", "strong", "R3")
    assert result.evidence["direction_coherence"] > 0.99
    assert result.evidence["projections"] == sorted(OUTPUT_SIDE)
    assert result.evidence["layers_list"] == list(range(TOY_LAYERS))


def test_direction_removal_from_the_record_alone_is_weak(tmp_path, rng):
    base = toy_model(rng)
    edited = _remove_direction(base, [_unit(rng, TOY_HIDDEN)] * TOY_LAYERS)
    result = _classify(tmp_path, base, edited, features=False)
    assert (result.kind, result.tier, result.rule) == ("direction-removal", "weak", "R3")
    assert "not measured" in result.reason


def test_unrelated_rank_one_directions_read_as_a_rank_one_lora(tmp_path, rng):
    base = toy_model(rng)
    edited = _remove_direction(base, [_unit(rng, TOY_HIDDEN) for _ in range(TOY_LAYERS)])
    result = _classify(tmp_path, base, edited)
    assert (result.kind, result.tier, result.rule) == ("merged-lora", "strong", "R5")
    assert result.evidence["r"] == 1
    assert result.evidence["direction_coherence"] < 0.5


def test_merged_lora_rank_is_measured_above_the_rounding_floor(tmp_path, rng):
    base = toy_model(rng)
    edited = _lora(rng, base, ("self_attn.q_proj.weight", "self_attn.v_proj.weight"), rank=4)
    result = _classify(tmp_path, base, edited)
    assert (result.kind, result.tier, result.rule) == ("merged-lora", "strong", "R5")
    assert result.evidence["r"] == 4
    assert result.evidence["projections"] == ["self_attn.q_proj.weight", "self_attn.v_proj.weight"]


def test_merged_lora_from_the_record_alone_is_weak(tmp_path, rng):
    base = toy_model(rng)
    edited = _lora(rng, base, tuple(TOY_PROJECTIONS), rank=4)
    result = _classify(tmp_path, base, edited, features=False)
    assert (result.kind, result.tier, result.rule) == ("merged-lora", "weak", "R5")
    assert "--features" in result.reason


def test_lora_with_a_trained_embedding_is_still_a_lora(tmp_path, rng):
    base = toy_model(rng)
    edited = _lora(rng, base, ("self_attn.q_proj.weight",), rank=2)
    embed = base["model.embed_tokens.weight"]
    edited["model.embed_tokens.weight"] = round_bf16(
        embed + 0.01 * rng.standard_normal(embed.shape)
    )
    result = _classify(tmp_path, base, edited)
    assert (result.kind, result.rule) == ("merged-lora", "R5")
    assert result.evidence["embeddings_changed"] == ["model.embed_tokens.weight"]


def test_high_rank_changes_without_norms_are_not_a_lora(tmp_path, rng):
    base = toy_model(rng)
    edited = {
        k: round_bf16(v + _scaled(rng.standard_normal(v.shape), v, 0.05)) if v.ndim == 2 else v
        for k, v in base.items()
    }
    result = _classify(tmp_path, base, edited)
    assert (result.kind, result.tier, result.rule) == ("dense-finetune", "weak", "R6")


def test_dense_fine_tune_moves_norms_and_has_high_rank(tmp_path, rng):
    base = toy_model(rng)
    edited = {
        k: round_bf16(v + _scaled(rng.standard_normal(v.shape), v, 0.05)) for k, v in base.items()
    }
    result = _classify(tmp_path, base, edited)
    assert (result.kind, result.tier, result.rule) == ("dense-finetune", "strong", "R6")
    assert result.evidence["vector_layers"] == TOY_LAYERS


def test_few_column_rank_one_edit_is_a_strong_direct_edit(tmp_path, rng):
    base = toy_model(rng)
    name = toy_name(3, "mlp.down_proj.weight")
    edited = dict(base)
    columns = base[name].copy()
    columns[:, :4] -= np.outer(_unit(rng, TOY_HIDDEN), [0.05, 0.04, 0.03, 0.02])
    edited[name] = round_bf16(columns)
    result = _classify(tmp_path, base, edited)
    assert (result.kind, result.tier, result.rule) == ("direct-edit", "strong", "R4")
    assert "4 cols" in result.reason


def test_whole_matrix_rank_one_edit_is_a_weak_direct_edit(tmp_path, rng):
    base = toy_model(rng)
    name = toy_name(5, "mlp.down_proj.weight")
    edited = dict(base)
    edited[name] = round_bf16(
        base[name] + 0.01 * np.outer(_unit(rng, TOY_HIDDEN), _unit(rng, TOY_INNER))
    )
    result = _classify(tmp_path, base, edited)
    assert (result.kind, result.tier, result.rule) == ("direct-edit", "weak", "R4")


def test_direction_removal_plus_other_changes_is_mixed(tmp_path, rng):
    base = toy_model(rng)
    edited = _remove_direction(base, [_unit(rng, TOY_HIDDEN)] * TOY_LAYERS)
    for layer in range(TOY_LAYERS):
        name = toy_name(layer, "self_attn.q_proj.weight")
        edited[name] = round_bf16(
            base[name] + _scaled(rng.standard_normal(base[name].shape), base[name], 0.05)
        )
    result = _classify(tmp_path, base, edited)
    assert (result.kind, result.tier, result.rule) == ("mixed", "weak", "M2")
    assert "plus 8 other" in result.reason


def test_vocabulary_resize_caps_the_tier(tmp_path, rng):
    base = toy_model(rng)
    edited = _lora(rng, base, ("self_attn.q_proj.weight",), rank=2)
    embed = base["model.embed_tokens.weight"]
    edited["model.embed_tokens.weight"] = np.concatenate([embed, embed[:8]])
    result = _classify(tmp_path, base, edited)
    assert (result.kind, result.tier, result.rule) == ("merged-lora", "weak", "R5")
    assert result.evidence["resized"] == ["model.embed_tokens.weight"]
    assert "resized" in result.reason


def test_changed_norms_alone_are_unknown_with_the_closest_rule(tmp_path, rng):
    base = toy_model(rng)
    edited = dict(base)
    for layer in (1, 2):
        name = toy_name(layer, "input_layernorm.weight")
        edited[name] = round_bf16(base[name] * 1.1)
    result = _classify(tmp_path, base, edited)
    assert (result.kind, result.tier, result.rule) == ("unknown", "weak", "U")
    assert result.reason.startswith("no rule fired; the closest, R4, failed: every change")
    assert set(result.evidence["failed"]) == {"R1", "R2", "R3", "R4", "R5", "R6", "M2"}


def test_spec_vectors_are_unknown():
    record = json.loads((VECTORS / "diff.json").read_text(encoding="utf-8"))
    result = classify_record(record)
    assert (result.kind, result.rule) == ("unknown", "U")
    assert "R4, failed: nothing may be added, removed, or quantized" in result.reason


@pytest.mark.parametrize(
    ("receipt", "tier"), [("E001/diff.json", "strong"), ("E008/diff-rankcap.json", "weak")]
)
def test_receipts_are_direct_edits(receipt, tier):
    record = json.loads((RECEIPTS / receipt).read_text(encoding="utf-8"))
    result = classify_record(record)
    assert (result.kind, result.tier, result.rule) == ("direct-edit", tier, "R4")
    assert result.evidence["layers_known"] is False


def test_bf16_to_f16_conversion_is_quantization(tmp_path, rng):
    base = toy_model(rng)
    base["model.layers.0.mlp.up_proj.weight"][0, :4] = [1e-6, -2e-7, 3e-8, 5e-9]  # F16 subnormal
    base = {k: round_bf16(v) for k, v in base.items()}
    halves = {k: np.asarray(v, dtype=np.float16) for k, v in base.items()}
    result = _classify(tmp_path, base, halves, dtype_b="F16")
    assert (result.kind, result.rule) == ("quantization", "R2")
    changed = result.evidence["casts"]["BF16->F16"]  # values below F16's normal range round
    assert changed >= 1
    assert changed + result.evidence["cast_lossless"] == len(base)


GGUF_RECEIPTS = RECEIPTS / "E011"


@pytest.mark.parametrize(
    ("receipt", "kind", "rule"),
    [
        ("smollm2-q4_k_m.diff.json", "quantization", "R2"),
        ("smollm2-f16.diff.json", "quantization", "R2"),
        ("qwen-q4_k_m.diff.json", "mixed", "M3"),
    ],
)
def test_gguf_quantization_comparisons(receipt, kind, rule):
    record = json.loads((GGUF_RECEIPTS / receipt).read_text(encoding="utf-8"))
    result = classify_record(record)
    assert (result.kind, result.rule) == (kind, rule)


def _hybrid(base):
    """Layers 6 and 7 hold linear attention instead of the q, k, v projections."""
    hybrid = dict(base)
    for layer in (6, 7):
        for module in ("q_proj", "k_proj", "v_proj"):
            del hybrid[toy_name(layer, f"self_attn.{module}.weight")]
        hybrid[toy_name(layer, "linear_attn.in_proj.weight")] = base[
            toy_name(layer, "mlp.up_proj.weight")
        ]
    return hybrid


def test_lora_on_a_hybrid_stack_is_consistent_per_module(tmp_path, rng):
    base = _hybrid(toy_model(rng))
    edited = dict(base)
    for name, values in base.items():
        if values.ndim == 2 and name.startswith("model.layers"):
            delta = rng.standard_normal((values.shape[0], 4)) @ rng.standard_normal(
                (4, values.shape[1])
            )
            edited[name] = round_bf16(values + _scaled(delta, values, 0.01))
    result = _classify(tmp_path, base, edited)
    assert (result.kind, result.tier, result.rule) == ("merged-lora", "strong", "R5")
    assert result.evidence["r"] == 4


def test_a_frozen_vision_tower_does_not_dilute_a_fine_tune(tmp_path, rng):
    base = toy_model(rng)
    for block in range(24):
        for part in ("attn.qkv.weight", "mlp.fc1.weight", "norm1.weight", "norm2.weight"):
            base[f"model.visual.blocks.{block}.{part}"] = round_bf16(rng.standard_normal(16) * 0.02)
    edited = {
        k: round_bf16(v + _scaled(rng.standard_normal(v.shape), v, 0.05))
        if k.startswith("model.layers") or k == "model.norm.weight"
        else v
        for k, v in base.items()
    }
    result = _classify(tmp_path, base, edited)
    assert (result.kind, result.rule) == ("dense-finetune", "R6")
    assert (result.evidence["layers"], result.evidence["stack"]) == (TOY_LAYERS, "model.layers")


def test_direction_removed_from_both_sides_is_a_direction_removal(tmp_path, rng):
    base = toy_model(rng)
    r = _unit(rng, TOY_HIDDEN)
    edited = _remove_direction(base, [r] * TOY_LAYERS)
    for layer in range(TOY_LAYERS):
        for module in ("self_attn.q_proj.weight", "mlp.up_proj.weight"):
            name = toy_name(layer, module)
            edited[name] = round_bf16(base[name] - np.outer(base[name] @ r, r))
    result = _classify(tmp_path, base, edited)
    assert (result.kind, result.tier, result.rule) == ("direction-removal", "strong", "R3")


def test_lora_stored_in_f16_is_a_lora_not_a_quantization(tmp_path, rng):
    base = toy_model(rng)
    edited = _lora(rng, base, ("self_attn.q_proj.weight", "self_attn.v_proj.weight"), rank=4)
    halves = {k: np.asarray(v, dtype=np.float16) for k, v in edited.items()}
    result = _classify(tmp_path, base, halves, dtype_b="F16")
    assert (result.kind, result.tier, result.rule) == ("merged-lora", "strong", "R5")
    assert result.evidence["r"] == 4


def test_features_under_the_snr_floor_fall_back_to_the_record():
    from wvc.classify_input import SNR_MIN, build_view, validate_record

    entry = {
        "name": "model.layers.0.mlp.down_proj.weight",
        "status": "changed",
        "shape_a": [4, 4],
        "shape_b": [4, 4],
        "features": {"signal_rank": 2, "energy_rank": 1, "signal_top1": 0.6, "snr": SNR_MIN / 2},
    }
    record = {
        "summary": {
            "tensors": 1,
            "by_status": dict(unchanged=0, changed=1, added=0, removed=0, reshaped=0),
        },
        "tensors": [entry],
    }
    assert build_view(validate_record(record)).moved[0].features is None
    entry["features"]["snr"] = None  # float32 and wider: no rounding floor to clear
    assert build_view(validate_record(record)).moved[0].features is not None


def test_a_lora_record_without_unchanged_tensors_uses_the_common_module_set(tmp_path, rng):
    base = toy_model(rng)
    edited = _lora(rng, base, ("self_attn.q_proj.weight", "self_attn.v_proj.weight"), rank=4)
    a = write_toy(tmp_path / "a.safetensors", base)
    b = write_toy(tmp_path / "b.safetensors", edited)
    run = run_pair(str(a), str(b), rank=True, rank_cap=1 << 26, features=False)
    record = {
        "summary": {
            "tensors": len(run.diffs),
            "by_status": dict(Counter(d.status for d in run.diffs)),
        },
        "tensors": [d.to_dict() for d in run.diffs if d.status != "unchanged"],
    }
    for status in ("unchanged", "changed", "added", "removed", "reshaped"):
        record["summary"]["by_status"].setdefault(status, 0)
    result = classify_record(record)
    assert (result.kind, result.tier, result.rule) == ("merged-lora", "weak", "R5")
    assert result.evidence["layers_known"] is False
