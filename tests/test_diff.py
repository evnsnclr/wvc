import numpy as np
import pytest
from helpers import bf16_bits_to_f32, f32_to_bf16_bits, write_safetensors

from wvc.diff import DEFAULT_RANK_CAP, _one_sided, diff_checkpoints, summarize
from wvc.hashing import tensor_digests
from wvc.safetensors_io import load_checkpoint


def _diff(tmp_path, tensors_a, tensors_b, **kwargs):
    ckpt_a = load_checkpoint(write_safetensors(tmp_path / "a.safetensors", tensors_a))
    ckpt_b = load_checkpoint(write_safetensors(tmp_path / "b.safetensors", tensors_b))
    diffs = diff_checkpoints(
        ckpt_a, ckpt_b, tensor_digests(ckpt_a), tensor_digests(ckpt_b), **kwargs
    )
    return {d.name: d for d in diffs}, summarize(diffs)


def test_identical_checkpoints_are_unchanged(tmp_path, rng):
    a = rng.standard_normal((4, 4)).astype(np.float32)
    by_name, summary = _diff(tmp_path, {"w": ("F32", a)}, {"w": ("F32", a.copy())})
    assert by_name["w"].status == "unchanged"
    assert by_name["w"].changed == 0
    assert summary["by_status"]["unchanged"] == 1
    assert summary["elements_changed"] == 0


def test_rank_one_column_edit_is_localised(tmp_path, rng):
    base = rng.standard_normal((16, 32)).astype(np.float32)
    direction = rng.standard_normal(16).astype(np.float32)
    cols = [3, 7, 9]
    edited = base.copy()
    edited[:, cols] -= np.outer(direction, np.array([0.5, 1.0, 2.0], dtype=np.float32))
    by_name, summary = _diff(tmp_path, {"w": ("F32", base)}, {"w": ("F32", edited)}, rank=True)
    d = by_name["w"]
    assert d.status == "changed"
    assert d.changed == 48
    assert d.rows_touched == 16
    assert d.cols_touched == 3
    assert d.rank == 1
    assert d.top1_energy > 0.999
    assert d.max_abs_delta > 0
    assert 0 < d.rel_delta_norm < 1
    assert summary["elements_changed"] == 48
    assert summary["fraction_changed"] == 48 / 512


def test_default_rank_cap_covers_a_9b_down_projection():
    # E008: Qwen3.5-9B's mlp.down_proj.weight is [4096, 12288]; the default must not skip it.
    # Spec section 2 states the number; change both together.
    assert DEFAULT_RANK_CAP == 1 << 26
    assert DEFAULT_RANK_CAP >= 4096 * 12288


def test_rank_cap_skips_the_svd_when_the_touched_block_is_larger(tmp_path, rng):
    base = rng.standard_normal((16, 32)).astype(np.float32)
    edited = base.copy()
    edited[:, [3, 7, 9]] -= 0.5  # touched block: 16 rows x 3 columns = 48 elements
    pair = ({"w": ("F32", base)}, {"w": ("F32", edited)})
    capped, _ = _diff(tmp_path, *pair, rank=True, rank_cap=47)
    uncapped, _ = _diff(tmp_path, *pair, rank=True, rank_cap=48)
    assert capped["w"].rank is None
    assert capped["w"].top1_energy is None
    assert uncapped["w"].rank == 1
    assert uncapped["w"].top1_energy > 0.999


def test_bf16_edit_counts_exact_elements(tmp_path, rng):
    base = rng.standard_normal((8, 8)).astype(np.float32)
    edited = base.copy()
    edited[2, 5] += 1.0
    edited[6, 1] -= 1.0
    bits_a, bits_b = f32_to_bf16_bits(base), f32_to_bf16_bits(edited)
    by_name, _ = _diff(tmp_path, {"w": ("BF16", bits_a)}, {"w": ("BF16", bits_b)})
    assert by_name["w"].changed == 2
    assert by_name["w"].rows_touched == 2
    assert by_name["w"].cols_touched == 2


def test_added_removed_reshaped_and_dtype_change(tmp_path, rng):
    a = rng.standard_normal(4).astype(np.float32)
    b = rng.standard_normal((2, 2)).astype(np.float32)
    by_name, summary = _diff(
        tmp_path,
        {"gone": ("F32", a), "shape": ("F32", a), "cast": ("F32", a)},
        {
            "new": ("F32", b),
            "shape": ("F32", a.reshape(2, 2)),
            "cast": ("BF16", f32_to_bf16_bits(a)),
        },
    )
    assert by_name["gone"].status == "removed"
    assert by_name["new"].status == "added"
    assert by_name["shape"].status == "reshaped"
    assert by_name["shape"].changed is None
    cast = by_name["cast"]
    assert cast.status == "reshaped"
    assert cast.dtype_b == "BF16"
    assert cast.changed == int((a != bf16_bits_to_f32(f32_to_bf16_bits(a))).sum())
    assert summary["by_status"] == {
        "unchanged": 0,
        "changed": 0,
        "added": 1,
        "removed": 1,
        "reshaped": 2,
    }


def test_different_bytes_are_never_unchanged(tmp_path):
    """Review finding H1: a dtype change, signed zeros, big ints, NaN payloads, bool bytes."""
    exact = np.array([1.0, 2.5, -3.0, 0.5], dtype=np.float32)
    by_name, _ = _diff(tmp_path, {"t": ("BF16", f32_to_bf16_bits(exact))}, {"t": ("F32", exact)})
    assert by_name["t"].status == "reshaped"
    assert by_name["t"].changed == 0
    cases = {
        "zero": (
            ("F32", np.array([0.0], dtype=np.float32)),
            ("F32", np.array([-0.0], dtype=np.float32)),
        ),
        "int": (
            ("I64", np.array([2**24 + 1], dtype="<i8")),
            ("I64", np.array([2**24], dtype="<i8")),
        ),
        "nan": (
            ("U32", np.array([0x7FC00000], dtype="<u4").view("<f4")),
            ("U32", np.array([0x7FC00001], dtype="<u4").view("<f4")),
        ),
        "bool": (("U8", np.array([1], dtype=np.uint8)), ("U8", np.array([2], dtype=np.uint8))),
    }
    for name, (left, right) in cases.items():
        if name == "nan":  # store the NaN patterns as F32 bytes
            left, right = ("F32", left[1]), ("F32", right[1])
        (tmp_path / name).mkdir()
        by_name, summary = _diff(tmp_path / name, {name: left}, {name: right})
        d = by_name[name]
        assert d.status == "changed", name
        assert d.changed == 1, name
        assert summary["elements_changed"] == 1
        if name == "int":
            assert d.max_abs_delta == 1.0
        if name == "zero":
            assert d.max_abs_delta == 0.0
        if name == "nan":
            assert d.nonfinite_changed == 1
            assert d.max_abs_delta is None
            assert d.to_dict()["max_abs_delta"] is None


def test_undecodable_dtype_gets_byte_level_count_only(tmp_path):
    a = np.arange(4, dtype="<c8")
    b = a.copy()
    b[1] += 1
    by_name, _ = _diff(tmp_path, {"c": ("C64", a)}, {"c": ("C64", b)})
    d = by_name["c"]
    assert d.status == "changed"
    assert d.changed == 1
    assert d.max_abs_delta is None


def test_fp8_decodes_exactly(tmp_path):
    from wvc.safetensors_io import fp8_table

    table = fp8_table("F8_E4M3")
    assert table[0x7F] != table[0x7F]
    assert table[0x38] == 1.0
    assert table[0x7E] == 448.0
    assert fp8_table("F8_E5M2")[0x7C] == np.inf
    assert fp8_table("F8_E8M0")[127] == 1.0
    codes = np.array([0x38, 0x40, 0x7E], dtype=np.uint8)
    edited = np.array([0x38, 0x41, 0x7E], dtype=np.uint8)
    by_name, _ = _diff(tmp_path, {"q": ("F8_E4M3", codes)}, {"q": ("F8_E4M3", edited)})
    assert by_name["q"].changed == 1
    assert by_name["q"].max_abs_delta == float(table[0x41] - table[0x40])


def test_one_dimensional_change_reports_rows_only(tmp_path, rng):
    a = rng.standard_normal(10).astype(np.float32)
    b = a.copy()
    b[4] = 0.0
    by_name, _ = _diff(tmp_path, {"bias": ("F32", a)}, {"bias": ("F32", b)})
    assert by_name["bias"].changed == 1
    assert by_name["bias"].rows_touched == 1
    assert by_name["bias"].cols_touched is None
    assert by_name["bias"].rank is None


def test_nan_in_both_is_not_a_change(tmp_path):
    a = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    b = np.array([1.0, np.nan, 4.0], dtype=np.float32)
    by_name, _ = _diff(tmp_path, {"v": ("F32", a)}, {"v": ("F32", b)})
    assert by_name["v"].changed == 1
    assert by_name["v"].max_abs_delta == 1.0


def test_one_sided_requires_exactly_one_checkpoint():
    with pytest.raises(ValueError, match="missing from both"):
        _one_sided("x", None, None)


def test_chunked_statistics_match_direct_computation(tmp_path, rng, monkeypatch):
    """Force many small chunks and compare every statistic with the in-memory formula."""
    import wvc.diff as diff_module

    monkeypatch.setattr(diff_module, "CHUNK_ELEMENTS", 7)
    base = rng.standard_normal((9, 11)).astype(np.float32)
    edited = base.copy()
    edited[2, :] += 0.5
    edited[5, 3] = np.nan
    edited[7, 10] -= 2.0
    base[5, 3] = np.nan  # NaN in both: not a change
    base[1, 1] = np.nan  # NaN only in base: a change with a non-finite delta
    by_name, _ = _diff(tmp_path, {"w": ("F32", base)}, {"w": ("F32", edited)}, rank=True)
    d = by_name["w"]
    mask = (base != edited) & ~(np.isnan(base) & np.isnan(edited))
    delta = (edited - base)[mask]
    finite = np.isfinite(delta)
    assert d.changed == int(mask.sum())
    assert d.max_abs_delta == pytest.approx(float(np.abs(delta[finite]).max()))
    assert d.delta_norm == pytest.approx(float(np.linalg.norm(delta[finite].astype(np.float64))))
    base_norm = float(np.linalg.norm(base[np.isfinite(base)].astype(np.float64)))
    assert d.rel_delta_norm == pytest.approx(d.delta_norm / base_norm)
    assert d.rows_touched == int(mask.any(axis=1).sum())
    assert d.cols_touched == int(mask.any(axis=0).sum())
    assert d.rank is None  # the touched submatrix holds a non-finite delta


def test_rank_of_a_rank_one_edit_survives_chunking(tmp_path, rng, monkeypatch):
    import wvc.diff as diff_module

    monkeypatch.setattr(diff_module, "CHUNK_ELEMENTS", 5)
    base = rng.standard_normal((16, 32)).astype(np.float32)
    edited = base.copy()
    edited[:, [3, 7, 9]] -= np.outer(rng.standard_normal(16).astype(np.float32), [0.5, 1.0, 2.0])
    by_name, _ = _diff(tmp_path, {"w": ("F32", base)}, {"w": ("F32", edited)}, rank=True)
    assert by_name["w"].rank == 1
    assert by_name["w"].top1_energy > 0.999
