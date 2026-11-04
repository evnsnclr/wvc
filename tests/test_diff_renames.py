"""Pairing tensors across a rename (spec section 2): by id, then by a leading name prefix."""

import json
from dataclasses import replace

import numpy as np
import pytest
from helpers import toy_model, write_safetensors, write_toy

from wvc.classify_input import ClassifyError, validate_record
from wvc.cli import main, run_pair
from wvc.cli_classify import classify_run
from wvc.diff import diff_checkpoints, summarize
from wvc.hashing import tensor_digests
from wvc.safetensors_io import load_checkpoint

WRAP = "language_model."


def _diff(tmp_path, tensors_a, tensors_b, **kwargs):
    ckpt_a = load_checkpoint(write_safetensors(tmp_path / "a.safetensors", tensors_a))
    ckpt_b = load_checkpoint(write_safetensors(tmp_path / "b.safetensors", tensors_b))
    digests_a, digests_b = tensor_digests(ckpt_a), tensor_digests(ckpt_b)
    diffs = diff_checkpoints(ckpt_a, ckpt_b, digests_a, digests_b, **kwargs)
    return {d.name: d for d in diffs}, summarize(diffs), (digests_a, digests_b)


def _matrix(rng, shape=(8, 6)):
    return rng.standard_normal(shape).astype(np.float32)


def test_identical_bytes_under_an_unrelated_name_pair_by_id(tmp_path, rng):
    w, v = _matrix(rng), _matrix(rng)
    a = {"transformer.wte.weight": ("F32", w), "shared": ("F32", v)}
    b = {"model.embed_tokens.weight": ("F32", w.copy()), "shared": ("F32", v)}
    by_name, summary, _ = _diff(tmp_path, a, b)
    assert set(by_name) == {"transformer.wte.weight", "shared"}
    paired = by_name["transformer.wte.weight"]
    assert (paired.status, paired.renamed_to, paired.changed) == (
        "unchanged",
        "model.embed_tokens.weight",
        0,
    )
    assert paired.sha256 is None
    assert by_name["shared"].renamed_to is None
    assert summary["renamed"] == 1
    assert summary["by_status"] == {
        "unchanged": 2,
        "changed": 0,
        "added": 0,
        "removed": 0,
        "reshaped": 0,
    }
    assert "renamed_to" not in by_name["shared"].to_dict()
    assert paired.to_dict()["renamed_to"] == "model.embed_tokens.weight"


@pytest.mark.parametrize(
    ("name_a", "name_b"),
    [
        ("model.layers.3.mlp.down_proj.weight", WRAP + "model.layers.3.mlp.down_proj.weight"),
        ("base_model.model.model.norm.weight", "model.norm.weight"),
        ("model.lm_head.weight", "lm_head.weight"),
    ],
    ids=["prefix-added", "prefix-dropped", "model-dropped"],
)
def test_a_leading_prefix_pairs_by_suffix_in_both_directions(tmp_path, rng, name_a, name_b):
    base = _matrix(rng)
    edited = base.copy()
    edited[2, :] += 1.0
    by_name, summary, _ = _diff(tmp_path, {name_a: ("F32", base)}, {name_b: ("F32", edited)})
    assert list(by_name) == [name_a]
    d = by_name[name_a]
    assert (d.status, d.renamed_to, d.changed, d.rows_touched) == ("changed", name_b, 6, 1)
    assert summary["renamed"] == 1
    assert summary["by_status"]["added"] == summary["by_status"]["removed"] == 0


def test_a_renamed_pair_is_compared_as_a_same_name_pair_with_rank(tmp_path, rng):
    base = _matrix(rng, (16, 32))
    edited = base.copy()
    edited[:, [3, 7, 9]] -= np.outer(_matrix(rng, (16,)), [0.5, 1.0, 2.0]).astype(np.float32)
    name = "model.layers.0.self_attn.o_proj.weight"
    same, _, _ = _diff(tmp_path, {name: ("F32", base)}, {name: ("F32", edited)}, rank=True)
    renamed, _, _ = _diff(
        tmp_path, {name: ("F32", base)}, {WRAP + name: ("F32", edited)}, rank=True
    )
    d = renamed[name]
    assert (d.rank, d.rows_touched, d.cols_touched, d.changed) == (1, 16, 3, 48)
    assert d.top1_energy > 0.999
    assert replace(d, renamed_to=None) == same[name]


def test_identical_renamed_tensors_need_no_read_when_the_id_pass_is_ambiguous(tmp_path):
    ones = np.ones(8, dtype=np.float32)
    a = {f"model.layers.{i}.norm.weight": ("F32", ones) for i in range(3)}
    b = {WRAP + name: value for name, value in a.items()}
    by_name, summary, _ = _diff(tmp_path, a, b)
    # three copies of one id: the id pass pairs none, the suffix pass pairs each with its layer
    assert summary["renamed"] == 3
    assert all(d.status == "unchanged" and d.renamed_to == WRAP + n for n, d in by_name.items())


def test_ambiguous_ids_leave_every_copy_unpaired(tmp_path, rng):
    w = _matrix(rng)
    a = {"first": ("F32", w), "second": ("F32", w.copy())}
    b = {"other": ("F32", w.copy())}
    by_name, summary, (digests_a, digests_b) = _diff(tmp_path, a, b)
    assert {n: d.status for n, d in by_name.items()} == {
        "first": "removed",
        "second": "removed",
        "other": "added",
    }
    assert summary["renamed"] == 0
    assert by_name["first"].sha256 == by_name["other"].sha256 == digests_b["other"]
    assert by_name["second"].sha256 == digests_a["second"]


@pytest.mark.parametrize("direction", ["one-removed-two-added", "two-removed-one-added"])
def test_ambiguous_suffixes_leave_every_candidate_unpaired(tmp_path, rng, direction):
    short = {"x.weight": ("F32", _matrix(rng))}
    long = {"p.x.weight": ("F32", _matrix(rng)), "q.x.weight": ("F32", _matrix(rng))}
    a, b = (short, long) if direction == "one-removed-two-added" else (long, short)
    by_name, summary, _ = _diff(tmp_path, a, b)
    assert summary["renamed"] == 0
    assert {d.status for n, d in by_name.items() if n in a} == {"removed"}
    assert {d.status for n, d in by_name.items() if n in b} == {"added"}


def test_suffix_ambiguity_on_one_name_does_not_block_another(tmp_path, rng):
    a = {"x.weight": ("F32", _matrix(rng)), "y.weight": ("F32", _matrix(rng))}
    b = {
        "p.x.weight": ("F32", _matrix(rng)),
        "q.x.weight": ("F32", _matrix(rng)),
        "p.y.weight": ("F32", _matrix(rng)),
    }
    by_name, summary, _ = _diff(tmp_path, a, b)
    assert summary["renamed"] == 1
    assert by_name["y.weight"].renamed_to == "p.y.weight"
    assert by_name["x.weight"].status == "removed"


@pytest.mark.parametrize(
    "layout_b",
    [("F16", (8, 6)), ("F32", (6, 8)), ("F32", (48,))],
    ids=["dtype", "transposed", "flattened"],
)
def test_no_pairing_across_dtype_or_shape(tmp_path, rng, layout_b):
    w = _matrix(rng)
    dtype, shape = layout_b
    other = w.reshape(shape).astype(np.float16 if dtype == "F16" else np.float32)
    by_name, summary, _ = _diff(
        tmp_path, {"model.w": ("F32", w)}, {WRAP + "model.w": (dtype, other)}
    )
    assert summary["renamed"] == 0
    assert (by_name["model.w"].status, by_name[WRAP + "model.w"].status) == ("removed", "added")


def test_equal_bytes_of_another_dtype_are_not_one_tensor(tmp_path):
    raw = np.arange(4, dtype=np.int32)
    a = {"alpha": ("I32", raw)}
    b = {"beta": ("F32", raw.view(np.float32))}
    by_name, summary, _ = _diff(tmp_path, a, b)
    assert by_name["alpha"].sha256 == by_name["beta"].sha256
    assert summary["renamed"] == 0


def test_quantization_packings_stay_added_and_removed(tmp_path, rng):
    a = {"model.layers.0.mlp.up_proj.weight": ("F16", _matrix(rng).astype(np.float16))}
    prefix = "model.layers.0.mlp.up_proj."
    b = {
        prefix + "qweight": ("I32", rng.integers(0, 1 << 30, (8, 1)).astype(np.int32)),
        prefix + "scales": ("F16", _matrix(rng, (1, 6)).astype(np.float16)),
        prefix + "qzeros": ("I32", rng.integers(0, 1 << 30, (1, 1)).astype(np.int32)),
    }
    by_name, summary, (_, digests_b) = _diff(tmp_path, a, b)
    assert summary["renamed"] == 0
    assert summary["by_status"]["added"] == 3
    assert all(by_name[n].sha256 == digests_b[n] for n in b)


def _wrapped_pair(tmp_path, rng, *, edit: bool):
    base = toy_model(rng)
    edited = {name: value.copy() for name, value in base.items()}
    if edit:
        edited["model.layers.2.mlp.down_proj.weight"][:, 5] += 0.5
    a = write_toy(tmp_path / "a.safetensors", base)
    b = write_toy(tmp_path / "b.safetensors", {WRAP + n: v for n, v in edited.items()})
    return a, b, len(base)


def test_the_table_names_both_sides_of_a_rename(tmp_path, rng, capsys):
    a, b, count = _wrapped_pair(tmp_path, rng, edit=True)
    assert main(["diff", str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert f"renamed: {count} of the changed and unchanged tensors" in out
    name = "model.layers.2.mlp.down_proj.weight"
    assert f"{name} -> {WRAP}{name}" in out
    assert f"model.norm.weight -> {WRAP}model.norm.weight" in out  # unchanged, but renamed
    assert main(["diff", str(a), str(a)]) == 0
    assert "renamed" not in capsys.readouterr().out


def test_the_json_record_carries_renamed_to(tmp_path, rng, capsys):
    a, b, count = _wrapped_pair(tmp_path, rng, edit=False)
    assert main(["diff", "--json", "--all", str(a), str(b)]) == 1
    record = json.loads(capsys.readouterr().out)
    assert record["summary"]["renamed"] == count
    assert all(t["renamed_to"] == WRAP + t["name"] for t in record["tensors"])
    assert all("sha256" not in t for t in record["tensors"])


def test_the_agent_view_counts_and_lists_renames(tmp_path, rng, capsys):
    a, b, count = _wrapped_pair(tmp_path, rng, edit=True)
    assert main(["diff", "--format", "agent", str(a), str(b)]) == 1
    text = capsys.readouterr().out
    assert f"- renamed: {count} of the changed and unchanged tensors" in text
    assert f"- renamed ({count}):" in text
    assert "  - `model.embed_tokens.weight` -> `language_model.model.embed_tokens.weight` " in text
    assert f"  - and {count - 8} more" in text
    assert "`model.layers.*.mlp.down_proj.weight`" in text  # grouped under the names in a
    assert main(["diff", "--format", "agent", str(a), str(a)]) == 0
    assert "renamed" not in capsys.readouterr().out


def test_a_renamed_but_identical_upload_is_a_reupload(tmp_path, rng):
    a, b, count = _wrapped_pair(tmp_path, rng, edit=False)
    result = classify_run(run_pair(str(a), str(b), rank=True, rank_cap=1 << 26, features=True))
    assert (result.kind, result.tier, result.rule) == ("reupload", "strong", "R1")
    assert result.evidence["renamed"] == count
    assert result.evidence["added"] == result.evidence["removed"] == 0
    assert f"{count} of them under another name" in result.reason


def test_features_read_a_renamed_tensor_under_its_name_in_b(tmp_path, rng):
    a, b, _ = _wrapped_pair(tmp_path, rng, edit=True)
    run = run_pair(str(a), str(b), rank=True, rank_cap=1 << 26, features=True)
    assert run.features is not None
    assert "model.layers.2.mlp.down_proj.weight" in run.features[0]
    assert classify_run(run).evidence["renamed"] == len(run.diffs)


def test_a_renamed_lossless_widening_says_so(tmp_path, rng):
    base = toy_model(rng)
    a = write_toy(tmp_path / "a.safetensors", base)
    widened = {WRAP + n: v for n, v in base.items()}
    widened["extra.bias"] = np.zeros(4, dtype=np.float32)  # a global tensor added
    b = write_toy(tmp_path / "b.safetensors", widened)
    result = classify_run(run_pair(str(a), str(b), rank=False, rank_cap=1 << 26, features=False))
    assert (result.kind, result.rule) == ("reupload", "R1b")
    assert f"{len(base)} under another name" in result.reason


@pytest.mark.parametrize("key", ["renamed_to", "sha256"])
def test_the_classifier_refuses_a_non_string_rename_field(key):
    entry = {"name": "w", "status": "unchanged", key: 7}
    by_status = {"unchanged": 1, "changed": 0, "added": 0, "removed": 0, "reshaped": 0}
    record = {"summary": {"tensors": 1, "by_status": by_status}, "tensors": [entry]}
    with pytest.raises(ClassifyError, match=key):
        validate_record(record)
