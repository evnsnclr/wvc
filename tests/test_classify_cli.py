"""``wvc classify``, ``wvc diff --classify``, and the diff record the classifier accepts."""

import json

import numpy as np
import pytest
from helpers import (
    TOY_HIDDEN,
    TOY_LAYERS,
    round_bf16,
    toy_model,
    toy_name,
    write_safetensors,
    write_toy,
)

from wvc.classify_input import ClassifyError, validate_record
from wvc.cli import main
from wvc.cli_classify import side_argument


def _pair(tmp_path, rng):
    base = toy_model(rng)
    edited = dict(base)
    r = rng.standard_normal(TOY_HIDDEN)
    r /= np.linalg.norm(r)
    for layer in range(TOY_LAYERS):
        for module in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
            name = toy_name(layer, module)
            edited[name] = round_bf16(base[name] - np.outer(r, r @ base[name]))
    return write_toy(tmp_path / "a.safetensors", base), write_toy(
        tmp_path / "b.safetensors", edited
    )


def test_classify_two_checkpoints_prints_kind_and_evidence(tmp_path, rng, capsys):
    a, b = _pair(tmp_path, rng)
    assert main(["classify", str(a), str(b), "--features"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("direction-removal (strong): rank-1 changes in mlp.down_proj")
    assert out[0].endswith("[R3]")
    assert "  layers: 8" in out


def test_classify_json_document(tmp_path, rng, capsys):
    a, b = _pair(tmp_path, rng)
    assert main(["classify", str(a), str(b), "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["format"] == "wvc-classification/1"
    assert document["rules"] == "v2"
    assert (document["kind"], document["tier"], document["features"]) == (
        "direction-removal",
        "weak",
        False,
    )


def test_diff_classify_appends_to_the_record_and_the_table(tmp_path, rng, capsys):
    a, b = _pair(tmp_path, rng)
    assert main(["diff", str(a), str(b), "--rank", "--classify"]) == 1
    out = capsys.readouterr().out
    assert "classification: direction-removal (weak)" in out
    assert main(["diff", str(a), str(b), "--rank", "--classify", "--features", "--json"]) == 1
    record = json.loads(capsys.readouterr().out)
    assert record["classification"]["kind"] == "direction-removal"
    assert record["features"]["hidden_size"] == TOY_HIDDEN
    assert record["features"]["directions"] == 2 * TOY_LAYERS
    changed = [t for t in record["tensors"] if t["status"] == "changed"]
    assert all(t["features"]["signal_rank"] >= 1 for t in changed)
    assert all("direction" not in t["features"] for t in changed)
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert main(["classify", str(path), "--json"]) == 0
    again = json.loads(capsys.readouterr().out)
    assert (again["kind"], again["tier"]) == ("direction-removal", "strong")


def test_classify_record_with_features_reads_the_pair_it_names(tmp_path, rng, capsys):
    a, b = _pair(tmp_path, rng)
    assert main(["diff", str(a), str(b), "--rank", "--json"]) == 1
    path = tmp_path / "record.json"
    path.write_bytes(capsys.readouterr().out.encode("utf-8"))
    assert main(["classify", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tier"] == "weak"
    assert main(["classify", str(path), "--features", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tier"] == "strong"


def test_diff_features_needs_classify(tmp_path, rng, capsys):
    a, b = _pair(tmp_path, rng)
    with pytest.raises(SystemExit) as exc:
        main(["diff", str(a), str(b), "--features"])
    assert exc.value.code == 2
    assert "--features applies with --classify" in capsys.readouterr().err


def test_classify_takes_one_or_two_inputs(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["classify", "x", "y", "z"])
    assert exc.value.code == 2
    assert "give one diff record, or two checkpoints" in capsys.readouterr().err


def test_classify_refuses_a_record_that_is_not_json(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_bytes(b"{not json")
    assert main(["classify", str(path)]) == 2
    assert "diff record is not valid JSON" in capsys.readouterr().err


def _record(**changes):
    entry = {"name": "w", "status": "changed", "shape_a": [2, 2], "shape_b": [2, 2], "rank": 1}
    record = {
        "summary": {
            "tensors": 2,
            "by_status": {"unchanged": 1, "changed": 1, "added": 0, "removed": 0, "reshaped": 0},
        },
        "tensors": [entry],
    }
    for key, value in changes.items():
        if key.startswith("t_"):
            entry[key[2:]] = value
        else:
            record[key] = value
    return record


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ([], "must be a JSON object"),
        (_record(summary={"tensors": -1}), "non-negative integer 'tensors'"),
        (_record(summary={"tensors": 1, "by_status": {}}), "must give a count"),
        (
            _record(
                summary={
                    "tensors": 3,
                    "by_status": dict(unchanged=1, changed=1, added=0, removed=0, reshaped=0),
                }
            ),
            "must add up",
        ),
        (_record(tensors={}), "must be a list"),
        (_record(tensors=[1]), "tensors[0]: must be an object"),
        (_record(t_name=7), "tensors[0].name: must be a string"),
        (_record(t_status="moved"), "tensors[0].status: must be one of"),
        (_record(t_dtype_a=3), "tensors[0].dtype_a: must be null or a string"),
        (_record(t_shape_a=[1.5]), "shape_a: must be null or a list"),
        (_record(t_shape_a=[1] * 65), "shape_a: must be null or a list"),
        (_record(t_rank=-1), "tensors[0].rank: must be null or a non-negative integer"),
        (_record(t_top1_energy="x"), "top1_energy: must be null or a finite number"),
        (_record(t_features=[]), "features: must be null or an object"),
        (_record(t_features={"energy_rank": 1}), "features.signal_rank: is required"),
        (
            _record(t_features={"signal_rank": 1, "energy_rank": 1, "sigma": ["x"]}),
            "features.sigma: must be a list",
        ),
        (
            _record(t_features={"signal_rank": 1, "energy_rank": 1, "snr": True}),
            "features.snr: must be null or a finite number",
        ),
        (_record(t_quantization={"type": 4}), "quantization: must be null or an object"),
        (_record(features=[]), "record.features: must be null or an object"),
        (_record(features={"directions": -2}), "features.directions: must be null"),
        (_record(t_status="added"), "lists more tensors of a status"),
    ],
)
def test_validate_record_names_the_key(record, message):
    with pytest.raises(ClassifyError, match=None) as exc:
        validate_record(record, "record")
    assert message in str(exc.value)


def test_validate_record_refuses_repeats_and_missing_changes():
    record = _record()
    record["tensors"].append(dict(record["tensors"][0]))
    record["summary"]["by_status"] = dict(unchanged=0, changed=2, added=0, removed=0, reshaped=0)
    with pytest.raises(ClassifyError, match="repeats 'w'"):
        validate_record(record, "record")
    missing = _record(tensors=[])
    with pytest.raises(ClassifyError, match="must list every tensor that is not unchanged"):
        validate_record(missing, "record")


def test_side_argument_names_a_path_or_a_pinned_hub_commit():
    sha = "a" * 40
    assert side_argument({"a": {"path": "x.safetensors"}}, "a") == "x.safetensors"
    assert side_argument({"b": {"hf": {"repo": "o/n", "sha": sha}}}, "b") == f"hf:o/n@{sha}"
    for bad in ({}, {"a": {"hf": {"repo": "o/n", "sha": "main"}}}, {"a": {"hf": {"repo": "x"}}}):
        with pytest.raises(ClassifyError, match="names no checkpoint"):
            side_argument(bad, "a")


def test_m8_quantization_field_counts_as_a_cast():
    entries = [
        {
            "name": f"model.layers.0.w{i}",
            "status": "reshaped",
            "dtype_a": "BF16",
            "dtype_b": "F32",
            "shape_a": [4, 4],
            "shape_b": [4, 4],
            "changed": 16,
            "rel_delta_norm": 0.05,
            "quantization": {"type": "Q4_K"},
        }
        for i in range(10)
    ]
    record = {
        "summary": {
            "tensors": 10,
            "by_status": dict(unchanged=0, changed=0, added=0, removed=0, reshaped=10),
        },
        "tensors": entries,
    }
    from wvc.classify import classify_record

    result = classify_record(record)
    assert (result.kind, result.rule) == ("quantization", "R2")
    assert result.evidence["tolerance"] == pytest.approx(2 * 2.0**-3.5)
    entries[0]["rel_delta_norm"] = 0.5
    assert classify_record(record).rule == "M1"
    for entry in entries:
        entry["quantization"] = {"type": "IQ_UNKNOWN"}
    result = classify_record(record)
    assert result.evidence["tolerance"] == pytest.approx(2 * 2.0**-24)  # F32: no ggml bound


def test_hub_pair_measures_features_while_the_shards_are_open(tmp_path, rng, fake_hub, capsys):
    a, b = _pair(tmp_path, rng)
    fake_hub.add_repo("base/model", {"model.safetensors": a.read_bytes()})
    fake_hub.add_repo("edit/model", {"model.safetensors": b.read_bytes()})
    args = ["diff", "hf:base/model", "hf:edit/model", "--rank", "--json"]
    assert main([*args, "--classify", "--features"]) == 1
    record = json.loads(capsys.readouterr().out)
    assert (record["classification"]["kind"], record["classification"]["tier"]) == (
        "direction-removal",
        "strong",
    )
    path = tmp_path / "record.json"
    path.write_text(json.dumps({k: record[k] for k in ("a", "b", "summary", "tensors")}))
    assert main(["classify", str(path), "--features", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tier"] == "strong"


def test_widened_fine_tune_is_compared_as_a_change(tmp_path, rng, capsys):
    base = toy_model(rng)
    edited = {k: (v * 1.01 if v.ndim == 1 else v) for k, v in base.items()}
    a = write_toy(tmp_path / "a.safetensors", base)
    b = write_safetensors(
        tmp_path / "b.safetensors",
        {k: ("F32", np.asarray(v, dtype=np.float32)) for k, v in edited.items()},
    )
    assert main(["classify", str(a), str(b), "--json", "--features"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["evidence"]["vectors_changed"] == 2 * TOY_LAYERS + 1
    assert document["evidence"]["cast"] == 0


def test_features_without_signal_fall_back_to_the_record():
    from wvc.classify_input import build_view

    record = {
        "summary": {
            "tensors": 1,
            "by_status": dict(unchanged=0, changed=1, added=0, removed=0, reshaped=0),
        },
        "tensors": [
            {
                "name": "model.layers.0.mlp.down_proj.weight",
                "status": "changed",
                "shape_a": [4, 4],
                "shape_b": [4, 4],
                "features": {"signal_rank": 0, "energy_rank": 0, "signal_top1": None},
            }
        ],
    }
    view = build_view(validate_record(record))
    assert view.moved[0].features is None
