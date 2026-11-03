"""PEFT LoRA adapters: load_adapter, merge_adapter, commits with factors, and wvc merge-adapter."""

import json
import math
import os
import re

import numpy as np
import pytest
from helpers import bf16_bits_to_f32, commit, f32_to_bf16_bits, write_safetensors

from wvc.checkout import checkout_commit
from wvc.cli import main
from wvc.hashing import build_manifest
from wvc.lowrank import LowRankError, LowRankFactors
from wvc.merge_adapter import merge_adapter
from wvc.peft import load_adapter
from wvc.repo import (
    ManifestComparison,
    commit_checkpoint,
    commit_manifest,
    compare_manifests,
)
from wvc.safetensors_io import SafetensorsError, load_checkpoint, read_header
from wvc.store import ObjectStore, StoreError

SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
Q_PROJ = "model.layers.0.self_attn.q_proj.weight"  # BF16 [16, 12] in shard 1
K_PROJ = "model.layers.0.self_attn.k_proj.weight"  # BF16 [8, 12], last in shard 1's buffer
DOWN_PROJ = "model.layers.1.mlp.down_proj.weight"  # F16 [12, 20] in shard 2
TARGETS = (Q_PROJ, DOWN_PROJ)  # sorted by name
Q_A = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
Q_B = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
DOWN_A = "base_model.model.model.layers.1.mlp.down_proj.lora_A.default.weight"
DOWN_B = "base_model.model.model.layers.1.mlp.down_proj.lora_B.default.weight"
R, ALPHA = 4, 8
NUMPY_FOR = {"F32": "<f4", "F16": "<f2"}
CONFIG = {
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "base_model_name_or_path": "synthetic",
    "r": R,
    "lora_alpha": ALPHA,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "down_proj"],
    "bias": "none",
    "use_rslora": False,
    "use_dora": False,
    "fan_in_fan_out": False,
    "rank_pattern": {},
    "alpha_pattern": {},
    "modules_to_save": None,
    "layers_to_transform": None,
    "layers_pattern": None,
    "init_lora_weights": True,
    "inference_mode": True,
}


def bf16(values):
    return f32_to_bf16_bits(np.asarray(values, dtype=np.float32))


def write_base(directory, rng):
    """A two-shard base whose headers list tensors out of buffer order."""
    directory.mkdir(parents=True)
    shard_1 = {
        "model.embed_tokens.weight": ("F32", rng.standard_normal((32, 12)).astype(np.float32)),
        Q_PROJ: ("BF16", bf16(rng.standard_normal((16, 12)))),
        K_PROJ: ("BF16", bf16(rng.standard_normal((8, 12)))),
    }
    shard_2 = {
        "model.layers.0.input_layernorm.weight": ("F32", np.ones(12, dtype=np.float32)),
        DOWN_PROJ: ("F16", rng.standard_normal((12, 20)).astype(np.float16)),
        "model.layers.1.mlp.up_proj.weight": ("F16", rng.standard_normal((20, 12)).astype("f2")),
    }
    for name, tensors in zip(SHARDS, (shard_1, shard_2), strict=True):
        write_safetensors(directory / name, tensors, header_order=sorted(tensors, reverse=True))
    return directory


def adapter_tensors(rng, rank=R):
    """LoRA factors for q_proj (F32, plain names) and down_proj (BF16, '.default.' names)."""
    return {
        Q_A: ("F32", (0.05 * rng.standard_normal((rank, 12))).astype(np.float32)),
        Q_B: ("F32", (0.05 * rng.standard_normal((16, rank))).astype(np.float32)),
        DOWN_A: ("BF16", bf16(0.05 * rng.standard_normal((rank, 20)))),
        DOWN_B: ("BF16", bf16(0.05 * rng.standard_normal((12, rank)))),
    }


def write_adapter(directory, tensors, **config):
    """An adapter directory: ``CONFIG`` updated by ``config``, and ``tensors``."""
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(json.dumps({**CONFIG, **config}))
    write_safetensors(directory / "adapter_model.safetensors", tensors)
    return directory


@pytest.fixture
def base_dir(tmp_path, rng):
    return write_base(tmp_path / "base", rng)


@pytest.fixture
def base_commit(store, base_dir):
    return commit(store, base_dir, "base", kind="pretrain-checkpoint")


@pytest.fixture
def adapter_dir(tmp_path, rng):
    return write_adapter(tmp_path / "adapter", adapter_tensors(rng))


def merge(store, adapter, out, ref="HEAD", message="merge"):
    return merge_adapter(store, ref, adapter, out, message=message, author="tester")


def tensor_f64(path, name):
    """A tensor of a checkpoint decoded to float64 without wvc."""
    ckpt = load_checkpoint(path)
    info, raw = ckpt.tensors[name], np.asarray(ckpt.raw(name))
    if info.dtype == "BF16":
        values = bf16_bits_to_f32(raw.view("<u2"))
    else:
        values = raw.view(NUMPY_FOR[info.dtype])
    return values.astype(np.float64).reshape(info.shape), info.dtype


def spacing(values, dtype):
    """One unit in the last place of ``dtype`` at each magnitude."""
    magnitude = np.abs(values)
    if dtype == "F16":
        return np.spacing(magnitude.astype(np.float16)).astype(np.float64)
    gap = np.spacing(magnitude.astype(np.float32)).astype(np.float64)
    return gap * 2.0**16 if dtype == "BF16" else gap


def test_load_adapter_maps_each_pair_to_its_base_tensor(adapter_dir):
    adapter = load_adapter(adapter_dir)
    assert (adapter.r, adapter.lora_alpha, adapter.scale) == (R, ALPHA, 2.0)
    assert adapter.target_modules == ("down_proj", "q_proj")
    assert list(adapter.targets) == list(TARGETS)
    assert [(a.name, b.name) for a, b in adapter.targets.values()] == [
        (Q_A, Q_B),
        (DOWN_A, DOWN_B),
    ]
    ckpt = load_checkpoint(adapter_dir / "adapter_model.safetensors")
    assert adapter.tensors_root == build_manifest(ckpt)["root"]
    factors = adapter.factors(Q_PROJ)
    assert isinstance(factors, LowRankFactors)
    assert not isinstance(factors.a_raw, np.memmap)
    assert (factors.a_dtype, factors.a_shape, factors.b_dtype, factors.b_shape) == (
        "F32",
        (R, 12),
        "F32",
        (16, R),
    )
    assert factors.a_raw.tobytes() == np.asarray(ckpt.raw(Q_A)).tobytes()
    assert factors.scale == 2.0


@pytest.mark.parametrize(
    ("config", "scale", "operation"),
    [
        ({"use_rslora": True}, 4.0, {"lora_alpha": 8, "scale": "4.0"}),
        ({"lora_alpha": 6.5}, 1.625, {"lora_alpha": "6.5", "scale": "1.625"}),
        ({"lora_alpha": 16, "use_rslora": True, "r": 4}, 8.0, {"scale": "8.0"}),
        ({"target_modules": "q_proj|down_proj"}, 2.0, {"target_modules": "q_proj|down_proj"}),
        ({"target_modules": None}, 2.0, {"target_modules": None}),
        (
            {"modules_to_save": [], "layers_pattern": [], "use_dora": None, "rank_pattern": {}},
            2.0,
            {"scale": "2.0"},
        ),
    ],
)
def test_scale_and_operation_follow_the_config(tmp_path, rng, config, scale, operation):
    adapter = load_adapter(write_adapter(tmp_path / "adapter", adapter_tensors(rng), **config))
    assert adapter.scale == scale
    recorded = adapter.operation()
    assert recorded == {**recorded, **operation}
    assert set(recorded) == {
        "peft_type", "r", "lora_alpha", "scale", "target_modules", "tensors_root"
    }  # fmt: skip
    assert recorded["peft_type"] == "LORA"
    assert float(recorded["scale"]) == scale


def test_rslora_scale_is_alpha_over_the_square_root_of_r(tmp_path, rng):
    tensors = adapter_tensors(rng, rank=3)
    adapter = load_adapter(
        write_adapter(tmp_path / "adapter", tensors, r=3, lora_alpha=7, use_rslora=True)
    )
    assert adapter.scale == 7 / math.sqrt(3)
    assert adapter.operation()["scale"] == repr(7 / math.sqrt(3))


CONFIG_REFUSALS = [
    ({"peft_type": "LOHA"}, "peft_type 'LOHA' is not supported"),
    ({"peft_type": None}, "peft_type None is not supported"),
    ({"use_dora": True}, "use_dora is True"),
    ({"fan_in_fan_out": True}, "fan_in_fan_out is True"),
    ({"rank_pattern": {"q_proj": 8}}, "rank_pattern is"),
    ({"alpha_pattern": {"q_proj": 16}}, "alpha_pattern is"),
    ({"modules_to_save": ["lm_head"]}, "modules_to_save is"),
    ({"layers_pattern": "layers"}, "layers_pattern is 'layers'"),
    ({"layers_to_transform": [0]}, r"layers_to_transform is \[0\]"),
    ({"layers_to_transform": 0}, "layers_to_transform is 0"),
    ({"use_dora": "false"}, "use_dora must be true or false"),
    ({"use_rslora": 1}, "use_rslora must be true or false"),
    ({"r": 0}, "r must be a positive integer"),
    ({"r": "4"}, "r must be a positive integer"),
    ({"r": True}, "r must be a positive integer"),
    ({"lora_alpha": "8"}, "lora_alpha must be a finite number"),
    ({"lora_alpha": None}, "lora_alpha must be a finite number"),
    ({"lora_alpha": float("nan")}, "lora_alpha must be a finite number"),
    ({"lora_alpha": 10**400}, "lora_alpha must be a finite number"),
    ({"target_modules": 3}, "target_modules must be a list of names or a pattern"),
    ({"target_modules": ["q_proj", 3]}, "target_modules must be a list of names or a pattern"),
]


@pytest.mark.parametrize(("config", "message"), CONFIG_REFUSALS)
def test_unsupported_configs_are_refused(tmp_path, rng, config, message):
    adapter = write_adapter(tmp_path / "adapter", adapter_tensors(rng), **config)
    with pytest.raises(StoreError, match=message):
        load_adapter(adapter)


def without(tensors, *names):
    return {name: value for name, value in tensors.items() if name not in names}


TENSOR_REFUSALS = {
    "missing-half": (lambda t: without(t, Q_B), "q_proj has lora_A but no lora_B"),
    "missing-other-half": (lambda t: without(t, DOWN_A), "down_proj has lora_B but no lora_A"),
    "lora-bias": (
        lambda t: {**t, Q_B.replace(".weight", ".bias"): ("F32", np.zeros(16, np.float32))},
        "is not a LoRA factor named",
    ),
    "base-bias": (
        lambda t: {**t, "base_model.model.model.norm.bias": ("F32", np.zeros(12, np.float32))},
        "is not a LoRA factor named",
    ),
    "embedding": (
        lambda t: {
            **t,
            "base_model.model.model.embed_tokens.lora_embedding_A": (
                "F32",
                np.zeros((R, 32), np.float32),
            ),
        },
        "is not a LoRA factor named",
    ),
    "no-prefix": (
        lambda t: {name.removeprefix("base_model.model."): value for name, value in t.items()},
        "is not a LoRA factor named",
    ),
    "other-rank": (
        lambda t: {
            **t,
            Q_A: ("F32", np.zeros((R - 1, 12), np.float32)),
            Q_B: ("F32", np.zeros((16, R - 1), np.float32)),
        },
        r"q_proj has lora_A \[3, 12\] and lora_B \[16, 3\], which are not of rank r = 4",
    ),
    "integer-factor": (
        lambda t: {**t, Q_A: ("I8", np.zeros((R, 12), np.int8))},
        r"lora_A.weight is I8\[4, 12\]; LoRA factors must be 2-D F32, F16, BF16",
    ),
    "conv-factor": (
        lambda t: {**t, Q_A: ("F32", np.zeros((R, 12, 1, 1), np.float32))},
        "LoRA factors must be 2-D",
    ),
    "two-adapters": (
        lambda t: {
            **t,
            Q_A.replace("lora_A.", "lora_A.other."): t[Q_A],
            Q_B.replace("lora_B.", "lora_B.other."): t[Q_B],
        },
        f"two adapters update {re.escape(Q_PROJ)}",
    ),
    "empty": (lambda t: {}, "holds no LoRA weights"),
}


@pytest.mark.parametrize(
    ("change", "message"), TENSOR_REFUSALS.values(), ids=TENSOR_REFUSALS.keys()
)
def test_adapter_files_holding_anything_but_lora_pairs_are_refused(tmp_path, rng, change, message):
    adapter = write_adapter(tmp_path / "adapter", change(adapter_tensors(rng)))
    with pytest.raises(StoreError, match=message):
        load_adapter(adapter)


def test_missing_or_malformed_adapter_files_are_refused(tmp_path, rng):
    adapter = write_adapter(tmp_path / "adapter", adapter_tensors(rng))
    (adapter / "adapter_model.safetensors").unlink()
    with pytest.raises(StoreError, match=re.escape("adapter_model.safetensors is missing")):
        load_adapter(adapter)
    (adapter / "adapter_config.json").unlink()
    with pytest.raises(StoreError, match=re.escape("adapter_config.json is missing")):
        load_adapter(adapter)
    (adapter / "adapter_config.json").write_text("{not json")
    with pytest.raises(StoreError, match="is not UTF-8 JSON"):
        load_adapter(adapter)
    (adapter / "adapter_config.json").write_text("[]")
    with pytest.raises(StoreError, match="does not hold a JSON object"):
        load_adapter(adapter)
    (adapter / "adapter_config.json").write_text(json.dumps(CONFIG))
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 4)
    with pytest.raises(SafetensorsError, match="shorter than the 8-byte header"):
        load_adapter(adapter)


def test_deeply_nested_or_oversized_adapter_config_is_refused(tmp_path, rng):
    adapter = write_adapter(tmp_path / "adapter", adapter_tensors(rng))
    (adapter / "adapter_config.json").write_text("[" * 100_000 + "]" * 100_000)
    with pytest.raises(StoreError, match=re.escape("adapter_config.json is not UTF-8 JSON")):
        load_adapter(adapter)
    (adapter / "adapter_config.json").write_text(" " * (1 << 20) + "{}")
    with pytest.raises(StoreError, match="more than the 1048576 bytes allowed"):
        load_adapter(adapter)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are POSIX only")
@pytest.mark.parametrize("name", ["adapter_config.json", "adapter_model.safetensors"])
def test_adapter_files_that_are_fifos_are_refused_without_blocking(tmp_path, rng, name):
    adapter = write_adapter(tmp_path / "adapter", adapter_tensors(rng))
    (adapter / name).unlink()
    os.mkfifo(adapter / name)
    with pytest.raises(StoreError, match="not a regular file"):
        load_adapter(adapter)


def test_merge_adapter_stores_the_targets_as_lowrank_deltas(
    store, base_commit, base_dir, adapter_dir, tmp_path
):
    out = tmp_path / "merged"
    result = merge(store, adapter_dir, out)
    assert (result.parent_id, store.head()) == (base_commit.commit_id, result.commit_id)
    assert (result.tensors, result.reused, result.stored_full) == (6, 4, 0)
    assert result.stored_delta == {"delta-lowrank": 2}
    record = store.get_commit(result.commit_id)
    adapter_root = build_manifest(load_checkpoint(adapter_dir / "adapter_model.safetensors"))
    assert record["operation"] == {
        "kind": "adapter-merge",
        "adapter": {
            "peft_type": "LORA",
            "r": R,
            "lora_alpha": ALPHA,
            "scale": "2.0",
            "target_modules": ["down_proj", "q_proj"],
            "tensors_root": adapter_root["root"],
        },
    }
    old, new = (commit_manifest(store, c) for c in (base_commit.commit_id, result.commit_id))
    assert compare_manifests(old, new) == ManifestComparison(4, TARGETS, (), ())
    for name in TARGETS:
        obj = store.get_object(new["tensors"][name]["sha256"])
        assert (obj["form"], obj["base"], obj["depth"]) == (
            "delta-lowrank",
            old["tensors"][name]["sha256"],
            1,
        )
        assert (obj["rank"], obj["scale"], obj["procedure"]) == (R, "2.0", "lowrank-merge/0")
    files = [(f["name"], f["header"], f["nbytes"]) for f in record["files"]]
    base_files = store.get_commit(base_commit.commit_id)["files"]
    assert files == [(f["name"], f["header"], f["nbytes"]) for f in base_files]
    assert sorted(path.name for path in out.iterdir()) == list(SHARDS)


def test_only_the_target_byte_ranges_change(store, base_commit, base_dir, adapter_dir, tmp_path):
    out = tmp_path / "merged"
    merge(store, adapter_dir, out)
    for name in SHARDS:
        before = np.frombuffer((base_dir / name).read_bytes(), dtype=np.uint8)
        after = np.frombuffer((out / name).read_bytes(), dtype=np.uint8)
        file = read_header(out / name)
        inside = np.zeros(after.size, dtype=bool)
        for info in file.tensors.values():
            if info.name in TARGETS:
                inside[file.data_start + info.begin : file.data_start + info.end] = True
        assert before.size == after.size
        assert inside.any()
        assert np.array_equal(before[~inside], after[~inside])
        assert (before[inside] != after[inside]).mean() > 0.3


def test_checkout_of_the_merge_reproduces_the_merged_files(
    store, base_commit, adapter_dir, tmp_path
):
    result = merge(store, adapter_dir, tmp_path / "merged")
    checked = checkout_commit(store, result.commit_id, tmp_path / "again")
    assert [file.name for file in checked.files] == list(SHARDS)
    for name in SHARDS:
        assert (tmp_path / "again" / name).read_bytes() == (tmp_path / "merged" / name).read_bytes()


def test_merged_values_match_a_float64_reference(
    store, base_commit, base_dir, adapter_dir, tmp_path
):
    out = tmp_path / "merged"
    merge(store, adapter_dir, out)
    adapter = adapter_dir / "adapter_model.safetensors"
    for target, (a_name, b_name) in ((Q_PROJ, (Q_A, Q_B)), (DOWN_PROJ, (DOWN_A, DOWN_B))):
        w, dtype = tensor_f64(base_dir, target)
        a, b = (tensor_f64(adapter, name)[0] for name in (a_name, b_name))
        reference = w + 2.0 * (b @ a)
        result = tensor_f64(out, target)[0]
        # Half an ulp of the base dtype, plus half a float32 ulp for the rounding to float32
        # that comes first, plus the float64 rounding by which the two sums can differ.
        tolerance = 0.5 * spacing(result, dtype) + 0.5 * spacing(reference, "F32")
        magnitude = np.abs(w) + 2.0 * (np.abs(b) @ np.abs(a))
        tolerance += (R + 2) * np.finfo(np.float64).eps * magnitude
        assert np.all(np.abs(result - reference) <= tolerance), target
        assert (result != w).mean() > 0.3


def test_merging_again_chains_lowrank_deltas(store, base_commit, adapter_dir, tmp_path, rng):
    first = merge(store, adapter_dir, tmp_path / "first")
    second_adapter = write_adapter(tmp_path / "second-adapter", adapter_tensors(rng))
    second = merge(store, second_adapter, tmp_path / "second", message="again")
    assert second.parent_id == first.commit_id
    assert second.stored_delta == {"delta-lowrank": 2}
    entries = commit_manifest(store, second.commit_id)["tensors"]
    assert {store.get_object(entries[name]["sha256"])["depth"] for name in TARGETS} == {2}
    checkout_commit(store, second.commit_id, tmp_path / "again")
    for name in SHARDS:
        assert (tmp_path / "again" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()


def test_depth_limit_falls_back_to_full_objects(store, base_commit, adapter_dir, tmp_path):
    store.write_config({"format": "wvc-repo/0", "max_delta_depth": 0})
    result = merge(store, adapter_dir, tmp_path / "merged")
    assert (result.reused, result.stored_full, result.stored_delta) == (4, 2, {})
    entries = commit_manifest(store, result.commit_id)["tensors"]
    assert {store.get_object(entries[name]["sha256"])["form"] for name in TARGETS} == {
        "zstd-planes"
    }
    checkout_commit(store, result.commit_id, tmp_path / "again")
    for name in SHARDS:
        assert (tmp_path / "again" / name).read_bytes() == (tmp_path / "merged" / name).read_bytes()


def test_merge_onto_an_earlier_commit(store, base_commit, adapter_dir, tmp_path):
    first = merge(store, adapter_dir, tmp_path / "first")
    # H3: the base is not the head of main, so the merge needs a branch of its own; the refusal
    # comes before the output directory is written.
    with pytest.raises(StoreError, match=r"not the head of branch 'main'.*--branch"):
        merge(store, adapter_dir, tmp_path / "again", ref=base_commit.commit_id[:7])
    assert not (tmp_path / "again").exists()
    assert store.head() == first.commit_id
    again = merge_adapter(
        store,
        base_commit.commit_id[:7],
        adapter_dir,
        tmp_path / "again",
        message="merge",
        author="tester",
        branch="again",
    )
    assert again.parent_id == base_commit.commit_id
    assert store.head() == again.commit_id
    assert (store.current_branch(), store.read_ref("main")) == ("again", first.commit_id)
    assert again.manifest_root == first.manifest_root  # the same merge, so every tensor reused
    assert (again.reused, again.stored_delta, again.payload_bytes) == (6, {}, 0)


def refused_merge(store, adapter, out, message):
    head, before = store.head(), sorted(store.dir.rglob("*"))
    with pytest.raises(StoreError, match=message):
        merge(store, adapter, out)
    assert store.head() == head
    assert sorted(store.dir.rglob("*")) == before


def test_targets_the_base_cannot_take_are_refused_before_writing(store, base_commit, tmp_path, rng):
    tensors = adapter_tensors(rng)
    renamed = {name.replace("layers.1.mlp", "layers.7.mlp"): v for name, v in tensors.items()}
    cases = {
        "missing": (renamed, "'model.layers.7.mlp.down_proj.weight' is not a tensor of the base"),
        "in-dim": (
            {**tensors, Q_A: ("F32", np.zeros((R, 13), np.float32))},
            r"A \[4, 13\] and B \[16, 4\] do not factor W \[16, 12\]",
        ),
        "out-dim": (
            {**tensors, Q_B: ("F32", np.zeros((17, R), np.float32))},
            r"A \[4, 12\] and B \[17, 4\] do not factor W \[16, 12\]",
        ),
        "not-2-d": (
            {
                "base_model.model.model.layers.0.input_layernorm.lora_A.weight": tensors[Q_A],
                "base_model.model.model.layers.0.input_layernorm.lora_B.weight": tensors[Q_B],
            },
            "W: a shape must be two non-negative integers",
        ),
    }
    for name, (factors, message) in cases.items():
        adapter = write_adapter(tmp_path / f"adapter-{name}", factors)
        out = tmp_path / f"out-{name}"
        refused_merge(store, adapter, out, message)
        assert not out.exists()


def test_integer_targets_and_odd_file_names_are_refused(store, tmp_path, rng):
    ints = {Q_PROJ: ("I32", rng.integers(-5, 5, size=(16, 12), dtype=np.int32))}
    single = write_safetensors(tmp_path / "weights.bin", ints)
    with pytest.raises(StoreError, match=r"'weights\.bin' is not an allowed file name"):  # H5
        commit_checkpoint(store, single, message="ints", operation_kind="unknown", author="t")
    adapter = write_adapter(tmp_path / "adapter", without(adapter_tensors(rng), DOWN_A, DOWN_B))
    (tmp_path / "ints").mkdir()
    ints = {Q_PROJ: ("I32", rng.integers(-5, 5, size=(16, 12), dtype=np.int32))}
    named = write_safetensors(tmp_path / "ints" / "model.safetensors", ints)
    commit_checkpoint(store, named.parent, message="named", operation_kind="unknown", author="t")
    refused_merge(store, adapter, tmp_path / "out", "is I32 in the base")
    assert not (tmp_path / "out").exists()


def test_a_non_empty_output_directory_or_bad_text_is_refused(
    store, base_commit, adapter_dir, tmp_path
):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    refused_merge(store, adapter_dir, out, "non-empty directory")
    assert [path.name for path in out.iterdir()] == ["keep.txt"]
    with pytest.raises(StoreError, match="message is empty"):
        merge(store, adapter_dir, tmp_path / "other", message=" ")
    assert not (tmp_path / "other").exists()
    with pytest.raises(StoreError, match="no commit matches"):
        merge(store, adapter_dir, tmp_path / "other", ref="1234567")


def test_merge_checks_what_it_finds_in_the_checkout(
    store, base_commit, adapter_dir, tmp_path, monkeypatch
):
    def broken(*args):
        raise LowRankError("broken merge")

    cases = {
        "are not the base tensor": ("wvc.merge_adapter.digest_bytes", lambda view: "0" * 64),
        "do not hold 'model.layers.0": ("wvc.merge_adapter._locate", lambda out, files: {}),
        "adapter target 'model.layers.0.self_attn.q_proj.weight': broken merge": (
            "wvc.merge_adapter.merge_lowrank",
            broken,
        ),
    }
    for index, (message, (target, replacement)) in enumerate(cases.items()):
        monkeypatch.setattr(target, replacement)
        with pytest.raises(StoreError, match=re.escape(message)):
            merge(store, adapter_dir, tmp_path / f"out-{index}")
        monkeypatch.undo()
        assert store.head() == base_commit.commit_id


def test_commit_checkpoint_checks_its_lowrank_and_operation_arguments(
    store, base_commit, base_dir, adapter_dir, tmp_path
):
    factors = load_adapter(adapter_dir).factors(Q_PROJ)
    zero_b = LowRankFactors(  # A merge that changes nothing, against an edited K_PROJ
        factors.a_raw, "F32", (R, 12), np.zeros(8 * R * 4, np.uint8), "F32", (8, R), 2.0
    )
    cases = {
        "must not set 'kind'": {"operation_extra": {"kind": "finetune"}},
        "forbids floats": {"operation_extra": {"lr": 1e-4}},
        "lacks: \\['model.missing.weight'\\]": {"lowrank": {"model.missing.weight": factors}},
        "did not follow the procedure": {"lowrank": {K_PROJ: zero_b}},
    }
    edited = tmp_path / "edited"
    checkout_commit(store, "HEAD", edited)
    path = edited / SHARDS[0]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1  # the last byte of K_PROJ, the last tensor of shard 1, so something changed
    path.write_bytes(bytes(data))
    for message, options in cases.items():
        head, before = store.head(), sorted(store.dir.rglob("*"))
        with pytest.raises(StoreError, match=message):
            commit_checkpoint(
                store, edited, message="m", operation_kind="direct-edit", author="t", **options
            )
        assert store.head() == head
        assert sorted(store.dir.rglob("*")) == before


def test_lowrank_factors_need_a_parent_tensor(store, base_dir, adapter_dir):
    factors = load_adapter(adapter_dir).factors(Q_PROJ)
    with pytest.raises(StoreError, match="needs the parent's tensor of the same name"):
        commit_checkpoint(
            store,
            base_dir,
            message="m",
            operation_kind="adapter-merge",
            author="t",
            lowrank={Q_PROJ: factors},
        )


# The command line


def run(capsys, *argv):
    code = main([str(arg) for arg in argv])
    out, err = capsys.readouterr()
    return code, out, err


def test_cli_merge_adapter_prints_the_commit_and_the_targets(
    tmp_path, base_dir, adapter_dir, capsys, monkeypatch
):
    monkeypatch.setenv("WVC_AUTHOR", "Evan")
    repo = tmp_path / "repo"
    assert run(capsys, "init", repo)[0] == 0
    code, out, _ = run(capsys, "commit", base_dir, "-m", "base", "--repo", repo)
    base_id = re.match(r"commit ([0-9a-f]{64}) ", out)[1]
    merged = tmp_path / "merged"
    code, out, err = run(
        capsys, "merge-adapter", "HEAD", adapter_dir, "--out", merged, "-m", "merge", "--repo", repo
    )
    lines = out.splitlines()
    assert (code, err, len(lines)) == (0, "", 4)
    assert re.fullmatch(rf"commit [0-9a-f]{{64}} \(main, parent {base_id[:12]}\)", lines[0])
    assert lines[1] == (
        "tensors: 6 (4 reused by hash, 0 stored full, 2 stored as delta: 2 delta-lowrank)"
    )
    assert re.fullmatch(r"payload bytes written: [\d,]+ in \d+\.\d\d s", lines[2])
    assert lines[3] == "adapter targets merged: 2 (r 4, lora_alpha 8, scale 2.0)"
    store = ObjectStore(repo)
    record = store.get_commit(store.head())
    assert (record["author"], record["operation"]["kind"]) == ("Evan", "adapter-merge")
    code, out, _ = run(capsys, "log", "--repo", repo)
    assert code == 0
    assert "adapter-merge" in out.splitlines()[0]


def test_cli_merge_adapter_refusals_exit_2(tmp_path, base_dir, adapter_dir, capsys, monkeypatch):
    monkeypatch.setenv("WVC_AUTHOR", "Evan")
    repo = tmp_path / "repo"
    run(capsys, "init", repo)
    run(capsys, "commit", base_dir, "-m", "base", "--repo", repo)
    bad = write_adapter(tmp_path / "dora", {}, use_dora=True)
    for adapter, message in ((bad, "use_dora is True"), (tmp_path, "adapter_config.json")):
        code, out, err = run(
            capsys, "merge-adapter", "HEAD", adapter, "--out", tmp_path / "o", "-m", "m",
            "--repo", repo,
        )  # fmt: skip
        assert (code, out) == (2, "")
        assert err.startswith("wvc: ")
        assert message in err
    assert not (tmp_path / "o").exists()
    with pytest.raises(SystemExit) as exit_info:
        main(["merge-adapter", "HEAD", str(bad), "--repo", str(repo)])
    assert exit_info.value.code == 2
    assert "--out" in capsys.readouterr().err


def test_cli_merge_adapter_onto_an_earlier_commit_needs_a_branch(
    tmp_path, base_dir, adapter_dir, capsys, monkeypatch
):
    monkeypatch.setenv("WVC_AUTHOR", "Evan")
    repo = tmp_path / "repo"
    run(capsys, "init", repo)
    code, out, _ = run(capsys, "commit", base_dir, "-m", "base", "--repo", repo)
    base_id = re.match(r"commit ([0-9a-f]{64}) ", out)[1]
    merge_argv = ("merge-adapter", base_id[:7], adapter_dir, "-m", "merge", "--repo", repo)
    assert run(capsys, *merge_argv, "--out", tmp_path / "first")[0] == 0
    code, out, err = run(capsys, *merge_argv, "--out", tmp_path / "again")
    assert (code, out) == (2, "")
    assert "--branch" in err
    assert not (tmp_path / "again").exists()
    code, out, err = run(capsys, *merge_argv, "--out", tmp_path / "again", "--branch", "again")
    assert code == 0
    assert re.fullmatch(
        rf"commit [0-9a-f]{{64}} \(again, parent {base_id[:12]}\)", out.splitlines()[0]
    )
