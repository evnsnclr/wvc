"""GGUF uploads on the Hub: ``hf:OWNER/NAME[@REVISION]:FILE.gguf``, split files, all offline."""

import hashlib
import json
from pathlib import Path

import gguf as gguf_py
import numpy as np
import pytest
from gguf import GGMLQuantizationType, quants
from gguf_helpers import INT32, UINT16, UINT32, Tensor, Value, gguf_bytes, plain
from helpers import FakeSibling, write_safetensors

from wvc.cli import main
from wvc.gguf_io import FILE_TYPES, GgufError, load_gguf, parse_gguf, read_gguf
from wvc.gguf_split import check_parts, split_names
from wvc.hashing import build_manifest, gguf_hash_lines
from wvc.hub import (
    Hub,
    HubError,
    HubRef,
    HubRepo,
    gguf_choices,
    load_hub_gguf,
    parse_hub_argument,
    parse_hub_checkpoint,
)
from wvc.safetensors_io import load_checkpoint

HIDDEN, FFN, VOCAB = 32, 64, 8
QWEN_NAMES = {  # Hub name -> GGUF name under the qwen2 table
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "model.layers.0.input_layernorm.weight": "blk.0.attn_norm.weight",
    "model.layers.0.mlp.down_proj.weight": "blk.0.ffn_down.weight",
}
SHAPES = {
    "model.embed_tokens.weight": (VOCAB, HIDDEN),
    "model.norm.weight": (HIDDEN,),
    "model.layers.0.input_layernorm.weight": (HIDDEN,),
    "model.layers.0.mlp.down_proj.weight": (HIDDEN, FFN),
}
QUANTIZED = ("model.embed_tokens.weight", "model.layers.0.mlp.down_proj.weight")
SPLIT = [f"m-{i:05d}-of-00003.gguf" for i in (1, 2, 3)]


def _base(rng):
    return {name: rng.standard_normal(shape).astype(np.float32) for name, shape in SHAPES.items()}


def _quantized(base, dtype="Q8_0"):
    """The GGUF tensors of ``base`` under qwen2 names, matrices quantized to ``dtype``."""
    tensors = {}
    for name, values in base.items():
        if name in QUANTIZED:
            raw = quants.quantize(values, GGMLQuantizationType[dtype]).tobytes()
            tensors[QWEN_NAMES[name]] = Tensor(dtype, values.shape, raw)
        else:
            tensors[QWEN_NAMES[name]] = plain("F32", values)
    return tensors


def _meta(**extra):
    return {"general.architecture": "qwen2", "general.file_type": 7, **extra}


def _split_meta(no, count=3, total=4, **extra):
    split = {
        "split.no": Value(UINT16, no),
        "split.count": Value(UINT16, count),
        "split.tensors.count": Value(INT32, total),
    }
    return {**(_meta() if no == 0 else {}), **split, **extra}


def _split_files(tensors, metas=None):
    """Three parts of ``tensors`` (two, one, and one tensors), as gguf-split lays them out."""
    names = list(tensors)
    groups = [names[:2], names[2:3], names[3:]]
    metas = metas or [_split_meta(index) for index in range(3)]
    return {
        part: gguf_bytes({name: tensors[name] for name in group}, meta)
        for part, group, meta in zip(SPLIT, groups, metas, strict=True)
    }


def _calls(fake_hub, method):
    return [kwargs for name, kwargs in fake_hub.calls if name == method]


def _downloads(fake_hub):
    return sorted(call["filename"] for call in _calls(fake_hub, "hf_hub_download"))


def _diff_json(capsys, *args):
    status = main(["diff", "--json", "--all", *args])
    captured = capsys.readouterr()
    return status, json.loads(captured.out), captured.err


# Argument forms


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("hf:owner/name", HubRef("owner/name", "main")),
        ("hf:owner/name@v1.0", HubRef("owner/name", "v1.0")),
        ("hf:owner/name:m-Q4_K_M.gguf", HubRef("owner/name", "main", "m-Q4_K_M.gguf")),
        ("hf:owner/name@v1.0:BF16/m.gguf", HubRef("owner/name", "v1.0", "BF16/m.gguf")),
        ("hf:o/n@refs/pr/3:a/b/c+d.gguf", HubRef("o/n", "refs/pr/3", "a/b/c+d.gguf")),
        ("hf:o/n:m-00002-of-00003.gguf", HubRef("o/n", "main", "m-00002-of-00003.gguf")),
    ],
)
def test_every_hub_argument_form_parses(arg, expected):
    assert parse_hub_argument(arg) == expected


@pytest.mark.parametrize(
    ("arg", "message"),
    [
        ("hf:owner/name:", "must be a .gguf file"),
        ("hf:owner/name:model.safetensors", "must be a .gguf file"),
        ("hf:owner/name:m.GGUF", "must be a .gguf file"),
        ("hf:owner/name:../m.gguf", "relative path"),
        ("hf:owner/name:/m.gguf", "relative path"),
        ("hf:owner/name:a//m.gguf", "relative path"),
        ("hf:owner/name:./m.gguf", "relative path"),
        ("hf:owner/name:.hidden.gguf", "relative path"),
        ("hf:owner/name:a b.gguf", "relative path"),
        ("hf:owner/name:a\\m.gguf", "relative path"),
        ("hf:owner/name:m.gguf:x.gguf", "relative path"),
        ("hf:owner/name:" + "/".join("d" * 9) + ".gguf", "at most 8 parts"),
        ("hf:owner/name@:m.gguf", "OWNER/NAME"),
        ("hf:gpt2:m.gguf", "OWNER/NAME"),
        ("hf://owner/name:m.gguf", "names a wvc remote"),
        ("owner/name:m.gguf", "OWNER/NAME"),
    ],
)
def test_malformed_gguf_arguments_are_refused(arg, message):
    with pytest.raises(HubError, match=message):
        parse_hub_argument(arg)


def test_the_repository_parser_refuses_a_file_argument():
    with pytest.raises(HubError, match="names one file"):
        parse_hub_checkpoint("hf:owner/name:m.gguf")


# Choosing a file


def test_a_bare_argument_on_a_gguf_repository_lists_the_files_to_choose(rng, fake_hub, capsys):
    tensors = _quantized(_base(rng))
    q8, q4 = gguf_bytes(tensors, _meta()), gguf_bytes(tensors, _meta(**{"general.file_type": 15}))
    split = _split_files(tensors)
    files = {"m-Q8_0.gguf": q8, "sub/m-Q4_K_M.gguf": q4, **split, "README.md": b"# card\n"}
    files.update(
        {"bad name.gguf": q8, "orphan-00002-of-00004.gguf": q8, "x-00000-of-00002.gguf": q8}
    )
    fake_hub.add_repo("quant/m-GGUF", files)
    assert main(["hash", "hf:quant/m-GGUF"]) == 2
    err = capsys.readouterr().err
    lines = err.splitlines()
    assert lines[0] == "GGUF checkpoints in hf:quant/m-GGUF@main:"
    assert lines[1:6] == [
        f"  m-00001-of-00003.gguf  {sum(map(len, split.values())):,} bytes in 3 parts",
        f"  m-Q8_0.gguf  {len(q8):,} bytes",
        f"  orphan-00001-of-00004.gguf  {len(q8):,} bytes in 4 parts (1 listed)",
        f"  sub/m-Q4_K_M.gguf  {len(q4):,} bytes",
        f"  x-00000-of-00002.gguf  {len(q8):,} bytes",
    ]
    assert lines[6] == (
        "wvc: hf:quant/m-GGUF@main: no top-level *.safetensors files, but 5 GGUF checkpoints; "
        "name one as hf:quant/m-GGUF@main:FILE.gguf"
    )
    assert "bad name" not in err
    assert _downloads(fake_hub) == []


def test_a_one_file_gguf_repository_says_checkpoint(rng, fake_hub, capsys):
    fake_hub.add_repo("quant/one", {"m.gguf": gguf_bytes(_quantized(_base(rng)), _meta())})
    assert main(["diff", "hf:quant/one", "hf:quant/one"]) == 2
    assert "but 1 GGUF checkpoint; name one as" in capsys.readouterr().err


def test_a_repository_with_safetensors_and_gguf_files_reads_the_safetensors(
    tmp_path, rng, fake_hub, capsys
):
    base = _base(rng)
    st = write_safetensors(tmp_path / "model.safetensors", {k: ("F32", v) for k, v in base.items()})
    files = {"model.safetensors": st.read_bytes(), "m-Q8_0.gguf": gguf_bytes(_quantized(base))}
    fake_hub.add_repo("base/m", files)
    assert main(["hash", "hf:base/m"]) == 0
    with load_checkpoint(st) as ckpt:
        assert json.loads(capsys.readouterr().out) == build_manifest(ckpt)
    assert _downloads(fake_hub) == ["model.safetensors"]


def test_a_file_the_repository_does_not_list_is_refused(rng, fake_hub, capsys):
    fake_hub.add_repo("quant/m", {"m-Q8_0.gguf": gguf_bytes(_quantized(_base(rng)), _meta())})
    sha = fake_hub.sha_of("quant/m")
    assert main(["hash", "hf:quant/m:m-Q4_K_M.gguf"]) == 2
    assert capsys.readouterr().err == (
        f"wvc: hf:quant/m@main:m-Q4_K_M.gguf: the repository lists no file m-Q4_K_M.gguf at {sha}\n"
    )
    assert _downloads(fake_hub) == []


# Hashing and verifying one file


def test_hash_gguf_hash_of_a_hub_file_matches_the_local_file(tmp_path, rng, fake_hub, capsys):
    data = gguf_bytes(_quantized(_base(rng)), _meta())
    local = tmp_path / "m-Q8_0.gguf"
    local.write_bytes(data)
    fake_hub.add_repo("quant/m", {"sub/m-Q8_0.gguf": data, "m-Q4_K_M.gguf": data + b"x" * 32})
    sha = fake_hub.sha_of("quant/m")
    assert main(["hash", "--format", "gguf-hash", "hf:quant/m:sub/m-Q8_0.gguf"]) == 0
    captured = capsys.readouterr()
    with load_gguf(local) as ckpt:
        assert captured.out == "\n".join(gguf_hash_lines(ckpt)) + "\n"
        manifest = build_manifest(ckpt)
    digest = hashlib.sha256(data).hexdigest()
    assert captured.err.splitlines() == [
        f"pinned: hf:quant/m@main:sub/m-Q8_0.gguf at {sha}",
        f"file: sub/m-Q8_0.gguf ({len(data):,} bytes, sha256 {digest} checked)",
    ]
    assert [(c["filename"], c["revision"]) for c in _calls(fake_hub, "hf_hub_download")] == [
        ("sub/m-Q8_0.gguf", sha)
    ]
    assert main(["hash", "hf:quant/m:sub/m-Q8_0.gguf"]) == 0
    assert json.loads(capsys.readouterr().out) == manifest


def test_verify_a_hub_gguf_file_carries_the_files_it_read(tmp_path, rng, fake_hub, capsys):
    data = gguf_bytes(_quantized(_base(rng)), _meta())
    local = tmp_path / "m.gguf"
    local.write_bytes(data)
    fake_hub.add_repo("quant/m", {"m.gguf": data})
    manifest = tmp_path / "manifest.json"
    with load_gguf(local) as ckpt:
        manifest.write_text(json.dumps(build_manifest(ckpt)), encoding="utf-8")
    assert main(["verify", "--json", "hf:quant/m:m.gguf", str(manifest)]) == 0
    record = json.loads(capsys.readouterr().out)
    digest = hashlib.sha256(data).hexdigest()
    assert record["hf"] == {
        "repo": "quant/m",
        "revision": "main",
        "sha": fake_hub.sha_of("quant/m"),
        "file": "m.gguf",
        "files": [{"name": "m.gguf", "nbytes": len(data), "sha256": digest}],
    }
    assert record["root_match"] is True


def test_a_download_whose_sha256_differs_from_the_listing_is_refused(rng, fake_hub, capsys):
    data = gguf_bytes(_quantized(_base(rng)), _meta())
    fake_hub.add_repo("quant/m", {"m.gguf": data})
    real = fake_hub.hf_hub_download

    def tampered(repo_id, filename, **kwargs):
        local = Path(real(repo_id, filename, **kwargs))
        local.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))  # same size, other bytes
        return str(local)

    fake_hub.hf_hub_download = tampered
    assert main(["hash", "hf:quant/m:m.gguf"]) == 2
    err = capsys.readouterr().err
    assert "m.gguf downloaded with SHA-256" in err
    assert "the Hub listed " + hashlib.sha256(data).hexdigest() in err


def test_a_plain_blob_gguf_file_is_read_without_a_sha256(rng, fake_hub, capsys):
    data = gguf_bytes(_quantized(_base(rng)), _meta())
    fake_hub.add_repo("quant/m", {"m.gguf": data})
    fake_hub.plain.add("m.gguf")
    assert main(["hash", "--root-only", "hf:quant/m:m.gguf"]) == 0
    assert f"file: m.gguf ({len(data):,} bytes, no LFS sha256 listed)" in capsys.readouterr().err


# Split files


def test_any_part_of_a_split_file_fetches_every_part(tmp_path, rng, fake_hub, capsys):
    tensors = _quantized(_base(rng))
    whole = tmp_path / "whole.gguf"
    whole.write_bytes(gguf_bytes(tensors, _meta()))
    fake_hub.add_repo("quant/split", {**_split_files(tensors), "other.gguf": whole.read_bytes()})
    with load_gguf(whole) as ckpt:
        expected = build_manifest(ckpt)
    for part in SPLIT:
        fake_hub.calls.clear()
        assert main(["hash", f"hf:quant/split:{part}"]) == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out) == expected
        assert [c["filename"] for c in _calls(fake_hub, "hf_hub_download")] == SPLIT
        assert sum(line.startswith("file: m-0000") for line in captured.err.splitlines()) == 3
    assert main(["hash", "--format", "gguf-hash", f"hf:quant/split:{SPLIT[1]}"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[-1] for line in lines] == [
        f"{SPLIT[0]}:token_embd.weight", f"{SPLIT[0]}:output_norm.weight", SPLIT[0],
        f"{SPLIT[1]}:blk.0.attn_norm.weight", SPLIT[1],
        f"{SPLIT[2]}:blk.0.ffn_down.weight", SPLIT[2],
    ]  # fmt: skip


def test_parts_written_by_the_reference_writer_assemble(tmp_path, rng):
    writer = gguf_py.GGUFWriter(tmp_path / "ref.gguf", "qwen2", split_max_tensors=2)
    writer.add_file_type(gguf_py.LlamaFileType.MOSTLY_Q8_0)
    for index in range(5):
        writer.add_tensor(f"t{index}", rng.standard_normal(32).astype(np.float32))
    writer.write_header_to_file(tmp_path / "ref.gguf")
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    names = split_names("ref-00002-of-00003.gguf")
    assert names == [f"ref-{i:05d}-of-00003.gguf" for i in (1, 2, 3)]
    parts = [read_gguf(tmp_path / name) for name in names]
    check_parts(parts)
    assert parts[0].file_type == "Q8_0"
    assert [list(part.tensors) for part in parts] == [["t0", "t1"], ["t2", "t3"], ["t4"]]


def test_a_missing_part_is_refused_before_any_download(rng, fake_hub, capsys):
    files = _split_files(_quantized(_base(rng)))
    del files[SPLIT[2]]
    fake_hub.add_repo("quant/split", files)
    assert main(["hash", f"hf:quant/split:{SPLIT[0]}"]) == 2
    assert capsys.readouterr().err == (
        f"wvc: hf:quant/split@main:{SPLIT[0]}: 1 of the 3 parts of this split checkpoint are "
        f"not in the repository, the first {SPLIT[2]}\n"
    )
    assert _downloads(fake_hub) == []


@pytest.mark.parametrize(
    ("metas", "message"),
    [
        ([_split_meta(0), _split_meta(2), _split_meta(2)], "split.no is 2, but the name makes"),
        ([_split_meta(0), _split_meta(1, count=4), _split_meta(2)], "split.count is 4, but"),
        ([_split_meta(0), _split_meta(1, total=5), _split_meta(2)], "is 5, but part 1 says 4"),
        ([_split_meta(i, total=5) for i in range(3)], "is 5, but the 3 parts hold 4 tensors"),
        (
            [_split_meta(0), {"split.no": Value(UINT16, 1)}, _split_meta(2)],
            "part 2 of 3 has no split.count",
        ),
        (
            [_split_meta(0), _split_meta(1, **{"split.no": Value(UINT32, 1)}), _split_meta(2)],
            "split.no must be a UINT16",
        ),
        (
            [_split_meta(0), _split_meta(1, **{"split.tensors.count": 4}), _split_meta(2)],
            "split.tensors.count must be a INT32",
        ),
        (
            [_split_meta(0), _split_meta(1, **{"general.architecture": "llama"}), _split_meta(2)],
            "general.architecture disagrees with part 1",
        ),
    ],
)
def test_parts_that_disagree_are_refused(rng, fake_hub, capsys, metas, message):
    fake_hub.add_repo("quant/split", _split_files(_quantized(_base(rng)), metas))
    assert main(["hash", f"hf:quant/split:{SPLIT[1]}"]) == 2
    assert message in capsys.readouterr().err


def test_a_tensor_in_two_parts_is_refused(rng, fake_hub, capsys):
    tensors = _quantized(_base(rng))
    files = _split_files(tensors)
    names = list(tensors)
    files[SPLIT[2]] = gguf_bytes({names[0]: tensors[names[0]]}, _split_meta(2))
    fake_hub.add_repo("quant/split", files)
    assert main(["hash", f"hf:quant/split:{SPLIT[0]}"]) == 2
    assert "appears in both" in capsys.readouterr().err


def test_a_renamed_part_is_refused(rng, fake_hub, capsys):
    tensors = _quantized(_base(rng))
    names = list(tensors)
    part = gguf_bytes({name: tensors[name] for name in names[:2]}, _split_meta(0))
    fake_hub.add_repo("quant/m", {"m.gguf": part})
    assert main(["hash", "hf:quant/m:m.gguf"]) == 2
    assert "split.count is 3, so this is one part of a split checkpoint" in (
        capsys.readouterr().err
    )


def test_a_single_part_without_split_keys_is_a_whole_checkpoint(rng, fake_hub, capsys):
    data = gguf_bytes(_quantized(_base(rng)), _meta())
    fake_hub.add_repo("quant/m", {"m-q8_0-00001-of-00001.gguf": data})
    assert main(["hash", "--root-only", "hf:quant/m:m-q8_0-00001-of-00001.gguf"]) == 0
    assert capsys.readouterr().out.strip()


def test_a_part_number_no_writer_makes_is_refused(rng, fake_hub, capsys):
    data = gguf_bytes(_quantized(_base(rng)), _meta())
    fake_hub.add_repo("quant/m", {"m-00000-of-00002.gguf": data})
    assert main(["hash", "hf:quant/m:m-00000-of-00002.gguf"]) == 2
    assert "part 0 of 2 is not a part of a split checkpoint" in capsys.readouterr().err
    with pytest.raises(GgufError, match="part 3 of 2"):
        split_names("m-00003-of-00002.gguf")


# Diffs


def _publish_pair(tmp_path, rng, fake_hub, dtype="Q8_0"):
    base = _base(rng)
    st = write_safetensors(tmp_path / "model.safetensors", {k: ("F32", v) for k, v in base.items()})
    local = tmp_path / "m.gguf"
    local.write_bytes(gguf_bytes(_quantized(base, dtype), _meta()))
    fake_hub.add_repo("base/m", {"model.safetensors": st.read_bytes(), "config.json": b"{}"})
    fake_hub.add_repo("quant/m-GGUF", {"m-Q8_0.gguf": local.read_bytes()})
    return st, local


def test_a_hub_gguf_against_its_hub_base_matches_the_local_diff(tmp_path, rng, fake_hub, capsys):
    st, local = _publish_pair(tmp_path, rng, fake_hub)
    status, record, err = _diff_json(capsys, "--rank", "hf:base/m", "hf:quant/m-GGUF:m-Q8_0.gguf")
    local_status, local_record, _ = _diff_json(capsys, "--rank", str(st), str(local))
    assert status == local_status == 1
    assert record["tensors"] == local_record["tensors"]
    assert record["summary"] == local_record["summary"]
    assert {t["name"]: t["status"] for t in record["tensors"]} == {
        name: "changed" if name in QUANTIZED else "unchanged" for name in SHAPES
    }
    sha_a, sha_b = fake_hub.sha_of("base/m"), fake_hub.sha_of("quant/m-GGUF")
    assert record["comparison"] == {
        **local_record["comparison"],
        "files": {"a": ["model.safetensors"], "b": ["m-Q8_0.gguf"]},
        "revisions": {"a": sha_a, "b": sha_b},
        "quantizations": {"a": None, "b": "Q8_0"},
    }
    assert local_record["comparison"] == {
        "kind": "quantization", "gguf": ["b"], "mapping": "qwen2", "values": "float64"
    }  # fmt: skip
    assert record["a"] == {"hf": {"repo": "base/m", "revision": "main", "sha": sha_a},
                           "root": local_record["a"]["root"]}  # fmt: skip
    b = record["b"]
    assert b["format"] == "gguf"
    assert b["gguf"] == local_record["b"]["gguf"]
    assert b["root"] == local_record["b"]["root"]
    assert b["hf"]["file"] == "m-Q8_0.gguf"
    assert [file["name"] for file in b["hf"]["files"]] == ["m-Q8_0.gguf"]
    assert f"pinned: hf:quant/m-GGUF@main:m-Q8_0.gguf at {sha_b}" in err
    assert sorted((c["repo_id"], c["filename"]) for c in _calls(fake_hub, "hf_hub_download")) == [
        ("base/m", "model.safetensors"), ("quant/m-GGUF", "m-Q8_0.gguf")
    ]  # fmt: skip


def test_a_local_base_against_a_hub_gguf_names_both_sides(tmp_path, rng, fake_hub, capsys):
    st, _ = _publish_pair(tmp_path, rng, fake_hub)
    _, record, _ = _diff_json(capsys, str(st), "hf:quant/m-GGUF:m-Q8_0.gguf")
    assert record["comparison"]["files"] == {"a": ["model.safetensors"], "b": ["m-Q8_0.gguf"]}
    assert record["comparison"]["revisions"]["a"] is None


def test_the_agent_view_names_the_hub_file_and_its_file_type(tmp_path, rng, fake_hub, capsys):
    _publish_pair(tmp_path, rng, fake_hub)
    assert main(["diff", "--format", "agent", "hf:base/m", "hf:quant/m-GGUF:m-Q8_0.gguf"]) == 1
    out = capsys.readouterr().out
    sha_b = fake_hub.sha_of("quant/m-GGUF")
    assert f"- b: `hf:quant/m-GGUF@{sha_b}`, file `m-Q8_0.gguf` (revision `main`)" in out
    assert "; file type b `Q8_0`" in out


def test_the_agent_view_keeps_a_long_file_name_whole(rng, fake_hub, capsys):
    name = "a" * 100 + "-Q4_K_M.gguf"
    data = gguf_bytes(_quantized(_base(rng)), _meta())
    fake_hub.add_repo("quant/long-name-GGUF", {name: data})
    arg = f"hf:quant/long-name-GGUF:{name}"
    assert main(["diff", "--format", "agent", arg, arg]) == 0
    assert f"file `{name}`" in capsys.readouterr().out


def test_a_split_gguf_diffs_like_the_whole_file(tmp_path, rng, fake_hub, capsys):
    base = _base(rng)
    st = write_safetensors(tmp_path / "model.safetensors", {k: ("F32", v) for k, v in base.items()})
    tensors = _quantized(base)
    whole = tmp_path / "whole.gguf"
    whole.write_bytes(gguf_bytes(tensors, _meta()))
    fake_hub.add_repo("quant/split", _split_files(tensors))
    _, record, _ = _diff_json(capsys, str(st), f"hf:quant/split:{SPLIT[2]}")
    _, local, _ = _diff_json(capsys, str(st), str(whole))
    assert record["tensors"] == local["tensors"]
    assert record["b"]["gguf"] == {**local["b"]["gguf"], "parts": 3}
    assert record["comparison"]["files"]["b"] == SPLIT
    assert main(["diff", "--format", "agent", str(st), f"hf:quant/split:{SPLIT[0]}"]) == 1
    assert "file `m-00001-of-00003.gguf` (revision `main`, 3 parts)" in capsys.readouterr().out


def test_two_hub_gguf_files_of_one_quantization_compare_as_stored(rng, fake_hub, capsys):
    tensors = _quantized(_base(rng))
    edited = dict(tensors)
    norm = np.frombuffer(tensors["output_norm.weight"].data, "<f4").copy()
    norm[3] += 1.0
    edited["output_norm.weight"] = plain("F32", norm)
    fake_hub.add_repo("quant/a", {"m.gguf": gguf_bytes(tensors, _meta())})
    fake_hub.add_repo("quant/b", {"m.gguf": gguf_bytes(edited, _meta())})
    status, record, _ = _diff_json(capsys, "hf:quant/a:m.gguf", "hf:quant/b:m.gguf")
    assert status == 1
    assert record["comparison"]["gguf"] == ["a", "b"]
    assert record["comparison"]["mapping"] is None
    assert record["comparison"]["quantizations"] == {"a": "Q8_0", "b": "Q8_0"}
    by_name = {t["name"]: t for t in record["tensors"]}
    assert set(by_name) == set(tensors)
    assert by_name["output_norm.weight"]["status"] == "changed"
    assert by_name["output_norm.weight"]["changed"] == 1
    assert "compared" not in by_name["output_norm.weight"]  # same plain type: by bytes
    assert {name for name, t in by_name.items() if t["status"] == "unchanged"} == (
        set(tensors) - {"output_norm.weight"}
    )


def test_a_hub_gguf_against_a_bare_gguf_repository_lists_the_choices(rng, fake_hub, capsys):
    data = gguf_bytes(_quantized(_base(rng)), _meta())
    fake_hub.add_repo("quant/a", {"m.gguf": data})
    assert main(["diff", "hf:quant/a:m.gguf", "hf:quant/a"]) == 2
    err = capsys.readouterr().err
    assert "GGUF checkpoints in hf:quant/a@main:" in err
    assert "name one as hf:quant/a@main:FILE.gguf" in err


# Library edges


def test_file_types_follow_the_reference_names(tmp_path):
    reference = {member.value: member.name for member in gguf_py.LlamaFileType}
    for value, name in FILE_TYPES.items():
        assert reference[value] in (f"MOSTLY_{name}", f"ALL_{name}")
    data = gguf_bytes({}, {"general.file_type": 99})
    assert parse_gguf(Path("x.gguf"), memoryview(data)).file_type == "file_type 99"
    assert parse_gguf(Path("x.gguf"), memoryview(gguf_bytes({}, {}))).file_type is None
    bad = parse_gguf(Path("x.gguf"), memoryview(gguf_bytes({}, {"general.file_type": "Q4"})))
    with pytest.raises(GgufError, match=r"general\.file_type must be an integer"):
        _ = bad.file_type


def test_library_calls_without_a_file_are_refused(fake_hub):
    hub = Hub()
    with pytest.raises(HubError, match="names no GGUF file"):
        hub.describe_gguf(HubRef("owner/name", "main"))
    with pytest.raises(HubError, match="names no GGUF file"):
        load_hub_gguf(hub, HubRepo("owner/name", "main", "0" * 40, ()))


def test_choices_skip_names_that_cannot_be_named():
    siblings = [FakeSibling(name, 10, None) for name in ("a.gguf", "b c.gguf", None)]
    assert [choice.name for choice in gguf_choices("hf:o/n@main", siblings)] == ["a.gguf"]
