"""Hub checkpoints: ``hf:OWNER/NAME[@REVISION]`` for hash, diff, and verify, all offline."""

import dataclasses
import hashlib
import inspect
import json
import struct

import numpy as np
import pytest
from helpers import REAL_MAKE_HUB_CLIENT, FakeHubError, FakeOperationAdd, write_safetensors

from wvc.cli import main
from wvc.hashing import build_manifest
from wvc.hub import Hub, HubError, parse_hub_checkpoint
from wvc.safetensors_io import load_checkpoint
from wvc.transport import HubClient

S1, S2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
EXTRA = "extra.safetensors"


def _pair(tmp_path, rng):
    """Local copies of two repositories: shard 1 byte-identical in both, shard 2 edited."""
    w = rng.standard_normal((8, 8)).astype(np.float32)
    v = rng.standard_normal(16).astype(np.float32)
    u = rng.standard_normal((4, 4)).astype(np.float32)
    edited = u.copy()
    edited[0, 0] += 1.0
    base, edit = tmp_path / "base", tmp_path / "edit"
    for directory in (base, edit):
        directory.mkdir()
        write_safetensors(
            directory / S1, {"w": ("F32", w), "v": ("F32", v)}, header_order=["v", "w"]
        )
    write_safetensors(base / S2, {"u": ("F32", u)})
    write_safetensors(edit / S2, {"u": ("F32", edited)})
    return base, edit


def _publish(fake_hub, repo_id, directory, **extra):
    """Put a directory's files in a fake repository, with card files and two decoys."""
    files = {path.name: path.read_bytes() for path in directory.iterdir()}
    files.update({"README.md": b"# card\n", "config.json": b"{}\n"})
    files.update({"sub/nested.safetensors": b"\x00" * 8, ".hidden.safetensors": b"\x00" * 8})
    files.update(extra)
    fake_hub.add_repo(repo_id, files)
    return fake_hub.sha_of(repo_id)


def _calls(fake_hub, method):
    return [kwargs for name, kwargs in fake_hub.calls if name == method]


def _downloads(fake_hub):
    return sorted((c["repo_id"], c["filename"]) for c in _calls(fake_hub, "hf_hub_download"))


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("hf:owner/name", ("owner/name", "main")),
        ("hf:owner/name@v1.0", ("owner/name", "v1.0")),
        ("hf:Qwen/Qwen3.5-9B@refs/pr/3", ("Qwen/Qwen3.5-9B", "refs/pr/3")),
    ],
)
def test_hub_checkpoint_arguments_parse(arg, expected):
    assert parse_hub_checkpoint(arg) == expected


@pytest.mark.parametrize("arg", ["hf:gpt2", "hf:", "hf:owner/name@", "hf:a--b/c", "owner/name"])
def test_malformed_hub_checkpoint_arguments_are_refused(arg):
    with pytest.raises(HubError, match="OWNER/NAME"):
        parse_hub_checkpoint(arg)


def test_a_remote_url_is_not_a_checkpoint():
    with pytest.raises(HubError, match="names a wvc remote"):
        parse_hub_checkpoint("hf://owner/name")


def test_hash_fetches_every_shard_at_the_pinned_commit(tmp_path, rng, fake_hub, capsys):
    base, _ = _pair(tmp_path, rng)
    sha = _publish(fake_hub, "base/model", base)
    assert main(["hash", "hf:base/model"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == build_manifest(load_checkpoint(base))
    (info,) = _calls(fake_hub, "model_info")
    assert info == {"repo_id": "base/model", "revision": "main", "files_metadata": True}
    downloads = _calls(fake_hub, "hf_hub_download")
    assert sorted(call["filename"] for call in downloads) == [S1, S2]  # no decoys, no card
    assert {(c["repo_id"], c["revision"], c["repo_type"]) for c in downloads} == {
        ("base/model", sha, "model")
    }
    assert _calls(fake_hub, "read_range") == []
    assert "pinned: hf:base/model@main at " + sha in captured.err


def test_diff_of_two_hub_repositories_skips_byte_identical_shards(tmp_path, rng, fake_hub, capsys):
    base, edit = _pair(tmp_path, rng)
    sha_a = _publish(fake_hub, "base/model", base)
    sha_b = _publish(fake_hub, "edit/model", edit)
    assert main(["diff", "hf:base/model", "hf:edit/model@main", "--json", "--all"]) == 1
    payload = json.loads(capsys.readouterr().out)
    shard = base / S1
    assert payload["a"] == {
        "hf": {"repo": "base/model", "revision": "main", "sha": sha_a},
        "root": None,  # unknown: the skipped shard's tensors were never hashed
        "files": {
            "skipped": [
                {
                    "name": S1,
                    "nbytes": shard.stat().st_size,
                    "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                }
            ],
            "fetched": [S2],
        },
    }
    assert payload["b"]["hf"]["sha"] == sha_b
    assert payload["b"]["files"]["fetched"] == [S2]
    assert payload["summary"]["tensors"] == 3
    assert payload["summary"]["by_status"]["changed"] == 1
    by_name = {t["name"]: t for t in payload["tensors"]}
    assert [*by_name] == ["u", "v", "w"]
    assert by_name["u"]["changed"] == 1
    assert {k: by_name["w"][k] for k in ("status", "dtype_a", "shape_b", "numel", "changed")} == {
        "status": "unchanged", "dtype_a": "F32", "shape_b": [8, 8], "numel": 64, "changed": 0
    }  # fmt: skip
    assert _downloads(fake_hub) == [("base/model", S2), ("edit/model", S2)]
    reads = _calls(fake_hub, "read_range")
    assert {(r["repo_id"], r["filename"], r["revision"]) for r in reads} == {
        ("base/model", S1, sha_a)
    }
    assert (reads[0]["start"], reads[0]["length"]) == (0, 8)  # the size field, then the header

    assert main(["diff", "hf:base/model", "hf:edit/model"]) == 1
    out = capsys.readouterr().out
    assert (
        f"hub: a = hf:base/model@main ({sha_a[:12]}), b = hf:edit/model@main ({sha_b[:12]})" in out
    )
    assert "shards: 2 in a, 2 in b; 1 byte-identical on the Hub and skipped" in out
    assert "fetched 1 from a and 1 from b" in out
    assert "1 changed, 2 unchanged" in out


def test_diff_of_identical_hub_repositories_downloads_nothing(tmp_path, rng, fake_hub, capsys):
    base, _ = _pair(tmp_path, rng)
    _publish(fake_hub, "base/model", base)
    _publish(fake_hub, "copy/model", base)
    assert main(["diff", "hf:base/model", "hf:copy/model"]) == 0
    out = capsys.readouterr().out
    assert "0 changed, 3 unchanged" in out
    assert "2 byte-identical on the Hub and skipped" in out
    assert _downloads(fake_hub) == []
    assert sorted({c["filename"] for c in _calls(fake_hub, "read_range")}) == [S1, S2]


def test_shards_present_on_one_side_only_are_fetched(tmp_path, rng, fake_hub, capsys):
    base, edit = _pair(tmp_path, rng)
    extra = write_safetensors(tmp_path / EXTRA, {"n": ("I8", np.arange(4, dtype=np.int8))})
    _publish(fake_hub, "base/model", base)
    _publish(fake_hub, "edit/model", edit, **{EXTRA: extra.read_bytes()})
    assert main(["diff", "hf:base/model", "hf:edit/model", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["b"]["files"]["fetched"] == [EXTRA, S2]
    assert {t["name"]: t["status"] for t in payload["tensors"]} == {"n": "added", "u": "changed"}
    assert _downloads(fake_hub) == [("base/model", S2), ("edit/model", EXTRA), ("edit/model", S2)]


def test_diff_against_a_local_checkpoint_fetches_every_shard(tmp_path, rng, fake_hub, capsys):
    base, edit = _pair(tmp_path, rng)
    sha = _publish(fake_hub, "base/model", base)
    assert main(["diff", "hf:base/model", str(edit), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["a"] == {
        "hf": {"repo": "base/model", "revision": "main", "sha": sha},
        "root": build_manifest(load_checkpoint(base))["root"],
    }
    assert payload["b"] == {
        "path": str(edit),
        "root": build_manifest(load_checkpoint(edit))["root"],
    }
    assert _downloads(fake_hub) == [("base/model", S1), ("base/model", S2)]


def test_verify_a_hub_checkpoint_against_a_manifest(tmp_path, rng, fake_hub, capsys):
    base, edit = _pair(tmp_path, rng)
    _publish(fake_hub, "base/model", base)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(build_manifest(load_checkpoint(edit))), encoding="utf-8")
    assert main(["verify", "hf:base/model", str(manifest)]) == 1
    assert "1 mismatch" in capsys.readouterr().out
    manifest.write_text(json.dumps(build_manifest(load_checkpoint(base))), encoding="utf-8")
    assert main(["verify", "hf:base/model", str(manifest)]) == 0
    assert "root ok" in capsys.readouterr().out


def test_hub_errors_name_the_checkpoint(fake_hub, capsys):
    assert main(["hash", "hf:nobody/missing"]) == 2
    err = capsys.readouterr().err
    assert "hf:nobody/missing@main" in err
    assert "404" in err
    assert main(["diff", "hf:owner/name", "hf://owner/name"]) == 2
    assert "names a wvc remote" in capsys.readouterr().err


def test_a_repository_without_top_level_shards_is_refused(fake_hub, capsys):
    fake_hub.add_repo("text/only", {"README.md": b"", "model.bin": b"\x80", "a/b.safetensors": b""})
    assert main(["hash", "hf:text/only"]) == 2
    assert "no top-level *.safetensors" in capsys.readouterr().err


def test_a_tensor_in_a_skipped_and_a_fetched_shard_is_refused(tmp_path, rng, fake_hub, capsys):
    base, edit = _pair(tmp_path, rng)
    zeros = np.zeros((8, 8), np.float32)
    write_safetensors(edit / S2, {"u": ("F32", zeros[:4, :4]), "w": ("F32", zeros)})  # w twice
    _publish(fake_hub, "base/model", base)
    _publish(fake_hub, "edit/model", edit)
    assert main(["diff", "hf:base/model", "hf:edit/model"]) == 2
    assert "appears in both" in capsys.readouterr().err


def test_a_shard_the_hub_lists_without_a_size_is_an_error(tmp_path, rng, fake_hub, capsys):
    base, _ = _pair(tmp_path, rng)
    _publish(fake_hub, "base/model", base)
    original = fake_hub.model_info
    fake_hub.model_info = lambda repo_id, **kw: original(repo_id, **{**kw, "files_metadata": False})
    assert main(["hash", "hf:base/model"]) == 2
    assert "no valid size" in capsys.readouterr().err


def test_hub_calls_bind_to_the_installed_client(tmp_path, rng, fake_hub, monkeypatch):
    """Offline check of the calls and result attributes against the real ``huggingface_hub``."""
    hub = pytest.importorskip("huggingface_hub")
    real = REAL_MAKE_HUB_CLIENT()  # builds an HfApi, which makes no request
    base, edit = _pair(tmp_path, rng)
    sha_a = _publish(fake_hub, "base/model", base)
    _publish(fake_hub, "edit/model", edit)
    client = HubClient(
        api=fake_hub,
        operation_add=hub.CommitOperationAdd,
        errors=real.errors,
        read_range=fake_hub.read_range,
    )
    monkeypatch.setattr("wvc.transport.make_hub_client", lambda: client)
    assert main(["diff", "hf:base/model", "hf:edit/model"]) == 1
    names = {name for name, _ in fake_hub.calls}
    assert names == {"model_info", "hf_hub_download", "read_range"}
    for name, kwargs in fake_hub.calls:
        if name == "read_range":
            inspect.signature(real.read_range).bind(*kwargs.values())
        else:
            inspect.signature(getattr(real.api, name)).bind(**kwargs)
    # A ModelInfo built from an API-shaped payload describes through the real attribute names.
    shard = base / S1
    payload = {
        "rfilename": S1,
        "size": shard.stat().st_size,
        "blobId": "b" * 40,
        "lfs": {"size": shard.stat().st_size, "sha256": "c" * 64, "pointerSize": 134},
    }
    info = hub.ModelInfo(id="base/model", sha=sha_a, siblings=[payload, {"rfilename": "README.md"}])
    fake_hub.model_info = lambda repo_id, **kwargs: info
    repo = Hub(client).describe("base/model", "main")
    assert (repo.sha, [(f.name, f.nbytes, f.sha256) for f in repo.files]) == (
        sha_a,
        [(S1, shard.stat().st_size, "c" * 64)],
    )
    assert issubclass(hub.errors.HfHubHTTPError, real.errors)


def _raw(path, header: bytes, buffer: bytes, size_field: int | None = None):
    """A safetensors file from raw header bytes; ``size_field`` overrides the stated length."""
    length = len(header) if size_field is None else size_field
    path.write_bytes(struct.pack("<Q", length) + header + buffer)
    return path


ENTRY = b'{"dtype":"F32","shape":[2],"data_offsets":[0,8]}'
BAD_OFFSETS = b'{"w":{"dtype":"F32","shape":[32],"data_offsets":[0,128]}}'
BAD_SHAPE = b'{"w":{"dtype":"F32","shape":5,"data_offsets":[0,8]}}'
HOSTILE_HEADERS = {
    "offsets beyond the buffer": (BAD_OFFSETS, b"\0" * 64, None, "fall outside"),
    "duplicate key": (b'{"w":' + ENTRY + b',"w":' + ENTRY + b"}", b"\0" * 8, None, "duplicate key"),
    "control character": (b'{"w\\u0007":' + ENTRY + b"}", b"\0" * 8, None, "control char"),
    "non-integer shape": (BAD_SHAPE, b"", None, "shape must be"),
    "not an object": (b"[]", b"", None, "does not start with"),
    "truncated JSON": (b'{"w":{"dtype":"F32"', b"", None, "not valid JSON"),
    "header longer than the file": (b"{}", b"", 1000, "exceeds the"),
}  # fmt: skip


@pytest.mark.parametrize("case", sorted(HOSTILE_HEADERS))
def test_hostile_headers_on_the_skip_path_are_refused_before_any_download(
    tmp_path, rng, fake_hub, capsys, case
):
    header, buffer, size_field, message = HOSTILE_HEADERS[case]
    base, edit = _pair(tmp_path, rng)
    for directory in (base, edit):
        _raw(directory / S1, header, buffer, size_field)  # byte-identical, so skipped
    _publish(fake_hub, "base/model", base)
    _publish(fake_hub, "edit/model", edit)
    assert main(["diff", "hf:base/model", "hf:edit/model"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("wvc: ")
    assert message in err
    assert S1 in err
    assert _downloads(fake_hub) == []


def test_an_empty_header_on_the_skip_path_makes_no_zero_length_range_request(
    tmp_path, rng, fake_hub, monkeypatch, capsys
):
    base, edit = _pair(tmp_path, rng)
    for directory in (base, edit):
        _raw(directory / S1, b"", b"\0" * 64)  # header-size field 0, byte-identical, so skipped
    _publish(fake_hub, "base/model", base)
    _publish(fake_hub, "edit/model", edit)
    lengths = []
    read_range = fake_hub.read_range

    def recording(repo_id, filename, revision, start, length):
        lengths.append((filename, start, length))
        return read_range(repo_id, filename, revision, start, length)

    monkeypatch.setattr(fake_hub, "read_range", recording)
    client = HubClient(fake_hub, FakeOperationAdd, (FakeHubError,), recording)
    monkeypatch.setattr("wvc.transport.make_hub_client", lambda: client)
    assert main(["diff", "hf:base/model", "hf:edit/model"]) == 2
    err = capsys.readouterr().err
    assert "does not start with" in err
    assert S1 in err
    assert lengths == [(S1, 0, 8)]  # the size field only; no bytes=8-7 request
    assert _downloads(fake_hub) == []


def test_a_plain_blob_shard_is_fetched_not_skipped(tmp_path, rng, fake_hub, capsys):
    base, edit = _pair(tmp_path, rng)
    _publish(fake_hub, "base/model", base)
    _publish(fake_hub, "edit/model", edit)
    fake_hub.plain.add(S1)  # the Hub lists it without an LFS sha256
    assert main(["diff", "hf:base/model", "hf:edit/model"]) == 1
    assert "0 byte-identical on the Hub and skipped" in capsys.readouterr().out
    assert _downloads(fake_hub) == [
        ("base/model", S1), ("base/model", S2), ("edit/model", S1), ("edit/model", S2)
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sha", "main", "no commit sha"),
        ("lfs", {"sha256": "zz"}, "malformed LFS"),
        ("size", -1, "no valid size"),
    ],
)
def test_malformed_listings_are_refused(tmp_path, rng, fake_hub, capsys, field, value, message):
    base, _ = _pair(tmp_path, rng)
    _publish(fake_hub, "base/model", base)
    original = fake_hub.model_info

    def tampered(repo_id, **kwargs):
        info = original(repo_id, **kwargs)
        if field == "sha":
            return dataclasses.replace(info, sha=value)
        siblings = [
            dataclasses.replace(s, **{field: value}) if s.rfilename == S1 else s
            for s in info.siblings
        ]
        return dataclasses.replace(info, siblings=siblings)

    fake_hub.model_info = tampered
    assert main(["hash", "hf:base/model"]) == 2
    assert message in capsys.readouterr().err
    assert _downloads(fake_hub) == []


def test_a_download_of_the_wrong_size_is_refused(tmp_path, rng, fake_hub, capsys):
    base, _ = _pair(tmp_path, rng)
    _publish(fake_hub, "base/model", base)
    original = fake_hub.hf_hub_download

    def truncated(repo_id, filename, **kwargs):
        path = original(repo_id, filename, **kwargs)
        data = (tmp_path / "hub-downloads" / "base--model" / filename).read_bytes()
        (tmp_path / "hub-downloads" / "base--model" / filename).write_bytes(data[:-1])
        return path

    fake_hub.hf_hub_download = truncated
    assert main(["hash", "hf:base/model"]) == 2
    assert "the Hub listed" in capsys.readouterr().err


def test_resharded_repositories_fetch_everything(tmp_path, rng, fake_hub, capsys):
    base, _ = _pair(tmp_path, rng)
    single = tmp_path / "single"
    single.mkdir()
    ckpt = load_checkpoint(base)
    tensors = {n: (ckpt.tensors[n].dtype, ckpt.as_float32(n)) for n in ("u", "v", "w")}
    write_safetensors(single / "model.safetensors", tensors)
    _publish(fake_hub, "base/model", base)
    _publish(fake_hub, "one/model", single)
    assert main(["diff", "hf:base/model", "hf:one/model"]) == 0
    out = capsys.readouterr().out
    assert "shards: 2 in a, 1 in b; 0 byte-identical" in out
    assert "3 unchanged" in out
    assert len(_downloads(fake_hub)) == 3


def test_revisions_resolve_or_fail_as_the_hub_answers(tmp_path, rng, fake_hub, capsys):
    base, _ = _pair(tmp_path, rng)
    sha = _publish(fake_hub, "base/model", base)
    assert main(["hash", "--root-only", f"hf:base/model@{sha}"]) == 0
    assert f"pinned: hf:base/model@{sha} at {sha}" in capsys.readouterr().err
    assert main(["hash", "--root-only", "hf:base/model@nope"]) == 2
    assert "404" in capsys.readouterr().err
    assert _downloads(fake_hub) == [
        ("base/model", S1),
        ("base/model", S2),
    ]


def test_rank_through_the_hub_pair(tmp_path, rng, fake_hub, capsys):
    base, edit = _pair(tmp_path, rng)
    _publish(fake_hub, "base/model", base)
    _publish(fake_hub, "edit/model", edit)
    assert main(["diff", "hf:base/model", "hf:edit/model", "--rank", "--json"]) == 1
    (changed,) = json.loads(capsys.readouterr().out)["tensors"]
    assert (changed["name"], changed["rank"], changed["rows_touched"]) == ("u", 1, 1)


def test_verify_json_names_the_pinned_commit_and_reads_the_manifest_first(
    tmp_path, rng, fake_hub, capsys
):
    base, _ = _pair(tmp_path, rng)
    sha = _publish(fake_hub, "base/model", base)
    assert main(["verify", "hf:base/model", str(tmp_path / "missing.json")]) == 2
    assert _downloads(fake_hub) == []  # a bad manifest costs no download
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(build_manifest(load_checkpoint(base))), encoding="utf-8")
    assert main(["verify", "hf:base/model", str(manifest), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hf"] == {"repo": "base/model", "revision": "main", "sha": sha}


@pytest.mark.parametrize(
    "name", ["C:evil.safetensors", "CON.safetensors", "a\\b.safetensors", "my model.safetensors"]
)
@pytest.mark.parametrize("command", [["hash"], ["hash", "--format", "gguf-hash"], ["diff"]])
def test_hub_shard_names_outside_the_portable_rule_are_refused_before_fetching(
    tmp_path, rng, fake_hub, capsys, name, command
):
    """C-L4: an alternate data stream, a device name, a backslash, or whitespace (which a
    gguf-hash line cannot carry) is refused at the listing, before any download."""
    base, edit = _pair(tmp_path, rng)
    _publish(fake_hub, "base/model", base, **{name: (base / S1).read_bytes()})
    _publish(fake_hub, "edit/model", edit)
    extra = ["hf:edit/model"] if command == ["diff"] else []
    assert main([*command, "hf:base/model", *extra]) == 2
    err = capsys.readouterr().err
    assert "hf:base/model@main" in err
    assert "is not an allowed file name" in err
    assert _downloads(fake_hub) == []
    assert _calls(fake_hub, "read_range") == []


def test_a_hub_listing_entry_without_a_file_name_is_skipped(tmp_path, rng, fake_hub, capsys):
    base, _ = _pair(tmp_path, rng)
    _publish(fake_hub, "base/model", base)
    original = fake_hub.model_info

    def nameless(repo_id, **kwargs):
        info = original(repo_id, **kwargs)
        return dataclasses.replace(info, siblings=[*info.siblings, object()])

    fake_hub.model_info = nameless
    assert main(["hash", "--root-only", "hf:base/model"]) == 0
    assert len(capsys.readouterr().out.strip()) == 64
