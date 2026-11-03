"""Extras, the non-tensor files of a checkpoint directory (spec section 3.3a).

What a commit records and skips, the size caps, dedup across commits, the nothing-changed rule,
checkout beside the shards, forged records refused at put_commit and at checkout, tampered
payloads, show, fsck, and push, pull, and clone.
"""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import pytest
from helpers import (
    DENSE,
    GONE,
    SAME,
    SHARDS,
    SPARSE,
    commit,
    link_or_skip,
    plant_commit,
    replace_or_skip,
)

from wvc import checkout as checkout_module
from wvc import extras as extras_module
from wvc.checkout import checkout_commit
from wvc.cli import main
from wvc.extras import (
    MAX_EXTRA_BYTES,
    SYMLINK_OUTSIDE,
    commit_extras,
    is_extra_name,
    plan_extras,
    read_extra,
    validate_extra_records,
)
from wvc.fsck import fsck
from wvc.hashing import build_manifest
from wvc.remote import add_remote, clone, pull, push
from wvc.repo import init_repository, show_commit
from wvc.safetensors_io import load_checkpoint
from wvc.store import IntegrityError, ObjectStore, StoreError
from wvc.transport import DirTransport

INDEX = {
    "metadata": {"total_size": 0},
    "weight_map": {SPARSE: SHARDS[0], DENSE: SHARDS[0], SAME: SHARDS[1], GONE: SHARDS[1]},
}
TEXTS = {
    "config.json": b'{"architectures": ["ToyForCausalLM"], "hidden_size": 64}\n',
    "chat_template.jinja": b"{% for m in messages %}{{ m['content'] }}{% endfor %}\n",
    "LICENSE": b"Apache License, Version 2.0 (a test fixture)\n",
    "README.md": b"# toy model\n\nA test fixture.\n",
    "model.safetensors.index.json": json.dumps(INDEX, indent=2).encode(),
}
GENERATION = b'{"do_sample": false}\n'
RECORDED = (  # sorted by code point, as the commit record lists them
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
)
SKIPPED = [
    ".gitattributes (dot file)",
    "custom_modeling.py (unsupported name)",
    "eval (a directory)",
    "large.json (too large)",
    "preprocessor_config.json (not a regular file)",
    "weights.bin (unsupported name)",
]


@pytest.fixture
def model_dir(tmp_path, revisions):
    """The two base shards, seven extras (one a link into a subdirectory), six skipped entries."""
    directory = tmp_path / "model"
    shutil.copytree(revisions.base, directory)
    write_extras(directory, tmp_path / "elsewhere")
    return directory


def tokenizer_bytes():
    """A few kilobytes of tokenizer JSON."""
    vocab = {f"token{i}": i for i in range(400)}
    return json.dumps({"model": {"type": "BPE", "vocab": vocab}}).encode()


def write_extras(directory, elsewhere):
    """Write what a model directory holds besides its shards into ``directory``."""
    for name, data in TEXTS.items():
        (directory / name).write_bytes(data)
    (directory / "tokenizer.json").write_bytes(tokenizer_bytes())
    elsewhere.mkdir()
    (directory / ".gitattributes").write_bytes(b"*.safetensors filter=lfs\n")
    (directory / "eval").mkdir()
    (directory / "eval" / "results.json").write_bytes(b"{}\n")
    (directory / "eval" / "generation_config.json").write_bytes(GENERATION)
    (directory / "generation_config.json").symlink_to(Path("eval", "generation_config.json"))
    (directory / "preprocessor_config.json").symlink_to(elsewhere / "missing.json")  # dangling
    (directory / "weights.bin").write_bytes(bytes(16))
    (directory / "custom_modeling.py").write_bytes(b"print('custom code ran')\n")
    with (directory / "large.json").open("wb") as handle:
        handle.truncate(MAX_EXTRA_BYTES + (1 << 20))  # 65 MiB, sparse; it is never read


def hub_snapshot(root, files):
    """A Hugging Face cache entry under ``root``: snapshots/<sha>/<name> -> ../../blobs/<hash>."""
    repo = root / "models--toy--model"
    blobs, snapshot = repo / "blobs", repo / "snapshots" / ("ab" * 20)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for name, data in files.items():
        digest = hashlib.sha256(data).hexdigest()
        (blobs / digest).write_bytes(data)
        link_or_skip(snapshot / name, Path("..", "..", "blobs", digest))
    return snapshot


def expected_extras(directory):
    """The extras records a commit of ``directory`` holds."""
    records = []
    for name in RECORDED:
        data = (directory / name).read_bytes()  # a link is followed
        records.append(
            {"name": name, "payload": hashlib.sha256(data).hexdigest(), "nbytes": len(data)}
        )
    return records


def payload_path(store, pid):
    return store.dir / "payloads" / pid[:2] / pid


def extra_entry(store, commit_id, name):
    return next(entry for entry in store.get_commit(commit_id)["extras"] if entry["name"] == name)


def run(capsys, *argv):
    """Run the CLI; return its exit code, stdout, and stderr."""
    code = main([str(arg) for arg in argv])
    out, err = capsys.readouterr()
    return code, out, err


def columns(line):
    return re.split(r"\s{2,}", line)


# Names and plans


@pytest.mark.parametrize(
    "name",
    [
        "config.json", "tokenizer.json", "chat_template.jinja", "tokenizer.model",
        "cl100k_base.tiktoken", "README.md", "merges.txt", "params.yaml", "params.yml",
        "LICENSE", "LICENCE", "NOTICE", "LICENSE-MIT", "LICENSE.txt",
        "model.safetensors.index.json",
    ],
)  # fmt: skip
def test_allowed_extra_names(name):
    assert is_extra_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "evil.py", "../x.json", "a/b.json", ".hidden.json", "weights.bin", "model.safetensors",
        "LICENSE.py", "LICENSE.safetensors", "NOTICE.sh", "config.JSON", "license", "CON.json",
        "nul.txt", "a b.json", "modèle.json", "config.", "-x.json", "x" * 196 + ".json", "",
        None, 7,
    ],
)  # fmt: skip
def test_refused_extra_names(name):
    """Code, shards, unlisted types, and names a checkout could not write back."""
    assert not is_extra_name(name)


def test_plan_lists_the_extras_and_the_skipped_entries_with_reasons(model_dir):
    plans, skipped = plan_extras(model_dir)
    assert [plan.name for plan in plans] == list(RECORDED)
    assert [plan.path for plan in plans] == [model_dir / name for name in RECORDED]
    assert [plan.nbytes for plan in plans] == [
        entry["nbytes"] for entry in expected_extras(model_dir)
    ]
    assert skipped == SKIPPED  # shards appear in neither list


def test_the_size_cap_of_one_extra_is_inclusive(model_dir, monkeypatch):
    config = len(TEXTS["config.json"])
    monkeypatch.setattr(extras_module, "MAX_EXTRA_BYTES", config)
    plans, skipped = plan_extras(model_dir)
    assert "config.json" in [plan.name for plan in plans]
    assert "tokenizer.json (too large)" in skipped


def test_extras_that_differ_only_in_case_are_refused(model_dir, monkeypatch):
    """Both names exist on a case-sensitive disk; on a case-insensitive one the listing is faked."""
    try:
        with (model_dir / "readme.md").open("xb") as handle:
            handle.write(b"# another card\n")
    except FileExistsError:  # README.md is that file on a case-insensitive disk
        real = os.listdir
        monkeypatch.setattr(extras_module.os, "listdir", lambda path: [*real(path), "readme.md"])
    with pytest.raises(StoreError, match="differ only in case"):
        plan_extras(model_dir)


# Commit


def test_commit_records_the_extras_and_returns_what_it_skipped(store, model_dir):
    result = commit(store, model_dir)
    record = store.get_commit(result.commit_id)
    assert record["extras"] == expected_extras(model_dir)
    assert [entry["name"] for entry in record["files"]] == list(SHARDS)
    total = sum(entry["nbytes"] for entry in record["extras"])
    assert (result.extras, result.extras_bytes, result.skipped) == (7, total, tuple(SKIPPED))
    for entry in record["extras"]:
        assert store.get_payload(entry["payload"]) == (model_dir / entry["name"]).read_bytes()
    # Extras never enter the manifest: identity stays tensors-only.
    assert result.manifest_root == build_manifest(load_checkpoint(model_dir))["root"]


def test_a_single_file_and_no_extras_record_none(store, model_dir):
    single = commit(store, model_dir / SHARDS[0])
    bare = commit(store, model_dir, extras=False)
    for result in (single, bare):
        assert "extras" not in store.get_commit(result.commit_id)
        assert (result.extras, result.extras_bytes, result.skipped) == (0, 0, ())


def test_a_single_file_after_a_directory_carries_the_parents_extras(store, model_dir, tmp_path):
    """C-M4: the file commit keeps the loadable checkout of E005, extras byte for byte."""
    first = commit(store, model_dir)
    single = commit(store, model_dir / SHARDS[0], message="one shard")
    record = store.get_commit(single.commit_id)
    assert record["extras"] == store.get_commit(first.commit_id)["extras"]
    assert record["extras"] == expected_extras(model_dir)
    assert (single.extras, single.extras_bytes) == (7, first.extras_bytes)
    assert (single.extras_carried, single.parent_extras, single.skipped) == (True, 7, ())
    assert single.payload_bytes == 0  # no extra was stored again
    out = tmp_path / "out"
    checked = checkout_commit(store, single.commit_id, out)
    assert [extra.name for extra in checked.extras] == list(RECORDED)
    assert sorted(path.name for path in out.iterdir()) == sorted([SHARDS[0], *RECORDED])
    for name in (SHARDS[0], *RECORDED):
        assert (out / name).read_bytes() == (model_dir / name).read_bytes()
    with pytest.raises(StoreError, match="nothing changed"):
        commit(store, model_dir / SHARDS[0], message="again")


def test_a_single_file_with_no_extras_records_none_after_a_directory(store, model_dir):
    commit(store, model_dir)
    bare = commit(store, model_dir / SHARDS[0], extras=False)
    assert "extras" not in store.get_commit(bare.commit_id)
    assert (bare.extras, bare.extras_carried, bare.parent_extras) == (0, False, 7)


def test_a_carried_extra_whose_payload_is_missing_refuses_the_commit(store, model_dir):
    first = commit(store, model_dir)
    payload_path(store, extra_entry(store, first.commit_id, "LICENSE")["payload"]).unlink()
    with pytest.raises(StoreError, match="cannot carry the extra LICENSE of parent"):
        commit(store, model_dir / SHARDS[0], message="one shard")
    assert store.head() == first.commit_id


def test_extras_over_the_total_cap_refuse_the_commit_before_anything_is_stored(
    store, model_dir, monkeypatch
):
    monkeypatch.setattr(extras_module, "MAX_EXTRAS_TOTAL", 1000)
    with pytest.raises(StoreError, match="more than the 1,000 one commit may record"):
        commit(store, model_dir)
    assert store.head() is None
    assert not any((store.dir / "payloads").iterdir())
    assert not any((store.dir / "objects").iterdir())


def test_identical_extras_are_stored_once_across_commits(tmp_path, revisions, model_dir):
    edited = tmp_path / "edited-model"
    shutil.copytree(revisions.edited, edited)
    write_extras(edited, tmp_path / "elsewhere-2")
    store = ObjectStore(init_repository(tmp_path / "with"))
    bare_store = ObjectStore(init_repository(tmp_path / "without"))
    first, second = [commit(store, path) for path in (model_dir, edited)]
    bare = [commit(bare_store, path, extras=False) for path in (model_dir, edited)]
    extras = [store.get_commit(result.commit_id)["extras"] for result in (first, second)]
    assert extras[0] == extras[1] == expected_extras(model_dir)  # the same payload ids
    assert first.payload_bytes == bare[0].payload_bytes + first.extras_bytes
    assert second.payload_bytes == bare[1].payload_bytes  # no extra was written again
    assert second.extras_bytes == first.extras_bytes


def test_nothing_changed_compares_the_extras(store, model_dir):
    first = commit(store, model_dir)
    with pytest.raises(StoreError, match=r"nothing changed: .* and the extras of parent"):
        commit(store, model_dir, message="again")
    readme = b"# toy model\n\nNow with a usage section.\n"
    (model_dir / "README.md").write_bytes(readme)
    card = commit(store, model_dir, message="the card only")
    assert (card.manifest_root, card.reused, card.stored_full) == (first.manifest_root, 4, 0)
    assert card.stored_delta == {}
    assert card.payload_bytes == len(readme)
    before, after = (store.get_commit(r.commit_id)["extras"] for r in (first, card))
    assert [new["name"] for old, new in zip(before, after, strict=True) if old != new] == [
        "README.md"
    ]
    dropped = commit(store, model_dir, message="no extras", extras=False)
    assert "extras" not in store.get_commit(dropped.commit_id)
    with pytest.raises(StoreError, match="nothing changed"):
        commit(store, model_dir, message="still none", extras=False)


def test_an_extra_that_changes_while_it_is_committed_is_refused(store, model_dir, monkeypatch):
    plans = {plan.name: plan for plan in plan_extras(model_dir)[0]}
    with (model_dir / "config.json").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(StoreError, match="changed size since the commit listed it"):
        read_extra(plans["config.json"])
    real_open = extras_module.open_input

    def open_then_replace(path):
        fd = real_open(path)
        sibling = path.with_name(path.name + ".partial")
        sibling.write_bytes(path.read_bytes())
        replace_or_skip(sibling, path)  # a new file under the name, as a trainer saves one
        return fd

    monkeypatch.setattr(extras_module, "open_input", open_then_replace)
    with pytest.raises(StoreError, match="changed while it was being committed"):
        read_extra(plans["tokenizer.json"])
    with pytest.raises(StoreError, match="changed while it was being committed"):
        commit(store, model_dir)
    assert store.head() is None
    assert not any((store.dir / "commits").iterdir())


# Checkout


def test_checkout_writes_every_extra_beside_the_shards(store, model_dir, tmp_path):
    result = commit(store, model_dir)
    out = tmp_path / "out"
    checked = checkout_commit(store, result.commit_id, out)
    assert [file.name for file in checked.files] == list(SHARDS)
    assert [extra.name for extra in checked.extras] == list(RECORDED)
    assert sorted(path.name for path in out.iterdir()) == sorted([*SHARDS, *RECORDED])
    for name in (*SHARDS, *RECORDED):
        assert (out / name).read_bytes() == (model_dir / name).read_bytes()
        assert (out / name).is_file()
        assert not (out / name).is_symlink()
    for extra, entry in zip(checked.extras, expected_extras(model_dir), strict=True):
        assert (extra.path, extra.sha256, extra.nbytes) == (
            out / extra.name,
            entry["payload"],
            entry["nbytes"],
        )
    assert checked.nbytes == sum(f.nbytes for f in (*checked.files, *checked.extras))


def test_symlinked_extras_of_a_hub_snapshot_check_out_byte_identical(store, revisions, tmp_path):
    """A Hugging Face cache snapshot is a directory of links into a blob store."""
    files = {name: (revisions.base / name).read_bytes() for name in SHARDS}
    files["config.json"] = TEXTS["config.json"]
    files["tokenizer.json"] = tokenizer_bytes()
    snapshot = hub_snapshot(tmp_path, files)
    result = commit(store, snapshot)
    record = store.get_commit(result.commit_id)
    assert [entry["name"] for entry in record["extras"]] == ["config.json", "tokenizer.json"]
    assert result.skipped == ()
    out = tmp_path / "out"
    checkout_commit(store, "HEAD", out)
    for name, data in files.items():
        assert (out / name).read_bytes() == data
        assert not (out / name).is_symlink()


# Symbolic links (second-pass code review H2, security review M1)


def test_a_link_to_a_file_inside_the_checkpoint_is_followed(store, model_dir):
    config = b'{"padding_side": "left"}\n'
    (model_dir / "eval" / "tokenizer_config.json").write_bytes(config)
    link_or_skip(model_dir / "tokenizer_config.json", model_dir / "eval" / "tokenizer_config.json")
    plans, skipped = plan_extras(model_dir)
    assert "tokenizer_config.json" in [plan.name for plan in plans]
    assert skipped == SKIPPED
    result = commit(store, model_dir)
    entry = extra_entry(store, result.commit_id, "tokenizer_config.json")
    assert store.get_payload(entry["payload"]) == config
    assert extra_entry(store, result.commit_id, "generation_config.json")["nbytes"] == len(
        GENERATION
    )


def test_a_link_outside_the_checkpoint_is_skipped_and_the_commit_succeeds_without_it(
    tmp_path, model_dir, capsys, monkeypatch
):
    """An archive's ``tokenizer_config.json -> ~/.ssh/id_ed25519`` is never read or stored."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "id_ed25519").write_bytes(b"-----BEGIN FAKE KEY-----\nFAKE-SECRET-123\n")
    link_or_skip(model_dir / "tokenizer_config.json", home / "id_ed25519")
    link_or_skip(model_dir / "keys", home, directory=True)  # escapes through a linked directory
    link_or_skip(model_dir / "special_tokens_map.json", Path("keys", "id_ed25519"))
    skipped = sorted(
        [
            *SKIPPED,
            "keys (a directory)",
            f"special_tokens_map.json ({SYMLINK_OUTSIDE})",
            f"tokenizer_config.json ({SYMLINK_OUTSIDE})",
        ]
    )
    assert plan_extras(model_dir)[1] == skipped
    monkeypatch.setenv("WVC_AUTHOR", "Evan")
    repo = tmp_path / "repo"
    run(capsys, "init", repo)
    code, out, err = run(capsys, "commit", model_dir, "-m", "base", "--repo", repo)
    assert code == 0
    assert err == "skipped: " + ", ".join(skipped) + "\n"
    assert out.splitlines()[2].startswith("extras: 7 files, ")
    store = ObjectStore(repo)
    assert store.get_commit(store.head())["extras"] == expected_extras(model_dir)
    for path in (store.dir / "payloads").rglob("*"):
        assert not path.is_file() or b"FAKE-SECRET" not in path.read_bytes()


def test_a_link_into_a_sibling_directory_not_named_blobs_is_skipped(tmp_path):
    snapshot = hub_snapshot(tmp_path, {"config.json": TEXTS["config.json"]})
    refs = snapshot.parent.parent / "refs"  # a cache entry also holds refs/main
    refs.mkdir()
    (refs / "main.json").write_bytes(b"{}\n")
    link_or_skip(snapshot / "tokenizer.json", Path("..", "..", "refs", "main.json"))
    plans, skipped = plan_extras(snapshot)
    assert [plan.name for plan in plans] == ["config.json"]
    assert skipped == [f"tokenizer.json ({SYMLINK_OUTSIDE})"]
    model, elsewhere = tmp_path / "model", tmp_path / "elsewhere"  # a plain sibling
    model.mkdir()
    elsewhere.mkdir()
    (elsewhere / "config.json").write_bytes(TEXTS["config.json"])
    link_or_skip(model / "config.json", elsewhere / "config.json")
    assert plan_extras(model) == ([], [f"config.json ({SYMLINK_OUTSIDE})"])


def test_blobs_admit_links_only_beside_snapshots_and_only_when_blobs_is_no_link(tmp_path):
    """``../../blobs`` admits nothing unless the checkpoint sits in ``snapshots``, and a
    ``blobs`` that is itself a link to elsewhere admits nothing either."""
    snapshot = hub_snapshot(tmp_path / "a", {"config.json": TEXTS["config.json"]})
    plain = snapshot.parent.parent / "checkpoints" / "x"
    plain.parent.mkdir()
    shutil.copytree(snapshot, plain, symlinks=True)
    assert (plain / "config.json").read_bytes() == TEXTS["config.json"]
    assert plan_extras(plain) == ([], [f"config.json ({SYMLINK_OUTSIDE})"])
    linked = hub_snapshot(tmp_path / "b", {"config.json": TEXTS["config.json"]})
    blobs = linked.parent.parent / "blobs"
    blobs.rename(tmp_path / "b-blobs")
    link_or_skip(blobs, tmp_path / "b-blobs", directory=True)
    assert (linked / "config.json").read_bytes() == TEXTS["config.json"]
    assert plan_extras(linked) == ([], [f"config.json ({SYMLINK_OUTSIDE})"])


def test_links_to_directories_are_skipped_as_directories_and_broken_links_as_not_regular(
    tmp_path,
):
    directory, outside = tmp_path / "model", tmp_path / "outside"
    (directory / "sub").mkdir(parents=True)
    outside.mkdir()
    link_or_skip(directory / "inner.json", directory / "sub", directory=True)
    link_or_skip(directory / "outer.json", outside, directory=True)
    link_or_skip(directory / "dangling.json", outside / "missing.json")
    link_or_skip(directory / "loop.json", Path("loop.json"))
    assert plan_extras(directory) == (
        [],
        [
            "dangling.json (not a regular file)",
            "inner.json (a directory)",
            "loop.json (not a regular file)",
            "outer.json (a directory)",
            "sub (a directory)",
        ],
    )


def test_a_symlinked_checkpoint_directory_keeps_its_files_and_its_blob_links(
    store, revisions, tmp_path
):
    files = {name: (revisions.base / name).read_bytes() for name in SHARDS}
    files["config.json"] = TEXTS["config.json"]
    snapshot = hub_snapshot(tmp_path, files)
    (snapshot / "README.md").write_bytes(TEXTS["README.md"])  # an ordinary file
    alias = tmp_path / "alias"
    link_or_skip(alias, snapshot, directory=True)
    plans, skipped = plan_extras(alias)
    assert [(plan.name, plan.path) for plan in plans] == [
        ("README.md", alias / "README.md"),
        ("config.json", alias / "config.json"),
    ]
    assert skipped == []
    record = store.get_commit(commit(store, alias).commit_id)
    assert [entry["name"] for entry in record["extras"]] == ["README.md", "config.json"]
    for entry in record["extras"]:
        assert store.get_payload(entry["payload"]) == (snapshot / entry["name"]).read_bytes()


def test_a_commit_without_extras_or_with_an_empty_list_is_valid(store, revisions, tmp_path):
    """Older commits have no extras key; a record may also hold an empty list."""
    record = store.get_commit(commit(store, revisions.base).commit_id)
    assert "extras" not in record
    assert commit_extras(record, "record") == []
    empty = store.put_commit({**record, "message": "empty extras", "extras": []})
    assert store.get_commit(empty)["extras"] == []
    assert checkout_commit(store, empty, tmp_path / "out").extras == ()


def rename(old, new):
    def change(extras):
        return sorted(
            ({**entry, "name": new} if entry["name"] == old else entry for entry in extras),
            key=lambda entry: entry["name"],
        )

    return change


def change_entry(name, **values):
    return lambda extras: [{**e, **values} if e["name"] == name else e for e in extras]


def resize(name, delta):
    return lambda extras: [
        {**e, "nbytes": e["nbytes"] + delta} if e["name"] == name else e for e in extras
    ]


def add_case_twin(extras):
    readme = next(entry for entry in extras if entry["name"] == "README.md")
    return sorted([*extras, {**readme, "name": "readme.md"}], key=lambda entry: entry["name"])


FORGERIES = {
    "code": (rename("config.json", "evil.py"), "not an allowed extra name"),
    "escape": (rename("config.json", "../x.json"), "not an allowed extra name"),
    "licence-code": (rename("LICENSE", "LICENSE.py"), "not an allowed extra name"),
    "shard": (rename("LICENSE", "LICENSE.safetensors"), "not an allowed extra name"),
    "wrong-size": (resize("config.json", 1), r"holds \d+ bytes, not the \d+"),
    "over-cap": (change_entry("config.json", nbytes=MAX_EXTRA_BYTES + 1), "byte count of at most"),
    "missing-payload": (change_entry("config.json", payload="0" * 64), "0{64} is not stored"),
    "unsorted": (lambda extras: extras[::-1], "sorted by name"),
    "case-twin": (add_case_twin, "differ only in case"),
    "keys": (lambda extras: [{**extras[0], "mode": "0755"}, *extras[1:]], "exactly the keys"),
    "not-a-list": (lambda extras: {"config.json": extras[3]}, "extras must be a list"),
}


@pytest.mark.parametrize(("change", "message"), list(FORGERIES.values()), ids=list(FORGERIES))
def test_forged_extras_are_refused_at_put_commit_and_at_checkout(
    store, model_dir, tmp_path, change, message
):
    """put_commit refuses such records; a received store may still hold them."""
    first = commit(store, model_dir)
    record = store.get_commit(first.commit_id)
    forged = {
        **record,
        "parents": [first.commit_id],
        "message": "forged",
        "extras": change(record["extras"]),
    }
    with pytest.raises(StoreError, match=message):
        store.put_commit(forged)
    out = tmp_path / "out"
    with pytest.raises(StoreError, match=message):
        checkout_commit(store, plant_commit(store, forged), out)
    assert not out.exists() or list(out.iterdir()) == []
    assert not (tmp_path / "x.json").exists()


def test_a_tampered_extras_payload_fails_checkout_and_changes_no_file(store, model_dir, tmp_path):
    result = commit(store, model_dir)
    deploy = tmp_path / "deploy"
    checkout_commit(store, result.commit_id, deploy)
    before = {path.name: path.read_bytes() for path in deploy.iterdir()}
    payload = payload_path(store, extra_entry(store, result.commit_id, "config.json")["payload"])
    data = payload.read_bytes()
    payload.write_bytes(bytes([data[0] ^ 0xFF]) + data[1:])  # same size, other bytes
    out = tmp_path / "out"
    with pytest.raises(IntegrityError, match="stored bytes hash to"):
        checkout_commit(store, result.commit_id, out)
    assert list(out.iterdir()) == []
    with pytest.raises(IntegrityError, match="stored bytes hash to"):
        checkout_commit(store, result.commit_id, deploy, force=True)
    assert {path.name: path.read_bytes() for path in deploy.iterdir()} == before


def test_a_directory_at_an_extras_name_refuses_the_checkout_before_anything_is_written(
    store, model_dir, tmp_path, monkeypatch
):
    """C-H1 as reproduced: an old shard and a directory named config.json used to end mixed."""
    result = commit(store, model_dir)
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    (deploy / SHARDS[0]).write_bytes(b"OLD SHARD")
    (deploy / "config.json").mkdir()
    (deploy / "config.json" / "inside.txt").write_bytes(b"mine")

    def no_write(path):
        raise AssertionError(f"wrote {path} although the checkout is refused")

    monkeypatch.setattr(checkout_module, "create_exclusive", no_write)
    with pytest.raises(StoreError, match=r"config\.json is a directory, and a checkout replaces"):
        checkout_commit(store, result.commit_id, deploy, force=True)
    assert (deploy / SHARDS[0]).read_bytes() == b"OLD SHARD"
    assert [p.name for p in (deploy / "config.json").iterdir()] == ["inside.txt"]
    assert sorted(p.name for p in deploy.iterdir()) == sorted([SHARDS[0], "config.json"])


def test_a_directory_made_during_the_writes_refuses_the_renames(
    store, model_dir, tmp_path, monkeypatch
):
    """The target check runs again after the last verification, before any rename."""
    result = commit(store, model_dir)
    deploy = tmp_path / "deploy"
    checkout_commit(store, result.commit_id, deploy)
    before = {path.name: path.read_bytes() for path in deploy.iterdir()}
    real = checkout_module._write_extra

    def write_then_block(store, entry, temporary, target):
        written = real(store, entry, temporary, target)
        if entry["name"] == "config.json":
            target.unlink()
            target.mkdir()  # another process puts a directory there meanwhile
        return written

    monkeypatch.setattr(checkout_module, "_write_extra", write_then_block)
    with pytest.raises(StoreError, match=r"config\.json is a directory"):
        checkout_commit(store, result.commit_id, deploy, force=True)
    del before["config.json"]
    assert (deploy / "config.json").is_dir()
    assert {p.name: p.read_bytes() for p in deploy.iterdir() if p.is_file()} == before


# What --force may replace

PROJECT = {  # S-M2: a project directory whose files share names with a hostile store's extras
    "package.json": b'{"name": "my-app", "scripts": {"start": "node main.js"}}\n',
    "requirements.txt": b"numpy==2.1.0\n",
    "docker-compose.yml": b"services:\n  app:\n    build: .\n",
    "main.py": b"print('my project')\n",
}
HOSTILE = ("docker-compose.yml", "package.json", "requirements.txt")


@pytest.fixture
def hostile_store(tmp_path, revisions):
    """A store whose commit records extras named like a project's own files."""
    model = tmp_path / "hostile-model"
    shutil.copytree(revisions.base, model)
    for name in HOSTILE:
        (model / name).write_bytes(f"from the store: {name}\n".encode())
    store = ObjectStore(init_repository(tmp_path / "hostile"))
    commit(store, model)
    return store


def project_dir(tmp_path):
    directory = tmp_path / "project"
    directory.mkdir()
    for name, data in PROJECT.items():
        (directory / name).write_bytes(data)
    return directory


def contents(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def test_force_refuses_to_replace_files_no_checkout_of_the_store_wrote(
    hostile_store, tmp_path, monkeypatch
):
    project = project_dir(tmp_path)

    def no_write(path):
        raise AssertionError(f"wrote {path} although the checkout is refused")

    monkeypatch.setattr(checkout_module, "create_exclusive", no_write)
    names = ", ".join(HOSTILE)
    with pytest.raises(StoreError, match=f"already holds {names}, which no earlier checkout"):
        checkout_commit(hostile_store, "HEAD", project, force=True)
    with pytest.raises(StoreError, match=r"already holds docker-compose\.yml, requirements\.txt,"):
        checkout_commit(hostile_store, "HEAD", project, force=True, overwrite=["package.json"])
    assert contents(project) == PROJECT


def test_force_with_overwrite_replaces_the_named_files(hostile_store, tmp_path):
    project = project_dir(tmp_path)
    result = checkout_commit(hostile_store, "HEAD", project, force=True, overwrite=HOSTILE)
    assert result.replaced == HOSTILE
    assert (project / "main.py").read_bytes() == PROJECT["main.py"]
    for name in HOSTILE:
        assert (project / name).read_bytes() == f"from the store: {name}\n".encode()


def test_overwrite_must_name_a_file_of_the_commit(hostile_store, tmp_path):
    project = project_dir(tmp_path)
    with pytest.raises(StoreError, match=r"--overwrite names main\.py, which commit"):
        checkout_commit(hostile_store, "HEAD", project, force=True, overwrite=["main.py"])
    assert contents(project) == PROJECT


def test_force_replaces_an_earlier_checkouts_extras_and_a_stale_shard(store, model_dir, tmp_path):
    first = commit(store, model_dir)
    deploy = tmp_path / "deploy"
    checkout_commit(store, first.commit_id, deploy)
    (deploy / SHARDS[0]).write_bytes(b"STALE SHARD")  # any bytes: checkout manages shards
    (deploy / "keep.txt").write_bytes(b"mine")
    config = b'{"architectures": ["ToyForCausalLM"], "hidden_size": 128}\n'
    (model_dir / "config.json").write_bytes(config)
    second = commit(store, model_dir, message="new config")
    result = checkout_commit(store, second.commit_id, deploy, force=True)
    assert result.replaced == (*SHARDS, *RECORDED)
    assert (deploy / "config.json").read_bytes() == config
    assert (deploy / SHARDS[0]).read_bytes() == (model_dir / SHARDS[0]).read_bytes()
    assert (deploy / "keep.txt").read_bytes() == b"mine"


def test_a_file_made_at_an_extras_name_during_the_writes_refuses_the_renames(
    hostile_store, tmp_path, monkeypatch
):
    """The replace check covers files that appear after it, before any rename."""
    deploy = tmp_path / "deploy"
    real = checkout_module._write_extra

    def write_then_plant(store, entry, temporary, target):
        written = real(store, entry, temporary, target)
        if entry["name"] == "package.json":
            target.write_bytes(PROJECT["package.json"])  # another process writes it meanwhile
        return written

    monkeypatch.setattr(checkout_module, "_write_extra", write_then_plant)
    with pytest.raises(StoreError, match=r"package\.json appeared in .* during the checkout"):
        checkout_commit(hostile_store, "HEAD", deploy)
    assert contents(deploy) == {"package.json": PROJECT["package.json"]}


# Show, fsck, and the command line


def test_show_lists_the_extras(store, model_dir):
    result = commit(store, model_dir)
    shown = show_commit(store, "HEAD")
    assert shown.record["extras"] == expected_extras(model_dir)
    assert shown.to_dict()["commit"]["extras"] == shown.record["extras"]
    assert result.commit_id == shown.commit_id


def test_fsck_reports_a_damaged_or_missing_extras_payload(store, model_dir):
    result = commit(store, model_dir)
    assert fsck(store).ok
    pid = extra_entry(store, result.commit_id, "LICENSE")["payload"]
    data = payload_path(store, pid).read_bytes()
    payload_path(store, pid).write_bytes(bytes([data[0] ^ 0xFF]) + data[1:])
    problems = fsck(store).problems
    where = f"commit {result.commit_id}: record: extra LICENSE: payload {pid}"
    assert f"{where} does not hash to its id" in problems
    assert any(problem.startswith(f"payload {pid}: ") for problem in problems)
    payload_path(store, pid).unlink()
    assert fsck(store).problems == [f"{where} is not stored"]


def test_cli_commit_show_and_checkout_print_the_extras(tmp_path, model_dir, capsys, monkeypatch):
    monkeypatch.setenv("WVC_AUTHOR", "Evan")
    repo = tmp_path / "repo"
    run(capsys, "init", repo)
    code, out, err = run(capsys, "commit", model_dir, "-m", "base", "--repo", repo)
    store = ObjectStore(repo)
    extras = store.get_commit(store.head())["extras"]
    total = sum(entry["nbytes"] for entry in extras)
    assert code == 0
    assert out.splitlines()[2] == f"extras: 7 files, {total:,} bytes"
    assert err == "skipped: " + ", ".join(SKIPPED) + "\n"
    code, out, _ = run(capsys, "show", "HEAD", "--repo", repo)
    lines = out.splitlines()
    last_file = max(i for i, line in enumerate(lines) if line.startswith("file: "))
    assert lines[last_file + 1 : last_file + 8] == [
        f"extra: {entry['name']} ({entry['nbytes']:,} bytes)" for entry in extras
    ]
    payload = json.loads(run(capsys, "show", "HEAD", "--json", "--repo", repo)[1])
    assert payload["commit"]["extras"] == extras
    code, out, _ = run(capsys, "checkout", "HEAD", "--out", tmp_path / "out", "--repo", repo)
    lines = out.splitlines()
    assert re.search(r": 2 files and 7 extras, [\d,]+ bytes in \d+\.\d\d s$", lines[0])
    rows = [columns(line) for line in lines[1:-1]]
    assert [row[0] for row in rows] == [*SHARDS, *RECORDED]
    assert all(row[2] == "sha256 ok" for row in rows)
    code, out, err = run(capsys, "commit", model_dir, "-m", "bare", "--no-extras", "--repo", repo)
    assert code == 0
    assert len(out.splitlines()) == 3
    assert err == "extras: the parent recorded 7; this commit records none (--no-extras)\n"
    assert "extras" not in store.get_commit(store.head())


def test_cli_a_single_file_commit_reports_the_carried_extras(
    tmp_path, model_dir, capsys, monkeypatch
):
    monkeypatch.setenv("WVC_AUTHOR", "Evan")
    repo = tmp_path / "repo"
    run(capsys, "init", repo)
    run(capsys, "commit", model_dir, "-m", "base", "--repo", repo)
    store = ObjectStore(repo)
    total = sum(entry["nbytes"] for entry in store.get_commit(store.head())["extras"])
    code, out, err = run(capsys, "commit", model_dir / SHARDS[0], "-m", "one", "--repo", repo)
    assert (code, err) == (0, "")
    assert out.splitlines()[2] == f"extras: 7 files carried from the parent, {total:,} bytes"
    code, out, err = run(
        capsys, "commit", model_dir / SHARDS[1], "-m", "two", "--no-extras", "--repo", repo
    )
    assert code == 0
    assert len(out.splitlines()) == 3
    assert err == "extras: the parent recorded 7; this commit records none (--no-extras)\n"
    assert "extras" not in store.get_commit(store.head())


def test_cli_a_directory_without_extras_after_one_with_extras_prints_a_note(
    tmp_path, model_dir, revisions, capsys, monkeypatch
):
    monkeypatch.setenv("WVC_AUTHOR", "Evan")
    repo = tmp_path / "repo"
    run(capsys, "init", repo)
    run(capsys, "commit", model_dir, "-m", "base", "--repo", repo)
    code, out, err = run(capsys, "commit", revisions.edited, "-m", "edit", "--repo", repo)
    assert code == 0
    assert len(out.splitlines()) == 3
    assert err == "extras: the parent recorded 7; this commit records none\n"


def test_cli_checkout_force_refuses_project_files_until_named(hostile_store, tmp_path, capsys):
    project = project_dir(tmp_path)
    repo = hostile_store.repo
    argv = ("checkout", "HEAD", "--out", project, "--force", "--repo", repo)
    code, out, err = run(capsys, *argv)
    assert (code, out) == (2, "")
    assert err == (
        f"wvc: refusing to check out: {project} already holds docker-compose.yml, package.json, "
        "requirements.txt, which no earlier checkout of this store wrote; nothing was changed. "
        "Pass --overwrite NAME (repeatable) to replace it\n"
    )
    assert contents(project) == PROJECT
    named = [arg for name in HOSTILE for arg in ("--overwrite", name)]
    code, out, err = run(capsys, *argv, *named)
    lines = out.splitlines()
    assert (code, err) == (0, "")
    assert lines[0].endswith(" s, replacing 3 existing files")
    rows = {columns(line)[0]: columns(line)[2:] for line in lines[1:-1]}
    assert all(rows[name] == ["sha256 ok", "replaced"] for name in HOSTILE)
    assert all(rows[name] == ["sha256 ok"] for name in SHARDS)
    assert (project / "main.py").read_bytes() == PROJECT["main.py"]


# Remotes


class Recording(DirTransport):
    """A directory transport that records the files it writes, in order."""

    def __init__(self, root) -> None:
        super().__init__(root)
        self.written = []

    def write(self, relpath, data):
        self.written.append(relpath)
        super().write(relpath, data)


def test_push_pull_and_clone_carry_the_extras(tmp_path, model_dir, monkeypatch):
    origin = ObjectStore(init_repository(tmp_path / "origin"))
    first = commit(origin, model_dir)
    recording = Recording(tmp_path / "remote")
    monkeypatch.setattr("wvc.remote.transport_for", lambda url: recording)
    add_remote(origin, "origin", str(recording.root))
    push(origin)
    record = origin.get_commit(first.commit_id)
    position = {relpath: index for index, relpath in enumerate(recording.written)}
    for entry in record["extras"]:  # every extra's payload precedes the commit record
        pid = entry["payload"]
        assert position[f"payloads/{pid[:2]}/{pid}"] < position[f"commits/{first.commit_id}.json"]
    clone(str(recording.root), tmp_path / "copy")
    copy = ObjectStore(tmp_path / "copy")
    assert copy.get_commit(first.commit_id)["extras"] == record["extras"]
    checkout_commit(copy, first.commit_id, tmp_path / "out")
    for name in (*SHARDS, *RECORDED):
        assert (tmp_path / "out" / name).read_bytes() == (model_dir / name).read_bytes()
    readme = b"# toy model\n\nRevised.\n"
    (model_dir / "README.md").write_bytes(readme)
    second = commit(origin, model_dir, message="the card only")
    push(origin)
    pulled = pull(copy)
    assert (pulled.commits, pulled.objects, pulled.payloads) == (1, 0, 1)  # the new card only
    assert copy.head() == second.commit_id
    checkout_commit(copy, second.commit_id, tmp_path / "out-2")
    assert (tmp_path / "out-2" / "README.md").read_bytes() == readme
    assert fsck(copy).ok


@pytest.mark.parametrize(
    ("extras", "message"),
    [
        ({}, "extras must be a list"),
        ([["config.json"]], "exactly the keys"),
        ([{"name": "a.json", "payload": "A" * 64, "nbytes": 1}], "64-hex"),
        ([{"name": "a.json", "payload": "a" * 64, "nbytes": True}], "byte count"),
        ([{"name": "a.json", "payload": "a" * 64, "nbytes": -1}], "byte count"),
        (
            [
                {"name": f"{letter}.json", "payload": "a" * 64, "nbytes": MAX_EXTRA_BYTES}
                for letter in "abcde"
            ],
            "the extras hold more than",
        ),
    ],
    ids=["not-a-list", "not-a-record", "payload", "bool-size", "negative-size", "total"],
)
def test_extras_records_are_validated(extras, message):
    with pytest.raises(StoreError, match=message):
        validate_extra_records(extras, "record")


def test_an_extra_may_not_collide_with_a_file_name():
    extras = [{"name": "README.md", "payload": "a" * 64, "nbytes": 1}]
    assert validate_extra_records(extras, "record", ["m.safetensors"]) == extras
    with pytest.raises(StoreError, match="differ only in case"):
        validate_extra_records(extras, "record", ["readme.md"])
    with pytest.raises(StoreError, match="share a name"):
        validate_extra_records(extras, "record", ["README.md"])
