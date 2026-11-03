import re

import pytest
from helpers import commit

from wvc.card import CardResult, model_card, render_card, write_manifest
from wvc.cli import main
from wvc.sources import check_source, parse_source
from wvc.store import ObjectStore, StoreError

ID = r"[0-9a-f]{64}"
QWEN = {"kind": "hf", "repo": "Qwen/Qwen2.5-0.5B", "revision": "main"}
REVISION = "0123456789abcdef0123456789abcdef01234567"  # a commit as a revision


def run(capsys, *argv):
    """Run the CLI; return its exit code, stdout, and stderr."""
    code = main([str(arg) for arg in argv])
    out, err = capsys.readouterr()
    return code, out, err


def hf(repo, revision="main"):
    return {"kind": "hf", "repo": repo, "revision": revision}


def path_source(value):
    return {"kind": "path", "value": value}


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("hf:Qwen/Qwen2.5-0.5B@main", QWEN),
        ("hf:Qwen/Qwen2.5-0.5B", QWEN),
        ("hf:org_1/model.v2@refs/pr/3", hf("org_1/model.v2", "refs/pr/3")),
        (f"hf:a/b@{REVISION}", hf("a/b", REVISION)),
        ("./checkpoints/step-100", path_source("./checkpoints/step-100")),
        ("/data/models/qwen", path_source("/data/models/qwen")),
        ("Qwen/Qwen2.5-0.5B", path_source("Qwen/Qwen2.5-0.5B")),
        ("s3://bucket/key", path_source("s3://bucket/key")),
        ("hf", path_source("hf")),
    ],
)
def test_parse_source(spec, expected):
    assert parse_source(spec) == expected
    assert check_source(expected) == expected


@pytest.mark.parametrize(
    "spec",
    [
        "hf:gpt2",
        "hf:a/b/c",
        "hf:/name",
        "hf:owner/",
        "hf:",
        "hf:a/b@",
        "hf:a/b@has space",
        "hf:a/b@main..x",
        "hf:a--b/c",
        "hf:a/b..c",
        "hf:-a/b",
        "hf://a/b",
    ],
)
def test_parse_source_refuses_a_malformed_hf_spec(spec):
    with pytest.raises(StoreError, match="expected OWNER/NAME"):
        parse_source(spec)


@pytest.mark.parametrize("spec", ["", "   ", None])
def test_parse_source_refuses_empty_text(spec):
    with pytest.raises(StoreError, match="non-empty text"):
        parse_source(spec)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("hf:a/b", "must be a JSON object"),
        ({"kind": "s3", "value": "x"}, "source kind"),
        ({"kind": "hf", "repo": "a/b"}, "needs the keys"),
        ({"kind": "path", "value": "x", "extra": "y"}, "needs the keys"),
        ({"kind": "path", "value": 1.5}, "non-empty text"),
        ({"kind": "path", "value": " "}, "non-empty text"),
        ({"kind": "path", "value": "bad \udcff"}, "UTF-8"),
        (hf("gpt2"), "OWNER/NAME"),
        (hf("a/b", "bad revision"), "OWNER/NAME"),
    ],
)
def test_check_source_refuses_malformed_sources(source, message):
    with pytest.raises(StoreError, match=message):
        check_source(source)


def test_commit_records_the_source_in_the_hashed_record(store, revisions):
    first = commit(store, revisions.base, "base", source=QWEN).commit_id
    second = commit(store, revisions.edited, "edit").commit_id
    assert store.get_commit(first)["source"] == QWEN
    assert "source" not in store.get_commit(second)
    # The id covers the source: the same record with another source has another id.
    record = store.get_commit(first)
    assert store.put_commit({**record, "source": hf("Qwen/Qwen2.5-1.5B")}) != first


def test_a_malformed_source_is_refused_before_anything_is_stored(store, revisions):
    for source in ({"kind": "ftp", "value": "x"}, {"kind": "path", "value": 1.5}, ["hf:a/b"]):
        with pytest.raises(StoreError):
            commit(store, revisions.base, "base", source=source)
    assert store.head() is None
    assert not any((store.dir / "payloads").iterdir())


def test_card_reports_the_base_model_tags_and_provenance_and_writes_the_manifest(
    tmp_path, revisions, capsys, monkeypatch
):
    monkeypatch.setenv("WVC_AUTHOR", "Evan")
    repo = tmp_path / "repo"
    assert run(capsys, "init", repo)[0] == 0
    base = ("commit", revisions.base, "-m", "base", "--source", "hf:Qwen/Qwen2.5-0.5B@main")
    code, out, err = run(capsys, *base, "--repo", repo)
    assert (code, err) == (0, "")
    parent = re.match(f"commit ({ID}) ", out)[1]
    edit = ("commit", revisions.edited, "-m", "edit", "--op", "finetune")
    code, out, err = run(capsys, *edit, "--repo", repo)
    child = re.match(f"commit ({ID}) ", out)[1]
    store = ObjectStore(repo)
    assert store.get_commit(parent)["source"] == QWEN
    assert "source" not in store.get_commit(child)
    sidecar = tmp_path / "wvc-manifest.json"
    code, out, err = run(capsys, "card", "HEAD", "--manifest-out", sidecar, "--repo", repo)
    root = store.get_commit(child)["manifest"]
    assert (code, err) == (0, "")
    assert out == (
        "---\n"
        "base_model: Qwen/Qwen2.5-0.5B\n"
        "tags:\n"
        "- wvc\n"
        "- wvc:op:finetune\n"
        "---\n"
        "# wvc provenance\n"
        f"# manifest root: {root}\n"
        f"# commit: {child}\n"
    )
    code, printed, _ = run(capsys, "hash", revisions.edited)
    assert code == 0
    assert sidecar.read_text(encoding="utf-8") == printed


def test_base_model_is_the_nearest_hf_source_on_the_first_parent_chain(store, revisions):
    plain = commit(store, revisions.base, "no source").commit_id
    card = model_card(store, "HEAD")
    assert card.base_model is None
    assert "base_model" not in card.yaml
    tagged = commit(store, revisions.edited, "from the hub", source=hf("org/a")).commit_id
    local = commit(store, revisions.base, "a local file", source=path_source("/data/b")).commit_id
    assert model_card(store, "HEAD").base_model == "org/a"  # a path source is passed over
    newest = commit(
        store, revisions.edited, "another hub model", source=hf("org/c", "v1")
    ).commit_id
    assert model_card(store, "HEAD").base_model == "org/c"  # REF itself comes first
    assert model_card(store, local[:7]).base_model == "org/a"
    assert model_card(store, tagged).base_model == "org/a"
    assert model_card(store, plain).base_model is None
    card = model_card(store, newest)
    record = store.get_commit(newest)
    assert card == CardResult(
        commit_id=newest,
        manifest_root=record["manifest"],
        operation_kind="finetune",
        base_model="org/c",
        yaml=render_card(
            base_model="org/c",
            operation_kind="finetune",
            manifest_root=record["manifest"],
            commit_id=newest,
        ),
    )


def test_card_without_a_hub_source_omits_base_model(store, revisions):
    commit_id = commit(store, revisions.base, "base", kind="pretrain-checkpoint").commit_id
    root = store.get_commit(commit_id)["manifest"]
    assert model_card(store, "main").yaml.splitlines() == [
        "---",
        "tags:",
        "- wvc",
        "- wvc:op:pretrain-checkpoint",
        "---",
        "# wvc provenance",
        f"# manifest root: {root}",
        f"# commit: {commit_id}",
    ]


def test_card_refuses_a_hub_source_that_is_not_a_repo_id(store, revisions):
    record = store.get_commit(commit(store, revisions.base, "base").commit_id)
    forged = {**record, "message": "forged", "source": hf("x\nbase_model: other/model")}
    with pytest.raises(StoreError, match="not OWNER/NAME"):
        model_card(store, store.put_commit(forged))


def test_write_manifest_matches_the_stored_manifest(store, revisions, tmp_path):
    root = store.get_commit(commit(store, revisions.base, "base").commit_id)["manifest"]
    target = write_manifest(store, root, tmp_path / "m.json")
    assert target.read_bytes() == (store.dir / "manifests" / f"{root}.json").read_bytes()


def test_card_and_source_refusals_exit_2(tmp_path, revisions, capsys, monkeypatch):
    monkeypatch.setenv("WVC_AUTHOR", "Evan")
    repo = tmp_path / "repo"
    run(capsys, "init", repo)
    code, out, err = run(capsys, "card", "HEAD", "--repo", repo)
    assert (code, out) == (2, "")
    assert "no commits yet" in err
    with pytest.raises(SystemExit) as exit_info:
        main(["commit", str(revisions.base), "-m", "x", "--source", "hf:gpt2", "--repo", str(repo)])
    assert exit_info.value.code == 2
    assert "invalid hf source 'hf:gpt2'" in capsys.readouterr().err
    assert ObjectStore(repo).head() is None
