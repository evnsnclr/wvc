import json
import os
import re

import pytest
from helpers import DENSE, GONE, NEW, SHARDS, SPARSE

from wvc.cli import main
from wvc.store import ObjectStore

ID = r"[0-9a-f]{64}"
TIME = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"


def run(capsys, *argv):
    """Run the CLI; return its exit code, stdout, and stderr."""
    code = main([str(arg) for arg in argv])
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def repo(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("WVC_AUTHOR", "Evan")
    path = tmp_path / "repo"
    assert run(capsys, "init", path)[0] == 0
    return path


def commit_both(capsys, repo, revisions):
    """Commit the base, then the edit; return both commit ids."""
    ids = []
    for path, message, kind in (
        (revisions.base, "base model", "pretrain-checkpoint"),
        (revisions.edited, "edit\n\nwith a body", "direct-edit"),
    ):
        code, out, err = run(capsys, "commit", path, "-m", message, "--op", kind, "--repo", repo)
        assert (code, err) == (0, "")
        ids.append(re.match(f"commit ({ID}) ", out)[1])
    return ids


def columns(line):
    return re.split(r"\s{2,}", line)


def test_init_prints_the_store_and_refuses_to_reinit(tmp_path, capsys, monkeypatch):
    code, out, err = run(capsys, "init", tmp_path / "r")
    expected = (tmp_path / "r").resolve() / ".wvc"
    assert (code, out, err) == (0, f"initialised an empty wvc repository in {expected}\n", "")
    code, out, err = run(capsys, "init", tmp_path / "r")
    assert (code, out) == (2, "")
    assert err.startswith("wvc: ")
    assert "already exists" in err
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    assert run(capsys, "init")[0] == 0
    assert (tmp_path / "cwd" / ".wvc" / "HEAD").is_file()


def test_commit_prints_the_id_counts_and_bytes(repo, revisions, capsys):
    code, out, _ = run(capsys, "commit", revisions.base, "-m", "base", "--repo", repo)
    lines = out.splitlines()
    first = re.fullmatch(rf"commit ({ID}) \(main, no parent\)", lines[0])[1]
    assert lines[1] == "tensors: 4 (0 reused by hash, 4 stored full, 0 stored as delta)"
    assert re.fullmatch(r"payload bytes written: [\d,]+ in \d+\.\d\d s", lines[2])
    assert (code, len(lines)) == (0, 3)
    code, out, _ = run(capsys, "commit", revisions.edited, "-m", "edit", "--repo", repo)
    lines = out.splitlines()
    assert re.fullmatch(rf"commit {ID} \(main, parent {first[:12]}\)", lines[0])
    assert lines[1] == (
        "tensors: 4 (1 reused by hash, 1 stored full, "
        "2 stored as delta: 1 delta-sparse, 1 delta-xor-planes)"
    )
    store = ObjectStore(repo)
    assert store.get_commit(store.head())["operation"] == {"kind": "unknown"}  # the default op


def test_author_comes_from_the_option_or_the_environment(repo, revisions, capsys, monkeypatch):
    """S-L5: no silent OS user name; without --author or WVC_AUTHOR the commit is refused."""
    store = ObjectStore(repo)
    run(capsys, "commit", revisions.base, "-m", "a", "--repo", repo)
    assert store.get_commit(store.head())["author"] == "Evan"
    run(capsys, "commit", revisions.edited, "-m", "b", "--author", "Ada", "--repo", repo)
    assert store.get_commit(store.head())["author"] == "Ada"
    head = store.head()
    monkeypatch.delenv("WVC_AUTHOR")
    monkeypatch.setattr("getpass.getuser", lambda: "os-user")
    code, out, err = run(capsys, "commit", revisions.base, "-m", "c", "--repo", repo)
    assert (code, out) == (2, "")
    assert "pass --author NAME or set WVC_AUTHOR" in err
    assert store.head() == head
    run(capsys, "commit", revisions.base, "-m", "c", "--author", "Ada", "--repo", repo)
    assert store.get_commit(store.head())["author"] == "Ada"


def test_log_prints_one_line_per_commit_and_honours_n(repo, revisions, capsys):
    first, second = commit_both(capsys, repo, revisions)
    code, out, _ = run(capsys, "log", "--repo", repo)
    rows = [columns(line) for line in out.splitlines()]
    assert code == 0
    assert [row[0] for row in rows] == [second[:12], first[:12]]
    assert [row[2:] for row in rows] == [
        ["direct-edit", "4 tensors", "edit"],
        ["pretrain-checkpoint", "4 tensors", "base model"],
    ]
    assert all(re.fullmatch(TIME, row[1]) for row in rows)
    assert all(line == line.rstrip() for line in out.splitlines())
    code, out, _ = run(capsys, "log", "-n", "1", "--repo", repo)
    assert code == 0
    assert [columns(line)[0] for line in out.splitlines()] == [second[:12]]
    assert run(capsys, "log", "-n", "0", "--repo", repo)[:2] == (0, "")
    for bad in ("-1", "two"):
        with pytest.raises(SystemExit) as exit_info:
            main(["log", "-n", bad, "--repo", str(repo)])
        assert exit_info.value.code == 2
        assert "non-negative integer" in capsys.readouterr().err


def test_show_prints_the_record_and_the_comparison(repo, revisions, capsys):
    first, second = commit_both(capsys, repo, revisions)
    code, out, _ = run(capsys, "show", "HEAD", "--repo", repo)
    store = ObjectStore(repo)
    record = store.get_commit(second)
    head, table = out.rstrip("\n").rsplit("\n\n", 1)  # the table follows the last blank line
    assert head.splitlines() == [
        f"commit {second}",
        f"parents: {first}",
        f"manifest: {record['manifest']}",
        f"tensors: 4 (2 changed, 1 unchanged, 1 added, 1 removed against {first[:12]})",
        "operation: direct-edit",
        "author: Evan",
        f"time: {record['time']}",
        "message: edit",
        "",
        "  with a body",
        *(
            f"file: {entry['name']} ({entry['nbytes']:,} bytes, 2 tensors, "
            f"sha256 {entry['sha256']})"
            for entry in record["files"]
        ),
    ]
    assert [columns(line) for line in table.splitlines()] == [
        ["status", "tensor"],
        ["changed", DENSE],
        ["changed", SPARSE],
        ["added", NEW],
        ["removed", GONE],
    ]
    assert code == 0
    assert all(line == line.rstrip() for line in out.splitlines())
    code, out, _ = run(capsys, "show", first[:7], "--repo", repo)
    assert code == 0
    assert "tensors: 4 (no parent)" in out
    assert "status" not in out


def test_show_json(repo, revisions, capsys):
    first, second = commit_both(capsys, repo, revisions)
    code, out, _ = run(capsys, "show", second[:7], "--json", "--repo", repo)
    payload = json.loads(out)
    assert code == 0
    assert list(payload) == [
        "id",
        "commit",
        "tensor_count",
        "first_parent",
        "comparison",
    ]
    assert (payload["id"], payload["first_parent"], payload["tensor_count"]) == (second, first, 4)
    assert payload["commit"] == ObjectStore(repo).get_commit(second)
    assert payload["comparison"] == {
        "counts": {"changed": 2, "unchanged": 1, "added": 1, "removed": 1},
        "changed": [DENSE, SPARSE],
        "added": [NEW],
        "removed": [GONE],
    }
    payload = json.loads(run(capsys, "show", "main", "--json", "--repo", repo)[1])
    assert payload["id"] == second
    payload = json.loads(run(capsys, "show", first, "--json", "--repo", repo)[1])
    assert (payload["first_parent"], payload["comparison"]) == (None, None)


def test_checkout_prints_each_file_and_reproduces_it(repo, revisions, tmp_path, capsys):
    first, second = commit_both(capsys, repo, revisions)
    for commit_id, source in ((first, revisions.base), (second, revisions.edited)):
        out_dir = tmp_path / f"out-{commit_id[:7]}"
        code, out, _ = run(capsys, "checkout", commit_id[:7], "--out", out_dir, "--repo", repo)
        lines = out.splitlines()
        assert code == 0
        assert lines[0].startswith(f"checked out {commit_id[:12]} into {out_dir}: ")
        assert re.search(r": 2 files, [\d,]+ bytes in \d+\.\d\d s$", lines[0])
        assert lines[-1].startswith("manifest root ")
        assert " verified against the files" in lines[-1]
        assert len(lines[-1].split()[2]) == 64
        for line, name in zip(lines[1:-1], SHARDS, strict=True):
            size = (source / name).stat().st_size
            assert columns(line) == [name, f"{size:,} bytes", "sha256 ok"]
            assert (out_dir / name).read_bytes() == (source / name).read_bytes()


def test_checkout_refuses_a_non_empty_directory_unless_forced(repo, revisions, tmp_path, capsys):
    commit_both(capsys, repo, revisions)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "keep.txt").write_text("mine")
    code, out, err = run(capsys, "checkout", "HEAD", "--out", out_dir, "--repo", repo)
    assert (code, out) == (2, "")
    assert "non-empty directory" in err
    assert "--force" in err
    code, _, _ = run(capsys, "checkout", "HEAD", "--out", out_dir, "--force", "--repo", repo)
    assert code == 0
    assert (out_dir / SHARDS[1]).read_bytes() == (revisions.edited / SHARDS[1]).read_bytes()


def test_a_forced_checkout_marks_the_files_it_replaced(repo, revisions, tmp_path, capsys):
    """S-M2: each replaced file is marked and the summary counts them; a fresh checkout none."""
    commit_both(capsys, repo, revisions)
    out_dir = tmp_path / "out"
    code, out, _ = run(capsys, "checkout", "HEAD", "--out", out_dir, "--repo", repo)
    assert code == 0
    assert "replac" not in out
    (out_dir / SHARDS[0]).unlink()
    code, out, _ = run(capsys, "checkout", "HEAD", "--out", out_dir, "--force", "--repo", repo)
    lines = out.splitlines()
    assert code == 0
    assert lines[0].endswith(" s, replacing 1 existing file")
    first, second = (columns(line) for line in lines[1:3])
    assert (first[0], first[2:]) == (SHARDS[0], ["sha256 ok"])
    assert (second[0], second[2:]) == (SHARDS[1], ["sha256 ok", "replaced"])


def test_checkout_help_names_force_and_overwrite(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["checkout", "--help"])
    out = " ".join(capsys.readouterr().out.split())  # argparse wraps the help text
    assert exit_info.value.code == 0
    assert "--force" in out
    assert "--overwrite NAME" in out
    assert "(repeatable)" in out


def test_a_symlink_at_a_target_name_fails_checkout_with_exit_2(repo, revisions, tmp_path, capsys):
    commit_both(capsys, repo, revisions)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / SHARDS[1]).symlink_to(tmp_path / "elsewhere")
    code, out, err = run(capsys, "checkout", "HEAD", "--out", out_dir, "--force", "--repo", repo)
    assert (code, out) == (2, "")
    assert "is a symbolic link" in err
    assert "nothing was changed" in err
    assert [path.name for path in out_dir.iterdir()] == [SHARDS[1]]


def test_commands_find_the_repository_from_a_nested_directory(repo, revisions, capsys, monkeypatch):
    nested = repo / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    code, _, err = run(capsys, "commit", revisions.base, "-m", "from below")
    assert (code, err) == (0, "")
    assert run(capsys, "log")[0] == 0
    assert run(capsys, "show", "HEAD")[0] == 0
    code, out, _ = run(capsys, "checkout", "HEAD", "--out", "out")
    assert code == 0
    assert "sha256 ok" in out
    for name in SHARDS:
        assert (nested / "out" / name).read_bytes() == (revisions.base / name).read_bytes()


def test_tampered_payload_fails_checkout_with_exit_2(repo, revisions, tmp_path, capsys):
    commit_both(capsys, repo, revisions)
    store = ObjectStore(repo)
    tensor_id = store.get_commit(store.head())["files"][1]["tensors"][0]
    pid = store.get_object(tensor_id)["payload"]
    (store.dir / "payloads" / pid[:2] / pid).write_bytes(b"tampered")
    out_dir = tmp_path / "out"
    code, out, err = run(capsys, "checkout", "HEAD", "--out", out_dir, "--repo", repo)
    assert (code, out) == (2, "")
    assert "holds 8 bytes, not the" in err
    # S-M7: the first file was complete and verified, but nothing is renamed into place
    # unless every file verifies.
    assert list(out_dir.iterdir()) == []


def test_refusals_exit_2_with_a_message(repo, revisions, tmp_path, capsys, monkeypatch):
    def refused(*argv):
        code, out, err = run(capsys, *argv)
        assert (code, out) == (2, ""), (argv, code, out, err)
        assert err.startswith("wvc: "), (argv, code, out, err)
        return err

    assert "no commits yet" in refused("log", "--repo", repo)
    assert "no commits yet" in refused("show", "HEAD", "--repo", repo)
    commit_both(capsys, repo, revisions)
    assert "nothing changed" in refused("commit", revisions.edited, "-m", "again", "--repo", repo)
    for suffix in ("1", "2"):
        (repo / ".wvc" / "commits" / f"{'abcdef0' + suffix * 57}.json").write_text("{}")
    assert "ambiguous" in refused(
        "commit", revisions.base, "-m", "x", "--parent", "abcdef0", "--repo", repo
    )
    assert "no commit matches" in refused("show", "1234567", "--repo", repo)
    assert "unknown ref" in refused("checkout", "nope!", "--out", tmp_path / "o", "--repo", repo)
    assert "not a wvc repository" in refused("log", "--repo", tmp_path)
    assert "no *.safetensors" in refused("commit", tmp_path, "-m", "x", "--repo", repo)
    monkeypatch.chdir(tmp_path)
    assert "not inside a wvc repository" in refused("log")


def test_commit_parent_needs_branch_unless_it_is_the_head(repo, revisions, capsys):
    """H3: --parent other than the head is refused; --branch NAME commits there and moves HEAD."""
    first, second = commit_both(capsys, repo, revisions)
    argv = ("commit", revisions.edited, "-m", "side", "--parent", first[:7], "--repo", repo)
    code, out, err = run(capsys, *argv)
    assert (code, out) == (2, "")
    assert "not the head of branch 'main'" in err
    assert "--branch" in err
    code, out, err = run(capsys, *argv, "--branch", "side")
    assert (code, err) == (0, "")
    assert re.fullmatch(rf"commit {ID} \(side, parent {first[:12]}\)", out.splitlines()[0])
    store = ObjectStore(repo)
    assert (store.current_branch(), store.read_ref("main")) == ("side", second)
    code, out, err = run(capsys, *argv[:-2], "--branch", "main", "--repo", repo)
    assert code == 2
    assert "not an ancestor" in err


@pytest.mark.skipif(os.name == "nt", reason="Windows cannot name a directory with ESC in it")
def test_clone_escapes_the_url_it_prints(repo, revisions, tmp_path, capsys):
    """M3: the clone message escapes a URL's control characters like every other message."""
    commit_both(capsys, repo, revisions)
    remote = tmp_path / "shared\x1b[31m"
    assert run(capsys, "remote", "add", "origin", remote, "--repo", repo)[0] == 0
    assert run(capsys, "push", "--repo", repo)[0] == 0
    code, out, err = run(capsys, "clone", remote, tmp_path / "copy")
    assert (code, err) == (0, "")
    assert "\x1b" not in out
    assert out.startswith("cloned ")
    assert "shared\\x1b[31m" in out
