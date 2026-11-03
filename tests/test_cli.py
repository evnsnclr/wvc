import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
from helpers import child_env, run_wvc, write_safetensors

from wvc import cli
from wvc.cli import main
from wvc.cli_common import format_table, plural, short_id
from wvc.codec import CodecError
from wvc.errors import WvcError
from wvc.hashing import build_manifest, gguf_hash_lines
from wvc.hub import HubError
from wvc.ids import short
from wvc.lowrank import LowRankError
from wvc.records import StoreError
from wvc.safetensors_io import SafetensorsError, load_checkpoint
from wvc.store import manifest_json
from wvc.verify import ManifestError


def test_hash_prints_manifest_json(tmp_path, rng, capsys):
    a = rng.standard_normal((2, 2)).astype(np.float32)
    path = write_safetensors(tmp_path / "m.safetensors", {"a": ("F32", a)})
    assert main(["hash", str(path)]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["tensor_count"] == 1
    assert len(manifest["root"]) == 64
    assert main(["hash", "--root-only", str(path)]) == 0
    assert capsys.readouterr().out.strip() == manifest["root"]


def test_diff_table_and_json(tmp_path, rng, capsys):
    base = rng.standard_normal((4, 6)).astype(np.float32)
    edited = base.copy()
    edited[:, 2] += 1.0
    pa = write_safetensors(tmp_path / "a.safetensors", {"w": ("F32", base), "u": ("F32", base)})
    pb = write_safetensors(tmp_path / "b.safetensors", {"w": ("F32", edited), "u": ("F32", base)})
    assert main(["diff", str(pa), str(pb), "--rank"]) == 1
    out = capsys.readouterr().out
    assert "1 changed, 1 unchanged" in out
    assert "4/24" in out
    assert "\nchanged" in out
    assert main(["diff", str(pa), str(pb), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["elements_changed"] == 4
    assert [t["name"] for t in payload["tensors"]] == ["w"]
    assert main(["diff", str(pa), str(pa)]) == 0


def test_bad_path_returns_2(tmp_path, capsys):
    assert main(["hash", str(tmp_path / "missing")]) == 2
    assert "wvc:" in capsys.readouterr().err


def test_diff_table_has_no_trailing_whitespace(tmp_path, rng, capsys):
    base = rng.standard_normal((3, 3)).astype(np.float32)
    edited = base.copy()
    edited[0, 0] += 1.0
    pa = write_safetensors(tmp_path / "a.safetensors", {"w": ("F32", base)})
    pb = write_safetensors(tmp_path / "b.safetensors", {"w": ("F32", edited)})
    main(["diff", str(pa), str(pb), "--rank"])
    assert all(line == line.rstrip() for line in capsys.readouterr().out.splitlines())


def test_documents_are_the_same_bytes_on_every_platform(tmp_path, rng):
    """L10: manifests, gguf-hash lines, and JSON are UTF-8 with LF, whatever the console is."""
    base = rng.standard_normal((4, 6)).astype(np.float32)
    edited = base.copy()
    edited[:, 2] += 1.0
    tensors = {"modèle.w": ("F32", base), "b": ("F32", np.ones(3, dtype=np.float32))}
    pa = write_safetensors(tmp_path / "a.safetensors", tensors)
    pb = write_safetensors(tmp_path / "b.safetensors", {**tensors, "modèle.w": ("F32", edited)})
    ckpt = load_checkpoint(pa)
    code, out, err = run_wvc("hash", pa, cwd=tmp_path)
    assert (code, err) == (0, b"")
    assert out == manifest_json(build_manifest(ckpt))
    code, out, err = run_wvc("hash", "--root-only", pa, cwd=tmp_path)
    assert (code, err) == (0, b""), err
    assert re.fullmatch(rb"[0-9a-f]{64}\n", out), out
    ascii_console = {"PYTHONIOENCODING": "ascii"}
    code, out, err = run_wvc("hash", "--format", "gguf-hash", pa, env=ascii_console, cwd=tmp_path)
    assert (code, err) == (0, b"")
    assert out == ("\n".join(gguf_hash_lines(ckpt)) + "\n").encode("utf-8")
    code, out, err = run_wvc("diff", pa, pb, "--json", cwd=tmp_path)
    assert (code, err) == (1, b"")
    assert b"\r" not in out
    assert json.loads(out)["summary"]["elements_changed"] == 4
    # Messages still go through the text layer (platform newlines), escaped rather than
    # crashing on a narrow console.
    code, out, err = run_wvc("diff", pa, pb, env=ascii_console, cwd=tmp_path)
    assert code == 1, err
    assert b"Traceback" not in err, err
    assert b"mod\\xe8le.w" in out


def test_repository_documents_are_lf_utf8_bytes(tmp_path, rng):
    """L10 for show --json, card, and fsck --json, through the real standard output."""
    path = write_safetensors(
        tmp_path / "m.safetensors", {"w": ("F32", rng.standard_normal(8).astype(np.float32))}
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    assert run_wvc("init", repo, cwd=tmp_path)[0] == 0
    commit = ["commit", path, "-m", "naïve 模型", "--op", "finetune", "--author", "Évan"]
    code, out, err = run_wvc(*commit, "--repo", repo, cwd=tmp_path)
    assert (code, err) == (0, b""), err
    documents = {}
    for name, argv in {
        "show": ["show", "HEAD", "--json"],
        "card": ["card", "HEAD"],
        "fsck": ["fsck", "--json"],
    }.items():
        code, out, err = run_wvc(*argv, "--repo", repo, cwd=tmp_path)
        assert (code, err) == (0, b""), (name, err)
        assert b"\r" not in out, name
        assert out.endswith(b"\n"), name
        documents[name] = out
    record = json.loads(documents["show"])["commit"]
    assert record["message"] == "naïve 模型"
    assert record["author"] == "Évan"
    assert documents["card"].startswith(b"---\n")
    assert json.loads(documents["fsck"])["ok"] is True


PIPE_CAPACITY = 1 << 16  # bytes a pipe holds before a writer blocks (64 KiB on Linux and macOS)


@pytest.mark.skipif(os.name == "nt", reason="Windows reports a closed pipe as EINVAL, not EPIPE")
def test_a_reader_closing_the_pipe_ends_wvc_quietly(tmp_path, rng):
    """P-M3: 'wvc hash big | head -c 5' exits 0 and prints nothing about the pipe."""
    tensors = {
        f"model.layers.{i}.{'mlp.' * 40}weight": ("F32", rng.standard_normal(4).astype(np.float32))
        for i in range(400)
    }
    path = write_safetensors(tmp_path / "many.safetensors", tensors)
    manifest = manifest_json(build_manifest(load_checkpoint(path)))
    assert len(manifest) > 2 * PIPE_CAPACITY  # so the write cannot finish into the pipe
    with subprocess.Popen(
        [sys.executable, "-m", "wvc", "hash", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env(),
        cwd=tmp_path,
    ) as child:
        assert child.stdout is not None
        assert child.stderr is not None
        assert child.stdout.read(5) == manifest[:5]
        child.stdout.close()
        err = child.stderr.read()
        code = child.wait(timeout=60)
    assert (code, err) == (0, b""), err.decode("utf-8", "replace")


def _raising(exc):
    def command(args):
        raise exc

    return command


def test_a_bug_propagates_and_a_wvc_error_exits_2(monkeypatch, capsys):
    """P-M2: only wvc's own errors (and OS errors) become 'wvc: ...' with exit 2."""
    for bug in (RuntimeError("a bug"), ValueError("cannot reshape array"), KeyError("k")):
        monkeypatch.setattr(cli, "cmd_hash", _raising(bug))
        with pytest.raises(type(bug)):
            main(["hash", "x"])
    for error in (SafetensorsError("bad \x1b header"), WvcError("bad \x1b header")):
        monkeypatch.setattr(cli, "cmd_hash", _raising(error))
        assert main(["hash", "x"]) == 2
        assert capsys.readouterr() == ("", "wvc: bad \\x1b header\n")
    for error in (OSError("disk on fire"), RecursionError("too deep")):
        monkeypatch.setattr(cli, "cmd_hash", _raising(error))
        assert main(["hash", "x"]) == 2
        assert capsys.readouterr().err == f"wvc: {error}\n"
    monkeypatch.setattr(cli, "cmd_hash", _raising(MemoryError()))
    assert main(["hash", "x"]) == 2
    assert capsys.readouterr().err == "wvc: out of memory while running hash\n"


def test_ctrl_c_exits_130_with_one_line(monkeypatch, capsys):
    monkeypatch.setattr(cli, "cmd_hash", _raising(KeyboardInterrupt()))
    assert main(["hash", "x"]) == 130
    assert capsys.readouterr() == ("", "wvc: interrupted\n")


def test_a_broken_pipe_that_is_not_stdout_is_an_error(monkeypatch, capsys):
    """Standard output here is pytest's capture, no pipe, so the closed pipe was another one."""
    monkeypatch.setattr(cli, "cmd_hash", _raising(BrokenPipeError(32, "Broken pipe")))
    assert main(["hash", "x"]) == 2
    assert capsys.readouterr().err == "wvc: [Errno 32] Broken pipe\n"


def test_wvc_debug_reraises_errors(monkeypatch, capsys):
    monkeypatch.setattr(cli, "cmd_hash", _raising(SafetensorsError("bad header")))
    monkeypatch.setenv(cli.DEBUG_ENV, "")  # empty means off
    assert main(["hash", "x"]) == 2
    monkeypatch.setenv(cli.DEBUG_ENV, "1")
    with pytest.raises(SafetensorsError, match="bad header"):
        main(["hash", "x"])
    for error in (OSError("disk"), MemoryError(), KeyboardInterrupt()):
        monkeypatch.setattr(cli, "cmd_hash", _raising(error))
        with pytest.raises(type(error)):
            main(["hash", "x"])
    capsys.readouterr()


def test_wvc_errors_are_value_errors():
    """Every error root derives from WvcError, which stays a ValueError for older callers."""
    for root in (StoreError, SafetensorsError, ManifestError, HubError, CodecError, LowRankError):
        assert issubclass(root, WvcError)
        assert issubclass(root, ValueError)


def test_format_table_aligns_escapes_and_trims():
    table = format_table([("status", "tensor"), ("changed", "w\x1b"), ("ok", "")])
    assert table.splitlines() == ["status   tensor", "changed  w\\x1b", "ok"]
    assert format_table([]) == ""


def test_plural_and_short_id():
    expected = ["0 tensors", "1 tensor", "2 tensors", "1,234 tensors"]
    assert [plural(n, "tensor") for n in (0, 1, 2, 1234)] == expected
    assert short_id("0123456789abcdef" * 4) == "0123456789ab"
    assert short_id(None) == "none"
    assert short_id is short  # the library's messages and the CLI render ids alike


def test_top_level_help_lists_the_exit_codes(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    out = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert "exit codes:" in out
    assert "2    a wvc error or bad usage" in out
    assert "130  interrupted (Ctrl-C)" in out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["commit", "m.safetensors", "-m", "x", "--op", "retrain"], "invalid choice: 'retrain'"),
        (["diff", "a.safetensors", "b.safetensors", "--rank-cap", "-1"], "got '-1'"),
    ],
    ids=["unknown-op", "negative-rank-cap"],
)
def test_bad_option_values_are_usage_errors(argv, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    err = capsys.readouterr().err
    assert exit_info.value.code == 2
    assert err.startswith(f"usage: wvc {argv[0]}")
    assert message in err
