"""Checkpoint release discipline: bounded open readers, close, and many-shard checkpoints."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from helpers import write_safetensors

from wvc import safetensors_io
from wvc.cli import main
from wvc.diff import diff_checkpoints, summarize
from wvc.hashing import build_manifest, gguf_hash_lines, hash_files, tensor_digests
from wvc.safetensors_io import MAX_OPEN_READERS, TensorReader, load_checkpoint

SHARD_COUNT = 300  # more shards than a 256-descriptor limit allows open at once
DESCRIPTOR_LIMIT = 64
CHANGED_EVERY = 7  # shard i of side b differs from side a when i % 7 == 0
FD_DIR = Path("/dev/fd")


def _values(index: int, bump: float) -> np.ndarray:
    values = np.full((2, 2), float(index), dtype=np.float32)
    return values + np.float32(bump) if index % CHANGED_EVERY == 0 else values


def _write_sharded(directory: Path, *, bump: float = 0.0) -> Path:
    """``SHARD_COUNT`` one-tensor shards, ``t000`` to ``t299``."""
    directory.mkdir()
    for index in range(SHARD_COUNT):
        name = f"model-{index + 1:05d}-of-{SHARD_COUNT:05d}.safetensors"
        write_safetensors(directory / name, {f"t{index:03d}": ("F32", _values(index, bump))})
    return directory


def _write_single(path: Path) -> Path:
    """The same tensors as an unbumped sharded checkpoint, in one file."""
    tensors = {f"t{index:03d}": ("F32", _values(index, 0.0)) for index in range(SHARD_COUNT)}
    return write_safetensors(path, tensors)


def _open_descriptors() -> int:
    return len(os.listdir(FD_DIR))


@pytest.fixture
def few_descriptors():
    """Lower the soft RLIMIT_NOFILE to 64 for the test (POSIX only) and restore it afterwards."""
    resource = pytest.importorskip("resource", reason="RLIMIT_NOFILE is POSIX-only")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = DESCRIPTOR_LIMIT if hard == resource.RLIM_INFINITY else min(DESCRIPTOR_LIMIT, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
    try:
        yield limit
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


@pytest.fixture
def sharded(tmp_path):
    """Two 300-shard checkpoints (a, b) and a single-file copy of a, written before any limit."""
    return (
        _write_sharded(tmp_path / "a"),
        _write_sharded(tmp_path / "b", bump=1.0),
        _write_single(tmp_path / "single.safetensors"),
    )


def test_many_shards_hash_and_diff_under_a_descriptor_limit(sharded, few_descriptors):
    dir_a, dir_b, single = sharded
    expected_root = build_manifest(load_checkpoint(single))["root"]
    with load_checkpoint(dir_a) as ckpt_a, load_checkpoint(dir_b) as ckpt_b:
        digests_a, digests_b = tensor_digests(ckpt_a), tensor_digests(ckpt_b)
        assert build_manifest(ckpt_a, digests_a)["root"] == expected_root
        digests, files = hash_files(ckpt_a)
        assert digests == digests_a
        assert len(files) == SHARD_COUNT
        lines = gguf_hash_lines(ckpt_a)
        assert len(lines) == 2 * SHARD_COUNT
        diffs = diff_checkpoints(ckpt_a, ckpt_b, digests_a, digests_b)
    assert [d.name for d in diffs] == sorted(f"t{index:03d}" for index in range(SHARD_COUNT))
    changed = {d.name for d in diffs if d.status == "changed"}
    assert changed == {f"t{index:03d}" for index in range(0, SHARD_COUNT, CHANGED_EVERY)}
    assert all(d.changed == 4 for d in diffs if d.status == "changed")
    assert summarize(diffs)["by_status"]["unchanged"] == SHARD_COUNT - len(changed)


def test_many_shards_through_the_cli_under_a_descriptor_limit(sharded, few_descriptors, capsys):
    dir_a, dir_b, single = sharded
    expected_root = build_manifest(load_checkpoint(single))["root"]
    assert main(["hash", "--root-only", str(dir_a)]) == 0
    assert capsys.readouterr().out.strip() == expected_root
    assert main(["hash", "--format", "gguf-hash", str(dir_a)]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 2 * SHARD_COUNT
    manifest = dir_a.parent / "a.manifest.json"
    assert main(["hash", str(dir_a)]) == 0
    manifest.write_text(capsys.readouterr().out, encoding="utf-8")
    assert main(["verify", str(dir_a), str(manifest)]) == 0
    capsys.readouterr()
    assert main(["diff", "--json", str(dir_a), str(dir_b)]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["a"]["root"] == expected_root
    changed = len(range(0, SHARD_COUNT, CHANGED_EVERY))
    assert payload["summary"]["by_status"]["changed"] == changed
    assert payload["summary"]["by_status"]["unchanged"] == SHARD_COUNT - changed


def test_open_readers_are_bounded_and_released_on_close(tmp_path, monkeypatch):
    shard_count = MAX_OPEN_READERS + 9
    for index in range(shard_count):
        write_safetensors(
            tmp_path / f"s{index:03d}.safetensors", {f"t{index:03d}": ("F32", _values(index, 0))}
        )
    closed: list[Path] = []
    original_close = TensorReader.close

    def counting_close(self):
        closed.append(self.file.path)
        original_close(self)

    monkeypatch.setattr(TensorReader, "close", counting_close)
    counting = FD_DIR.is_dir()
    before = _open_descriptors() if counting else 0
    ckpt = load_checkpoint(tmp_path)
    for name in ckpt.tensors:
        assert ckpt.raw(name).nbytes == 16
        if counting:
            assert _open_descriptors() - before <= MAX_OPEN_READERS
    # The oldest readers were closed first, in the order their files were opened.
    assert closed == [file.path for file in ckpt.files[: shard_count - MAX_OPEN_READERS]]
    assert ckpt.raw("t024").nbytes == 16  # still open: no reader is closed or reopened
    assert len(closed) == shard_count - MAX_OPEN_READERS
    ckpt.raw("t000")  # reopened, so the least recently used reader (t009's) is closed
    assert closed[-1] == ckpt.files[MAX_OPEN_READERS - 7].path
    with ckpt:
        pass
    assert len(closed) == shard_count + 1  # every reader opened was closed exactly once
    if counting:
        assert _open_descriptors() == before
    with pytest.raises(ValueError, match="closed"):
        ckpt.raw("t000")


def test_closing_a_reader_releases_its_descriptor(tmp_path):
    if not FD_DIR.is_dir():
        pytest.skip("no /dev/fd to count descriptors")
    path = write_safetensors(tmp_path / "m.safetensors", {"a": ("F32", _values(1, 0))})
    before = _open_descriptors()
    reader = TensorReader(load_checkpoint(path).files[0])
    assert bytes(reader.raw("a")) == _values(1, 0).tobytes()
    assert _open_descriptors() == before + 1  # the map holds one descriptor
    reader.close()
    assert _open_descriptors() == before
    with pytest.raises(ValueError, match="closed"):
        reader.raw("a")
    # A view still alive at close keeps its bytes readable until it is dropped.
    reader = TensorReader(load_checkpoint(path).files[0])
    view = reader.raw("a")
    reader.close()
    assert bytes(view) == _values(1, 0).tobytes()
    del view
    assert _open_descriptors() == before


def test_lru_bound_is_read_at_use(tmp_path, monkeypatch):
    monkeypatch.setattr(safetensors_io, "MAX_OPEN_READERS", 1)
    for index in range(3):
        write_safetensors(
            tmp_path / f"s{index}.safetensors", {f"t{index}": ("F32", _values(index, 0))}
        )
    with load_checkpoint(tmp_path) as ckpt:
        assert {name: bytes(ckpt.raw(name)) for name in ("t0", "t1", "t2", "t0")} == {
            f"t{index}": _values(index, 0).tobytes() for index in range(3)
        }
        assert len(ckpt._readers) == 1
