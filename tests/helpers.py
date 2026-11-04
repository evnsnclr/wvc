"""Helpers for writing small synthetic safetensors files and planting store records, and an
offline fake Hub."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from wvc import transport
from wvc.canonical import object_id
from wvc.hashing import build_manifest
from wvc.repo import CommitResult, commit_checkpoint
from wvc.safetensors_io import ITEMSIZES, load_checkpoint

NUMPY_FOR = {
    "F32": "<f4", "F16": "<f2", "I32": "<i4", "I64": "<i8", "I8": "i1", "U8": "u1", "U32": "<u4",
    "BF16": "<u2", "C64": "<c8", "F8_E4M3": "u1", "F8_E5M2": "u1",
}  # fmt: skip

CHILD_TIMEOUT_SECONDS = 120  # a hung child process fails its test instead of stalling the run
CONSOLE_ENV = ("PYTHONUTF8", "PYTHONIOENCODING")  # a developer's console settings stay out

# Two revisions of a two-shard checkpoint (see ``write_revisions``).
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
SPARSE, DENSE = "layers.0.sparse", "layers.0.dense"  # shard 1: edited sparsely and densely
SAME, GONE, NEW = "layers.1.same", "layers.1.gone", "layers.1.new"  # shard 2: kept, removed, added


def replace_or_skip(source: Path, target: Path) -> None:
    """``os.replace``, or a skip on Windows, which refuses to replace a file that is open or mapped.

    The scenario (a trainer saving over a file wvc is reading) cannot happen there: the
    trainer's own rename fails instead, as this one does.
    """
    try:
        os.replace(source, target)
    except PermissionError:
        if os.name != "nt":
            raise
        Path(source).unlink(missing_ok=True)
        pytest.skip("Windows refuses to replace a file that wvc holds open")


def link_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    """``symlink_to``, or a skip on Windows when this account may not create symbolic links."""
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError:
        if os.name != "nt":
            raise
        pytest.skip("this Windows account may not create symbolic links")


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 values to bfloat16 (round-to-nearest-even) and return the 16-bit patterns."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, np.ndarray]],
    metadata: dict[str, str] | None = None,
    header_order: Sequence[str] | None = None,
) -> Path:
    """Write ``name -> (dtype, array)``; BF16 arrays must already hold uint16 bit patterns.

    The data buffer holds the tensors in the order of ``tensors``. ``header_order`` lists the
    names in the order the JSON header gives them (default: the buffer order), since real
    checkpoints often list tensors in another order than their bytes.
    """
    header: dict[str, object] = {}
    buffer = bytearray()
    for name, (dtype, array) in tensors.items():
        raw = np.ascontiguousarray(array).astype(NUMPY_FOR[dtype], copy=False).tobytes()
        begin = len(buffer)
        buffer += raw
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [begin, len(buffer)],
        }
    if header_order is not None:
        assert sorted(header_order) == sorted(header), "header_order must list every tensor once"
        header = {name: header[name] for name in header_order}
    if metadata:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(buffer))
    return path


def store_checkpoint_files(store, path: Path) -> tuple[str, list[dict]]:
    """Store a checkpoint's tensors, header payloads, and manifest through the store API.

    Returns the manifest root and the file layout records that a commit of it holds.
    """
    ckpt = load_checkpoint(path)
    files = []
    for file in ckpt.files:
        data = file.path.read_bytes()
        ordered = sorted(file.tensors.values(), key=lambda info: (info.begin, info.end, info.name))
        raw = [(np.asarray(ckpt.raw(info.name)), ITEMSIZES[info.dtype]) for info in ordered]
        files.append(
            {
                "name": file.path.name,
                "header": store.put_payload(data[: file.data_start]),
                "tensors": [store.put_tensor(view, itemsize)["id"] for view, itemsize in raw],
                "sha256": hashlib.sha256(data).hexdigest(),
                "nbytes": len(data),
            }
        )
    return store.put_manifest(build_manifest(ckpt)), files


def commit(
    store,
    path: Path,
    message: str = "m",
    *,
    kind: str = "finetune",
    author: str = "tester",
    **options,
) -> CommitResult:
    """Commit the checkpoint at ``path``; ``options`` go to ``commit_checkpoint`` unchanged."""
    return commit_checkpoint(
        store, path, message=message, operation_kind=kind, author=author, **options
    )


def plant_commit(store, record: dict) -> str:
    """Write a commit record straight into a store, as a received ``.wvc`` may hold it.

    Nothing is validated (``put_commit`` would refuse a forged record); returns its id.
    """
    commit_id = object_id(record)
    path = store.dir / "commits" / f"{commit_id}.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return commit_id


def sparse_edit(values: np.ndarray, rng: np.random.Generator, fraction: float = 0.01) -> np.ndarray:
    """Copy float32 ``values`` with a random ``fraction`` of the elements redrawn."""
    edited = values.copy()
    flat = edited.reshape(-1)  # a view of the fresh copy
    changed = rng.choice(flat.size, size=int(flat.size * fraction), replace=False)
    flat[changed] = rng.standard_normal(changed.size).astype(np.float32)
    return edited


def dense_noise(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Copy float32 ``values`` with every element scaled by 1 + 1e-3 * N(0, 1)."""
    return values * (1 + 1e-3 * rng.standard_normal(values.shape)).astype(np.float32)


@dataclass(frozen=True)
class Revisions:
    base: Path  # a directory holding the two SHARDS
    edited: Path


def _write_shards(directory: Path, shards: dict) -> Path:
    directory.mkdir(parents=True)
    for name, (tensors, header_order) in shards.items():
        write_safetensors(directory / name, tensors, header_order=header_order)
    return directory


def write_revisions(root: Path, rng: np.random.Generator) -> Revisions:
    """Two revisions of a two-shard checkpoint whose headers list tensors out of buffer order.

    The edit redraws 1% of SPARSE, perturbs every element of DENSE, keeps SAME (BF16), removes
    GONE, and adds NEW (I8). Shard 1 keeps its names, shapes, and offsets, so its header bytes
    do not change.
    """
    sparse = rng.standard_normal((64, 64)).astype(np.float32)
    dense = rng.standard_normal(4096).astype(np.float32)
    same = rng.integers(0, 2**16, size=(32, 32), dtype=np.uint16)
    gone = rng.standard_normal(256).astype(np.float32)
    new = rng.integers(-128, 128, size=1000, dtype=np.int8)
    base = {
        SHARDS[0]: ({SPARSE: ("F32", sparse), DENSE: ("F32", dense)}, [DENSE, SPARSE]),
        SHARDS[1]: ({SAME: ("BF16", same), GONE: ("F32", gone)}, [GONE, SAME]),
    }
    edited_tensors = {
        SPARSE: ("F32", sparse_edit(sparse, rng)),
        DENSE: ("F32", dense_noise(dense, rng)),
    }
    edited = {
        SHARDS[0]: (edited_tensors, [DENSE, SPARSE]),
        SHARDS[1]: ({NEW: ("I8", new), SAME: ("BF16", same)}, [SAME, NEW]),
    }
    return Revisions(_write_shards(root / "base", base), _write_shards(root / "edited", edited))


# An offline stand-in for the Hugging Face Hub (see ``fake_hub``).

REAL_MAKE_HUB_CLIENT = transport.make_hub_client  # captured before any test replaces it


class FakeHubError(Exception):
    """Stands in for the client's HTTP errors."""


class FakeOperationAdd:
    """Stands in for ``huggingface_hub.CommitOperationAdd``."""

    def __init__(self, *, path_in_repo: str, path_or_fileobj: bytes) -> None:
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


@dataclass(frozen=True)
class FakeSibling:
    """Stands in for ``RepoSibling``; ``size`` and ``lfs`` are None without ``files_metadata``."""

    rfilename: str
    size: int | None
    lfs: dict | None


@dataclass(frozen=True)
class FakeModelInfo:
    """The two attributes of ``ModelInfo`` that wvc reads."""

    sha: str
    siblings: list[FakeSibling]


class FakeHubApi:
    """The ``HfApi`` methods wvc calls, over in-memory model repositories.

    ``files`` is the default repository ``owner/name``; ``add_repo`` adds others. Every
    ``.safetensors`` and ``.gguf`` file is an LFS object, as the Hub's default
    ``.gitattributes`` makes it,
    so ``model_info`` lists its SHA-256; a repository's commit sha is derived from its contents.
    """

    DEFAULT_REPO = "owner/name"

    def __init__(self, download_dir: Path) -> None:
        self.files = {".gitattributes": b"*.bin filter=lfs\n", "README.md": b"# a model\n"}
        self.repos: dict[str, dict[str, bytes]] = {self.DEFAULT_REPO: self.files}
        self.plain: set[str] = set()  # weight file names the Hub stores as plain blobs (no LFS)
        self.commits: list[list[str]] = []  # the operation paths of each create_commit
        self.calls: list[tuple[str, dict]] = []  # (method, keyword arguments) in call order
        self.download_dir = download_dir

    @property
    def listings(self) -> int:
        return sum(name == "list_repo_files" for name, _ in self.calls)

    def add_repo(self, repo_id: str, files: dict[str, bytes]) -> None:
        self.repos[repo_id] = dict(files)

    def sha_of(self, repo_id: str) -> str:
        """A 40-hex commit sha that changes whenever the repository's files change."""
        files = self._repo(repo_id)
        listing = "".join(
            f"{name}\0{hashlib.sha256(data).hexdigest()}\n" for name, data in sorted(files.items())
        )
        return hashlib.sha1(listing.encode("utf-8"), usedforsecurity=False).hexdigest()

    def _repo(self, repo_id: str) -> dict[str, bytes]:
        if repo_id not in self.repos:
            raise FakeHubError(f"404 Client Error: Repository Not Found for url: .../{repo_id}")
        return self.repos[repo_id]

    def list_repo_files(self, repo_id, *, revision=None, repo_type=None):
        where = {"repo_id": repo_id, "revision": revision, "repo_type": repo_type}
        self.calls.append(("list_repo_files", where))
        return list(self._repo(repo_id))

    def hf_hub_download(self, repo_id, filename, *, revision=None, repo_type=None):
        where = {"repo_id": repo_id, "revision": revision, "repo_type": repo_type}
        self.calls.append(("hf_hub_download", {**where, "filename": filename}))
        files = self._repo(repo_id)
        if filename not in files:
            raise FakeHubError(f"404 Client Error: {filename}")
        path = self.download_dir / repo_id.replace("/", "--") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[filename])
        return str(path)

    def create_commit(self, repo_id, operations, *, commit_message, revision=None, repo_type=None):
        where = {"repo_id": repo_id, "revision": revision, "repo_type": repo_type}
        details = {"operations": operations, "commit_message": commit_message}
        self.calls.append(("create_commit", {**where, **details}))
        self.commits.append([operation.path_in_repo for operation in operations])
        self._repo(repo_id).update({op.path_in_repo: op.path_or_fileobj for op in operations})

    def _pinned(self, repo_id, revision):
        """Only ``main`` and the repository's own commit sha resolve, as on the Hub."""
        if revision not in (None, "main", self.sha_of(repo_id)):
            raise FakeHubError(f"404 Client Error: Revision Not Found for {repo_id}@{revision}")

    def read_range(self, repo_id, filename, revision, start, length):
        """Bytes ``start:start + length`` of a file at the pinned commit only.

        A zero-length range returns ``b""`` and is not recorded, as the real reader makes no
        request for it.
        """
        if length == 0:
            return b""
        where = {"repo_id": repo_id, "revision": revision, "start": start, "length": length}
        self.calls.append(("read_range", {**where, "filename": filename}))
        files = self._repo(repo_id)
        if revision != self.sha_of(repo_id):
            raise FakeHubError(f"404 Client Error: Revision Not Found for {repo_id}@{revision}")
        if filename not in files:
            raise FakeHubError(f"404 Client Error: {filename}")
        if start + length > len(files[filename]):
            raise FakeHubError(f"416 Client Error: Range Not Satisfiable for {filename}")
        return files[filename][start : start + length]

    def model_info(self, repo_id, *, revision=None, files_metadata=False):
        where = {"repo_id": repo_id, "revision": revision, "files_metadata": files_metadata}
        self.calls.append(("model_info", where))
        files = self._repo(repo_id)
        self._pinned(repo_id, revision)
        siblings = []
        for name, data in sorted(files.items()):
            lfs = None
            lfs_type = name.endswith((".safetensors", ".gguf"))
            if files_metadata and lfs_type and name not in self.plain:
                digest = hashlib.sha256(data).hexdigest()
                lfs = {"size": len(data), "sha256": digest, "pointer_size": 134}
            siblings.append(FakeSibling(name, len(data) if files_metadata else None, lfs))
        return FakeModelInfo(sha=self.sha_of(repo_id), siblings=siblings)


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """``os.environ`` without any ``WVC_*`` setting or console encoding override, plus ``extra``.

    A child process sees only what its test sets, not the developer's ``WVC_AUTHOR``,
    ``WVC_DEBUG``, or ``PYTHONUTF8``.
    """
    inherited = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("WVC_") and key not in CONSOLE_ENV
    }
    return {**inherited, **(extra or {})}


def run_wvc(
    *argv: object, cwd: Path, env: dict[str, str] | None = None
) -> tuple[int, bytes, bytes]:
    """Run the CLI in a child process in ``cwd``, through its real standard streams.

    The child gets ``child_env(env)``; its exit code, stdout, and stderr come back as bytes.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "wvc", *map(str, argv)],
        capture_output=True,
        env=child_env(env),
        cwd=cwd,
        check=False,
        timeout=CHILD_TIMEOUT_SECONDS,
    )
    return completed.returncode, completed.stdout, completed.stderr
