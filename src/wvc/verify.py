"""Check a checkpoint against a manifest written by ``wvc hash``.

Every tensor named by the manifest or the checkpoint gets one status: ``ok`` (same dtype,
shape, and SHA-256), ``mismatch`` (same dtype and shape, different SHA-256), ``reshaped``
(dtype or shape differ), ``missing`` (only in the manifest), or ``extra`` (only in the
checkpoint). Digests are taken over the stored bytes; nothing is decoded. The manifest
root is recomputed over every tensor of the checkpoint and compared with the stated root.
``validate_manifest`` and ``ManifestError`` live in ``wvc.records`` and are re-exported here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import wvc.store
from wvc.fsio import read_input
from wvc.hashing import build_manifest, tensor_digests
from wvc.records import ManifestError, validate_manifest
from wvc.safetensors_io import Checkpoint

STATUSES = ("ok", "mismatch", "reshaped", "missing", "extra")


@dataclass(frozen=True)
class TensorRecord:
    dtype: str
    shape: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class TensorCheck:
    name: str
    status: str  # ok | mismatch | reshaped | missing | extra
    expected: TensorRecord | None = None  # from the manifest; None when the tensor is extra
    actual: TensorRecord | None = None  # from the checkpoint; None when the tensor is missing

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VerifyReport:
    manifest_root: str
    computed_root: str
    tensors: tuple[TensorCheck, ...]  # sorted by name

    @property
    def root_match(self) -> bool:
        return self.manifest_root == self.computed_root

    @property
    def summary(self) -> dict[str, int]:
        return {status: sum(t.status == status for t in self.tensors) for status in STATUSES}

    @property
    def passed(self) -> bool:
        """True only when every tensor is ok and the recomputed root matches."""
        return self.root_match and all(t.status == "ok" for t in self.tensors)

    def to_dict(self, *, include_ok: bool = True) -> dict:
        return {
            "manifest_root": self.manifest_root,
            "computed_root": self.computed_root,
            "root_match": self.root_match,
            "summary": self.summary,
            "tensors": [t.to_dict() for t in self.tensors if include_ok or t.status != "ok"],
        }


def load_manifest(path: str | Path) -> dict:
    """Read a manifest written by ``wvc hash`` and validate its structure.

    The file must be (or link to) a regular file of at most ``RECORD_LIMIT`` bytes.
    """
    path = Path(path)
    raw = read_input(path, wvc.store.RECORD_LIMIT)  # looked up per call
    try:
        data = json.loads(raw)  # bytes: json detects UTF-8/16/32 and a BOM
    except (ValueError, RecursionError) as exc:  # UnicodeDecodeError and JSONDecodeError too
        raise ManifestError(f"{path}: manifest is not valid JSON: {exc}") from exc
    return validate_manifest(data, source=str(path))


def _check(name: str, expected: TensorRecord | None, actual: TensorRecord | None) -> TensorCheck:
    """Classify one tensor from its manifest record and its checkpoint record."""
    if expected is None and actual is None:
        raise ValueError(f"tensor {name!r} is in neither the manifest nor the checkpoint")
    if actual is None:
        status = "missing"
    elif expected is None:
        status = "extra"
    elif (expected.dtype, expected.shape) != (actual.dtype, actual.shape):
        status = "reshaped"
    else:
        status = "ok" if expected.sha256 == actual.sha256 else "mismatch"
    return TensorCheck(name, status, expected, actual)


def verify_checkpoint(ckpt: Checkpoint, manifest: dict) -> VerifyReport:
    """Check every tensor of ``ckpt`` against ``manifest`` and recompute the manifest root.

    Each checkpoint tensor is hashed once from its stored bytes: the tensors the manifest
    lists for the comparison, and extra tensors too so that the recomputed root is complete.
    """
    entries = validate_manifest(manifest)["tensors"]
    digests = tensor_digests(ckpt)
    expected = {
        name: TensorRecord(entry["dtype"], tuple(entry["shape"]), entry["sha256"])
        for name, entry in entries.items()
    }
    actual = {
        name: TensorRecord(info.dtype, info.shape, digests[name])
        for name, info in ckpt.tensors.items()
    }
    checks = tuple(
        _check(name, expected.get(name), actual.get(name))
        for name in sorted(expected.keys() | actual.keys())
    )
    return VerifyReport(
        manifest_root=manifest["root"],
        computed_root=build_manifest(ckpt, digests)["root"],
        tensors=checks,
    )
