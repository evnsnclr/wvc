"""Reuse of stored tensor objects and its verification markers (spec section 3.6).

A commit reuses the stored object of a tensor id only after decoding it through its delta
chain back to that id (``check_reusable``): a record planted for a public tensor id, pointing
at junk of the right size, would otherwise stand in for the tensor and its bytes would never be
stored. An object that fails is replaced by the tensor stored in full (``replace_tensor``).
Pull decodes every received object the same way before writing its record (``check_decodes``).

An empty file ``.wvc/verified/<aa>/<id>`` records that this store decoded the object of tensor
``id`` back to that id (on reuse, at pull, or in ``wvc fsck``) or encoded it from the tensor's
own bytes, so reuse may skip decoding it again. Markers are local hints: push and pull never
transfer them, ``fsck`` does not check their layout, and reads (``read_tensor``, checkout,
``fsck``) verify everything whatever they say. All are removed whenever a stored record or
payload is replaced. Where ``verified`` or one of its shards exists but is not a directory (a
symbolic link included), markers there are neither read nor written.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import numpy as np

from wvc.fsio import fsync_directory, temporary_name
from wvc.records import (
    StoreError,
    check_id,
    check_tensor_bytes,
    expected_payload_size,
    validate_object,
)

VERIFIED_DIR = "verified"


class ReuseMixin:
    """The reuse checks and markers of ``ObjectStore``, which provides the members stubbed
    here."""

    dir: Path  # the .wvc directory
    _checked: set[Path]  # shard directories found not to be links
    _cleaned: set[Path]  # directories rid of stale temporaries

    def _layout_path(self, *parts: str) -> Path:
        raise NotImplementedError

    def _write(self, path: Path, data: bytes) -> None:
        raise NotImplementedError

    def _tensor_limit(self) -> int:
        raise NotImplementedError

    def _decode(self, record: dict, base: np.ndarray | None) -> np.ndarray:
        raise NotImplementedError

    def get_object(self, tensor_id: str) -> dict:
        raise NotImplementedError

    def payload_size(self, pid: str) -> int | None:
        raise NotImplementedError

    def read_tensor(self, tensor_id: str) -> np.ndarray:
        raise NotImplementedError

    def put_tensor(
        self,
        raw: np.ndarray,
        itemsize: int,
        *,
        base_id: str | None = None,
        tensor_id: str | None = None,
        replace: bool = False,
    ) -> dict:
        raise NotImplementedError

    def check_reusable(self, tensor_id: str, nbytes: int) -> dict | None:
        """The object record of a stored tensor of ``nbytes`` bytes, when it may be reused.

        The record must be readable and valid (its errors are raised, as ``get_object`` raises
        them). It must claim ``nbytes``, and its payload must be a non-empty regular file of the
        size the record implies. Unless a marker says this store already checked the object, it
        is then decoded through its delta chain, must reproduce ``tensor_id``, and is marked.
        None when one of these checks fails: the caller then stores the tensor again with
        ``replace_tensor``. Running out of memory is not caught.
        """
        record = self.get_object(tensor_id)
        try:
            size = self.payload_size(record["payload"])
            if record["nbytes"] != nbytes or not size or size != expected_payload_size(record):
                return None
            if not self.is_verified(tensor_id):
                self.read_tensor(tensor_id)
                self.mark_verified(tensor_id)
        except StoreError:
            return None
        return record

    def replace_tensor(self, raw: np.ndarray, itemsize: int, *, tensor_id: str) -> dict:
        """Store a tensor in full under its id, over a stored object that ``check_reusable``
        refused, and return the new record.

        The tensor is encoded without a base (a stored chain may be what failed). Every marker
        is removed first, since objects stored as deltas on this one may carry markers; the old
        record is overwritten without being read. ``tensor_id`` is vouched for by the caller as
        the SHA-256 of ``raw``, as for ``put_tensor``.
        """
        self.clear_verified()
        return self.put_tensor(raw, itemsize, tensor_id=tensor_id, replace=True)

    def check_decodes(self, record: dict, base: np.ndarray | None = None) -> np.ndarray:
        """Decode an object record, stored or not, and return the tensor's bytes.

        Its payload and, for a delta, its base object must be stored; ``base`` may give the
        base's decoded bytes, which are otherwise rebuilt and checked. IntegrityError unless
        the record decodes to its id. Nothing is written.
        """
        validate_object(record)
        check_tensor_bytes(record, self._tensor_limit(), f"object {record['id']}")
        if "base" not in record:
            return self._decode(record, None)
        return self._decode(record, self.read_tensor(record["base"]) if base is None else base)

    def is_verified(self, tensor_id: str) -> bool:
        """True when a marker says this store checked that the object decodes to its id."""
        path = self._marker_path(tensor_id)
        return path is not None and stat.S_ISREG(_lstat_mode(path))

    def mark_verified(self, tensor_id: str) -> None:
        """Write the empty marker of an object checked here (atomically, like every write).

        ``verified`` and the shard are made when missing, even when removed meanwhile.
        """
        path = self._marker_path(tensor_id)
        if path is None or os.path.lexists(path):
            return
        path.parent.parent.mkdir(exist_ok=True)
        path.parent.mkdir(exist_ok=True)
        self._write(path, b"")

    def unmark_verified(self, tensor_id: str) -> None:
        """Remove the marker of an object found not to decode, when there is one."""
        path = self._marker_path(tensor_id)
        if path is not None and stat.S_ISREG(_lstat_mode(path)):
            path.unlink(missing_ok=True)

    def clear_verified(self) -> None:
        """Remove every marker: ``verified`` is renamed away in one step, then deleted.

        Anything other than a directory in its place is removed itself, never followed.
        """
        directory = self._layout_path(VERIFIED_DIR)
        mode = _lstat_mode(directory)
        if mode == 0:
            return
        if not stat.S_ISDIR(mode):
            directory.unlink()
            return
        doomed = self.dir / temporary_name()
        os.rename(directory, doomed)
        fsync_directory(self.dir)
        shutil.rmtree(doomed)
        self._checked = {path for path in self._checked if path.parent != directory}
        self._cleaned = {path for path in self._cleaned if path.parent != directory}

    def _marker_path(self, tensor_id: str) -> Path | None:
        """``verified/<aa>/<id>``, or None when ``verified`` or the shard is not a directory."""
        directory = self._layout_path(VERIFIED_DIR)
        shard = directory / check_id(tensor_id, "tensor id")[:2]
        for part in (directory, shard):
            mode = _lstat_mode(part)
            if mode != 0 and not stat.S_ISDIR(mode):
                return None
        return shard / tensor_id


def _lstat_mode(path: Path) -> int:
    """The ``st_mode`` of ``path`` itself (a link is not followed), or 0 when it is absent."""
    try:
        return os.lstat(path).st_mode
    except FileNotFoundError:
        return 0
