"""Fixtures shared by the test modules; the helpers they use live in ``helpers.py``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import FakeHubApi, FakeHubError, FakeOperationAdd, Revisions, write_revisions

from wvc import transport
from wvc.cli import DEBUG_ENV
from wvc.repo import AUTHOR_ENV, init_repository
from wvc.store import ObjectStore
from wvc.transport import HubClient


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def store(tmp_path: Path) -> ObjectStore:
    """An empty repository at ``tmp_path / "repo"``, as ``wvc init`` makes it."""
    return ObjectStore(init_repository(tmp_path / "repo"))


@pytest.fixture
def revisions(tmp_path: Path, rng: np.random.Generator) -> Revisions:
    return write_revisions(tmp_path / "checkpoints", rng)


@pytest.fixture(autouse=True)
def _no_wvc_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts without ``WVC_DEBUG`` and ``WVC_AUTHOR``, whatever the shell sets.

    With ``WVC_DEBUG`` set, a wvc error is re-raised instead of exiting 2; with ``WVC_AUTHOR``
    set, a commit without ``--author`` would succeed.
    """
    monkeypatch.delenv(DEBUG_ENV, raising=False)
    monkeypatch.delenv(AUTHOR_ENV, raising=False)


@pytest.fixture(autouse=True)
def _no_real_hub_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test builds a real Hub client, so none reaches the network; use ``fake_hub``."""

    def refuse() -> HubClient:
        raise AssertionError("a test tried to build a real Hugging Face Hub client")

    monkeypatch.setattr(transport, "make_hub_client", refuse)


@pytest.fixture
def fake_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHubApi:
    """An in-memory Hub model repository that ``HfTransport`` reaches instead of the Hub."""
    api = FakeHubApi(tmp_path / "hub-downloads")
    client = HubClient(
        api=api, operation_add=FakeOperationAdd, errors=(FakeHubError,), read_range=api.read_range
    )
    monkeypatch.setattr(transport, "make_hub_client", lambda: client)
    return api
