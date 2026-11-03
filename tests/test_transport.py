import http.server
import importlib
import inspect
import os
import re
import sys
import threading
import types
from collections.abc import Iterator

import pytest
from helpers import REAL_MAKE_HUB_CLIENT, FakeHubError

from wvc.fsck import fsck
from wvc.fsio import TEMP_NAME, remove_stale_temporaries
from wvc.remote import transport_for
from wvc.repo import init_repository
from wvc.store import IntegrityError, ObjectStore, StoreError
from wvc.transport import (
    RANGE_TIMEOUT_SECONDS,
    DirTransport,
    HfTransport,
    HubClient,
    Transport,
    _range_reader,
    check_relpath,
    fetch_range,
)

LIMIT = 100  # bytes: more than any file these tests read
UNSAFE = ["../escape", "a/../../b", "/etc/passwd", "a//b", "./a", "a/.", "", "a\\b", "refs/.."]


def files_under(root):
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def test_dir_transport_writes_reads_and_lists(tmp_path):
    root = tmp_path / "missing" / "store"
    remote = DirTransport(root)
    assert repr(remote) == f"DirTransport({str(root)!r})"
    assert not remote.exists("refs/heads/main")
    assert remote.list() == []
    remote.write("payloads/ab/abc", b"one")  # creates the root and the subdirectories
    remote.write_many([("objects/cd/cd.json", b"{}"), ("refs/heads/main", b"old\n")])
    remote.write("refs/heads/main", b"new\n")  # an existing file is replaced
    assert remote.read("payloads/ab/abc", limit=LIMIT) == b"one"
    assert remote.read("refs/heads/main", limit=LIMIT) == b"new\n"
    assert remote.exists("objects/cd/cd.json")
    assert not remote.exists("objects/cd")
    assert remote.list() == ["objects/cd/cd.json", "payloads/ab/abc", "refs/heads/main"]
    assert remote.list("refs/") == remote.list("refs") == ["refs/heads/main"]
    assert remote.list("payloads/a") == ["payloads/ab/abc"]
    assert remote.list("commits/") == []
    assert files_under(root) == ["objects/cd/cd.json", "payloads/ab/abc", "refs/heads/main"]
    with pytest.raises(StoreError, match=re.escape("lacks commits/none.json")):
        remote.read("commits/none.json", limit=LIMIT)


def test_dir_transport_list_skips_hidden_and_temporary_files(tmp_path):
    remote = DirTransport(tmp_path)
    remote.write("commits/a.json", b"{}")
    (tmp_path / "commits" / ".a.json.0123.tmp").write_bytes(b"partial")
    (tmp_path / "commits" / ".DS_Store").write_bytes(b"")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_bytes(b"")
    assert remote.list() == ["commits/a.json"]


def test_dir_transport_write_is_atomic(tmp_path, monkeypatch):
    remote = DirTransport(tmp_path)
    remote.write("refs/heads/main", b"old\n")

    def fail(source, target):
        raise OSError("disk full")

    monkeypatch.setattr("wvc.fsio.os.replace", fail)
    with pytest.raises(OSError, match="disk full"):
        remote.write("refs/heads/main", b"new\n")
    assert remote.read("refs/heads/main", limit=LIMIT) == b"old\n"
    assert files_under(tmp_path) == ["refs/heads/main"]  # the temporary file was removed


def test_dir_transport_reads_within_the_bound_it_is_given(tmp_path):
    remote = DirTransport(tmp_path)
    remote.write("payloads/ab/ab", b"0123456789")
    assert remote.read("payloads/ab/ab", limit=10) == b"0123456789"
    assert remote.read("payloads/ab/ab", limit=0, size=10) == b"0123456789"  # size decides
    with pytest.raises(IntegrityError, match="holds 10 bytes, more than the 9 allowed"):
        remote.read("payloads/ab/ab", limit=9)
    with pytest.raises(IntegrityError, match="holds 10 bytes, not the 11 its record implies"):
        remote.read("payloads/ab/ab", limit=LIMIT, size=11)


def test_dir_transport_temporaries_are_the_stores_own(tmp_path, monkeypatch):
    """C-M5: DirTransport writes through write_atomic, so a temporary that a killed push
    leaves in a remote repository's refs/heads is skipped by fsck and swept when stale."""
    store = ObjectStore(init_repository(tmp_path / "repo"))
    remote = DirTransport(store.dir)
    renamed = []
    real_replace = os.replace

    def spy(source, target):
        renamed.append((source, target))
        real_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr("wvc.fsio.os.replace", spy)
        remote.write("refs/heads/main", b"0" * 64 + b"\n")
    [(temporary, target)] = renamed
    heads = store.dir / "refs" / "heads"
    assert os.path.dirname(temporary) == os.path.dirname(target) == str(heads)
    assert TEMP_NAME.fullmatch(os.path.basename(temporary)) is not None
    assert sorted(os.listdir(heads)) == ["main"]  # nothing left beside the ref
    os.unlink(heads / "main")  # fsck then sees only the orphan below in refs/heads
    with open(temporary, "wb") as handle:  # what a push killed before its rename leaves
        handle.write(b"partial")
    assert remote.list("refs/") == []
    assert fsck(store, decode=False).ok
    os.utime(temporary, (0, 0))
    assert remove_stale_temporaries(heads) == 1
    assert os.listdir(heads) == []


@pytest.mark.parametrize("relpath", UNSAFE)
def test_dir_transport_refuses_absolute_and_parent_paths(tmp_path, relpath):
    remote = DirTransport(tmp_path / "store")
    calls = (
        remote.exists,
        lambda path: remote.read(path, limit=LIMIT),
        lambda path: remote.write(path, b"x"),
    )
    for call in calls:
        with pytest.raises(StoreError, match="remote path"):
            call(relpath)
    assert not (tmp_path / "store").exists()
    assert not (tmp_path / "escape").exists()
    with pytest.raises(StoreError):
        check_relpath(relpath)


@pytest.mark.parametrize("prefix", ["../", "/abs", "a//", 3])
def test_listing_prefixes_are_checked(tmp_path, prefix):
    with pytest.raises(StoreError):
        DirTransport(tmp_path).list(prefix)


def test_the_default_write_many_writes_in_order():
    class Log(Transport):
        def __init__(self) -> None:
            self.written = []

        def exists(self, relpath):
            return False

        def read(self, relpath):
            raise AssertionError

        def write(self, relpath, data):
            self.written.append((relpath, data))

        def list(self, prefix=""):
            return []

    log = Log()
    log.write_many([("b", b"2"), ("a", b"1")])
    assert log.written == [("b", b"2"), ("a", b"1")]


def test_hf_transport_keeps_the_store_under_wvc(fake_hub):
    fake_hub.files["wvc/refs/heads/main"] = b"old\n"
    remote = HfTransport("owner/name", "main")
    assert remote.url == "hf://owner/name@main"
    assert repr(remote) == "HfTransport('owner/name', 'main')"
    assert remote.exists("refs/heads/main")
    assert not remote.exists("commits/a.json")
    assert remote.list() == ["refs/heads/main"]  # files outside wvc/ are not the store's
    assert remote.read("refs/heads/main", limit=LIMIT) == b"old\n"
    remote.write_many([("payloads/ab/ab", b"\x00\x01"), ("objects/ab/ab.json", b"{}")])
    remote.write("refs/heads/main", b"new\n")
    remote.write_many([])  # nothing to send, so no Hub commit
    assert fake_hub.commits == [
        ["wvc/payloads/ab/ab", "wvc/objects/ab/ab.json"],
        ["wvc/refs/heads/main"],
    ]
    assert fake_hub.files["wvc/payloads/ab/ab"] == b"\x00\x01"
    # One listing per transport instance, kept current with the instance's own writes.
    assert remote.exists("payloads/ab/ab")
    assert remote.list("objects/") == ["objects/ab/ab.json"]
    assert remote.read("refs/heads/main", limit=LIMIT) == b"new\n"
    assert fake_hub.listings == 1
    assert HfTransport("owner/name").list("payloads") == ["payloads/ab/ab"]
    assert fake_hub.listings == 2
    targets = {(call["repo_id"], call["revision"], call["repo_type"]) for _, call in fake_hub.calls}
    assert targets == {("owner/name", "main", "model")}
    messages = [call["commit_message"] for name, call in fake_hub.calls if name == "create_commit"]
    assert messages == ["wvc: write 2 files", "wvc: write wvc/refs/heads/main"]


def test_hf_transport_refuses_missing_files_and_unsafe_paths(fake_hub):
    remote = HfTransport("owner/name", "dev")
    with pytest.raises(StoreError, match=re.escape("hf://owner/name@dev lacks wvc/commits/a.json")):
        remote.read("commits/a.json", limit=LIMIT)
    for relpath in UNSAFE:
        with pytest.raises(StoreError, match="remote path"):
            remote.write(relpath, b"x")
    with pytest.raises(StoreError, match="remote path"):
        remote.exists("../README.md")
    assert fake_hub.commits == []


def test_hf_transport_checks_the_cached_files_size_before_reading_it(fake_hub, monkeypatch):
    fake_hub.files["wvc/payloads/ab/ab"] = b"0123456789"
    remote = HfTransport("owner/name")
    assert remote.read("payloads/ab/ab", limit=0, size=10) == b"0123456789"
    with pytest.raises(IntegrityError, match="holds 10 bytes, more than the 9 allowed"):
        remote.read("payloads/ab/ab", limit=9)
    with pytest.raises(IntegrityError, match="holds 10 bytes, not the 8 its record implies"):
        remote.read("payloads/ab/ab", limit=LIMIT, size=8)
    real_fstat = os.fstat

    def stale(fd):  # the file grows after its fstat: at most size + 1 bytes are read
        info = real_fstat(fd)
        return os.stat_result((*info[:6], 8, *info[7:10]))

    monkeypatch.setattr("wvc.transport.os.fstat", stale)
    with pytest.raises(IntegrityError, match="changed size while it was being read"):
        remote.read("payloads/ab/ab", limit=LIMIT, size=8)


def test_hub_client_errors_become_store_errors(fake_hub, monkeypatch):
    def unavailable(**kwargs):
        raise FakeHubError("503 Service Unavailable")

    monkeypatch.setattr(fake_hub, "create_commit", unavailable)
    remote = HfTransport("owner/name")
    with pytest.raises(StoreError, match=re.escape("hf://owner/name@main: 503")):
        remote.write("refs/heads/main", b"x\n")
    monkeypatch.setattr(fake_hub, "list_repo_files", unavailable)
    with pytest.raises(StoreError, match="503"):
        HfTransport("owner/name").exists("refs/heads/main")


@pytest.mark.parametrize(
    ("repo_id", "revision"),
    [
        ("gpt2", "main"),
        ("a/b/c", "main"),
        ("owner/name", ""),
        ("owner/name", "a b"),
        ("a--b/c", "x"),
    ],
)
def test_hf_transport_checks_the_repository_before_building_a_client(repo_id, revision):
    # The autouse guard fails the test if a client is built, so validation comes first.
    with pytest.raises(StoreError, match="invalid Hub repository"):
        HfTransport(repo_id, revision)


def test_a_missing_huggingface_hub_names_the_wvc_hf_extra(monkeypatch):
    monkeypatch.setattr("wvc.transport.make_hub_client", REAL_MAKE_HUB_CLIENT)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # the import now fails
    for build in (lambda: HfTransport("owner/name"), lambda: transport_for("hf://owner/name@v1")):
        with pytest.raises(StoreError, match=re.escape("install wvc[hf]")):
            build()


def test_transport_for_parses_remote_urls(tmp_path, fake_hub):
    hub = transport_for("hf://owner/name@refs/pr/1")
    assert isinstance(hub, HfTransport)
    assert (hub.repo_id, hub.revision) == (
        "owner/name",
        "refs/pr/1",
    )
    assert transport_for("hf://owner/name").revision == "main"
    local = transport_for(str(tmp_path / "remote"))
    assert isinstance(local, DirTransport)
    assert local.root == tmp_path / "remote"
    for url in ("hf://gpt2", "hf://owner/name@", "s3://bucket/key", "file:///tmp/x", "", " "):
        with pytest.raises(StoreError):
            transport_for(url)


def test_hf_transport_calls_bind_to_the_installed_huggingface_hub(fake_hub, monkeypatch):
    """Offline check of HfTransport's calls against the real client's signatures."""
    hub = pytest.importorskip("huggingface_hub")
    real = REAL_MAKE_HUB_CLIENT()  # builds an HfApi, which makes no request
    assert isinstance(real.api, hub.HfApi)
    assert real.operation_add is hub.CommitOperationAdd
    errors = hub.errors
    for error in (
        errors.RepositoryNotFoundError,
        errors.RevisionNotFoundError,
        errors.HfHubHTTPError,
    ):
        assert issubclass(error, real.errors)
    client = HubClient(
        api=fake_hub,
        operation_add=hub.CommitOperationAdd,
        errors=real.errors,
        read_range=fake_hub.read_range,
    )
    monkeypatch.setattr("wvc.transport.make_hub_client", lambda: client)
    remote = HfTransport("owner/name")
    remote.write_many([("payloads/ab/ab", b"\x00" * 600), ("refs/heads/main", b"x\n")])
    assert remote.read("refs/heads/main", limit=LIMIT) == b"x\n"
    assert remote.list() == [
        "payloads/ab/ab",
        "refs/heads/main",
    ]
    assert {name for name, _ in fake_hub.calls} == {
        "list_repo_files",
        "hf_hub_download",
        "create_commit",
    }
    for name, kwargs in fake_hub.calls:
        inspect.signature(getattr(real.api, name)).bind(**kwargs)


# The real Hub range reader against a loopback HTTP server (no network).

BODY = bytes(range(256)) * 64  # 16 KiB served by the loopback server
ENDLESS_CAP = 256 * 1024 * 1024  # bytes: a safety stop should a client never hang up
TRANSPORT_READ = 64 * 1024  # bytes: the most httpcore takes from a socket in one read


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves ``BODY`` at ``/ok``; the other paths misbehave as their names say."""

    def log_message(self, *args):  # keep the test output quiet
        pass

    def do_GET(self):
        self.server.requests.append((self.path, dict(self.headers)))
        first, last = (int(n) for n in self.headers["range"].removeprefix("bytes=").split("-"))
        part = BODY[first : last + 1]
        if self.path == "/ok":
            self._head(206, len(part))
            self.wfile.write(part)
        elif self.path == "/ignores-range":  # a 200 with an endless body
            self._head(200, None)
            self._endless()
        elif self.path == "/declares-more":
            self._head(206, len(part) + 100)
            self.wfile.write(part + b"\0" * 100)
        elif self.path == "/sends-more":  # a 206 without Content-Length and an endless body
            self._head(206, None)
            self._endless()

    def _head(self, status, length):
        self.send_response(status)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.end_headers()

    def _endless(self):
        chunk = b"\xff" * 65536
        try:
            while self.server.sent < ENDLESS_CAP:
                self.wfile.write(chunk)
                self.server.sent += len(chunk)
        except OSError:  # the client hung up, as it should
            self.server.hung_up.set()


@pytest.fixture
def loopback():
    """A loopback HTTP/1.0 server in a thread; skipped where no socket can be bound."""
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    except OSError as exc:
        pytest.skip(f"cannot bind a loopback socket: {exc}")
    server.requests, server.sent, server.hung_up = [], 0, threading.Event()
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)


@pytest.fixture
def http_client():
    """A client of the HTTP library the installed huggingface_hub uses, counting body bytes.

    ``counting.received`` is the number of response body bytes the reader pulled from the
    transport; ``counting.timeouts`` holds each request's timeout settings.
    """
    utils = pytest.importorskip("huggingface_hub.utils")
    library = importlib.import_module(type(utils.get_session()).__module__.partition(".")[0])

    class Counted(library.SyncByteStream):
        def __init__(self, stream, owner) -> None:
            self.stream, self.owner = stream, owner

        def __iter__(self) -> Iterator[bytes]:
            for chunk in self.stream:
                self.owner.received += len(chunk)
                yield chunk

        def close(self):
            self.stream.close()

    class Counting(library.BaseTransport):
        def __init__(self) -> None:
            self.inner, self.received, self.timeouts = library.HTTPTransport(), 0, []

        def handle_request(self, request):
            self.timeouts.append(request.extensions.get("timeout"))
            response = self.inner.handle_request(request)
            return library.Response(
                response.status_code,
                headers=response.headers,
                stream=Counted(response.stream, self),
                extensions=response.extensions,
            )

        def close(self):
            self.inner.close()

    counting = Counting()
    with library.Client(transport=counting, trust_env=False) as client:
        client.counting = counting
        yield client


def test_a_satisfied_range_is_returned_exactly(loopback, http_client):
    data = fetch_range(http_client, f"{loopback.url}/ok", {"x-test": "1"}, 8, 300, "f")
    assert data == BODY[8:308]
    ((path, headers),) = loopback.requests
    assert path == "/ok"
    assert headers["range"] == "bytes=8-307"
    assert headers["x-test"] == "1"
    assert headers["accept-encoding"] == "identity"
    assert http_client.counting.received == 300
    expected = dict.fromkeys(("connect", "read", "write", "pool"), RANGE_TIMEOUT_SECONDS)
    assert http_client.counting.timeouts == [expected]


def test_a_server_that_ignores_the_range_is_refused_unread(loopback, http_client):
    with pytest.raises(StoreError, match=r"f: status 200 \(the server ignored the range\)"):
        fetch_range(http_client, f"{loopback.url}/ignores-range", {}, 8, 300, "f")
    assert http_client.counting.received == 0  # the reader pulled no body byte
    assert loopback.hung_up.wait(timeout=10)  # the endless body stopped: the client hung up
    assert loopback.sent < 64 * TRANSPORT_READ  # what the socket buffers took, at most


def test_a_zero_length_range_makes_no_request(loopback, http_client, monkeypatch):
    assert fetch_range(http_client, f"{loopback.url}/ok", {}, 8, 0, "f") == b""
    hub = types.SimpleNamespace(hf_hub_url=lambda *args, **kwargs: f"{loopback.url}/ok")
    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: http_client)
    monkeypatch.setattr("huggingface_hub.utils.build_hf_headers", lambda: {})
    read_range = _range_reader(hub)
    assert read_range("owner/name", "f.safetensors", "a" * 40, 8, 0) == b""
    assert loopback.requests == []  # the handler was never called
    assert read_range("owner/name", "f.safetensors", "a" * 40, 0, 8) == BODY[:8]
    assert [headers["range"] for _, headers in loopback.requests] == ["bytes=0-7"]


def test_a_declared_length_other_than_the_range_is_refused_unread(loopback, http_client):
    with pytest.raises(StoreError, match="Content-Length '400'; expected 206 with 300 bytes"):
        fetch_range(http_client, f"{loopback.url}/declares-more", {}, 8, 300, "f")
    assert http_client.counting.received == 0


def test_a_body_longer_than_the_range_is_refused_within_one_chunk(loopback, http_client):
    with pytest.raises(StoreError, match="more than 300 bytes arrived"):
        fetch_range(http_client, f"{loopback.url}/sends-more", {}, 8, 300, "f")
    assert 300 < http_client.counting.received <= 300 + TRANSPORT_READ  # one read too many
    assert loopback.hung_up.wait(timeout=10)
