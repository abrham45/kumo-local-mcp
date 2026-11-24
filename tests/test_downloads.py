import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kumo_local_mcp.core import Failure
from kumo_local_mcp.download import transfer_ranges


@pytest.fixture
def endpoint():
    payload = bytes(range(251)) * 35
    calls = []
    dropped = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.endswith("?expired=1"):
                self.send_error(403)
                return
            start, end = (int(v) for v in self.headers["Range"].removeprefix("bytes=").split("-"))
            calls.append((start, end))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            if self.path.endswith("?drop=1") and start == 0 and not dropped:
                dropped.append(True)
                self.wfile.write(payload[:73])
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(payload[start : end + 1])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/weights", payload, calls
    server.shutdown()
    server.server_close()
    thread.join()


def test_download_verifies_and_reuses_offline_cache(endpoint, tmp_path):
    url, payload, calls = endpoint
    destination = tmp_path / "model"
    digest = hashlib.sha256(payload).hexdigest()
    transfer_ranges(url, destination, len(payload), digest, workers=3, chunk_bytes=1000)
    assert destination.read_bytes() == payload
    count = len(calls)
    transfer_ranges("http://invalid.invalid", destination, len(payload), digest)
    assert len(calls) == count


def test_download_resumes_complete_parts_and_removes_corrupt_parts(endpoint, tmp_path):
    url, payload, calls = endpoint
    target = tmp_path / "model"
    parts = tmp_path / "model.parts"
    parts.mkdir()
    (parts / "0").write_bytes(payload[:1000])
    transfer_ranges(
        url, target, len(payload), hashlib.sha256(payload).hexdigest(), workers=2, chunk_bytes=1000
    )
    assert (0, 999) not in calls
    assert target.read_bytes() == payload
    target.unlink()
    parts.mkdir()
    (parts / "0").write_bytes(b"x" * 1000)
    with pytest.raises(Failure) as error:
        transfer_ranges(
            url,
            target,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
            workers=2,
            chunk_bytes=1000,
        )
    assert error.value.code == "MODEL_DIGEST_MISMATCH"
    assert not target.exists() and not list(parts.iterdir())


def test_invalid_model_digest_is_rejected_before_network(tmp_path):
    with pytest.raises(Failure) as error:
        transfer_ranges("http://invalid.invalid", tmp_path / "model", 100, "unverified")
    assert error.value.code == "MODEL_METADATA_INVALID"


def test_broken_response_resumes_from_last_received_byte(endpoint, tmp_path):
    url, payload, calls = endpoint
    target = tmp_path / "model"
    transfer_ranges(
        url + "?drop=1",
        target,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        workers=2,
        chunk_bytes=1000,
    )
    assert (73, 999) in calls
    assert target.read_bytes() == payload


def test_expired_url_refresh_keeps_pinned_content(endpoint, tmp_path):
    url, payload, calls = endpoint
    refreshed = []

    def renew():
        refreshed.append(True)
        return url

    target = tmp_path / "model"
    transfer_ranges(
        url + "?expired=1",
        target,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        workers=3,
        chunk_bytes=1000,
        refresh_url=renew,
    )
    assert len(refreshed) == 1
    assert target.read_bytes() == payload
