"""#913 — remote ``scheme://`` dataset sources load instead of mangling to a local path.

Found live (#888 E2E): ``iter_rows("http://…/data.jsonl")`` passed the URL to
``load_dataset`` as a PATH → ``/app/http:/…`` FileNotFoundError → every
readiness/prep op errored on exactly the dataset forms `FinetuneRequest` (#571)
REQUIRES with a ``compute_target``. Remote URLs must route through
``data_files=`` with an explicit builder. Hermetic: a stdlib http.server on
localhost serves a tmp JSONL — no network beyond the loopback.
"""
from __future__ import annotations

import http.server
import json
import threading

import pytest

from operator_dataset_checks._io import (
    _remote_builder,
    iter_rows,
    load_rows,
    scan_frame,
)

ROWS = [
    {"instruction": "q1", "output": "a1"},
    {"instruction": "q2", "output": "a2"},
    {"instruction": "q3", "output": "a3"},
]


@pytest.fixture()
def http_jsonl(tmp_path):
    """Serve a tmp JSONL over a localhost http.server; yield its URL."""
    f = tmp_path / "data.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in ROWS), encoding="utf-8")
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *a, directory=str(tmp_path), **kw
    )
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/data.jsonl"
    finally:
        srv.shutdown()
        t.join(timeout=5)


def test_remote_builder_classification():
    assert _remote_builder("http://h/x.jsonl") == "json"
    assert _remote_builder("https://h/x.csv") == "csv"
    assert _remote_builder("s3://bucket/x.parquet") == "parquet"
    assert _remote_builder("s3://bucket/x.unknown") == "json"  # jsonl default
    # NOT remote: local paths, hub ids, and file:// (normalized before this)
    assert _remote_builder("/tmp/x.jsonl") is None
    assert _remote_builder("org/name") is None


def test_iter_rows_streams_a_http_url(http_jsonl):
    assert list(iter_rows(http_jsonl)) == ROWS


def test_load_rows_reads_a_http_url(http_jsonl):
    assert load_rows(http_jsonl) == ROWS


def test_scan_frame_returns_none_for_remote(http_jsonl):
    # the columnar fast path is local-only; remote streams via iter_rows
    assert scan_frame(http_jsonl) is None
