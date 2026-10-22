"""Tests for dataset/provenance sidecar reading — local path, file:// and the
remote-URL fix (#937): a metadata_path http(s) URL must be fetched, not
Path-collapsed into a bogus local file (``http:/…`` → FileNotFoundError)."""
from __future__ import annotations

import json

from operator_dataset_checks.operators.provenance import _read_sidecar


def test_read_sidecar_local_path(tmp_path):
    p = tmp_path / "meta.json"
    p.write_text(json.dumps({"license": "cc-by-4.0"}), encoding="utf-8")
    assert json.loads(_read_sidecar(str(p)))["license"] == "cc-by-4.0"


def test_read_sidecar_file_uri(tmp_path):
    p = tmp_path / "meta.json"
    p.write_text(json.dumps({"source": "synthetic"}), encoding="utf-8")
    assert json.loads(_read_sidecar(f"file://{p}"))["source"] == "synthetic"


def test_read_sidecar_http_url_not_path_collapsed(monkeypatch):
    seen: dict[str, str] = {}
    payload = json.dumps({"license": "mit"}).encode("utf-8")

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return payload

    def _fake_urlopen(url, timeout=30):  # noqa: ARG001
        seen["url"] = url
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    out = json.loads(_read_sidecar("http://172.18.0.1:8765/check-meta.json"))
    assert out["license"] == "mit"
    # the URL reached urllib verbatim — the `//` was NOT collapsed to `/`
    assert seen["url"] == "http://172.18.0.1:8765/check-meta.json"
