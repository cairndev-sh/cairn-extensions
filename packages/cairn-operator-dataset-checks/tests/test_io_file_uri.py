"""``file://`` URI handling for the dataset loader (issue #584).

``FinetuneRequest`` (#571) accepts ``scheme://`` dataset paths and an ADR-0066
dataset pack can emit a ``file://`` ``dataset_uri``. The server-side loader must
resolve a ``file://`` URI to its local path — before the fix it fell through the
``Path(source).exists()`` check, was mis-read as an HF Hub id, and errored on a
mangled ``<cwd>/file:/…`` path. Non-``file`` schemes / Hub ids must pass through
unchanged.
"""
from __future__ import annotations

import json

import operator_dataset_checks._io as io_mod

SAMPLE = [
    {"instruction": "Say hello politely.", "output": "Hello! How can I help?"},
    {"instruction": "A synonym for happy.", "output": "Joyful."},
]


def _write_jsonl(tmp_path, rows, name="d.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_normalize_source_resolves_file_uri():
    assert io_mod._normalize_source("file:///tmp/x.jsonl") == "/tmp/x.jsonl"
    # host form + percent-escape both resolve to the plain local path
    assert io_mod._normalize_source("file://localhost/tmp/x.jsonl") == "/tmp/x.jsonl"
    assert io_mod._normalize_source("file:///tmp/a%20b.jsonl") == "/tmp/a b.jsonl"


def test_normalize_source_passes_other_schemes_through():
    for s in ("s3://bucket/train.jsonl", "https://h/d.jsonl", "org/name", "/abs/local.jsonl"):
        assert io_mod._normalize_source(s) == s


def test_load_rows_accepts_file_uri(tmp_path):
    p = _write_jsonl(tmp_path, SAMPLE)
    rows = io_mod.load_rows(f"file://{p}")  # p is absolute → file:///abs/d.jsonl
    assert len(rows) == 2
    assert rows[0]["instruction"].startswith("Say hello")


def test_iter_rows_accepts_file_uri(tmp_path):
    p = _write_jsonl(tmp_path, SAMPLE)
    rows = list(io_mod.iter_rows(f"file://{p}"))
    assert len(rows) == 2
    assert rows[1]["output"] == "Joyful."


def test_file_uri_matches_raw_path(tmp_path):
    p = _write_jsonl(tmp_path, SAMPLE)
    assert io_mod.load_rows(str(p)) == io_mod.load_rows(f"file://{p}")
