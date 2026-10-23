"""#2171 (C1) — ``ingest.normalize`` on the URI a registered version really carries.

The registry stores what the runner described: ``file://<abs>`` for a local
export (``cairn/core/dataflow/runner.py``), and the control plane hands the
kernel ``<uri>#sha256=<hash>``. Before the fix the verification step hashed
``Path("file:///…")`` — which pathlib collapses to ``file:/…`` — and the leg died
on a FileNotFoundError naming a path that had never existed. These pin the one
resolution rule for the five source forms and the verification on each.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import operator_dataset_checks.dataflow_stages as S
from operator_dataset_checks._io import _normalize_source

ROWS = [{"prompt": f"q{i}", "completion": f"a{i}"} for i in range(5)]


def _jsonl(tmp_path: Path) -> tuple[Path, str]:
    p = tmp_path / "golden.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


# ── the five source forms, one rule ─────────────────────────────────────────

@pytest.mark.parametrize("form, expected", [
    ("file:///abs/golden.jsonl", "/abs/golden.jsonl"),     # what the runner registers
    ("file:/abs/golden.jsonl", "/abs/golden.jsonl"),       # the pathlib-collapsed form
    ("/abs/golden.jsonl", "/abs/golden.jsonl"),            # a bare path
    ("s3://bucket/key/golden.jsonl", "s3://bucket/key/golden.jsonl"),   # fetched, not opened
    ("org/dataset-name", "org/dataset-name"),              # an HF Hub id
])
def test_the_five_source_forms_resolve_to_what_the_kernel_can_open(form, expected):
    assert _normalize_source(form) == expected


# ── verification on a registered file:// version ────────────────────────────

def test_a_registered_file_uri_with_its_hash_fragment_is_verified_and_read(tmp_path):
    p, h = _jsonl(tmp_path)
    wd = tmp_path / "wd"; wd.mkdir()
    out = S.ingest_normalize(wd, {"path": f"file://{p}#sha256={h}"}, [])
    assert out["rows"] == len(ROWS)
    assert out["source"] == f"file://{p}"          # lineage names what was READ, scheme kept
    assert out["source_hash"] == h                 # …and pins it by content (#1432)
    assert out["source_bytes"] == p.stat().st_size


def test_the_collapsed_single_slash_form_is_read_too(tmp_path):
    p, h = _jsonl(tmp_path)
    wd = tmp_path / "wd"; wd.mkdir()
    out = S.ingest_normalize(wd, {"path": f"file:{p}#sha256={h}"}, [])   # file:/abs/…
    assert out["rows"] == len(ROWS) and out["source_hash"] == h


def test_a_wrong_fragment_still_refuses_a_file_uri(tmp_path):
    """The fix must not weaken #1466: the NAME still has to guarantee the BYTES."""
    p, _ = _jsonl(tmp_path)
    wd = tmp_path / "wd"; wd.mkdir()
    with pytest.raises(RuntimeError, match="dataset verification FAILED"):
        S.ingest_normalize(wd, {"path": f"file://{p}#sha256={'0' * 64}"}, [])


def test_the_same_registered_version_reads_to_the_same_hash_twice(tmp_path):
    """C1 pin: a registered version re-read yields identical content evidence."""
    p, h = _jsonl(tmp_path)
    outs = []
    for i in range(2):
        wd = tmp_path / f"wd{i}"; wd.mkdir()
        outs.append(S.ingest_normalize(wd, {"path": f"file://{p}#sha256={h}"}, []))
    assert outs[0]["source_hash"] == outs[1]["source_hash"] == h
    assert outs[0]["rows"] == outs[1]["rows"]
