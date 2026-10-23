"""Atomic-write behavior of the JSONL writers (#1007, ADR-0078 S1).

A crash mid-write must never leave a truncated file at the real output path:
the path holds either its previous content or the complete new content. Success
output stays byte-compatible with ``"\\n".join(json.dumps(r) for r in rows)``.
"""
from __future__ import annotations

import json

import pytest

from operator_dataset_checks._io import JsonlWriter, ShardedJsonlWriter

ROWS = [{"text": f"row {i}"} for i in range(5)]


def _expected(rows: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)


def _no_tmp_leftovers(directory) -> bool:
    return not [p for p in directory.iterdir() if ".tmp-" in p.name]


# ── JsonlWriter ──────────────────────────────────────────────────────────────

def test_jsonl_writer_success_is_byte_compatible(tmp_path):
    out = tmp_path / "out.jsonl"
    with JsonlWriter(out) as w:
        for r in ROWS:
            w.write(r)
    assert out.read_text(encoding="utf-8") == _expected(ROWS)
    assert _no_tmp_leftovers(tmp_path)


def test_jsonl_writer_empty_stream_writes_empty_file(tmp_path):
    out = tmp_path / "empty.jsonl"
    with JsonlWriter(out):
        pass
    assert out.exists() and out.read_text(encoding="utf-8") == ""
    assert _no_tmp_leftovers(tmp_path)


def test_jsonl_writer_crash_leaves_no_partial(tmp_path):
    out = tmp_path / "out.jsonl"
    with pytest.raises(RuntimeError, match="boom"):
        with JsonlWriter(out) as w:
            w.write(ROWS[0])
            raise RuntimeError("boom")
    assert not out.exists()
    assert _no_tmp_leftovers(tmp_path)


def test_jsonl_writer_crash_preserves_previous_content(tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text("previous complete content", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with JsonlWriter(out) as w:
            w.write(ROWS[0])
            raise RuntimeError("boom")
    assert out.read_text(encoding="utf-8") == "previous complete content"
    assert _no_tmp_leftovers(tmp_path)


# ── ShardedJsonlWriter ───────────────────────────────────────────────────────

def test_sharded_writer_success_commits_all_shards(tmp_path):
    with ShardedJsonlWriter(tmp_path, "train", shard_size=2) as w:
        for r in ROWS:
            w.write(r)
    assert len(w.paths) == 3 and w.count == 5
    # each committed shard is complete and byte-compatible
    assert (tmp_path / "train.jsonl").read_text(encoding="utf-8") == _expected(ROWS[:2])
    assert (tmp_path / "train.00001.jsonl").read_text(encoding="utf-8") == _expected(ROWS[2:4])
    assert (tmp_path / "train.00002.jsonl").read_text(encoding="utf-8") == _expected(ROWS[4:])
    assert _no_tmp_leftovers(tmp_path)


def test_sharded_writer_crash_drops_only_inflight_shard(tmp_path):
    with pytest.raises(RuntimeError):
        with ShardedJsonlWriter(tmp_path, "train", shard_size=2) as w:
            for r in ROWS[:3]:  # shard 0 (2 rows) rolls + commits; shard 1 in flight
                w.write(r)
            raise RuntimeError("boom")
    # shard 0 was complete when it rolled → committed and intact
    assert (tmp_path / "train.jsonl").read_text(encoding="utf-8") == _expected(ROWS[:2])
    # the in-flight shard 1 never reached its final path, tmp cleaned up
    assert not (tmp_path / "train.00001.jsonl").exists()
    assert w.paths == [str(tmp_path / "train.jsonl")]
    assert _no_tmp_leftovers(tmp_path)


def test_sharded_writer_empty_stream_emits_empty_primary_shard(tmp_path):
    with ShardedJsonlWriter(tmp_path, "val") as w:
        pass
    assert w.paths == [str(tmp_path / "val.jsonl")]
    assert (tmp_path / "val.jsonl").read_text(encoding="utf-8") == ""
    assert _no_tmp_leftovers(tmp_path)
