"""dedup.exact backend=duckdb — the columnar path (e2e-fullstack pack run).

Explicit opt-in: the default stays the proven python body, so these pin
(1) the duckdb path's counts + first-occurrence-wins + stat evidence,
(2) that the default is untouched, (3) loud failure when opted in without
the package.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from operator_dataset_checks.dataflow_stages import dedup_exact

duckdb = pytest.importorskip("duckdb")


def _write(workdir: Path, rows: list[dict]) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "current.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _read(workdir: Path) -> list[dict]:
    return [json.loads(x) for x in
            (workdir / "current.jsonl").read_text().splitlines()]


def test_duckdb_drops_exact_dupes_first_wins(tmp_path):
    rows = [
        {"messages": [{"role": "user", "content": "a"}], "tag": 1},
        {"messages": [{"role": "user", "content": "b"}], "tag": 2},
        {"messages": [{"role": "user", "content": "a"}], "tag": 1},   # dup of 0
        {"messages": [{"role": "user", "content": "c"}], "tag": 3},
        {"messages": [{"role": "user", "content": "a"}], "tag": 1},   # dup of 0
    ]
    _write(tmp_path, rows)
    stats = dedup_exact(tmp_path, {"backend": "duckdb"}, [])
    assert stats["backend"] == "duckdb"
    assert stats["rows"] == 3 and stats["dropped"] == 2 == stats["dropped_exact"]
    kept = _read(tmp_path)
    assert [r["tag"] for r in kept] == [1, 2, 3]          # first occurrence, file order


def test_duckdb_normalize_folds_whitespace_and_case(tmp_path):
    _write(tmp_path, [
        {"text": "Hello  World"},
        {"text": "hello world"},          # same after normalize
        {"text": "hello, world"},         # different
    ])
    stats = dedup_exact(tmp_path, {"backend": "duckdb", "normalize": True}, [])
    assert stats["rows"] == 2
    strict = tmp_path / "strict"
    _write(strict, [{"text": "Hello  World"}, {"text": "hello world"}])
    stats = dedup_exact(strict, {"backend": "duckdb", "normalize": False}, [])
    assert stats["rows"] == 2             # byte-different rows both survive


def test_duckdb_streams_survivors_across_many_batches(tmp_path, monkeypatch):
    """#1382 — survivors leave DuckDB via bounded fetchmany batches, never one
    fetchall(). Pin correctness ACROSS batch boundaries: shrink the batch to 7
    so 100 unique rows (+dupes interleaved) span 15 partial batches, and both
    the counts and first-occurrence file order must still hold."""
    from operator_dataset_checks import dataflow_stages

    monkeypatch.setattr(dataflow_stages, "_DEDUP_FETCH_ROWS", 7)
    rows = []
    for i in range(100):
        rows.append({"text": f"row {i}", "tag": i})
        if i % 3 == 0:
            rows.append({"text": f"row {i}", "tag": i})  # 34 interleaved dupes
    _write(tmp_path, rows)
    stats = dedup_exact(tmp_path, {"backend": "duckdb"}, [])
    assert stats["rows"] == 100 and stats["dropped"] == 34
    kept = _read(tmp_path)
    assert [r["tag"] for r in kept] == list(range(100))   # order survives batching


def test_default_backend_is_still_python(tmp_path):
    _write(tmp_path, [{"text": "a"}, {"text": "a"}, {"text": "b"}])
    stats = dedup_exact(tmp_path, {}, [])
    assert stats["backend"] == "python"
    assert stats["rows"] == 2


def test_opted_in_without_duckdb_fails_loud(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_duckdb(name, *a, **k):
        if name == "duckdb":
            raise ImportError("no duckdb")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_duckdb)
    _write(tmp_path, [{"text": "a"}])
    with pytest.raises(RuntimeError, match="needs the duckdb package"):
        dedup_exact(tmp_path, {"backend": "duckdb"}, [])
