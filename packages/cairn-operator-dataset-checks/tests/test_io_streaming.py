"""Streaming-loader tests for ``_io.iter_rows`` / ``_io.scan_frame`` (issue #380).

``iter_rows`` must (a) yield the same rows as ``load_rows`` for a local JSONL
with nested ``messages`` preserved, (b) be lazy — a generator that reads on
demand, not a materialized list, and (c) drop null-valued keys like
``load_rows``. ``scan_frame`` returns a polars ``LazyFrame`` under the ``[fast]``
extra and ``None`` without it, so the suite stays hermetic when polars is absent.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Iterator

import pytest

import operator_dataset_checks._io as io_mod

SAMPLE = [
    {"messages": [{"role": "user", "content": "q1"},
                  {"role": "assistant", "content": "a1"}]},
    {"messages": [{"role": "user", "content": "q2"}], "meta": None},
    {"prompt": "p3", "completion": "c3"},
]


def _write_jsonl(tmp_path, rows, name="d.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_iter_rows_streams_jsonl_preserving_nested_and_dropping_nulls(tmp_path):
    p = _write_jsonl(tmp_path, SAMPLE)
    gen = io_mod.iter_rows(str(p))
    assert isinstance(gen, Iterator)  # lazy — not a materialized list
    rows = list(gen)
    assert len(rows) == 3
    # nested messages preserved verbatim
    assert rows[0]["messages"][1] == {"role": "assistant", "content": "a1"}
    # null-valued keys dropped (meta: None) — parity with load_rows
    assert "meta" not in rows[1]
    assert rows[2] == {"prompt": "p3", "completion": "c3"}


def test_iter_rows_is_lazy_reads_one_row_without_consuming_all(tmp_path):
    p = _write_jsonl(tmp_path, SAMPLE)
    gen = io_mod.iter_rows(str(p))
    first = next(gen)  # pulling one row must not require reading the whole file
    assert first["messages"][0]["content"] == "q1"


def test_iter_rows_skips_blank_lines(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert list(io_mod.iter_rows(str(p))) == [{"a": 1}, {"a": 2}]


def test_iter_rows_matches_load_rows_on_flat_jsonl(tmp_path):
    # load_rows goes through datasets; iter_rows streams via stdlib. For a flat,
    # homogeneous JSONL the row sets must match (content + order, nulls dropped).
    flat = [{"text": "one"}, {"text": "two"}, {"text": "three"}]
    p = _write_jsonl(tmp_path, flat, name="flat.jsonl")
    assert list(io_mod.iter_rows(str(p))) == io_mod.load_rows(str(p))


def test_iter_rows_routes_hub_id_to_streaming(monkeypatch):
    seen = {}

    def fake_load_dataset(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return iter([{"x": 1, "y": None}])

    monkeypatch.setattr(io_mod, "load_dataset", fake_load_dataset)
    rows = list(io_mod.iter_rows("org/name"))
    assert seen["args"][0] == "org/name"
    assert seen["kwargs"].get("streaming") is True  # streaming path, not materialized
    assert rows == [{"x": 1}]  # null dropped


def test_iter_rows_routes_local_csv_to_streaming(monkeypatch, tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b\n1,x\n", encoding="utf-8")
    seen = {}

    def fake_load_dataset(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return iter([{"a": 1, "b": "x"}])

    monkeypatch.setattr(io_mod, "load_dataset", fake_load_dataset)
    list(io_mod.iter_rows(str(p)))
    assert seen["args"][0] == "csv"  # suffix-inferred builder
    assert seen["kwargs"].get("streaming") is True
    assert seen["kwargs"].get("data_files") == str(p)


def test_scan_frame_none_without_polars(tmp_path, monkeypatch):
    # Simulate the base install (no [fast] extra): importing polars fails →
    # scan_frame returns None so callers fall back to iter_rows.
    monkeypatch.setitem(sys.modules, "polars", None)
    p = _write_jsonl(tmp_path, SAMPLE)
    assert io_mod.scan_frame(str(p)) is None


def test_scan_frame_none_for_hub_id():
    pytest.importorskip("polars")
    assert io_mod.scan_frame("org/name") is None  # Hub id — not lazily scannable


def test_scan_frame_returns_lazyframe_over_flat_jsonl(tmp_path):
    pl = pytest.importorskip("polars")
    p = tmp_path / "flat.jsonl"
    p.write_text('{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n', encoding="utf-8")
    lf = io_mod.scan_frame(str(p))
    assert lf is not None and isinstance(lf, pl.LazyFrame)  # lazy, not read yet
    assert lf.collect().to_dicts() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_scan_frame_reads_parquet(tmp_path):
    pl = pytest.importorskip("polars")
    p = tmp_path / "d.parquet"
    pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}).write_parquet(p)
    lf = io_mod.scan_frame(str(p))
    assert lf is not None
    assert lf.collect().to_dicts() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
