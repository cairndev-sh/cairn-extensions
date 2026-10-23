"""Malformed-row quarantine (#1008, ADR-0078 S2).

One bad line in a JSONL corpus must not kill the run: by default it is routed
to a ``<src>.quarantine.jsonl`` side file (line number + error + bounded raw
prefix), counted, and skipped. ``strict=True`` restores the old fail-fast.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from operator_dataset_checks._io import RowErrors, iter_rows, load_rows
from operator_dataset_checks.operators.profile import ProfileOp
from operator_dataset_checks.schemas.profile import ProfileInput


class _Ctx:
    pass


GOOD = [{"prompt": "p1", "output": "a perfectly fine response"},
        {"prompt": "p2", "output": "another good one"}]


def _mixed_file(tmp_path):
    """2 good rows + 3 bad lines (broken JSON, truncated object, non-object)."""
    src = tmp_path / "in.jsonl"
    src.write_text(
        "\n".join([
            json.dumps(GOOD[0]),
            '{"prompt": "broken',          # line 2: invalid JSON
            "",                             # blank — skipped, NOT quarantined
            json.dumps(GOOD[1]),
            '{"prompt": "p3", "output":',  # line 5: truncated
            "[1, 2, 3]",                   # line 6: valid JSON, not an object
        ]),
        encoding="utf-8",
    )
    return src


def test_iter_rows_quarantines_and_keeps_going(tmp_path):
    src = _mixed_file(tmp_path)
    errs = RowErrors()
    rows = list(iter_rows(str(src), errors=errs))
    assert rows == GOOD
    assert errs.count == 3
    q = tmp_path / "in.jsonl.quarantine.jsonl"
    assert errs.quarantine_path == str(q) and q.exists()
    entries = [json.loads(l) for l in q.read_text(encoding="utf-8").splitlines()]
    assert [e["line"] for e in entries] == [2, 5, 6]
    assert all("error" in e and "raw" in e for e in entries)


def test_load_rows_same_tolerance(tmp_path):
    src = _mixed_file(tmp_path)
    errs = RowErrors()
    assert load_rows(str(src), errors=errs) == GOOD
    assert errs.count == 3


def test_strict_restores_fail_fast(tmp_path):
    src = _mixed_file(tmp_path)
    with pytest.raises(json.JSONDecodeError):
        list(iter_rows(str(src), strict=True))
    # fail-fast wrote no quarantine file
    assert not (tmp_path / "in.jsonl.quarantine.jsonl").exists()


def test_clean_file_writes_no_quarantine(tmp_path):
    src = tmp_path / "clean.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in GOOD), encoding="utf-8")
    errs = RowErrors()
    assert list(iter_rows(str(src), errors=errs)) == GOOD
    assert errs.count == 0 and errs.quarantine_path is None
    assert not (tmp_path / "clean.jsonl.quarantine.jsonl").exists()


def test_raw_prefix_is_bounded(tmp_path):
    src = tmp_path / "big.jsonl"
    src.write_text('{"broken": "' + "x" * 5000, encoding="utf-8")
    errs = RowErrors()
    assert list(iter_rows(str(src), errors=errs)) == []
    q = json.loads((tmp_path / "big.jsonl.quarantine.jsonl").read_text().splitlines()[0])
    assert len(q["raw"]) <= 500


def test_profile_surfaces_quarantine_counts(tmp_path):
    src = _mixed_file(tmp_path)
    out = asyncio.run(ProfileOp().run(ProfileInput(path=str(src)), _Ctx()))
    assert out.total == 2
    assert out.quarantined_rows == 3
    assert out.quarantine_path == str(tmp_path / "in.jsonl.quarantine.jsonl")


def test_profile_strict_raises(tmp_path):
    src = _mixed_file(tmp_path)
    with pytest.raises(json.JSONDecodeError):
        asyncio.run(ProfileOp().run(ProfileInput(path=str(src), strict=True), _Ctx()))
