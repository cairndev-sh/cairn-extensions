"""Tests for opt-in reservoir sampling on the statistical checks (issue #656).

`dataset/format`, `dataset/pii_scan` and `dataset/profile` accept an optional
`sample_size`: when set (and the dataset is larger), they reservoir-sample down
to that many rows before checking, so a go/no-go readiness gate doesn't have to
scan 10M rows. The sample MUST be deterministic (fixed-seed reservoir) so runs
are replayable and the gate's verdict is reproducible. The default (no
`sample_size`) path is covered by the pre-existing golden tests, which must stay
green unchanged.
"""
from __future__ import annotations

import json

from cairn.core.operator import OperationContext
from operator_dataset_checks._io import _reservoir_sample
from operator_dataset_checks.operators.format import FormatOp
from operator_dataset_checks.operators.pii_scan import PiiScanOp
from operator_dataset_checks.operators.profile import ProfileOp
from operator_dataset_checks.schemas.format import FormatInput
from operator_dataset_checks.schemas.pii_scan import PiiInput
from operator_dataset_checks.schemas.profile import ProfileInput


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="sampling")


def _write(tmp_path, rows) -> str:
    p = tmp_path / "ds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def _chat(user: str, asst: str) -> dict:
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": asst},
    ]}


# ── the shared helper: deterministic, uniform, correctly bounded ──────────────
def test_reservoir_sample_is_deterministic():
    rows = [{"i": i} for i in range(1000)]
    a = _reservoir_sample(rows, 50)
    b = _reservoir_sample(rows, 50)
    assert a == b  # same fixed seed → byte-for-byte identical selection
    assert len(a) == 50
    # Every sampled row came from the source (a genuine subset, no fabrication).
    picked = {r["i"] for r in a}
    assert picked <= set(range(1000))
    assert len(picked) == 50  # no duplicates


def test_reservoir_sample_returns_all_when_k_ge_len():
    rows = [{"i": i} for i in range(10)]
    assert _reservoir_sample(rows, 10) == rows
    assert _reservoir_sample(rows, 99) == rows
    # A copy, not the same list object (callers may reassign `rows`).
    assert _reservoir_sample(rows, 99) is not rows


# ── dataset/format ────────────────────────────────────────────────────────────
async def test_format_no_sampling_when_size_none(tmp_path):
    rows = [_chat(f"q{i}", f"a reply number {i} here") for i in range(40)]
    out = await FormatOp().run(FormatInput(path=_write(tmp_path, rows)), _ctx())
    assert out.sampled is False
    assert out.sample_size is None
    assert out.total == 40


async def test_format_no_sampling_when_size_ge_len(tmp_path):
    rows = [_chat(f"q{i}", f"a reply number {i} here") for i in range(40)]
    out = await FormatOp().run(
        FormatInput(path=_write(tmp_path, rows), sample_size=100), _ctx(),
    )
    assert out.sampled is False
    assert out.sample_size is None
    assert out.total == 40


async def test_format_samples_and_reports(tmp_path):
    rows = [_chat(f"q{i}", f"a reply number {i} here") for i in range(200)]
    path = _write(tmp_path, rows)
    out = await FormatOp().run(FormatInput(path=path, sample_size=25), _ctx())
    assert out.sampled is True
    assert out.sample_size == 25
    assert out.total == 25
    # Deterministic: a second run over the same input is identical.
    out2 = await FormatOp().run(FormatInput(path=path, sample_size=25), _ctx())
    assert out.model_dump() == out2.model_dump()


# ── dataset/pii_scan ──────────────────────────────────────────────────────────
async def test_pii_no_sampling_when_size_none(tmp_path):
    rows = [{"text": f"row {i} mail bob{i}@example.com"} for i in range(40)]
    out = await PiiScanOp().run(
        PiiInput(path=_write(tmp_path, rows), backend="regex"), _ctx(),
    )
    assert out.sampled is False
    assert out.sample_size is None
    assert out.total == 40


async def test_pii_samples_and_reports(tmp_path):
    rows = [{"text": f"row {i} mail bob{i}@example.com"} for i in range(200)]
    path = _write(tmp_path, rows)
    out = await PiiScanOp().run(
        PiiInput(path=path, backend="regex", sample_size=30), _ctx(),
    )
    assert out.sampled is True
    assert out.sample_size == 30
    assert out.total == 30
    out2 = await PiiScanOp().run(
        PiiInput(path=path, backend="regex", sample_size=30), _ctx(),
    )
    assert out.model_dump() == out2.model_dump()


# ── dataset/profile ───────────────────────────────────────────────────────────
async def test_profile_no_sampling_when_size_none(tmp_path):
    rows = [_chat(f"q{i}", f"a reply number {i} here") for i in range(40)]
    out = await ProfileOp().run(ProfileInput(path=_write(tmp_path, rows)), _ctx())
    assert out.sampled is False
    assert out.sample_size is None
    assert out.total == 40


async def test_profile_samples_and_reports(tmp_path):
    rows = [_chat(f"q{i}", f"a reply number {i} here") for i in range(200)]
    path = _write(tmp_path, rows)
    out = await ProfileOp().run(ProfileInput(path=path, sample_size=25), _ctx())
    assert out.sampled is True
    assert out.sample_size == 25
    assert out.total == 25
    out2 = await ProfileOp().run(ProfileInput(path=path, sample_size=25), _ctx())
    assert out.model_dump() == out2.model_dump()
