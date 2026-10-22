"""#964 — batched PII scan/anonymize: chunk-batch output must equal per-string,
including across chunk boundaries. Uses the regex backend (hermetic; the presidio
batch path is the same code shape, exercised live)."""

from __future__ import annotations

import json

import pytest

from cairn.core.operator import OperationContext
from operator_dataset_checks import _pii
from operator_dataset_checks.operators.anonymize import AnonymizeOp
from operator_dataset_checks.operators.pii_scan import PiiScanOp
from operator_dataset_checks.schemas.anonymize import AnonymizeInput
from operator_dataset_checks.schemas.pii_scan import PiiInput

TEXTS = [
    "email me at alice@example.com",
    "nothing sensitive here",
    "key sk-ABCDEFGHIJKLMNOPQRSTUV and ip 10.0.0.1",
    "plain",
    "bob@corp.io plus AKIA0123456789ABCDEF",
]


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="batch")


def test_scan_many_matches_per_string_regex():
    assert _pii.scan_many(TEXTS, backend="regex") == [
        _pii.scan(t, backend="regex") for t in TEXTS
    ]


def test_anonymize_many_matches_per_string_regex():
    assert _pii.anonymize_many(TEXTS, backend="regex") == [
        _pii.anonymize(t, backend="regex") for t in TEXTS
    ]


@pytest.mark.asyncio
async def test_anonymize_op_batched_output_crosses_chunk_boundary(tmp_path, monkeypatch):
    # force a tiny chunk so 5 rows span multiple batches — proves the zip-back
    # (iter_strings / map_strings order) is correct across chunk edges.
    monkeypatch.setattr(_pii, "_PRESIDIO_BATCH_SIZE", 2)
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps({"text": t}) for t in TEXTS))
    out = tmp_path / "out.jsonl"
    res = await AnonymizeOp().run(
        AnonymizeInput(path=str(src), output_path=str(out), backend="regex"), _ctx()
    )
    # every row preserved, emails/keys/ip redacted
    lines = out.read_text().splitlines()
    assert len(lines) == len(TEXTS) == res.total
    joined = "\n".join(lines)
    assert "alice@example.com" not in joined and "bob@corp.io" not in joined
    assert "<EMAIL>" in joined and "<OPENAI_KEY>" in joined
    assert res.rows_redacted == 3  # rows 0, 2, 4 carry PII


@pytest.mark.asyncio
async def test_pii_scan_op_batched_counts_and_findings(tmp_path, monkeypatch):
    monkeypatch.setattr(_pii, "_PRESIDIO_BATCH_SIZE", 2)
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps({"text": t}) for t in TEXTS))
    res = await PiiScanOp().run(
        PiiInput(path=str(src), backend="regex", sample_size=None), _ctx()
    )
    assert res.total == len(TEXTS)
    assert res.rows_with_pii == 3
    # findings carry the ORIGINAL row indices (0,2,4) even across chunk edges
    assert sorted(f.row for f in res.findings) == [0, 2, 4]
    assert res.counts.get("email") == 2
