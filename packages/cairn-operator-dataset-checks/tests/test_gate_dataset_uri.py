"""#579 / ADR-0066 — the dataset/gate operator EMITS a dataset_uri (the validated
dataset location, passed through) so a downstream step can train on it. The gate
stays a pure function of its inputs — dataset_uri is pass-through, no I/O."""
from __future__ import annotations

import asyncio

from cairn.core.operator.protocol import OperationContext
from operator_dataset_checks.operators.gate import GateOp
from operator_dataset_checks.schemas.gate import GateInput


def _run(**kw):
    return asyncio.run(GateOp().run(GateInput(**kw), OperationContext(pipeline_id="t", node_name="gate")))


def test_gate_passes_dataset_uri_through():
    out = _run(dataset_uri="s3://bucket/train.jsonl", format_conformant_pct=99.0)
    assert out.dataset_uri == "s3://bucket/train.jsonl"
    assert out.verdict in ("pass", "warn", "fail")


def test_gate_dataset_uri_defaults_empty():
    out = _run(format_conformant_pct=99.0)
    assert out.dataset_uri == ""


def test_gate_emits_uri_even_on_fail():
    # The consumer decides whether to use it (only when verdict != fail); the gate
    # still emits the ref so a bundle can surface "which dataset failed".
    out = _run(dataset_uri="hf://org/ds", format_conformant_pct=1.0)
    assert out.dataset_uri == "hf://org/ds"
