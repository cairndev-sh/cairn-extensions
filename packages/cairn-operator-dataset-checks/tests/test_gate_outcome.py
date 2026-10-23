"""ADR-0075 — dataset/gate publishes a normalized run-level `outcome` list so the
pilot report / Playground read one honest verdict off the standard channel.
Deterministic: it mirrors the verdict the gate computes."""
from __future__ import annotations

import asyncio

from operator_dataset_checks.operators.gate import GateOp
from operator_dataset_checks.schemas.gate import GateInput


def _run(inp: GateInput):
    return asyncio.run(GateOp().run(inp, None))  # type: ignore[arg-type]


def _named(outcome, name):
    return next((o for o in outcome if o["name"] == name), None)


def test_outcome_headline_matches_verdict_on_fail():
    # format below bar (BLOCKING) + non-secret PII (WARNING) → fail
    out = _run(GateInput(format_conformant_pct=66.7, pii_rows=1, pii_counts={"email": 1}))
    assert out.verdict == "fail"
    head = _named(out.outcome, "readiness")
    assert head == {"name": "readiness", "value": "fail", "pass": False, "target": "pass", "severity": "block"}
    # severity mirrors the gate's own block/warn split: format blocks, email warns
    assert _named(out.outcome, "format conformant")["severity"] == "block"
    assert _named(out.outcome, "PII rows")["severity"] == "warn"


def test_outcome_headline_passes_on_clean():
    out = _run(GateInput(format_conformant_pct=100.0, pii_rows=0, pii_counts={}, dedup_pct=0.0, prov_ok=True))
    assert out.verdict == "pass"
    assert _named(out.outcome, "readiness")["pass"] is True
    assert all(o["severity"] == "ok" for o in out.outcome)


def test_outcome_only_includes_checks_that_ran():
    # only format ran → outcome has the headline + format, nothing else
    out = _run(GateInput(format_conformant_pct=100.0))
    names = {o["name"] for o in out.outcome}
    assert names == {"readiness", "format conformant"}


def test_outcome_secret_pii_blocks():
    out = _run(GateInput(pii_rows=2, pii_counts={"aws_access_key": 1, "email": 1}))
    assert out.verdict == "fail"  # hard secret blocks
    assert _named(out.outcome, "readiness")["pass"] is False
    assert _named(out.outcome, "PII rows")["severity"] == "block"  # secret present


def test_outcome_warn_only_verdict():
    # non-secret PII only → warn (not fail); headline severity is 'warn'
    out = _run(GateInput(pii_rows=1, pii_counts={"email": 1}, format_conformant_pct=100.0))
    assert out.verdict == "warn"
    assert _named(out.outcome, "readiness")["severity"] == "warn"
    assert _named(out.outcome, "PII rows")["severity"] == "warn"
