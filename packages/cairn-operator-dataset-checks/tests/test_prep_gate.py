"""dataset/prep_gate — the config-gated post-scrub PII gate (#882) + the #889
certify-not-just-count refinement.

Hermetic: pure function of (rows, sampled, backend, mode). "clean" (0 found) is
only a GUARANTEE (`verified`) when the scan was FULL and on a certifying NER
backend; a sampled/regex clean is WARN, never a clean PASS.
"""
from __future__ import annotations

from cairn.core.operator import OperationContext
from operator_dataset_checks.operators.prep_gate import PrepGateOp
from operator_dataset_checks.schemas.prep_gate import PrepGateInput


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="prep-gate")


async def _run(**kw):
    return await PrepGateOp().run(PrepGateInput(**kw), _ctx())


# ── certified clean = 0 PII, full scan, presidio NER ──────────────────────────


async def test_certified_clean_passes_in_every_mode():
    for mode in ("off", "block", "hitl"):
        out = await _run(post_scrub_pii_rows=0, post_scrub_pii_backend="presidio", mode=mode)
        assert out.clean is True and out.verified is True
        assert out.action == "pass" and out.verdict == "pass"
        assert out.outcome[0]["severity"] == "ok" and out.outcome[0]["verified"] is True


# ── #889 — 0 PII but the scan can't CERTIFY → WARN, never a clean PASS ─────────


async def test_sampled_clean_is_warn_not_certified():
    out = await _run(post_scrub_pii_rows=0, post_scrub_pii_backend="presidio",
                     post_scrub_pii_sampled=True, mode="block")
    assert out.verified is False and out.verdict == "warn" and out.clean is True
    assert out.outcome[0]["severity"] == "warn" and out.outcome[0]["name"].endswith("(unverified)")
    assert any("certify" in b.lower() for b in out.blocking)


async def test_regex_clean_is_warn_not_certified():
    out = await _run(post_scrub_pii_rows=0, post_scrub_pii_backend="regex", mode="block")
    assert out.verified is False and out.verdict == "warn"
    assert any("NER" in b for b in out.blocking)


async def test_uncertified_clean_in_hitl_routes_to_review():
    out = await _run(post_scrub_pii_rows=0, post_scrub_pii_backend="regex", mode="hitl")
    assert out.action == "review" and out.verdict == "warn" and out.verified is False


# ── dirty (rows > 0) still enforces per mode (unchanged from #882/#883) ────────


async def test_dirty_block_fails_closed():
    out = await _run(post_scrub_pii_rows=3, post_scrub_pii_counts={"email": 3},
                     post_scrub_pii_backend="presidio", mode="block")
    assert out.action == "block" and out.verdict == "fail" and out.verified is False
    assert out.outcome[0]["severity"] == "block"


async def test_dirty_hitl_routes_to_review():
    out = await _run(post_scrub_pii_rows=2, post_scrub_pii_backend="presidio", mode="hitl")
    assert out.action == "review" and out.verdict == "fail"


async def test_dirty_off_surfaces_only():
    out = await _run(post_scrub_pii_rows=3, post_scrub_pii_backend="presidio", mode="off")
    assert out.action == "pass" and out.verdict == "pass"  # off = surface via the report warning
