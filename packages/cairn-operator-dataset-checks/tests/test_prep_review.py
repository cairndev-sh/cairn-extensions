"""dataset/prep_review — the HITL sign-off body (#883).

The PAUSE is the approval gate's job (requires_approval, ADR-0034); this body runs
only on approve. Here we just pin its recorded sign-off.
"""
from __future__ import annotations

from cairn.core.operator import OperationContext
from operator_dataset_checks.operators.prep_review import PrepReviewOp
from operator_dataset_checks.schemas.prep_review import PrepReviewInput


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="prep-review")


async def test_records_signoff_with_evidence():
    out = await PrepReviewOp().run(
        PrepReviewInput(post_scrub_pii_rows=3, reasons=["post-scrub PII: 3 row(s) (3 email)"]),
        _ctx(),
    )
    assert out.reviewed is True
    assert out.residual_pii_rows == 3
    assert "3 residual PII row(s)" in out.note and "3 email" in out.note


async def test_mock_matches_run():
    inp = PrepReviewInput(post_scrub_pii_rows=1)
    assert (await PrepReviewOp().mock(inp, _ctx())).reviewed is True
