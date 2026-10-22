"""dataset/prep_review — the HITL review of a dirty scrub (#883).

Declares ``requires_approval`` (pack.toml, ADR-0034): the durable approval gate
fires at the chokepoint BEFORE this runs, so the prep run PAUSES
``awaiting_approval`` — a human reviews the residual-PII evidence and approves or
rejects. Reject is handled by the gate (``last_action_status=rejected``); this
body runs ONLY on approve, recording the sign-off. Reuses the same approval
machinery as ``model/register`` — no new mechanism.
"""
from __future__ import annotations

from cairn.core.operator import Operation, OperationContext, register_operator
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks.schemas.prep_review import (
    PrepReviewInput,
    PrepReviewOutput,
)


@register_operator(
    name="dataset/prep_review",
    version="0.1.0",
    capabilities=["dataset.prep_review"],
    implements=["contract://dataset.prep_review/1.0.0"],  # #2242 V1-D — cairn-contracts-dataset
)
class PrepReviewOp(Operation[PrepReviewInput, PrepReviewOutput]):
    """Record a human's sign-off on a scrub that still carries residual PII."""

    input_schema = PrepReviewInput
    output_schema = PrepReviewOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    # #1539 — governance declarations; the [operator.*] table is generated from these
    requires_approval = True
    supports_mock = True

    async def run(self, inp: PrepReviewInput, ctx: OperationContext) -> PrepReviewOutput:
        why = f" ({'; '.join(inp.reasons)})" if inp.reasons else ""
        return PrepReviewOutput(
            reviewed=True,
            residual_pii_rows=int(inp.post_scrub_pii_rows or 0),
            note=(
                f"Human approved the scrub despite {inp.post_scrub_pii_rows} residual "
                f"PII row(s){why}."
            ),
        )

    async def mock(self, inp: PrepReviewInput, ctx: OperationContext) -> PrepReviewOutput:
        # --mock skips the interrupt (dev/CI); the sign-off record is deterministic.
        return await self.run(inp, ctx)
