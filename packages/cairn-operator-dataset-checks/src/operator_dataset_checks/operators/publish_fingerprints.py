"""dataset/publish_fingerprints — index a cleaned batch for cross-batch dedup.

The CONTROL-SIDE publish step #1050 listed and never built. Cross-batch dedup
(ADR-0080 P2) works in two halves: ``probe`` asks "have I seen these row hashes
in an EARLIER batch of this dataset?", and ``publish`` is what makes a future
probe meaningful. Until this operator, the only publisher in the tree was the
fix trio's ``_fix_pipeline.py`` (since deleted, #2425). The ``clean`` dataflow
published nothing, so every clean run silently degraded dedup to within-batch
only, and deleting the trio would have removed the last writer with no error and
no failing test.

Why it is an OPERATOR and not a dataflow stage: the data plane runs on a box with
no database access (the gap found live in #1050). The engine fetches the export
output back control-side, so the control plane can hash the cleaned rows and write
the index — which is exactly what this does.

``read_only`` risk: it writes cairn's OWN index, touches no user data and no
external system. Never raises — a missing store or an unreadable path degrades to
``published=0`` with a ``skipped`` reason, because losing the index must not fail
an otherwise-successful remediation run. The report surfaces the count, and
``tests/test_publish_fingerprints.py`` fails when a real batch publishes zero.
"""
from __future__ import annotations

import logging

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks._canon import canonical_hash
from operator_dataset_checks._io import iter_rows
from operator_dataset_checks.schemas.publish_fingerprints import (
    PublishFingerprintsInput,
    PublishFingerprintsOutput,
)

logger = logging.getLogger("operator_dataset_checks.publish_fingerprints")


@register_operator(
    name="dataset/publish_fingerprints",
    version="0.1.0",
    capabilities=["dataset.publish_fingerprints"],
    implements=["contract://dataset.publish_fingerprints/1.0.0"],  # #2242 V1-D — cairn-contracts-dataset
)
class PublishFingerprintsOp(
    Operation[PublishFingerprintsInput, PublishFingerprintsOutput]
):
    """Hash a cleaned batch's rows and publish them to the fingerprint index."""

    input_schema = PublishFingerprintsInput
    output_schema = PublishFingerprintsOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report

    async def run(
        self, inp: PublishFingerprintsInput, ctx: OperationContext
    ) -> PublishFingerprintsOutput:
        def _skip(reason: str, note: str) -> PublishFingerprintsOutput:
            logger.info("fingerprint publish skipped (%s): %s", reason, note)
            return PublishFingerprintsOutput(
                published=0, dataset_id=inp.dataset_id, batch_id=inp.batch_id,
                skipped=reason, summary=f"no fingerprints published — {note}",
            )

        if not inp.dataset_id.strip():
            # No cross-batch identity → nothing a future probe could match on.
            return _skip("no_dataset_id", "the run carries no dataset_id")

        try:
            hashes = [canonical_hash(row, inp.normalize) for row in iter_rows(inp.path)]
        except Exception as e:  # noqa: BLE001 — an unreadable output must not fail the run
            return _skip("no_rows", f"could not read {inp.path}: {type(e).__name__}")

        if not hashes:
            return _skip("no_rows", f"{inp.path} produced no rows")

        try:
            # The index is the context's handle (#2453 B): opened on first use,
            # closed by the runtime after the run — this pack owns no database.
            store = ctx.fingerprint_store()
            published = await store.publish(inp.dataset_id, inp.batch_id, hashes)
        except Exception as e:  # noqa: BLE001 — no index is a degraded run, not a failed one
            logger.warning(
                "fingerprint store unavailable (%s) — %d hashes NOT indexed for %s/%s",
                e, len(hashes), inp.dataset_id, inp.batch_id,
            )
            return _skip("store_unavailable", f"{type(e).__name__}: {e}")

        return PublishFingerprintsOutput(
            published=int(published),
            dataset_id=inp.dataset_id,
            batch_id=inp.batch_id,
            skipped=None,
            summary=f"published {published} fingerprints for {inp.dataset_id}/{inp.batch_id}",
        )
