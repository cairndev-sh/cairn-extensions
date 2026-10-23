"""dataset/dedup_apply — drop duplicate rows and write a cleaned dataset.

The *remediation* half of dataset/dedup (#991, ADR-0076 D3). dedup DETECTS +
reports (read_only, runs inside the readiness gate); dedup_apply DROPS the
duplicates and writes a new JSONL (external_write → HITL-gated in prod). It
reuses dedup's exact (canonical-hash) + near-dup (MinHash/LSH) detection, so a
row it drops is exactly one the check flagged.

Determinism: within each duplicate group / near-dup cluster it keeps the
EARLIEST row (lowest source index) and drops the rest, preserving source order —
so the fixed dataset is a stable, replayable function of its input.

STREAMING (#1009, ADR-0078 S3): rows are never materialized. The exact path is
ONE pass — stream → seen-hash set → write survivors (memory = the hash set).
The minhash path is TWO passes: pass 1 streams to collect per-row MinHash
signatures (memory = signatures, the same ceiling the detection op already has
— never the rows) + the exact-drop indices; pass 2 streams again and writes
every row not flagged. Decision-identical to the old materialized
implementation, so the output is byte-for-byte the same.
"""
from __future__ import annotations

from pathlib import Path

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks._canon import canonical_hash
from operator_dataset_checks._io import JsonlWriter, iter_rows
from operator_dataset_checks._minhash import cluster_signatures, resolve, row_minhash
from operator_dataset_checks.schemas.dedup_apply import (
    DedupApplyInput,
    DedupApplyOutput,
)


@register_operator(
    name="dataset/dedup_apply",
    version="0.1.0",
    capabilities=["dataset.dedup_apply"],
)
class DedupApplyOp(Operation[DedupApplyInput, DedupApplyOutput]):
    """Drop exact (+ optionally near-) duplicate rows; write the survivors."""

    input_schema = DedupApplyInput
    output_schema = DedupApplyOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    # #1539 — governance declarations; the [operator.*] table is generated from these
    risk = 'external_write'

    async def run(self, inp: DedupApplyInput, ctx: OperationContext) -> DedupApplyOutput:
        return self.sync_run(inp)

    @staticmethod
    def sync_run(inp: DedupApplyInput) -> DedupApplyOutput:
        # #2242 V1-B (M7a) — the body, sync and context-free: the dataflow kernels call it
        # as `run_dedup_apply` (no event loop, no stand-in context); `run` is the operator's door.
        near = inp.method in ("minhash", "both")
        if near:
            resolve("minhash")  # clear RuntimeError if the [minhash] extra is absent

        out = Path(inp.output_path)
        seen: set[str] = set()

        if not near:
            # exact only — ONE streaming pass: first occurrence survives.
            total = 0
            dropped_exact = 0
            with JsonlWriter(out) as writer:
                for row in iter_rows(inp.path):
                    total += 1
                    h = canonical_hash(row, inp.normalize)
                    if h in seen:
                        dropped_exact += 1
                    else:
                        seen.add(h)
                        writer.write(row)
            kept = total - dropped_exact
            return DedupApplyOutput(
                output_path=inp.output_path, total=total, kept=kept,
                dropped=dropped_exact, dropped_exact=dropped_exact,
                dropped_near=0, method_used=inp.method,
            )

        # near-dup — pass 1: exact-drop indices + per-row signatures (no rows held).
        exact_drop: set[int] = set()
        sigs: list = []
        total = 0
        for i, row in enumerate(iter_rows(inp.path)):
            total += 1
            h = canonical_hash(row, inp.normalize)
            if h in seen:
                exact_drop.add(i)
            else:
                seen.add(h)
            sigs.append(row_minhash(row, ngram=inp.ngram, num_perm=inp.num_perm))
        seen.clear()  # pass 2 doesn't need the hashes — free them before clustering

        # cluster; within each cluster keep the earliest not-already-dropped row.
        near_drop: set[int] = set()
        for cluster in cluster_signatures(sigs, threshold=inp.threshold, num_perm=inp.num_perm):
            kept_one = False
            for idx in sorted(cluster):
                if idx in exact_drop:
                    continue
                if not kept_one:
                    kept_one = True  # earliest survivor stays
                else:
                    near_drop.add(idx)

        # pass 2: stream again, write survivors in source order.
        with JsonlWriter(out) as writer:
            for i, row in enumerate(iter_rows(inp.path)):
                if i not in exact_drop and i not in near_drop:
                    writer.write(row)

        dropped = len(exact_drop) + len(near_drop)
        return DedupApplyOutput(
            output_path=inp.output_path, total=total, kept=total - dropped,
            dropped=dropped, dropped_exact=len(exact_drop),
            dropped_near=len(near_drop), method_used=inp.method,
        )


#: #2242 V1-B (M7a) — the shared helper the `dedup.exact` and `dedup.near` kernels call directly.
run_dedup_apply = DedupApplyOp.sync_run
