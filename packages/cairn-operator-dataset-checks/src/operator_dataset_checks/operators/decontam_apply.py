"""dataset/decontam_apply — drop train↔eval contaminated rows; write a cleaned set.

The *remediation* half of dataset/decontam (#991, ADR-0076 D3). decontam DETECTS
+ reports (read_only, runs in the readiness gate); decontam_apply DROPS the
training rows that leak the eval/benchmark set and writes a new JSONL
(external_write → HITL-gated in prod). Per-row decisions are identical to
decontam's exact/ngram/minhash detection (same primitives, same thresholds), so
a row it drops is exactly one the check flagged. Source order is preserved
among survivors.

No ``eval_path`` → nothing to check against, so it copies the dataset through
unchanged (``checked=false``) rather than failing — mirroring dataset/decontam.

STREAMING (#1009, ADR-0078 S3): the eval side (small — a benchmark set) is
loaded and indexed ONCE (hash set / shingle sets / LSH index); the train side —
the big side — then streams in ONE pass, each row checked against the prebuilt
index and written or dropped. Memory = the eval index, never the train set.
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
from operator_dataset_checks._io import JsonlWriter, iter_rows, load_rows
from operator_dataset_checks._minhash import SEED, resolve, row_minhash
from operator_dataset_checks._ngram import jaccard, row_text, shingles
from operator_dataset_checks.schemas.decontam_apply import (
    DecontamApplyInput,
    DecontamApplyOutput,
)


@register_operator(
    name="dataset/decontam_apply",
    version="0.1.0",
    capabilities=["dataset.decontam_apply"],
)
class DecontamApplyOp(Operation[DecontamApplyInput, DecontamApplyOutput]):
    """Drop training rows that appear in the eval set; write the survivors."""

    input_schema = DecontamApplyInput
    output_schema = DecontamApplyOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    # #1539 — governance declarations; the [operator.*] table is generated from these
    risk = 'external_write'

    async def run(self, inp: DecontamApplyInput, ctx: OperationContext) -> DecontamApplyOutput:
        return self.sync_run(inp)

    @staticmethod
    def sync_run(inp: DecontamApplyInput) -> DecontamApplyOutput:
        # #2242 V1-B (M7a) — the body, sync and context-free: the dataflow kernels call it
        # as `run_decontam_apply` (no event loop, no stand-in context); `run` is the operator's door.
        out = Path(inp.output_path)

        if not inp.eval_path:
            # nothing to decontaminate against — stream a copy through unchanged.
            total = 0
            with JsonlWriter(out) as writer:
                for row in iter_rows(inp.path):
                    writer.write(row)
                    total += 1
            return DecontamApplyOutput(
                output_path=inp.output_path, total=total, kept=total,
                dropped=0, checked=False, method_used=inp.method,
            )

        run_exact = inp.method in ("exact", "both")
        run_ngram = inp.method in ("ngram", "both")
        run_minhash = inp.method == "minhash"

        # ── build the eval-side index ONCE (the small side) ──────────────────
        eval_rows = load_rows(inp.eval_path)
        eval_hashes: set[str] = (
            {canonical_hash(r, inp.normalize) for r in eval_rows} if run_exact else set()
        )
        eval_shingle_sets = (
            [shingles(row_text(r), inp.ngram) for r in eval_rows] if run_ngram else []
        )
        lsh = None
        if run_minhash:
            resolve("minhash")  # clear RuntimeError if the [minhash] extra is absent
            from datasketch import MinHashLSH  # optional dependency (minhash extra)

            lsh = MinHashLSH(threshold=inp.overlap_threshold, num_perm=inp.num_perm)
            for j, row in enumerate(eval_rows):
                lsh.insert(
                    f"eval-{j}",
                    row_minhash(row, ngram=inp.ngram, num_perm=inp.num_perm, seed=SEED),
                )
        del eval_rows  # the indexes carry everything the scan needs

        # ── stream the train side (the big side) in ONE pass ─────────────────
        total = 0
        dropped = 0
        with JsonlWriter(out) as writer:
            for row in iter_rows(inp.path):
                total += 1
                contaminated = (
                    run_exact and canonical_hash(row, inp.normalize) in eval_hashes
                )
                if not contaminated and run_ngram:
                    ts = shingles(row_text(row), inp.ngram)
                    contaminated = any(
                        jaccard(ts, es) >= inp.overlap_threshold for es in eval_shingle_sets
                    )
                if not contaminated and run_minhash:
                    sig = row_minhash(row, ngram=inp.ngram, num_perm=inp.num_perm, seed=SEED)
                    contaminated = bool(lsh.query(sig))
                if contaminated:
                    dropped += 1
                else:
                    writer.write(row)

        return DecontamApplyOutput(
            output_path=inp.output_path, total=total, kept=total - dropped,
            dropped=dropped, checked=True, method_used=inp.method,
        )


#: #2242 V1-B (M7a) — the shared helper the `decontam.apply` kernel calls directly.
run_decontam_apply = DecontamApplyOp.sync_run
