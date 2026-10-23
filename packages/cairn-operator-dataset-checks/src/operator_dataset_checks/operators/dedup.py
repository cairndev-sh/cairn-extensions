"""dataset/dedup — exact + near-duplicate detection over a JSONL dataset.

Canonicalizes each example (sorted-key JSON, optionally whitespace-collapsed),
hashes it, and groups identical rows. Reports the effective (deduped) size —
the number of training rows that actually carry distinct signal. The default
``method='exact'`` path is deterministic and dependency-free.

Issue #382 adds an OPTIONAL near-duplicate method: MinHash + LSH (``_minhash.py``
via ``datasketch``, the ``[minhash]`` extra) clusters rows whose word-shingle
Jaccard similarity is at/above a threshold — catching reworded/paraphrased dups
that a byte-identical hash misses, without the O(n²) all-pairs comparison.
``method='exact'`` (the default) preserves today's output exactly;
``'minhash'``/``'both'`` are opt-in.
"""
from __future__ import annotations

from itertools import islice
from typing import Optional

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec
from cairn.core.operator.sharded import Shardable, ShardSpec

from operator_dataset_checks._canon import canonical_hash
from operator_dataset_checks._io import iter_rows, load_rows
from operator_dataset_checks._minhash import cluster_signatures, resolve, row_minhash
from operator_dataset_checks.schemas.dedup import (
    DedupConfig,
    DedupInput,
    DedupOutput,
    DuplicateGroup,
    NearDuplicateGroup,
)


def _shard_signals(rows: list[dict], inp: DedupInput) -> dict:
    """Per-row dedup SIGNALS for a slice of rows (picklable): the canonical hash
    of every row (always) and, when the method needs it, each row's MinHash
    signature. The CPU-heavy per-row work (canonicalize / shingle / MinHash);
    the actual grouping + LSH clustering is the global reduce in :meth:`_report`."""
    hashes = [canonical_hash(row, inp.normalize) for row in rows]
    sigs = None
    if resolve(inp.method) in ("minhash", "both"):
        sigs = [row_minhash(row, ngram=inp.ngram, num_perm=inp.num_perm) for row in rows]
    return {"hashes": hashes, "sigs": sigs}


@register_operator(
    name="dataset/dedup",
    version="0.2.0",
    capabilities=["dataset.dedup"],
)
class DedupOp(Operation[DedupInput, DedupOutput], Shardable):
    """Find exact (+ optionally near-) duplicate rows and report the deduped size.

    #962 — Shardable: exact + near-dup detection are GLOBAL (a duplicate pair can
    span shards), so shards do only the parallel per-row work — compute each row's
    canonical hash + (optional) MinHash signature — and the merge does the global
    reduce: group hashes and cluster all signatures at once, so a cross-shard dup
    is still caught. Parity-exact with the inline path by construction."""

    input_schema = DedupInput
    output_schema = DedupOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    config_schema = DedupConfig

    @staticmethod
    def _report(hashes: list[str], signatures: Optional[list], inp: DedupInput) -> DedupOutput:
        # hashes are in GLOBAL row order (shards concatenated in order), so the
        # positional index i IS the global row index — matching the inline path.
        groups: dict[str, list[int]] = {}
        for i, h in enumerate(hashes):
            groups.setdefault(h, []).append(i)
        total = len(hashes)
        unique = len(groups)
        exact_duplicates = total - unique
        dup_groups = [
            DuplicateGroup(hash=h[:12], rows=idxs)
            for h, idxs in groups.items()
            if len(idxs) > 1
        ]
        pct = round(exact_duplicates / total * 100, 1) if total else 0.0

        near_groups: list[NearDuplicateGroup] = []
        near_duplicates = 0
        if signatures is not None:  # method included minhash/both; cluster globally
            clusters = cluster_signatures(
                signatures, threshold=inp.threshold, num_perm=inp.num_perm
            )
            near_groups = [NearDuplicateGroup(rows=c) for c in clusters]
            near_duplicates = sum(len(c) - 1 for c in clusters)

        return DedupOutput(
            total=total,
            unique=unique,
            exact_duplicates=exact_duplicates,
            effective_size=unique,
            duplicate_pct=pct,
            duplicate_groups=dup_groups,
            near_duplicate_groups=near_groups,
            near_duplicates=near_duplicates,
        )

    async def run(self, inp: DedupInput, ctx: OperationContext) -> DedupOutput:
        s = _shard_signals(load_rows(inp.path), inp)
        return self._report(s["hashes"], s["sigs"], inp)

    # ── #962 Shardable: signals computed per shard, reduced once in merge ──
    def shard_total(self, inp: DedupInput) -> int:
        return sum(1 for _ in iter_rows(inp.path))

    def run_shard(self, inp: DedupInput, spec: ShardSpec) -> dict:
        rows = list(islice(iter_rows(inp.path), spec.start, spec.end))
        return _shard_signals(rows, inp)

    def merge(self, inp: DedupInput, parts: list[dict]) -> DedupOutput:
        # concatenate shard signals IN ORDER -> global row order
        hashes = [h for p in parts for h in p["hashes"]]
        sigs: Optional[list] = None
        if parts and parts[0]["sigs"] is not None:
            sigs = [s for p in parts for s in p["sigs"]]
        return self._report(hashes, sigs, inp)
