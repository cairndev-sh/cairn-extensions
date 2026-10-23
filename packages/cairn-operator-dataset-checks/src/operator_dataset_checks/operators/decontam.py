"""dataset/decontam — train↔eval contamination check.

Hashes every example in the eval/benchmark set (shared canonicalization with
dataset/dedup) and flags any TRAINING row whose content also appears there —
i.e. eval examples that leaked into training. Deterministic, catches
byte-identical leaks.

Issue #381 adds an OPTIONAL second, complementary method: word n-gram
(shingle) Jaccard overlap (``_ngram.py``), which also catches NEAR-duplicate
leaks — an eval question reworded, truncated, or copy-pasted with extra
whitespace hashes differently but still leaks the answer. ``method='exact'``
(the default) preserves today's behaviour exactly; ``'ngram'``/``'both'`` are
opt-in.

Issue #382 adds a THIRD, scalable near-dup method: ``method='minhash'``
(``_minhash.py`` via ``datasketch``, the ``[minhash]`` extra). It approximates
the same shingle-Jaccard overlap the n-gram path computes exactly, but via a
MinHash+LSH index over the eval set — one LSH query per train row instead of
the ngram path's O(train × eval) all-pairs comparison — so it stays near-linear
on large corpora. Requires the ``[minhash]`` extra; the hermetic exact/ngram/
both paths never touch it.
"""
from __future__ import annotations

from itertools import islice

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec
from cairn.core.operator.sharded import Shardable, ShardSpec

from operator_dataset_checks._canon import canonical_hash
from operator_dataset_checks._io import iter_rows, load_rows
from operator_dataset_checks._minhash import contaminated_by_minhash, resolve
from operator_dataset_checks._ngram import jaccard, row_text, shingles
from operator_dataset_checks.schemas.decontam import (
    DecontamConfig,
    DecontamInput,
    DecontamOutput,
)


def _check_contamination(
    train: list[dict], eval_rows: list[dict], inp: DecontamInput, *, offset: int = 0
) -> tuple[list[int], list[int], list[int]]:
    """Contamination check for a train slice against the eval set — returns the
    (exact, ngram, minhash) contaminated row-index lists with GLOBAL indices
    (``offset + local``). Shared by the inline ``run`` and a sharded ``run_shard``
    so both paths compute identical verdicts. Each row is checked independently
    against the eval index, so a train slice is exactly a shard of the whole."""
    run_exact = inp.method in ("exact", "both")
    run_ngram = inp.method in ("ngram", "both")
    run_minhash = inp.method == "minhash"
    if run_minhash:
        resolve("minhash")  # clear RuntimeError if the [minhash] extra is absent

    exact: list[int] = []
    if run_exact:
        eval_hashes = {canonical_hash(r, inp.normalize) for r in eval_rows}
        exact = [
            offset + i for i, row in enumerate(train)
            if canonical_hash(row, inp.normalize) in eval_hashes
        ]

    ngram: list[int] = []
    if run_ngram:
        exact_set = set(exact)
        ngram = [
            offset + i for i in _ngram_matches(train, eval_rows, inp.ngram, inp.overlap_threshold)
            if (offset + i) not in exact_set
        ]

    minhash: list[int] = []
    if run_minhash:
        minhash = [
            offset + i for i in contaminated_by_minhash(
                train, eval_rows, threshold=inp.overlap_threshold,
                ngram=inp.ngram, num_perm=inp.num_perm,
            )
        ]
    return exact, ngram, minhash


def _ngram_matches(
    train: list[dict], eval_rows: list[dict], n: int, threshold: float,
) -> list[int]:
    """Training-row indices whose shingle set is >= threshold Jaccard-similar
    to ANY eval-row shingle set. O(train * eval) — fine at decontam's normal
    scale (one eval/benchmark set vs one training set); see _ngram.py's
    verify-upstream note for the MinHash/LSH scale-up path (#382)."""
    eval_shingle_sets = [shingles(row_text(r), n) for r in eval_rows]
    matches = []
    for i, row in enumerate(train):
        train_shingles = shingles(row_text(row), n)
        if any(jaccard(train_shingles, es) >= threshold for es in eval_shingle_sets):
            matches.append(i)
    return matches


@register_operator(
    name="dataset/decontam",
    version="0.3.0",
    capabilities=["dataset.decontam"],
    implements=["contract://dataset.decontam/1.0.0"],  # #2242 V1-B — cairn-contracts-dataset
)
class DecontamOp(Operation[DecontamInput, DecontamOutput], Shardable):
    """Flag training rows that also appear in the eval/benchmark set.

    #962 — Shardable: the TRAIN set is split across a budgeted process pool; each
    shard checks its slice against the (shared) eval index and returns GLOBAL
    contaminated indices, which merge concatenates."""

    input_schema = DecontamInput
    output_schema = DecontamOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    config_schema = DecontamConfig

    async def run(self, inp: DecontamInput, ctx: OperationContext) -> DecontamOutput:
        return self.sync_run(inp)

    @staticmethod
    def sync_run(inp: DecontamInput) -> DecontamOutput:
        # #2381 (ADR-0118 M7a) — the body, sync and context-free: the `decontam.scan` kernel
        # calls it as `run_decontam` (no event loop, no stand-in context); `run` is the door.
        train = load_rows(inp.path)

        # No eval set → report "not checked" rather than failing the run.
        if not inp.eval_path:
            return DecontamOutput(
                checked=False,
                train_total=len(train),
                eval_total=0,
                contaminated=0,
                overlap_pct=0.0,
                contaminated_rows=[],
                ngram_contaminated_rows=[],
                minhash_contaminated_rows=[],
            )

        eval_rows = load_rows(inp.eval_path)
        exact, ngram, minhash = _check_contamination(train, eval_rows, inp)
        return DecontamOp._output(len(train), len(eval_rows), [(exact, ngram, minhash)])

    @staticmethod
    def _output(train_total: int, eval_total: int, groups) -> DecontamOutput:
        """Assemble the DecontamOutput from one or more (exact, ngram, minhash)
        index groups — one group inline, one-per-shard when merged."""
        exact: list[int] = []
        ngram: list[int] = []
        minhash: list[int] = []
        for e, n, m in groups:
            exact += e
            ngram += n
            minhash += m
        total = len(exact) + len(ngram) + len(minhash)
        pct = round(total / train_total * 100, 1) if train_total else 0.0
        return DecontamOutput(
            checked=True,
            train_total=train_total,
            eval_total=eval_total,
            contaminated=total,
            overlap_pct=pct,
            contaminated_rows=sorted(exact),
            ngram_contaminated_rows=sorted(ngram),
            minhash_contaminated_rows=sorted(minhash),
        )

    # ── #962 Shardable: split the TRAIN set across shards; each checks its slice
    #    against the (shared) eval index. No eval set → shard_total 0 → inline. ──
    def shard_total(self, inp: DecontamInput) -> int:
        if not inp.eval_path:
            return 0
        return sum(1 for _ in iter_rows(inp.path))

    def run_shard(self, inp: DecontamInput, spec: ShardSpec) -> dict:
        eval_rows = load_rows(inp.eval_path)
        train = list(islice(iter_rows(inp.path), spec.start, spec.end))
        exact, ngram, minhash = _check_contamination(train, eval_rows, inp, offset=spec.start)
        return {
            "exact": exact, "ngram": ngram, "minhash": minhash,
            "train_total": len(train), "eval_total": len(eval_rows),
        }

    def merge(self, inp: DecontamInput, parts: list[dict]) -> DecontamOutput:
        train_total = sum(p["train_total"] for p in parts)
        eval_total = parts[0]["eval_total"] if parts else 0
        groups = [(p["exact"], p["ngram"], p["minhash"]) for p in parts]
        return self._output(train_total, eval_total, groups)


run_decontam = DecontamOp.sync_run
