"""dataset/replay — rehearsal sampling over governed dataset versions.

ADR-0095 §2.3 (#1625). Forgetting was MEASURED (verifiable-finetune's control
split) but never DEFENDED: nothing mixed prior-cycle rows back into cycle N's
train slice. Replay is the workhorse fix, and in cairn it is a DATASET
OPERATION, not an algorithm — EWC/GEM-style regularization is rejected outright
(ADR-0095 §2.2), so there is nothing here but sampling and concatenation over
the dataset versions the registry (#1466) already retains.

Determinism is the contract: every draw goes through ONE ``random.Random(seed)``
consumed in ``sources`` order, so the same inputs + seed reproduce the mix
byte-for-byte from the receipt. The output row order is fixed too — the new
rows first, then each source's block in ``sources`` order, each block in its own
source order — which is what makes the per-row-GROUP provenance in ``mix``
meaningful without stamping extra keys onto the training rows (an unknown key
can break a trainer's formatter).

Memory: the new slice STREAMS (it is written through as it is read); the replay
sources are materialized, because sampling k of n needs random access. Sources
are prior train slices, so this is the same ceiling the split/curate ops accept.
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks._canon import canonical_hash
from operator_dataset_checks._io import JsonlWriter, iter_rows, load_rows, upload_s3
from operator_dataset_checks.schemas.replay import (
    ReplayConfig,
    ReplayInput,
    ReplayMixEntry,
    ReplayOutput,
)

# ADR-0095 Amendment 1 item 2 — `strategy` is an OPEN enum from day one. These
# two are SPEC'd but need the per-row, per-version eval history eval/forgetting
# (#1626) produces, so they fail loudly rather than silently degrading to
# uniform: a run that thinks it prioritized forgotten rows and didn't would be
# worse than one that refused to start.
_RESERVED = {
    "hard_example": "rows the current model gets wrong",
    "forgetting_aware": "rows whose score degraded between adapter versions",
}
_IMPLEMENTED = ("uniform", "recency_weighted")

# D6 payload budget (see the note at the top of sample.py): this op puts COUNTS
# and LOCATIONS in its findings and never a row's text, so `mix` stays a handful
# of small records however large the corpora are.


def _strip_pin(source: str) -> str:
    """Drop the ``#sha256=<hash>`` pin a launch-resolved ``dataset:`` ref carries
    (#1466), the same way the fetching kernels do (``dataflow_stages``). The pin
    is a RETRIEVAL rule verified by whoever fetches the bytes; the reader just
    needs the location. The un-stripped string is what lands in ``mix``, so the
    receipt still names the exact version that was mixed in."""
    return source.partition("#")[0]


def _allocate(weights: list[float], caps: list[int], k: int) -> list[int]:
    """Spread ``k`` rows across the sources in proportion to ``weights``, capped
    by each source's ``caps`` (its eligible row count).

    Largest-remainder apportionment, ties broken by index, so the split is a
    deterministic function of the inputs — no RNG is consumed here, the seed
    only chooses WHICH rows a source contributes. A source that cannot fill its
    share is capped and its shortfall is re-apportioned over the sources that
    still have room; when everything is exhausted the caller reports the true
    (lower) ratio rather than oversampling duplicates to hit the target.
    """
    quota = [0] * len(weights)
    active = {i for i in range(len(weights)) if caps[i] > 0 and weights[i] > 0}
    remaining = k
    while remaining > 0 and active:
        total_w = sum(weights[i] for i in active)
        if total_w <= 0:
            break
        raw = {i: remaining * weights[i] / total_w for i in active}
        share = {i: int(raw[i]) for i in active}
        for i in sorted(active, key=lambda j: (-(raw[j] - share[j]), j))[
            : remaining - sum(share.values())
        ]:
            share[i] += 1
        for i in sorted(active):
            take = min(share[i], caps[i] - quota[i])
            quota[i] += take
            remaining -= take
            if quota[i] >= caps[i]:
                active.discard(i)
    return quota


@register_operator(
    name="dataset/replay",
    version="0.1.0",
    capabilities=["dataset.replay"],
    implements=["contract://dataset.replay/1.0.0"],  # #158 — the retrain-cycle slice
)
class ReplayOp(Operation[ReplayInput, ReplayOutput]):
    """Mix prior-cycle rows into the new train slice at a governed ratio."""

    input_schema = ReplayInput
    output_schema = ReplayOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    config_schema = ReplayConfig
    # #1539 — governance declarations; the [operator.*] table is generated from these
    risk = 'external_write'

    async def run(self, inp: ReplayInput, ctx: OperationContext) -> ReplayOutput:
        strategy = inp.strategy.strip().lower()
        if strategy in _RESERVED:
            raise ValueError(
                f"dataset/replay strategy {strategy!r} ({_RESERVED[strategy]}) is "
                "RESERVED, not implemented: it needs the per-row, per-version eval "
                "history eval/forgetting (#1626) produces, passed as `eval_report`. "
                f"Use one of {list(_IMPLEMENTED)} until then (ADR-0095 Amendment 1)."
            )
        if strategy not in _IMPLEMENTED:
            raise ValueError(
                f"dataset/replay: unknown strategy {strategy!r} — "
                f"expected one of {list(_IMPLEMENTED)} "
                f"(reserved for v2: {sorted(_RESERVED)})."
            )

        rng = random.Random(inp.seed)
        # An s3 out_path writes to a private temp file and uploads on success.
        # JsonlWriter is local-only — before this branch, Path("s3://…") wrote a
        # LOCAL ghost file at a directory literally named "s3:/", and the train
        # leg (possibly on another box) 404'd on the real key. The declared
        # out_path stays the output's truth either way.
        out_is_s3 = str(inp.out_path).startswith("s3://")
        if out_is_s3:
            out = Path(tempfile.mkdtemp(prefix="replay-mix-")) / "mix.jsonl"
        else:
            out = Path(inp.out_path)

        with JsonlWriter(out) as writer:
            # the new slice streams straight through; hashing it as it passes
            # gives dedup its reference set without a second read.
            new_rows = 0
            new_hashes: set[str] = set()
            for row in iter_rows(_strip_pin(inp.new_rows_path)):
                writer.write(row)
                new_rows += 1
                if inp.dedup_against_new:
                    new_hashes.add(canonical_hash(row, inp.normalize))

            # eligible rows per source — dedup FIRST so `rows_available` (and
            # therefore the apportionment) counts only rows that can be taken.
            pools: list[list[dict]] = []
            for source in inp.sources:
                rows = load_rows(_strip_pin(source))
                if inp.dedup_against_new:
                    rows = [
                        r for r in rows
                        if canonical_hash(r, inp.normalize) not in new_hashes
                    ]
                pools.append(rows)
            caps = [len(p) for p in pools]

            # `ratio` is the replay share of the OUTPUT, so the target solves
            # k / (new + k) = ratio  →  k = new * ratio / (1 - ratio).
            target = (
                int(round(new_rows * inp.ratio / (1.0 - inp.ratio)))
                if inp.ratio > 0 else 0
            )
            if strategy == "recency_weighted":
                # `sources` is OLDEST first, so position+1 is a linear recency
                # ramp: the last (most recent) cycle gets the largest share.
                weights = [float(i + 1) for i in range(len(pools))]
            else:  # uniform — share ∝ eligible rows, i.e. every prior row equally likely
                weights = [float(c) for c in caps]
            quota = _allocate(weights, caps, target)

            mix: list[ReplayMixEntry] = []
            replay_rows = 0
            for source, rows, take in zip(inp.sources, pools, quota):
                # sample WITHOUT replacement, then restore source order so the
                # block is a stable slice of its source, not a shuffle.
                picked = sorted(rng.sample(range(len(rows)), take)) if take else []
                for i in picked:
                    writer.write(rows[i])
                replay_rows += len(picked)
                mix.append(ReplayMixEntry(
                    source=source, rows_taken=len(picked), rows_available=len(rows),
                ))

        if out_is_s3:
            upload_s3(out, str(inp.out_path))

        total = new_rows + replay_rows
        return ReplayOutput(
            out_path=inp.out_path,
            total_rows=total,
            new_rows=new_rows,
            replay_rows=replay_rows,
            ratio_actual=round(replay_rows / total, 4) if total else 0.0,
            strategy_used=strategy,
            mix=mix,
        )
