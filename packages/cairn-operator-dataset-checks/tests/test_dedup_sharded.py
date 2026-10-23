"""#962 — dedup Shardable: exact dedup is GLOBAL (a duplicate can span shards),
so shards compute per-row hashes and merge groups them once. The sharded result
must equal the inline result even when duplicate rows land in different shards.
(The minhash near-dup reduce uses the same shard-signals→cluster_signatures split
but needs the [minhash] extra, so it's verified on the box, not here.)"""

from __future__ import annotations

import json

import pytest

from cairn.core.operator import OperationContext
from cairn.core.operator.sharded import ShardSpec, shard_ranges
from operator_dataset_checks.operators.dedup import DedupOp
from operator_dataset_checks.schemas.dedup import DedupInput

# rows 0 & 6 are identical; 2 & 9 are identical — duplicates that WILL span shards
# under k>=2, so this exercises the cross-shard grouping in merge.
ROWS = [
    {"text": "alpha duplicated row"},   # 0  == 6
    {"text": "bravo unique"},           # 1
    {"text": "charlie duplicated"},     # 2  == 9
    {"text": "delta unique"},           # 3
    {"text": "echo unique"},            # 4
    {"text": "foxtrot unique"},         # 5
    {"text": "alpha duplicated row"},   # 6  == 0
    {"text": "golf unique"},            # 7
    {"text": "hotel unique"},           # 8
    {"text": "charlie duplicated"},     # 9  == 2
]


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="dedup")


def _src(tmp_path) -> str:
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in ROWS))
    return str(p)


@pytest.mark.asyncio
@pytest.mark.parametrize("k", [2, 3, 5])
async def test_sharded_exact_dedup_equals_inline(tmp_path, k):
    src = _src(tmp_path)
    op = DedupOp()
    inp = DedupInput(path=src, method="exact")

    inline = await op.run(inp, _ctx())
    total = op.shard_total(inp)
    parts = [op.run_shard(inp, ShardSpec(i, s, e)) for i, (s, e) in enumerate(shard_ranges(total, k))]
    merged = op.merge(inp, parts)

    assert merged.total == inline.total == 10
    assert merged.unique == inline.unique == 8       # 2 duplicate pairs collapse
    assert merged.exact_duplicates == inline.exact_duplicates == 2
    assert merged.effective_size == inline.effective_size
    # the duplicate GROUPS carry the same global row indices, spanning shards
    inline_groups = sorted(sorted(g.rows) for g in inline.duplicate_groups)
    merged_groups = sorted(sorted(g.rows) for g in merged.duplicate_groups)
    assert merged_groups == inline_groups == [[0, 6], [2, 9]]
    assert merged.duplicate_pct == inline.duplicate_pct


@pytest.mark.asyncio
async def test_all_unique_no_duplicates(tmp_path):
    p = tmp_path / "u.jsonl"
    p.write_text("\n".join(json.dumps({"text": f"row {i}"}) for i in range(9)))
    op = DedupOp()
    inp = DedupInput(path=str(p), method="exact")
    inline = await op.run(inp, _ctx())
    total = op.shard_total(inp)
    parts = [op.run_shard(inp, ShardSpec(i, s, e)) for i, (s, e) in enumerate(shard_ranges(total, 4))]
    merged = op.merge(inp, parts)
    assert merged.unique == merged.total == 9 == inline.unique
    assert merged.duplicate_groups == [] and inline.duplicate_groups == []
