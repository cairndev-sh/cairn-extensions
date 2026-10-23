"""#962 — pii_scan Shardable: a multi-shard split/merge must equal the inline
scan (same counts, rows, findings, truncation). The process-pool dispatch itself
is covered by tests/core/test_batch_cpu_budget.py; here we prove the sharding
LOGIC (plan → run_shard×k → merge) is parity-exact, driven in-process."""

from __future__ import annotations

import json

import pytest

from cairn.core.operator import OperationContext
from cairn.core.operator.sharded import ShardSpec, shard_ranges, should_shard
from operator_dataset_checks.operators.pii_scan import PiiScanOp
from operator_dataset_checks.schemas.pii_scan import PiiInput

# 12 rows, PII on a known subset — enough to span >1 shard with global indices.
ROWS = [
    {"text": "email alice@example.com"},        # 0  pii
    {"text": "clean row"},                        # 1
    {"text": "sk-ABCDEFGHIJKLMNOPQRSTUV"},        # 2  pii
    {"text": "nothing"},                          # 3
    {"text": "ip 10.0.0.1 and bob@corp.io"},      # 4  pii (2 kinds)
    {"text": "plain text"},                       # 5
    {"text": "AKIA0123456789ABCDEF"},             # 6  pii
    {"text": "another clean"},                    # 7
    {"text": "ghp_" + "a" * 36},                  # 8  pii
    {"text": "still clean"},                      # 9
    {"text": "carol@x.org"},                      # 10 pii
    {"text": "end"},                              # 11
]


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="shard")


def _write(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in ROWS))
    return str(src)


def _manual_sharded(op: PiiScanOp, inp: PiiInput, k: int):
    total = op.shard_total(inp)
    parts = [op.run_shard(inp, ShardSpec(i, s, e)) for i, (s, e) in enumerate(shard_ranges(total, k))]
    return op.merge(inp, parts)


@pytest.mark.asyncio
@pytest.mark.parametrize("k", [2, 3, 5])
async def test_sharded_scan_equals_inline(tmp_path, k):
    src = _write(tmp_path)
    op = PiiScanOp()
    inp = PiiInput(path=src, backend="regex", sample_size=None)
    inline = await op.run(inp, _ctx())
    sharded = _manual_sharded(op, inp, k)

    assert sharded.total == inline.total == 12
    assert sharded.rows_with_pii == inline.rows_with_pii == 6
    assert sharded.counts == inline.counts
    # findings carry GLOBAL row indices and stay in row order across shards
    assert [f.row for f in sharded.findings] == [f.row for f in inline.findings] == [0, 2, 4, 6, 8, 10]
    assert sharded.findings_truncated == inline.findings_truncated


@pytest.mark.asyncio
async def test_sharded_respects_findings_cap_globally(tmp_path):
    src = _write(tmp_path)
    op = PiiScanOp()
    inp = PiiInput(path=src, backend="regex", sample_size=None, max_findings=3)
    inline = await op.run(inp, _ctx())
    sharded = _manual_sharded(op, inp, 4)
    # global first-3 PII rows, truncated, exact counts regardless of shard boundary
    assert [f.row for f in sharded.findings] == [f.row for f in inline.findings] == [0, 2, 4]
    assert sharded.findings_truncated is True and inline.findings_truncated is True
    assert sharded.rows_with_pii == 6 and sharded.counts == inline.counts


def test_should_shard_gating(tmp_path, monkeypatch):
    src = _write(tmp_path)
    op = PiiScanOp()
    full = PiiInput(path=src, backend="regex", sample_size=None)
    sampled = PiiInput(path=src, backend="regex", sample_size=5)

    monkeypatch.setenv("CAIRN_BATCH_THRESHOLD_ROWS", "100")
    assert should_shard(op, full) == (False, 12)      # below threshold -> inline
    monkeypatch.setenv("CAIRN_BATCH_THRESHOLD_ROWS", "0")
    assert should_shard(op, full) == (True, 12)       # at/over threshold -> shard
    assert should_shard(op, sampled) == (False, 0)    # sampled -> never shardable
