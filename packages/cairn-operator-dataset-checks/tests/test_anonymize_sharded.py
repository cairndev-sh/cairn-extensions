"""#962 — anonymize Shardable: sharded redaction must be BYTE-IDENTICAL to the
inline redaction (concatenated shard files == single-writer output), with the
same counts. Split/merge logic driven in-process (pool dispatch covered by
tests/core/test_batch_cpu_budget.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cairn.core.operator import OperationContext
from cairn.core.operator.sharded import ShardSpec, shard_ranges
from operator_dataset_checks.operators.anonymize import AnonymizeOp
from operator_dataset_checks.schemas.anonymize import AnonymizeInput

ROWS = [
    {"text": "email alice@example.com", "id": 1},
    {"text": "clean", "id": 2},
    {"text": "key sk-ABCDEFGHIJKLMNOPQRSTUV", "id": 3},
    {"text": "ip 10.0.0.1 bob@corp.io", "id": 4},
    {"text": "plain", "id": 5},
    {"text": "AKIA0123456789ABCDEF", "id": 6},
    {"text": "clean2", "id": 7},
    {"text": "carol@x.org", "id": 8},
    {"text": "nine", "id": 9},
]


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="anon")


def _src(tmp_path) -> str:
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in ROWS))
    return str(p)


@pytest.mark.asyncio
@pytest.mark.parametrize("k", [2, 3, 4])
async def test_sharded_anonymize_is_byte_identical_to_inline(tmp_path, k):
    src = _src(tmp_path)
    op = AnonymizeOp()

    out_inline = str(tmp_path / "inline.jsonl")
    inline = await op.run(
        AnonymizeInput(path=src, output_path=out_inline, backend="regex"), _ctx()
    )

    out_sharded = str(tmp_path / "sharded.jsonl")
    inp = AnonymizeInput(path=src, output_path=out_sharded, backend="regex")
    workdir = tmp_path / "shards"
    workdir.mkdir()
    total = op.shard_total(inp)
    parts = [
        op.run_shard(inp, ShardSpec(i, s, e, str(workdir)))
        for i, (s, e) in enumerate(shard_ranges(total, k))
    ]
    merged = op.merge(inp, parts)

    # the cleaned bytes that actually train must be identical either way
    assert Path(out_sharded).read_text() == Path(out_inline).read_text()
    assert merged.total == inline.total == 9
    assert merged.rows_redacted == inline.rows_redacted == 5
    assert merged.counts == inline.counts
    # and it's actually redacted + still valid JSONL, all rows preserved in order
    lines = Path(out_sharded).read_text().splitlines()
    assert len(lines) == 9
    assert [json.loads(x)["id"] for x in lines] == [r["id"] for r in ROWS]
    assert "alice@example.com" not in Path(out_sharded).read_text()


@pytest.mark.asyncio
async def test_sharded_merge_creates_missing_output_dir(tmp_path):
    # regression (live-caught on the box): merge wrote output_path with a raw
    # open() and failed when the parent dir didn't exist, unlike the inline
    # JsonlWriter path which mkdirs.
    src = _src(tmp_path)
    op = AnonymizeOp()
    out = str(tmp_path / "does" / "not" / "exist" / "out.jsonl")  # nested, absent
    inp = AnonymizeInput(path=src, output_path=out, backend="regex")
    total = op.shard_total(inp)
    parts = [op.run_shard(inp, ShardSpec(i, s, e, str(tmp_path)))
             for i, (s, e) in enumerate(shard_ranges(total, 3))]
    merged = op.merge(inp, parts)
    assert Path(out).exists() and merged.total == 9
