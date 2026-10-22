"""#962 — decontam Shardable: splitting the TRAIN set across shards (each checked
against the shared eval set) must produce the SAME contaminated indices as the
inline check. Exact + ngram methods (both hermetic); minhash needs datasketch."""

from __future__ import annotations

import json

import pytest

from cairn.core.operator import OperationContext
from cairn.core.operator.sharded import ShardSpec, shard_ranges
from operator_dataset_checks.operators.decontam import DecontamOp
from operator_dataset_checks.schemas.decontam import DecontamInput

# train rows 2, 5, 9 are copied from the eval set → contaminated.
TRAIN = [
    {"text": "unique training example one"},       # 0
    {"text": "another distinct training row"},      # 1
    {"text": "the quick brown fox jumps over"},     # 2  <- in eval
    {"text": "training row number three here"},     # 3
    {"text": "yet more original training text"},    # 4
    {"text": "capital of france is paris"},         # 5  <- in eval
    {"text": "six six six training data"},          # 6
    {"text": "seventh row of the train set"},       # 7
    {"text": "eighth distinct example row"},        # 8
    {"text": "water boils at one hundred celsius"}, # 9  <- in eval
    {"text": "tenth and final training row"},       # 10
]
EVAL = [
    {"text": "the quick brown fox jumps over"},
    {"text": "capital of france is paris"},
    {"text": "water boils at one hundred celsius"},
]


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="decontam")


def _files(tmp_path):
    tr = tmp_path / "train.jsonl"
    ev = tmp_path / "eval.jsonl"
    tr.write_text("\n".join(json.dumps(r) for r in TRAIN))
    ev.write_text("\n".join(json.dumps(r) for r in EVAL))
    return str(tr), str(ev)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["exact", "ngram", "both"])
@pytest.mark.parametrize("k", [2, 3, 4])
async def test_sharded_decontam_equals_inline(tmp_path, method, k):
    tr, ev = _files(tmp_path)
    op = DecontamOp()
    inp = DecontamInput(path=tr, eval_path=ev, method=method)

    inline = await op.run(inp, _ctx())
    total = op.shard_total(inp)
    parts = [op.run_shard(inp, ShardSpec(i, s, e)) for i, (s, e) in enumerate(shard_ranges(total, k))]
    merged = op.merge(inp, parts)

    assert merged.train_total == inline.train_total == 11
    assert merged.eval_total == inline.eval_total == 3
    assert merged.contaminated == inline.contaminated
    assert merged.contaminated_rows == inline.contaminated_rows          # GLOBAL indices, sorted
    assert merged.ngram_contaminated_rows == inline.ngram_contaminated_rows
    # exact method catches the 3 verbatim copies at their global positions
    if method in ("exact", "both"):
        assert 2 in inline.contaminated_rows and 5 in inline.contaminated_rows and 9 in inline.contaminated_rows


def test_no_eval_set_opts_out_of_sharding(tmp_path):
    tr, _ = _files(tmp_path)
    op = DecontamOp()
    inp = DecontamInput(path=tr, eval_path=None, method="exact")
    assert op.shard_total(inp) == 0   # nothing to check against -> stays inline
