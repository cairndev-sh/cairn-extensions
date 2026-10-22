"""dataset/dedup_apply + dataset/decontam_apply — the remediation writers (#991).

The detection they reuse is covered by test_dedup / test_decontam; here we pin
that the *apply* half writes the survivors: drops the flagged rows, keeps the
earliest of each duplicate group, preserves source order, and copies through
when there's nothing to remove.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from operator_dataset_checks.operators.dedup_apply import DedupApplyOp
from operator_dataset_checks.operators.decontam_apply import DecontamApplyOp
from operator_dataset_checks.schemas.dedup_apply import DedupApplyInput
from operator_dataset_checks.schemas.decontam_apply import DecontamApplyInput


class _Ctx:
    pass


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


def _read(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def test_dedup_apply_drops_exact_keeps_earliest_in_order(tmp_path):
    a = {"messages": [{"role": "user", "content": "hi"}]}
    b = {"messages": [{"role": "user", "content": "bye"}]}
    c = {"messages": [{"role": "user", "content": "unique"}]}
    src = _write(tmp_path / "in.jsonl", [a, a, b, a, c])   # a x3
    out = str(tmp_path / "out.jsonl")
    o = asyncio.run(DedupApplyOp().run(DedupApplyInput(path=src, output_path=out), _Ctx()))
    assert (o.total, o.kept, o.dropped, o.dropped_exact) == (5, 3, 2, 0 + 2)
    assert _read(out) == [a, b, c]  # earliest survivor + source order preserved


def test_dedup_apply_no_dups_is_identity(tmp_path):
    rows = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(4)]
    src = _write(tmp_path / "in.jsonl", rows)
    out = str(tmp_path / "out.jsonl")
    o = asyncio.run(DedupApplyOp().run(DedupApplyInput(path=src, output_path=out), _Ctx()))
    assert o.dropped == 0 and _read(out) == rows


def test_decontam_apply_drops_leaked_rows(tmp_path):
    train = [{"text": "the answer is 42"}, {"text": "clean one"}, {"text": "clean two"}]
    ev = [{"text": "the answer is 42"}]
    tp = _write(tmp_path / "train.jsonl", train)
    ep = _write(tmp_path / "eval.jsonl", ev)
    out = str(tmp_path / "clean.jsonl")
    o = asyncio.run(DecontamApplyOp().run(
        DecontamApplyInput(path=tp, eval_path=ep, output_path=out, method="exact"), _Ctx()))
    assert (o.total, o.kept, o.dropped, o.checked) == (3, 2, 1, True)
    assert {r["text"] for r in _read(out)} == {"clean one", "clean two"}


def test_decontam_apply_no_eval_copies_through(tmp_path):
    train = [{"text": "a"}, {"text": "b"}]
    tp = _write(tmp_path / "train.jsonl", train)
    out = str(tmp_path / "clean.jsonl")
    o = asyncio.run(DecontamApplyOp().run(
        DecontamApplyInput(path=tp, output_path=out), _Ctx()))
    assert o.checked is False and o.dropped == 0 and _read(out) == train
