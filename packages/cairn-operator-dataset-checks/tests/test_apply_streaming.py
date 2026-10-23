"""Streaming apply ops (#1009, ADR-0078 S3).

dedup_apply / decontam_apply no longer materialize the train set — the exact
paths are one streaming pass, near-dup is two passes over signatures. These
tests pin the semantics the rewrite must preserve (earliest survivor, source
order, per-row decisions identical to detection) plus the new tolerance: a
malformed line is quarantined (#1008), never crashes, and never reaches the
cleaned output.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from operator_dataset_checks.operators.dedup_apply import DedupApplyOp
from operator_dataset_checks.operators.decontam_apply import DecontamApplyOp
from operator_dataset_checks.schemas.dedup_apply import DedupApplyInput
from operator_dataset_checks.schemas.decontam_apply import DecontamApplyInput


class _Ctx:
    pass


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def _read(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def test_decontam_apply_ngram_drops_paraphrase_leak(tmp_path):
    leak = {"prompt": "what is the capital of france", "output": "the capital of france is paris of course"}
    clean = {"prompt": "sum 2 and 2", "output": "4"}
    near_leak = {  # identical text + one trailing word — most shingles shared
        "prompt": "what is the capital of france",
        "output": "the capital of france is paris of course indeed",
    }
    src = _write(tmp_path / "train.jsonl", [clean, near_leak])
    ev = _write(tmp_path / "eval.jsonl", [leak])
    out = str(tmp_path / "out.jsonl")
    o = asyncio.run(DecontamApplyOp().run(DecontamApplyInput(
        path=src, eval_path=ev, output_path=out, method="ngram", overlap_threshold=0.5,
    ), _Ctx()))
    assert (o.total, o.kept, o.dropped, o.checked) == (2, 1, 1, True)
    assert _read(out) == [clean]


def test_dedup_apply_tolerates_malformed_lines(tmp_path):
    a = {"prompt": "p", "output": "a"}
    b = {"prompt": "q", "output": "b"}
    src = tmp_path / "in.jsonl"
    src.write_text(
        json.dumps(a) + "\n" + '{"broken":' + "\n" + json.dumps(a) + "\n" + json.dumps(b),
        encoding="utf-8",
    )
    out = str(tmp_path / "out.jsonl")
    o = asyncio.run(DedupApplyOp().run(DedupApplyInput(path=str(src), output_path=out), _Ctx()))
    # bad line quarantined (not counted, not written); dup of `a` dropped
    assert (o.total, o.kept, o.dropped) == (3, 2, 1)
    assert _read(out) == [a, b]
    assert (tmp_path / "in.jsonl.quarantine.jsonl").exists()


def test_decontam_apply_copy_through_streams(tmp_path):
    rows = [{"prompt": f"p{i}", "output": f"o{i}"} for i in range(4)]
    src = _write(tmp_path / "train.jsonl", rows)
    out = str(tmp_path / "out.jsonl")
    o = asyncio.run(DecontamApplyOp().run(
        DecontamApplyInput(path=src, output_path=out), _Ctx(),
    ))
    assert (o.total, o.kept, o.checked) == (4, 4, False)
    assert _read(out) == rows


def test_dedup_apply_minhash_two_pass_keeps_earliest(tmp_path):
    pytest.importorskip("datasketch")
    base = "the quick brown fox jumps over the lazy dog again and again in the yard"
    near = base + " today"
    other = "completely different content about training language models on governed rails"
    rows = [{"text": base}, {"text": other}, {"text": near}]
    src = _write(tmp_path / "in.jsonl", rows)
    out = str(tmp_path / "out.jsonl")
    o = asyncio.run(DedupApplyOp().run(DedupApplyInput(
        path=src, output_path=out, method="minhash", threshold=0.5,
    ), _Ctx()))
    assert o.dropped_near == 1 and o.kept == 2
    assert _read(out) == [{"text": base}, {"text": other}]  # earliest survivor, source order


def test_decontam_apply_minhash_streams_against_lsh(tmp_path):
    pytest.importorskip("datasketch")
    leak = {"text": "the eval question about governed fine tuning pipelines and their audit trails"}
    clean = {"text": "an unrelated training row about cooking pasta with tomatoes and basil"}
    src = _write(tmp_path / "train.jsonl", [leak, clean])
    ev = _write(tmp_path / "eval.jsonl", [leak])
    out = str(tmp_path / "out.jsonl")
    o = asyncio.run(DecontamApplyOp().run(DecontamApplyInput(
        path=src, eval_path=ev, output_path=out, method="minhash", overlap_threshold=0.5,
    ), _Ctx()))
    assert (o.dropped, o.kept) == (1, 1)
    assert _read(out) == [clean]
