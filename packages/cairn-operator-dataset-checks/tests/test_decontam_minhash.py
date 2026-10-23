"""Tests for dataset/decontam's MinHash+LSH contamination method (issue #382).

`method='minhash'` is additive to #381's 'exact'/'ngram'/'both' — it flags
train rows whose MinHash Jaccard vs any eval row >= `overlap_threshold`, found
via an LSH index over the eval set (near-linear vs the ngram path's
O(train×eval) all-pairs Jaccard). It catches paraphrased leakage that exact
hashing misses, at a scale the exact-shingle ngram path can't reach.

The #381 exact/ngram/both paths are covered by `test_decontam.py` and
`test_decontam_ngram.py`, which stay green unchanged. The `[minhash]` extra
(`datasketch`) tests here are gated with `pytest.importorskip("datasketch")` so
hermetic CI SKIPs them; a dedicated test hides `datasketch` and asserts a clear
`RuntimeError`. Determinism: a FIXED MinHash seed makes signatures/buckets
byte-reproducible, and the fixtures are tuned so a reworded leak lands above the
threshold while an unrelated row sits at ~0.
"""
from __future__ import annotations

import json
import sys

import pytest

from cairn.core.operator import OperationContext
from operator_dataset_checks.operators.decontam import DecontamOp
from operator_dataset_checks.schemas.decontam import DecontamInput


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="decontam")


def _write(tmp_path, name, rows) -> str:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def _ex(q, a):
    return {"messages": [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


_EVAL = _ex(
    "What is the capital of France and why is it historically significant",
    "Paris is the capital of France and has long been culturally important",
)
_REWORD = _ex(  # one word changed vs _EVAL → different hash, near-identical shingles
    "What is the capital of France and why is it historically important",
    "Paris is the capital of France and has long been culturally important",
)
_UNRELATED = _ex(
    "Compute the derivative of x squared with respect to the variable x",
    "The derivative equals two times x by the standard power rule of calculus",
)


# ── hermetic: minhash without the extra → clear error ───────────────────────
async def test_minhash_method_raises_without_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "datasketch", None)
    with pytest.raises(RuntimeError, match=r"\[minhash\]"):
        await DecontamOp().run(
            DecontamInput(
                path=_write(tmp_path, "train.jsonl", [_UNRELATED, _REWORD]),
                eval_path=_write(tmp_path, "eval.jsonl", [_EVAL]),
                method="minhash",
            ),
            _ctx(),
        )


async def test_both_method_stays_hermetic_without_extra(monkeypatch, tmp_path):
    """decontam's 'both' means exact+ngram (no datasketch) — must NOT raise
    just because the [minhash] extra is absent (regression guard for the
    _minhash.resolve('both') overlap with dedup's vocabulary)."""
    monkeypatch.setitem(sys.modules, "datasketch", None)
    out = await DecontamOp().run(
        DecontamInput(
            path=_write(tmp_path, "train.jsonl", [_UNRELATED, _REWORD]),
            eval_path=_write(tmp_path, "eval.jsonl", [_EVAL]),
            method="both",
            ngram=3,
            overlap_threshold=0.5,
        ),
        _ctx(),
    )
    assert out.checked is True
    # ngram path still catches the reworded leak; minhash list stays empty.
    assert out.ngram_contaminated_rows == [1]
    assert out.minhash_contaminated_rows == []


# ── MinHash path (needs datasketch) ─────────────────────────────────────────
async def test_minhash_catches_reworded_leak_exact_misses(tmp_path):
    pytest.importorskip("datasketch")
    train = [_UNRELATED, _REWORD]
    eval_rows = [_EVAL]

    # Exact misses the reworded leak.
    exact = await DecontamOp().run(
        DecontamInput(
            path=_write(tmp_path, "train.jsonl", train),
            eval_path=_write(tmp_path, "eval.jsonl", eval_rows),
            method="exact",
        ),
        _ctx(),
    )
    assert exact.contaminated == 0

    out = await DecontamOp().run(
        DecontamInput(
            path=_write(tmp_path, "train.jsonl", train),
            eval_path=_write(tmp_path, "eval.jsonl", eval_rows),
            method="minhash",
            ngram=3,
            overlap_threshold=0.5,
        ),
        _ctx(),
    )
    assert out.checked is True
    assert out.contaminated == 1
    assert out.contaminated_rows == []  # exact list untouched under 'minhash'
    assert out.ngram_contaminated_rows == []  # ngram list untouched too
    assert out.minhash_contaminated_rows == [1]  # only the reworded leak


async def test_minhash_no_false_positive_on_unrelated_only(tmp_path):
    pytest.importorskip("datasketch")
    train = [_UNRELATED]
    out = await DecontamOp().run(
        DecontamInput(
            path=_write(tmp_path, "train.jsonl", train),
            eval_path=_write(tmp_path, "eval.jsonl", [_EVAL]),
            method="minhash",
            ngram=3,
            overlap_threshold=0.5,
        ),
        _ctx(),
    )
    assert out.minhash_contaminated_rows == []
    assert out.contaminated == 0


async def test_minhash_threshold_sensitivity(tmp_path):
    pytest.importorskip("datasketch")
    train = [_UNRELATED, _REWORD]
    eval_rows = [_EVAL]
    # A 1-word edit sits below 0.9 Jaccard → NOT flagged at a high threshold.
    strict = await DecontamOp().run(
        DecontamInput(
            path=_write(tmp_path, "train.jsonl", train),
            eval_path=_write(tmp_path, "eval.jsonl", eval_rows),
            method="minhash",
            ngram=3,
            overlap_threshold=0.9,
        ),
        _ctx(),
    )
    assert strict.minhash_contaminated_rows == []
    assert strict.contaminated == 0


async def test_minhash_catches_exact_leak_too(tmp_path):
    pytest.importorskip("datasketch")
    leak = _ex("capital of France question here for the leak", "the leaked answer is Paris")
    train = [_UNRELATED, leak]
    out = await DecontamOp().run(
        DecontamInput(
            path=_write(tmp_path, "train.jsonl", train),
            eval_path=_write(tmp_path, "eval.jsonl", [leak]),
            method="minhash",
            ngram=3,
            overlap_threshold=0.5,
        ),
        _ctx(),
    )
    # A byte-identical leak is trivially a MinHash near-dup (Jaccard 1.0).
    assert out.minhash_contaminated_rows == [1]


async def test_minhash_no_eval_set_skips(tmp_path):
    pytest.importorskip("datasketch")
    out = await DecontamOp().run(
        DecontamInput(
            path=_write(tmp_path, "train.jsonl", [_ex("a", "A")]),
            method="minhash",
        ),
        _ctx(),
    )
    assert out.checked is False
    assert out.minhash_contaminated_rows == []


async def test_minhash_is_deterministic_across_runs(tmp_path):
    pytest.importorskip("datasketch")
    train = [_UNRELATED, _REWORD]
    tp = _write(tmp_path, "train.jsonl", train)
    ep = _write(tmp_path, "eval.jsonl", [_EVAL])
    kw = dict(path=tp, eval_path=ep, method="minhash", ngram=3, overlap_threshold=0.5)
    a = await DecontamOp().run(DecontamInput(**kw), _ctx())
    b = await DecontamOp().run(DecontamInput(**kw), _ctx())
    assert a.minhash_contaminated_rows == b.minhash_contaminated_rows == [1]
