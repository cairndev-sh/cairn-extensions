"""Tests for dataset/decontam's n-gram/shingle overlap method (issue #381).

The default (`method='exact'`) path is covered by the pre-existing golden
tests in `tests/operators/dataset_checks/test_decontam.py`, which must stay
green unchanged. These tests exercise the NEW opt-in `ngram`/`both` methods,
which catch near-duplicate (reworded) contamination that exact hashing misses.
No optional dependency is involved here — `_ngram.py` is pure stdlib — so
these tests always run (nothing to skip).
"""
from __future__ import annotations

import json

from cairn.core.operator import OperationContext
from operator_dataset_checks._ngram import jaccard, row_text, shingles
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


# ── _ngram.py unit tests ────────────────────────────────────────────────
def test_shingles_basic_window():
    s = shingles("the quick brown fox jumps", 3)
    assert ("the", "quick", "brown") in s
    assert ("quick", "brown", "fox") in s
    assert ("brown", "fox", "jumps") in s
    assert len(s) == 3


def test_shingles_short_text_yields_one_shingle():
    s = shingles("hi there", 5)
    assert s == {("hi", "there")}


def test_shingles_empty_text_is_empty_set():
    assert shingles("", 3) == set()


def test_jaccard_identical_sets_is_one():
    s = shingles("the quick brown fox", 2)
    assert jaccard(s, s) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    a = shingles("apples and oranges", 2)
    b = shingles("completely unrelated content", 2)
    assert jaccard(a, b) == 0.0


def test_jaccard_both_empty_is_zero_not_one():
    assert jaccard(set(), set()) == 0.0


def test_row_text_flattens_chat_format():
    row = _ex("what is 2+2?", "4")
    text = row_text(row)
    assert "what is 2+2?" in text
    assert "4" in text


# ── operator-level: method='exact' behaves exactly as before (regression) ──
async def test_exact_is_still_the_default_method(tmp_path):
    leak = _ex("capital of France?", "Paris")
    train = [_ex("foo", "bar"), leak]
    eval_rows = [leak]
    out = await DecontamOp().run(
        DecontamInput(
            path=_write(tmp_path, "train.jsonl", train),
            eval_path=_write(tmp_path, "eval.jsonl", eval_rows),
        ),
        _ctx(),
    )
    assert out.checked is True
    assert out.contaminated == 1
    assert out.contaminated_rows == [1]
    assert out.ngram_contaminated_rows == []


# ── operator-level: method='ngram' catches near-duplicates exact misses ────
async def test_ngram_method_catches_reworded_leak_exact_hash_misses(tmp_path):
    eval_row = _ex(
        "What is the capital of France, and why is it significant historically?",
        "Paris is the capital of France and has been significant for centuries.",
    )
    # Same content, trivially reworded (punctuation + one word) — a different
    # canonical_hash, but near-identical n-gram shingle sets.
    reworded_train_row = _ex(
        "What is the capital of France and why is it significant historically",
        "Paris is the capital of France and has been significant for centuries!",
    )
    unrelated_train_row = _ex("2+2?", "4")
    train = [unrelated_train_row, reworded_train_row]
    eval_rows = [eval_row]

    exact_out = await DecontamOp().run(
        DecontamInput(
            path=_write(tmp_path, "train.jsonl", train),
            eval_path=_write(tmp_path, "eval.jsonl", eval_rows),
            method="exact",
        ),
        _ctx(),
    )
    assert exact_out.contaminated == 0  # exact hash misses the reworded leak

    ngram_out = await DecontamOp().run(
        DecontamInput(
            path=_write(tmp_path, "train.jsonl", train),
            eval_path=_write(tmp_path, "eval.jsonl", eval_rows),
            method="ngram",
            ngram=3,
            overlap_threshold=0.5,
        ),
        _ctx(),
    )
    assert ngram_out.checked is True
    assert ngram_out.contaminated == 1
    assert ngram_out.contaminated_rows == []  # exact list untouched under 'ngram'
    assert ngram_out.ngram_contaminated_rows == [1]


async def test_ngram_method_threshold_excludes_dissimilar_rows(tmp_path):
    eval_row = _ex("a b c d e f g h i j", "answer one two three four five")
    dissimilar_row = _ex("completely different words here now", "another distinct reply text")
    train = [dissimilar_row]
    out = await DecontamOp().run(
        DecontamInput(
            path=_write(tmp_path, "train.jsonl", train),
            eval_path=_write(tmp_path, "eval.jsonl", [eval_row]),
            method="ngram",
            ngram=3,
            overlap_threshold=0.8,
        ),
        _ctx(),
    )
    assert out.ngram_contaminated_rows == []
    assert out.contaminated == 0


async def test_both_method_unions_exact_and_ngram_without_double_counting(tmp_path):
    exact_leak = _ex("capital of France?", "Paris")
    eval_reword = _ex(
        "What is the capital of France, and why is it significant historically?",
        "Paris is the capital of France and has been significant for centuries.",
    )
    reworded_train_row = _ex(
        "What is the capital of France and why is it significant historically",
        "Paris is the capital of France and has been significant for centuries!",
    )
    train = [exact_leak, reworded_train_row, _ex("2+2?", "4")]
    eval_rows = [exact_leak, eval_reword]

    out = await DecontamOp().run(
        DecontamInput(
            path=_write(tmp_path, "train.jsonl", train),
            eval_path=_write(tmp_path, "eval.jsonl", eval_rows),
            method="both",
            ngram=3,
            overlap_threshold=0.5,
        ),
        _ctx(),
    )
    assert out.contaminated_rows == [0]
    assert out.ngram_contaminated_rows == [1]
    assert out.contaminated == 2
    assert out.overlap_pct == round(2 / 3 * 100, 1)


async def test_no_eval_set_skips_ngram_too(tmp_path):
    train = [_ex("a", "A")]
    out = await DecontamOp().run(
        DecontamInput(
            path=_write(tmp_path, "train.jsonl", train), method="both",
        ),
        _ctx(),
    )
    assert out.checked is False
    assert out.ngram_contaminated_rows == []
