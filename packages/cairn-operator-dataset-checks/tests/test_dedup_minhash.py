"""Tests for dataset/dedup's MinHash+LSH near-duplicate method (issue #382).

The default (`method='exact'`) path is covered by the pre-existing golden tests
in `tests/operators/dataset_checks/test_dedup.py`, which MUST stay byte-identical
green — a regression test here re-asserts the exact output is unchanged.

The MinHash path needs the `[minhash]` extra (`datasketch`), so those tests are
gated with `pytest.importorskip("datasketch")` and SKIP in a hermetic CI run
that never installs the extra. A dedicated test hides `datasketch` via
`sys.modules` and asserts `method='minhash'` raises a clear `RuntimeError` —
mirroring the `[fastpii]`/presidio-absent behaviour — so the failure mode is
proven even without the extra installed.

Determinism: `_minhash.build_minhash` uses a FIXED seed, so the MinHash
signatures — and therefore the LSH buckets and the reported near-dup clusters —
are byte-reproducible. The fixtures below are tuned (num_perm=128, ngram=3) so a
1-word-edited copy lands comfortably above the 0.5 threshold while an unrelated
row sits at ~0 similarity — no flakiness from LSH's probabilistic nature.
"""
from __future__ import annotations

import json
import sys

import pytest

from cairn.core.operator import OperationContext
from operator_dataset_checks.operators.dedup import DedupOp
from operator_dataset_checks.schemas.dedup import DedupInput


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="dedup")


def _write(tmp_path, rows) -> str:
    p = tmp_path / "ds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def _ex(q, a):
    return {"messages": [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


# Base row + a 1-word-edited near-duplicate (>~0.5 shingle Jaccard) + an
# unrelated row (~0 similarity). Long enough that a single word swap barely
# moves the shingle set.
_BASE = _ex(
    "What is the capital of France and why is it historically significant",
    "Paris is the capital of France and has long been culturally important",
)
_NEAR = _ex(  # "significant" -> "important" in the question only
    "What is the capital of France and why is it historically important",
    "Paris is the capital of France and has long been culturally important",
)
_FAR = _ex(
    "Compute the derivative of x squared with respect to the variable x",
    "The derivative equals two times x by the standard power rule of calculus",
)


# ── hermetic: minhash requested without the extra → clear error ─────────────
async def test_minhash_method_raises_without_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "datasketch", None)
    with pytest.raises(RuntimeError, match=r"\[minhash\]"):
        await DedupOp().run(
            DedupInput(path=_write(tmp_path, [_BASE, _NEAR]), method="minhash"),
            _ctx(),
        )


async def test_both_method_also_raises_without_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "datasketch", None)
    with pytest.raises(RuntimeError, match=r"\[minhash\]"):
        await DedupOp().run(
            DedupInput(path=_write(tmp_path, [_BASE, _NEAR]), method="both"),
            _ctx(),
        )


# ── regression: exact stays the default and byte-identical to pre-#382 ──────
async def test_exact_is_default_and_reports_no_near_dups(tmp_path):
    dup = _ex("a", "A")
    rows = [dup, _ex("b", "B"), dup]
    out = await DedupOp().run(DedupInput(path=_write(tmp_path, rows)), _ctx())
    assert out.total == 3
    assert out.unique == 2
    assert out.exact_duplicates == 1
    assert out.duplicate_pct == 33.3
    assert out.duplicate_groups[0].rows == [0, 2]
    # New fields are inert under the default method.
    assert out.near_duplicate_groups == []
    assert out.near_duplicates == 0


# ── MinHash path (needs datasketch) ─────────────────────────────────────────
async def test_minhash_catches_reworded_near_dup_exact_misses(tmp_path):
    pytest.importorskip("datasketch")
    rows = [_BASE, _FAR, _NEAR]  # 0 ~ 2 near-dup; 1 unrelated

    # Exact mode: the 1-word edit hashes differently → NOT a duplicate.
    exact_out = await DedupOp().run(DedupInput(path=_write(tmp_path, rows)), _ctx())
    assert exact_out.exact_duplicates == 0
    assert exact_out.near_duplicate_groups == []

    out = await DedupOp().run(
        DedupInput(
            path=_write(tmp_path, rows),
            method="minhash",
            threshold=0.5,
            ngram=3,
        ),
        _ctx(),
    )
    assert out.exact_duplicates == 0  # exact untouched under 'minhash'
    # The near-dup pair (0, 2) is clustered; the unrelated row (1) is NOT.
    assert [g.rows for g in out.near_duplicate_groups] == [[0, 2]]
    assert out.near_duplicates == 1
    assert 1 not in {i for g in out.near_duplicate_groups for i in g.rows}


async def test_minhash_threshold_sensitivity_high_threshold_excludes_edit(tmp_path):
    pytest.importorskip("datasketch")
    rows = [_BASE, _FAR, _NEAR]
    # A 1-word edit sits ~0.6-0.7 Jaccard: below 0.9 it is NO LONGER a near-dup.
    out = await DedupOp().run(
        DedupInput(
            path=_write(tmp_path, rows),
            method="minhash",
            threshold=0.9,
            ngram=3,
        ),
        _ctx(),
    )
    assert out.near_duplicate_groups == []
    assert out.near_duplicates == 0


async def test_minhash_no_false_positive_on_all_unrelated(tmp_path):
    pytest.importorskip("datasketch")
    rows = [
        _ex("what is two plus two", "the answer is four exactly"),
        _ex("name a primary color of light", "red is a primary color of light"),
        _ex("who wrote the novel moby dick", "herman melville wrote moby dick"),
    ]
    out = await DedupOp().run(
        DedupInput(path=_write(tmp_path, rows), method="minhash", threshold=0.5, ngram=3),
        _ctx(),
    )
    assert out.near_duplicate_groups == []
    assert out.near_duplicates == 0


async def test_minhash_clusters_exact_dups_too(tmp_path):
    pytest.importorskip("datasketch")
    dup = _ex("identical row content here for clustering", "same answer text repeated")
    rows = [dup, _FAR, dup]  # 0 and 2 are byte-identical → also near-dups
    out = await DedupOp().run(
        DedupInput(path=_write(tmp_path, rows), method="minhash", threshold=0.5, ngram=3),
        _ctx(),
    )
    assert [g.rows for g in out.near_duplicate_groups] == [[0, 2]]
    assert out.near_duplicates == 1


async def test_both_method_reports_exact_and_near_together(tmp_path):
    pytest.importorskip("datasketch")
    dup = _ex("identical row content here for clustering", "same answer text repeated")
    rows = [dup, _BASE, _NEAR, dup]  # exact: (0,3); near: (1,2) + (0,3)
    out = await DedupOp().run(
        DedupInput(path=_write(tmp_path, rows), method="both", threshold=0.5, ngram=3),
        _ctx(),
    )
    # Exact report unchanged from the default path.
    assert out.exact_duplicates == 1
    assert out.duplicate_groups[0].rows == [0, 3]
    # Near-dup report has BOTH the exact-dup cluster and the reworded cluster.
    near = sorted(g.rows for g in out.near_duplicate_groups)
    assert [0, 3] in near
    assert [1, 2] in near


async def test_minhash_result_is_deterministic_across_runs(tmp_path):
    pytest.importorskip("datasketch")
    rows = [_BASE, _FAR, _NEAR]
    path = _write(tmp_path, rows)
    a = await DedupOp().run(
        DedupInput(path=path, method="minhash", threshold=0.5, ngram=3), _ctx(),
    )
    b = await DedupOp().run(
        DedupInput(path=path, method="minhash", threshold=0.5, ngram=3), _ctx(),
    )
    assert [g.rows for g in a.near_duplicate_groups] == [g.rows for g in b.near_duplicate_groups]
