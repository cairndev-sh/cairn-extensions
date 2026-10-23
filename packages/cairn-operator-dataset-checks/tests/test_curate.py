"""Tests for dataset/curate — LLM-judge quality selection (#384).

FULLY HERMETIC: no network, no real model. The judge chokepoint
``cairn.core.llm.client.llm_complete`` is imported into the operator module as a
top-level symbol, so we monkeypatch it THERE
(``operator_dataset_checks.operators.curate.llm_complete``) with a canned async
stub that returns litellm/openai-shaped responses whose text carries a known
score keyed off the example's content. That lets us assert exactly which rows are
kept vs dropped, that threshold and keep_fraction both work, that parse failures
are counted (not kept), that ``max_samples`` caps the paid calls, and that the
report numbers are correct.

The single ``@pytest.mark.expensive`` test at the bottom is the only one that
would hit a real model; it is skipped unless a live key is present.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from cairn.core.operator import OperationContext
from operator_dataset_checks.operators import curate as curate_mod
from operator_dataset_checks.operators.curate import CurateOp
from operator_dataset_checks.schemas.curate import CurateInput


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="curate")


def _resp(text: str) -> SimpleNamespace:
    """A minimal litellm/openai-shaped completion carrying ``text``."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _ex(q: str, a: str) -> dict:
    return {"messages": [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


def _write(tmp_path, rows) -> str:
    p = tmp_path / "ds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def _patch_scores(monkeypatch, score_for):
    """Patch llm_complete to return a score derived from the user message.

    ``score_for(user_content, is_complexity)`` -> the raw LLM reply text.
    """
    calls: list[dict] = []

    async def fake_llm_complete(messages, model, temperature=0.0, **kwargs):
        system = messages[0]["content"]
        user = messages[1]["content"]
        is_complexity = "COMPLEXITY" in system
        calls.append({"user": user, "complexity": is_complexity})
        return _resp(score_for(user, is_complexity))

    monkeypatch.setattr(curate_mod, "llm_complete", fake_llm_complete)
    return calls


# --- AlpaGasus: threshold selection ----------------------------------------

async def test_alpagasus_threshold_keeps_high_drops_low(tmp_path, monkeypatch):
    rows = [
        _ex("good q", "excellent answer"),   # -> 5
        _ex("meh q", "weak answer"),          # -> 2
        _ex("great q", "strong answer"),      # -> 4.5
    ]
    # Score by a marker word in the response text.
    def score_for(user, is_complexity):
        if "excellent" in user:
            return "Score: 5"
        if "strong" in user:
            return "4.5"
        return "2"

    _patch_scores(monkeypatch, score_for)
    out_path = str(tmp_path / "out.jsonl")
    out = await CurateOp().run(
        CurateInput(
            path=_write(tmp_path, rows),
            output_path=out_path,
            method="alpagasus",
            threshold=4.5,
        ),
        _ctx(),
    )

    assert out.method == "alpagasus"
    assert out.criterion == "threshold>=4.5"
    assert out.scored == 3
    assert out.kept == 2  # 5 and 4.5 clear the bar; 2 dropped
    assert out.dropped == 1
    assert out.parse_failures == 0
    # Curated file holds exactly the kept rows, verbatim.
    written = [json.loads(x) for x in open(out_path).read().splitlines() if x]
    assert len(written) == 2
    assert rows[0] in written and rows[2] in written and rows[1] not in written
    # Distribution buckets by integer floor.
    assert out.score_distribution == {"5": 1, "4": 1, "2": 1}
    assert out.mean_score == pytest.approx((5 + 2 + 4.5) / 3, rel=1e-3)


async def test_alpagasus_default_threshold_is_4_5(tmp_path, monkeypatch):
    rows = [_ex("q1", "a1"), _ex("q2", "a2")]
    _patch_scores(monkeypatch, lambda user, c: "4.6" if "a1" in user else "4.4")
    out = await CurateOp().run(
        CurateInput(path=_write(tmp_path, rows), output_path=str(tmp_path / "o.jsonl")),
        _ctx(),
    )
    assert out.criterion == "threshold>=4.5"
    assert out.kept == 1  # 4.6 kept, 4.4 dropped


# --- keep_fraction selection -----------------------------------------------

async def test_keep_fraction_keeps_top_slice(tmp_path, monkeypatch):
    # Scores 1..5; keep_fraction 0.4 of 5 rows -> top 2.
    rows = [_ex(f"q{i}", f"a{i}") for i in range(5)]
    scores = {"a0": "1", "a1": "2", "a2": "3", "a3": "4", "a4": "5"}
    _patch_scores(monkeypatch, lambda user, c: next(v for k, v in scores.items() if k in user))
    out = await CurateOp().run(
        CurateInput(
            path=_write(tmp_path, rows),
            output_path=str(tmp_path / "o.jsonl"),
            keep_fraction=0.4,
        ),
        _ctx(),
    )
    assert out.criterion == "keep_fraction=0.4"
    assert out.kept == 2  # top 2 of 5 by score (5 and 4)
    kept_scores = sorted(r.score for r in out.scores if r.kept)
    assert kept_scores == [4.0, 5.0]


# --- parse-failure handling ------------------------------------------------

async def test_parse_failure_counted_and_not_kept(tmp_path, monkeypatch):
    rows = [_ex("q1", "a1"), _ex("q2", "a2")]
    # Second row yields no parseable number.
    _patch_scores(
        monkeypatch,
        lambda user, c: "5" if "a1" in user else "not a number at all",
    )
    out = await CurateOp().run(
        CurateInput(
            path=_write(tmp_path, rows),
            output_path=str(tmp_path / "o.jsonl"),
            threshold=1.0,
        ),
        _ctx(),
    )
    assert out.parse_failures == 1
    assert out.scored == 1  # only the parseable row is counted as scored
    assert out.kept == 1
    assert out.dropped == 0


# --- max_samples cap -------------------------------------------------------

async def test_max_samples_caps_scoring(tmp_path, monkeypatch):
    rows = [_ex(f"q{i}", f"a{i}") for i in range(10)]
    calls = _patch_scores(monkeypatch, lambda user, c: "5")
    out = await CurateOp().run(
        CurateInput(
            path=_write(tmp_path, rows),
            output_path=str(tmp_path / "o.jsonl"),
            max_samples=3,
            threshold=1.0,
        ),
        _ctx(),
    )
    assert out.total_rows == 10
    assert out.extractable == 10
    assert out.scored == 3  # only 3 examples scored despite 10 available
    assert len(calls) == 3  # exactly 3 paid calls made
    assert out.kept == 3


# --- DEITA: complexity * quality -------------------------------------------

async def test_deita_uses_complexity_times_quality(tmp_path, monkeypatch):
    rows = [_ex("q1", "a1"), _ex("q2", "a2")]
    # a1: quality 4, complexity 3 -> 12 ; a2: quality 5, complexity 1 -> 5
    def score_for(user, is_complexity):
        if "a1" in user:
            return "3" if is_complexity else "4"
        return "1" if is_complexity else "5"

    calls = _patch_scores(monkeypatch, score_for)
    out = await CurateOp().run(
        CurateInput(
            path=_write(tmp_path, rows),
            output_path=str(tmp_path / "o.jsonl"),
            method="deita",
            threshold=10.0,
        ),
        _ctx(),
    )
    assert out.method == "deita"
    assert len(calls) == 4  # two calls (quality + complexity) per row
    by_idx = {r.index: r for r in out.scores}
    assert by_idx[0].score == pytest.approx(12.0)
    assert by_idx[0].quality == 4.0 and by_idx[0].complexity == 3.0
    assert by_idx[1].score == pytest.approx(5.0)
    assert out.kept == 1  # only score 12 clears the >=10 bar


async def test_unknown_method_raises(tmp_path, monkeypatch):
    _patch_scores(monkeypatch, lambda user, c: "5")
    with pytest.raises(ValueError):
        await CurateOp().run(
            CurateInput(
                path=_write(tmp_path, [_ex("q", "a")]),
                output_path=str(tmp_path / "o.jsonl"),
                method="bogus",
            ),
            _ctx(),
        )


async def test_score_clamped_to_0_5(tmp_path, monkeypatch):
    rows = [_ex("q1", "a1")]
    _patch_scores(monkeypatch, lambda user, c: "9.9")  # out of range
    out = await CurateOp().run(
        CurateInput(path=_write(tmp_path, rows), output_path=str(tmp_path / "o.jsonl")),
        _ctx(),
    )
    assert out.scores[0].score == 5.0  # clamped to the 0-5 ceiling


# --- live model (skipped in hermetic CI) -----------------------------------

def _has_live_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key) and "dummy" not in key and "test" not in key


@pytest.mark.expensive
@pytest.mark.skipif(not _has_live_key(), reason="no live LLM key; hermetic run")
async def test_curate_live_smoke(tmp_path):
    rows = [_ex("What is 2+2?", "4"), _ex("Explain gravity", "Things fall.")]
    out = await CurateOp().run(
        CurateInput(
            path=_write(tmp_path, rows),
            output_path=str(tmp_path / "o.jsonl"),
            method="alpagasus",
            max_samples=2,
        ),
        _ctx(),
    )
    assert out.scored == 2
