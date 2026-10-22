"""dataset/replay (#1625, ADR-0095 §2.3) — rehearsal sampling over governed
dataset versions. Hermetic: local JSONL fixtures, no registry, no network.

The properties that matter are the ones a training receipt has to be able to
re-derive: the ratio is respected against the OUTPUT row count, the same seed
reproduces the mix byte-for-byte, the per-source `mix` table matches the rows
actually written, dedup is honoured, and a reserved v2 strategy fails loudly
instead of quietly degrading to uniform.
"""
from __future__ import annotations

import json

import pytest

from cairn.core.operator import OperationContext
from operator_dataset_checks.operators.replay import ReplayOp
from operator_dataset_checks.schemas.replay import ReplayInput


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="replay")


def _write(tmp_path, name: str, rows) -> str:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def _row(tag: str, i: int) -> dict:
    return {"messages": [
        {"role": "user", "content": f"{tag} q{i}"},
        {"role": "assistant", "content": f"{tag} a{i}"},
    ]}


def _read(path: str) -> list[dict]:
    text = open(path, encoding="utf-8").read()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


async def _run(tmp_path, **kwargs) -> tuple:
    out = str(tmp_path / kwargs.pop("out_name", "mixed.jsonl"))
    res = await ReplayOp().run(ReplayInput(out_path=out, **kwargs), _ctx())
    return res, _read(out)


# ── ratio: the denominator is the OUTPUT, not the new slice ──────────────────
async def test_ratio_is_the_share_of_the_final_mix(tmp_path):
    new = _write(tmp_path, "new.jsonl", [_row("new", i) for i in range(80)])
    old = _write(tmp_path, "old.jsonl", [_row("old", i) for i in range(500)])

    res, rows = await _run(tmp_path, new_rows_path=new, sources=[old], ratio=0.2)

    # 80 new rows at a 20% replay share → 20 replay rows, 100 total.
    assert (res.new_rows, res.replay_rows, res.total_rows) == (80, 20, 100)
    assert res.ratio_actual == pytest.approx(0.2, abs=0.01)
    assert len(rows) == 100
    # group ordering: the new block first, then the source's block.
    assert all("new" in r["messages"][0]["content"] for r in rows[:80])
    assert all("old" in r["messages"][0]["content"] for r in rows[80:])


async def test_ratio_zero_writes_the_new_slice_untouched(tmp_path):
    new_rows = [_row("new", i) for i in range(10)]
    new = _write(tmp_path, "new.jsonl", new_rows)
    old = _write(tmp_path, "old.jsonl", [_row("old", i) for i in range(50)])

    res, rows = await _run(tmp_path, new_rows_path=new, sources=[old], ratio=0.0)

    assert (res.replay_rows, res.total_rows, res.ratio_actual) == (0, 10, 0.0)
    assert rows == new_rows
    assert res.mix[0].rows_taken == 0 and res.mix[0].rows_available == 50


# ── determinism: same inputs + seed → byte-identical output ──────────────────
async def test_seed_determinism_is_byte_identical(tmp_path):
    new = _write(tmp_path, "new.jsonl", [_row("new", i) for i in range(40)])
    a = _write(tmp_path, "a.jsonl", [_row("a", i) for i in range(200)])
    b = _write(tmp_path, "b.jsonl", [_row("b", i) for i in range(200)])

    kw = dict(new_rows_path=new, sources=[a, b], ratio=0.25, seed=7)
    first = str(tmp_path / "one.jsonl")
    second = str(tmp_path / "two.jsonl")
    r1 = await ReplayOp().run(ReplayInput(out_path=first, **kw), _ctx())
    r2 = await ReplayOp().run(ReplayInput(out_path=second, **kw), _ctx())

    assert open(first, "rb").read() == open(second, "rb").read()
    assert [m.rows_taken for m in r1.mix] == [m.rows_taken for m in r2.mix]

    # a different seed picks different rows (same counts) — the seed is real.
    other = str(tmp_path / "three.jsonl")
    r3 = await ReplayOp().run(ReplayInput(out_path=other, **{**kw, "seed": 8}), _ctx())
    assert open(other, "rb").read() != open(first, "rb").read()
    assert (r3.replay_rows, r3.total_rows) == (r1.replay_rows, r1.total_rows)


# ── provenance: the mix table must match what was written ────────────────────
async def test_mix_table_matches_the_written_rows(tmp_path):
    new = _write(tmp_path, "new.jsonl", [_row("new", i) for i in range(60)])
    a = _write(tmp_path, "a.jsonl", [_row("a", i) for i in range(100)])
    b = _write(tmp_path, "b.jsonl", [_row("b", i) for i in range(300)])

    res, rows = await _run(tmp_path, new_rows_path=new, sources=[a, b], ratio=0.25)

    assert [m.source for m in res.mix] == [a, b]
    assert [m.rows_available for m in res.mix] == [100, 300]
    assert sum(m.rows_taken for m in res.mix) == res.replay_rows
    # uniform ⇒ share ∝ eligible rows: b (300) contributes 3× a's rows (100).
    taken_a, taken_b = (m.rows_taken for m in res.mix)
    assert (taken_a, taken_b) == (5, 15)
    tags = [r["messages"][0]["content"].split()[0] for r in rows]
    assert tags == ["new"] * 60 + ["a"] * taken_a + ["b"] * taken_b
    # counts and locations only — no row text leaks into the findings payload.
    assert all(isinstance(m.source, str) and "\n" not in m.source for m in res.mix)


async def test_recency_weighted_favours_later_sources(tmp_path):
    new = _write(tmp_path, "new.jsonl", [_row("new", i) for i in range(60)])
    srcs = [
        _write(tmp_path, f"s{i}.jsonl", [_row(f"s{i}", j) for j in range(100)])
        for i in range(3)
    ]

    res, _ = await _run(
        tmp_path, new_rows_path=new, sources=srcs, ratio=0.25,
        strategy="recency_weighted",
    )

    # sources are OLDEST first → weights 1:2:3 over a 20-row quota.
    assert [m.rows_taken for m in res.mix] == [3, 7, 10]
    assert res.strategy_used == "recency_weighted"


# ── small sources: take everything, report the truth, never oversample ───────
async def test_short_sources_report_the_true_ratio(tmp_path):
    new = _write(tmp_path, "new.jsonl", [_row("new", i) for i in range(90)])
    tiny = _write(tmp_path, "tiny.jsonl", [_row("tiny", i) for i in range(4)])

    res, rows = await _run(tmp_path, new_rows_path=new, sources=[tiny], ratio=0.5)

    # 90 new rows at 50% would want 90 replay rows; only 4 exist.
    assert (res.replay_rows, res.total_rows) == (4, 94)
    assert res.ratio_actual == pytest.approx(4 / 94, abs=1e-4)
    assert res.mix[0].rows_taken == res.mix[0].rows_available == 4
    # no duplicates invented to hit the target
    assert len({json.dumps(r, sort_keys=True) for r in rows[90:]}) == 4


async def test_shortfall_spills_onto_the_sources_with_room(tmp_path):
    new = _write(tmp_path, "new.jsonl", [_row("new", i) for i in range(90)])
    tiny = _write(tmp_path, "tiny.jsonl", [_row("tiny", i) for i in range(2)])
    big = _write(tmp_path, "big.jsonl", [_row("big", i) for i in range(500)])

    res, _ = await _run(
        tmp_path, new_rows_path=new, sources=[tiny, big], ratio=0.25,
        strategy="recency_weighted",
    )

    # quota 30; recency weights 1:2 would give tiny 10, but it holds 2 — the
    # other 8 land on `big` rather than shrinking the mix.
    assert [m.rows_taken for m in res.mix] == [2, 28]
    assert res.replay_rows == 30


async def test_no_sources_is_a_passthrough_not_an_error(tmp_path):
    new_rows = [_row("new", i) for i in range(5)]
    new = _write(tmp_path, "new.jsonl", new_rows)

    res, rows = await _run(tmp_path, new_rows_path=new, sources=[], ratio=0.3)

    assert (res.replay_rows, res.total_rows, res.ratio_actual) == (0, 5, 0.0)
    assert res.mix == [] and rows == new_rows


async def test_empty_source_file(tmp_path):
    new = _write(tmp_path, "new.jsonl", [_row("new", i) for i in range(10)])
    empty = _write(tmp_path, "empty.jsonl", [])

    res, rows = await _run(tmp_path, new_rows_path=new, sources=[empty], ratio=0.3)

    assert res.mix[0].rows_available == 0 and res.mix[0].rows_taken == 0
    assert (res.replay_rows, res.total_rows) == (0, 10)
    assert len(rows) == 10


# ── dedup against the new rows ───────────────────────────────────────────────
async def test_dedup_against_new_drops_exact_repeats(tmp_path):
    shared = [_row("new", i) for i in range(10)]
    new = _write(tmp_path, "new.jsonl", shared)
    # the source is the SAME 10 rows plus 10 genuinely older ones.
    old = _write(tmp_path, "old.jsonl", shared + [_row("old", i) for i in range(10)])

    res, rows = await _run(
        tmp_path, new_rows_path=new, sources=[old], ratio=0.5, dedup_against_new=True,
    )

    # only the 10 non-overlapping rows are eligible …
    assert res.mix[0].rows_available == 10
    assert res.replay_rows == 10
    assert all("old" in r["messages"][0]["content"] for r in rows[10:])

    off = await _run(
        tmp_path, out_name="off.jsonl", new_rows_path=new, sources=[old],
        ratio=0.5, dedup_against_new=False,
    )
    # … and with the filter off every row of the source is fair game again.
    assert off[0].mix[0].rows_available == 20


async def test_dedup_normalizes_whitespace_like_dataset_dedup(tmp_path):
    new = _write(tmp_path, "new.jsonl", [{"prompt": "hello there", "completion": "hi"}])
    old = _write(tmp_path, "old.jsonl", [
        {"prompt": "hello   there", "completion": "hi"},   # same after collapsing
        {"prompt": "different", "completion": "row"},
    ])

    res, _ = await _run(tmp_path, new_rows_path=new, sources=[old], ratio=0.5)
    assert res.mix[0].rows_available == 1

    raw, _ = await _run(
        tmp_path, out_name="raw.jsonl", new_rows_path=new, sources=[old],
        ratio=0.5, normalize=False,
    )
    assert raw.mix[0].rows_available == 2


# ── the launch-resolved `#sha256=` pin ───────────────────────────────────────
async def test_resolved_ref_pin_is_stripped_for_reading_and_kept_in_the_mix(tmp_path):
    new = _write(tmp_path, "new.jsonl", [_row("new", i) for i in range(8)])
    old = _write(tmp_path, "old.jsonl", [_row("old", i) for i in range(50)])
    pinned = f"{old}#sha256={'a' * 64}"

    res, rows = await _run(tmp_path, new_rows_path=new, sources=[pinned], ratio=0.2)

    assert res.replay_rows == 2 and len(rows) == 10
    # the receipt keeps the pin — it names the exact version that was mixed in.
    assert res.mix[0].source == pinned


# ── v2 strategies are reserved, never silently downgraded ────────────────────
@pytest.mark.parametrize("strategy", ["hard_example", "forgetting_aware"])
async def test_reserved_strategies_fail_loudly(tmp_path, strategy):
    new = _write(tmp_path, "new.jsonl", [_row("new", i) for i in range(4)])
    old = _write(tmp_path, "old.jsonl", [_row("old", i) for i in range(4)])
    out = str(tmp_path / "never.jsonl")

    with pytest.raises(ValueError, match="RESERVED"):
        await ReplayOp().run(
            ReplayInput(new_rows_path=new, out_path=out, sources=[old],
                        strategy=strategy, eval_report="s3://b/forgetting.json"),
            _ctx(),
        )
    # it refused BEFORE writing anything.
    assert not (tmp_path / "never.jsonl").exists()


async def test_unknown_strategy_is_rejected(tmp_path):
    new = _write(tmp_path, "new.jsonl", [_row("new", i) for i in range(4)])
    with pytest.raises(ValueError, match="unknown strategy"):
        await ReplayOp().run(
            ReplayInput(new_rows_path=new, out_path=str(tmp_path / "x.jsonl"),
                        sources=[], strategy="ewc"),
            _ctx(),
        )


def test_ratio_of_one_is_rejected_by_the_schema():
    # ratio is the share of the OUTPUT, so 1.0 has no finite solution.
    with pytest.raises(Exception):
        ReplayInput(new_rows_path="a.jsonl", out_path="b.jsonl", ratio=1.0)
    with pytest.raises(Exception):
        ReplayInput(new_rows_path="a.jsonl", out_path="b.jsonl", ratio=-0.1)
