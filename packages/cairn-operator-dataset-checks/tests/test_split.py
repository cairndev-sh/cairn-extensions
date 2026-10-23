"""Tests for dataset/split (#383) — deterministic, group-aware, optionally
stratified train/val[/test] split.

The hash-based fallback (`backend='hash'`) is pure stdlib and MUST pass in a
base install with no extras — those tests need no `importorskip`. The
sklearn-backed path needs the `[split]` extra, so those tests are gated with
`pytest.importorskip("sklearn")` and skip cleanly in a hermetic CI run.
`backend='auto'` is exercised indirectly: with sklearn hidden it falls back
to hash automatically (asserted explicitly below), which is also how we prove
the whole pack stays hermetic end-to-end.
"""
from __future__ import annotations

import json
import sys

import pytest

from cairn.core.operator import OperationContext
from operator_dataset_checks.operators.split import SplitOp
from operator_dataset_checks.schemas.split import SplitInput


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="split")


def _write(tmp_path, rows, name="ds.jsonl") -> str:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def _row(i, *, group=None, label=None):
    row = {"prompt": f"q{i}", "completion": f"a{i}"}
    if group is not None:
        row["doc_id"] = group
    if label is not None:
        row["source"] = label
    return row


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ── hash fallback: hermetic, no extra needed ────────────────────────────────

async def test_hash_backend_splits_and_writes_files(tmp_path):
    rows = [_row(i) for i in range(200)]
    out_dir = tmp_path / "out"
    out = await SplitOp().run(
        SplitInput(
            path=_write(tmp_path, rows),
            output_dir=str(out_dir),
            seed=13,
            train_ratio=0.8,
            val_ratio=0.2,
            backend="hash",
        ),
        _ctx(),
    )
    assert out.backend_used == "hash"
    assert out.total == 200
    assert set(out.counts.keys()) == {"train", "val"}
    assert out.counts["train"] + out.counts["val"] == 200
    # Ratios achieved should be close to requested (hash bucketing over 200
    # distinct keys should land within a reasonable band of 0.8/0.2).
    assert 0.65 <= out.ratios_achieved["train"] <= 0.95
    assert out.no_leakage is True

    assert set(out.output_paths.keys()) == {"train", "val"}
    train_rows = _read_jsonl(out.output_paths["train"])
    val_rows = _read_jsonl(out.output_paths["val"])
    assert len(train_rows) == out.counts["train"]
    assert len(val_rows) == out.counts["val"]
    # No row lost or duplicated across the two files.
    all_written = {json.dumps(r, sort_keys=True) for r in train_rows + val_rows}
    all_source = {json.dumps(r, sort_keys=True) for r in rows}
    assert all_written == all_source


async def test_test_ratio_zero_omits_test_file(tmp_path):
    rows = [_row(i) for i in range(50)]
    out = await SplitOp().run(
        SplitInput(
            path=_write(tmp_path, rows),
            output_dir=str(tmp_path / "out"),
            backend="hash",
        ),
        _ctx(),
    )
    assert "test" not in out.counts
    assert "test" not in out.output_paths


async def test_test_ratio_positive_creates_three_splits(tmp_path):
    rows = [_row(i) for i in range(300)]
    out = await SplitOp().run(
        SplitInput(
            path=_write(tmp_path, rows),
            output_dir=str(tmp_path / "out"),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            backend="hash",
        ),
        _ctx(),
    )
    assert set(out.counts.keys()) == {"train", "val", "test"}
    assert sum(out.counts.values()) == 300
    assert set(out.output_paths.keys()) == {"train", "val", "test"}


# ── determinism ──────────────────────────────────────────────────────────

async def test_same_seed_twice_gives_identical_split(tmp_path):
    rows = [_row(i) for i in range(120)]
    path = _write(tmp_path, rows)

    out_a = await SplitOp().run(
        SplitInput(path=path, output_dir=str(tmp_path / "a"), seed=42, backend="hash"),
        _ctx(),
    )
    out_b = await SplitOp().run(
        SplitInput(path=path, output_dir=str(tmp_path / "b"), seed=42, backend="hash"),
        _ctx(),
    )
    assert out_a.counts == out_b.counts
    a_train = _read_jsonl(out_a.output_paths["train"])
    b_train = _read_jsonl(out_b.output_paths["train"])
    assert a_train == b_train


async def test_different_seed_gives_different_split(tmp_path):
    rows = [_row(i) for i in range(120)]
    path = _write(tmp_path, rows)

    out_a = await SplitOp().run(
        SplitInput(path=path, output_dir=str(tmp_path / "a"), seed=1, backend="hash"),
        _ctx(),
    )
    out_b = await SplitOp().run(
        SplitInput(path=path, output_dir=str(tmp_path / "b"), seed=2, backend="hash"),
        _ctx(),
    )
    a_train = {json.dumps(r, sort_keys=True) for r in _read_jsonl(out_a.output_paths["train"])}
    b_train = {json.dumps(r, sort_keys=True) for r in _read_jsonl(out_b.output_paths["train"])}
    assert a_train != b_train


# ── group-aware: no leakage ──────────────────────────────────────────────

async def test_group_by_keeps_group_in_one_split_hash_backend(tmp_path):
    # 40 groups of 5 rows each — if grouping is honored, every group of 5
    # lands entirely in train or entirely in val, never split across both.
    rows = []
    for g in range(40):
        for _ in range(5):
            rows.append(_row(len(rows), group=f"doc-{g}"))

    out = await SplitOp().run(
        SplitInput(
            path=_write(tmp_path, rows),
            output_dir=str(tmp_path / "out"),
            train_ratio=0.75,
            val_ratio=0.25,
            group_by="doc_id",
            backend="hash",
        ),
        _ctx(),
    )
    assert out.no_leakage is True
    assert out.group_by == "doc_id"
    assert out.group_count == 40

    train_rows = _read_jsonl(out.output_paths["train"])
    val_rows = _read_jsonl(out.output_paths["val"])
    train_groups = {r["doc_id"] for r in train_rows}
    val_groups = {r["doc_id"] for r in val_rows}
    assert train_groups.isdisjoint(val_groups)

    # Every doc_id's rows are entirely on one side.
    for g in range(40):
        key = f"doc-{g}"
        count_in_train = sum(1 for r in train_rows if r["doc_id"] == key)
        count_in_val = sum(1 for r in val_rows if r["doc_id"] == key)
        assert count_in_train == 0 or count_in_val == 0
        assert count_in_train + count_in_val == 5


async def test_group_by_too_few_groups_for_3way_split_does_not_crash(tmp_path):
    # Regression: a low-cardinality group_by (5 groups) + a 3-way split sent the
    # sklearn path into GroupShuffleSplit with <2 groups left after train, raising
    # "n_samples=1, test_size=0.5 → empty train set". The group-count guard must
    # keep the run alive: a valid partition, group-locality intact, no crash.
    pytest.importorskip("sklearn")
    rows = []
    for g in range(5):
        for _ in range(200):
            rows.append(_row(len(rows), group=f"src-{g}"))

    out = await SplitOp().run(
        SplitInput(
            path=_write(tmp_path, rows),
            output_dir=str(tmp_path / "out"),
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            group_by="doc_id",
            backend="sklearn",
        ),
        _ctx(),
    )
    assert out.total == 1000
    assert sum(out.counts.values()) == 1000  # every row assigned exactly once
    assert out.no_leakage is True
    # each group lands wholly in one split (no straddling)
    seen: set[str] = set()
    for path in out.output_paths.values():
        groups = {r["doc_id"] for r in _read_jsonl(path)}
        assert seen.isdisjoint(groups)
        seen |= groups
    assert seen == {f"src-{g}" for g in range(5)}


async def test_no_group_by_means_every_row_its_own_group(tmp_path):
    rows = [_row(i) for i in range(30)]
    out = await SplitOp().run(
        SplitInput(path=_write(tmp_path, rows), output_dir=str(tmp_path / "out"), backend="hash"),
        _ctx(),
    )
    assert out.group_by is None
    assert out.group_count is None
    assert out.no_leakage is True


# ── stratification (hash fallback, best-effort) ─────────────────────────

async def test_stratify_by_reports_per_stratum_summary(tmp_path):
    rows = []
    for i in range(100):
        label = "a" if i % 2 == 0 else "b"
        rows.append(_row(i, label=label))

    out = await SplitOp().run(
        SplitInput(
            path=_write(tmp_path, rows),
            output_dir=str(tmp_path / "out"),
            train_ratio=0.8,
            val_ratio=0.2,
            stratify_by="source",
            backend="hash",
        ),
        _ctx(),
    )
    assert out.stratify_by == "source"
    labels = {s.label for s in out.stratification_summary}
    assert labels == {"a", "b"}
    for stratum in out.stratification_summary:
        total_in_stratum = sum(stratum.counts.values())
        assert total_in_stratum == 50  # 50 'a' rows, 50 'b' rows


# ── input validation ─────────────────────────────────────────────────────

async def test_all_zero_ratios_rejected(tmp_path):
    with pytest.raises(ValueError):
        SplitInput(
            path=_write(tmp_path, [_row(0)]),
            output_dir=str(tmp_path / "out"),
            train_ratio=0,
            val_ratio=0,
            test_ratio=0,
        )


async def test_negative_ratio_rejected(tmp_path):
    with pytest.raises(ValueError):
        SplitInput(
            path=_write(tmp_path, [_row(0)]),
            output_dir=str(tmp_path / "out"),
            train_ratio=-0.1,
        )


# ── backend resolution / hermeticity ──────────────────────────────────────

async def test_sklearn_backend_raises_without_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    rows = [_row(i) for i in range(10)]
    with pytest.raises(RuntimeError, match=r"\[split\]"):
        await SplitOp().run(
            SplitInput(
                path=_write(tmp_path, rows),
                output_dir=str(tmp_path / "out"),
                backend="sklearn",
            ),
            _ctx(),
        )


async def test_auto_backend_falls_back_to_hash_when_sklearn_absent(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    rows = [_row(i) for i in range(30)]
    out = await SplitOp().run(
        SplitInput(path=_write(tmp_path, rows), output_dir=str(tmp_path / "out"), backend="auto"),
        _ctx(),
    )
    assert out.backend_used == "hash"


# ── sklearn-backed path (needs the [split] extra) ─────────────────────────

async def test_sklearn_backend_basic_split(tmp_path):
    pytest.importorskip("sklearn")
    rows = [_row(i) for i in range(200)]
    out = await SplitOp().run(
        SplitInput(
            path=_write(tmp_path, rows),
            output_dir=str(tmp_path / "out"),
            train_ratio=0.8,
            val_ratio=0.2,
            backend="sklearn",
            seed=7,
        ),
        _ctx(),
    )
    assert out.backend_used == "sklearn"
    assert sum(out.counts.values()) == 200
    assert out.no_leakage is True
    # sklearn's ShuffleSplit gives exact ratios (no group/stratify constraint).
    assert out.counts["train"] == 160
    assert out.counts["val"] == 40


async def test_sklearn_backend_group_aware_no_leakage(tmp_path):
    pytest.importorskip("sklearn")
    rows = []
    for g in range(30):
        for _ in range(4):
            rows.append(_row(len(rows), group=f"doc-{g}"))

    out = await SplitOp().run(
        SplitInput(
            path=_write(tmp_path, rows),
            output_dir=str(tmp_path / "out"),
            train_ratio=0.75,
            val_ratio=0.25,
            group_by="doc_id",
            backend="sklearn",
            seed=3,
        ),
        _ctx(),
    )
    assert out.backend_used == "sklearn"
    assert out.no_leakage is True
    train_rows = _read_jsonl(out.output_paths["train"])
    val_rows = _read_jsonl(out.output_paths["val"])
    train_groups = {r["doc_id"] for r in train_rows}
    val_groups = {r["doc_id"] for r in val_rows}
    assert train_groups.isdisjoint(val_groups)


async def test_sklearn_backend_stratified(tmp_path):
    pytest.importorskip("sklearn")
    rows = []
    for i in range(100):
        label = "a" if i % 2 == 0 else "b"
        rows.append(_row(i, label=label))

    out = await SplitOp().run(
        SplitInput(
            path=_write(tmp_path, rows),
            output_dir=str(tmp_path / "out"),
            train_ratio=0.8,
            val_ratio=0.2,
            stratify_by="source",
            backend="sklearn",
            seed=5,
        ),
        _ctx(),
    )
    assert out.backend_used == "sklearn"
    train_rows = _read_jsonl(out.output_paths["train"])
    val_rows = _read_jsonl(out.output_paths["val"])
    train_a = sum(1 for r in train_rows if r["source"] == "a")
    train_b = sum(1 for r in train_rows if r["source"] == "b")
    val_a = sum(1 for r in val_rows if r["source"] == "a")
    val_b = sum(1 for r in val_rows if r["source"] == "b")
    # Stratified split should keep each label close to its 80/20 split.
    assert train_a == 40 and val_a == 10
    assert train_b == 40 and val_b == 10


async def test_sklearn_backend_deterministic_same_seed(tmp_path):
    pytest.importorskip("sklearn")
    rows = [_row(i) for i in range(150)]
    path = _write(tmp_path, rows)

    out_a = await SplitOp().run(
        SplitInput(path=path, output_dir=str(tmp_path / "a"), seed=99, backend="sklearn"),
        _ctx(),
    )
    out_b = await SplitOp().run(
        SplitInput(path=path, output_dir=str(tmp_path / "b"), seed=99, backend="sklearn"),
        _ctx(),
    )
    a_train = _read_jsonl(out_a.output_paths["train"])
    b_train = _read_jsonl(out_b.output_paths["train"])
    assert a_train == b_train


async def test_sklearn_backend_three_way_split(tmp_path):
    pytest.importorskip("sklearn")
    rows = [_row(i) for i in range(400)]
    out = await SplitOp().run(
        SplitInput(
            path=_write(tmp_path, rows),
            output_dir=str(tmp_path / "out"),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            backend="sklearn",
            seed=11,
        ),
        _ctx(),
    )
    assert set(out.counts.keys()) == {"train", "val", "test"}
    assert sum(out.counts.values()) == 400
    assert out.no_leakage is True
    # No row appears in more than one split file.
    train_rows = _read_jsonl(out.output_paths["train"])
    val_rows = _read_jsonl(out.output_paths["val"])
    test_rows = _read_jsonl(out.output_paths["test"])
    all_written = [json.dumps(r, sort_keys=True) for r in train_rows + val_rows + test_rows]
    assert len(all_written) == len(set(all_written)) == 400


async def test_sklearn_group_and_stratify_combined(tmp_path):
    pytest.importorskip("sklearn")
    rows = []
    for g in range(60):
        label = "a" if g % 2 == 0 else "b"
        for _ in range(3):
            rows.append(_row(len(rows), group=f"doc-{g}", label=label))

    out = await SplitOp().run(
        SplitInput(
            path=_write(tmp_path, rows),
            output_dir=str(tmp_path / "out"),
            train_ratio=0.8,
            val_ratio=0.2,
            group_by="doc_id",
            stratify_by="source",
            backend="sklearn",
            seed=21,
        ),
        _ctx(),
    )
    assert out.backend_used == "sklearn"
    assert out.no_leakage is True
    train_rows = _read_jsonl(out.output_paths["train"])
    val_rows = _read_jsonl(out.output_paths["val"])
    train_groups = {r["doc_id"] for r in train_rows}
    val_groups = {r["doc_id"] for r in val_rows}
    assert train_groups.isdisjoint(val_groups)
