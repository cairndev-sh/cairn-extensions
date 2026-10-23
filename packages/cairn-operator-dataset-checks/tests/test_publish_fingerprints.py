"""dataset/publish_fingerprints — cross-batch dedup actually gets an index (#1050).

The bug these tests exist to prevent is INVISIBLE by construction: before this
operator the new `clean` dataflow published nothing, so cross-batch dedup silently
degraded to within-batch only. No error, no log, a plausible-looking `0` in the
report. `test_real_batch_publishes_nonzero` is the guard — it fails if a clean run
ever indexes zero rows again.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from operator_dataset_checks.operators.publish_fingerprints import PublishFingerprintsOp
from operator_dataset_checks.schemas.publish_fingerprints import PublishFingerprintsInput


class _Ctx:
    """OperationContext stand-in — this operator uses no ctx capabilities."""


class _Store:
    """FingerprintStore double: records what was published."""

    def __init__(self, *, boom: bool = False):
        self.calls: list[tuple[str, str, list[str]]] = []
        self.closed = False
        self._boom = boom

    async def publish(self, dataset_id, batch_id, hashes, **kw) -> int:
        if self._boom:
            raise RuntimeError("postgres is down")
        self.calls.append((dataset_id, batch_id, list(hashes)))
        return len(hashes)

    async def close(self) -> None:
        self.closed = True


def _dataset(tmp_path: Path, rows: list[dict], name: str = "clean.jsonl") -> str:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(p)


def _run(inp: PublishFingerprintsInput, store, monkeypatch):
    """The index reaches the operator as the context's handle (#2453 B): the
    deployment default is what ``build_fingerprint_store`` returns, opened on
    first use and closed by the runtime's ``release_handles`` — so the close
    assertion below is about the CONTEXT keeping its promise."""
    import cairn.core.storage.fingerprints as fp_mod
    from cairn.core.operator import OperationContext

    monkeypatch.setattr(fp_mod, "build_fingerprint_store", lambda: store)

    async def _go():
        ctx = OperationContext(pipeline_id="p", node_name="publish")
        try:
            return await PublishFingerprintsOp().run(inp, ctx)
        finally:
            await ctx.release_handles()

    return asyncio.run(_go())


# ── THE guard ────────────────────────────────────────────────────────────────

def test_real_batch_publishes_nonzero(tmp_path, monkeypatch):
    """A cleaned batch with rows MUST land hashes in the index.

    This is the assertion whose absence let #1050 hide: the old failure mode
    produced published=0 with no error at all.
    """
    path = _dataset(tmp_path, [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}])
    store = _Store()

    out = _run(PublishFingerprintsInput(
        path=path, dataset_id="corp/support", batch_id="b-1"), store, monkeypatch)

    assert out.published == 3, "a real batch must publish one hash per row"
    assert out.skipped is None
    assert store.calls == [("corp/support", "b-1", store.calls[0][2])]
    assert len(store.calls[0][2]) == 3
    assert all(isinstance(h, str) and h for h in store.calls[0][2])
    assert store.closed, "the store must be closed even on the happy path"


def test_hashes_match_the_dedup_stage(tmp_path, monkeypatch):
    """Published hashes must equal what a future probe computes, or the index is
    useless. Same canonical_hash + same `normalize` as the dedup stage."""
    from operator_dataset_checks._canon import canonical_hash

    rows = [{"text": "alpha"}, {"text": "beta"}]
    path = _dataset(tmp_path, rows)
    store = _Store()

    _run(PublishFingerprintsInput(
        path=path, dataset_id="d", batch_id="b", normalize=True), store, monkeypatch)

    assert store.calls[0][2] == [canonical_hash(r, True) for r in rows]


def test_normalize_false_is_honoured(tmp_path, monkeypatch):
    """`normalize` changes the hash — a mismatch with the dedup stage would make
    every future probe miss."""
    from operator_dataset_checks._canon import canonical_hash

    rows = [{"text": "a  b"}]
    path = _dataset(tmp_path, rows)
    store = _Store()

    _run(PublishFingerprintsInput(
        path=path, dataset_id="d", batch_id="b", normalize=False), store, monkeypatch)

    assert store.calls[0][2] == [canonical_hash(rows[0], False)]


# ── degradation: never fail a run whose remediation already succeeded ─────────

def test_missing_dataset_id_skips_with_a_reason(tmp_path, monkeypatch):
    path = _dataset(tmp_path, [{"text": "x"}])
    out = _run(PublishFingerprintsInput(
        path=path, dataset_id="", batch_id="b"), _Store(), monkeypatch)

    assert out.published == 0 and out.skipped == "no_dataset_id"
    assert "no fingerprints published" in out.summary


def test_empty_dataset_skips_with_a_reason(tmp_path, monkeypatch):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")

    out = _run(PublishFingerprintsInput(
        path=str(p), dataset_id="d", batch_id="b"), _Store(), monkeypatch)

    assert out.published == 0 and out.skipped == "no_rows"


def test_unreadable_path_skips_instead_of_raising(tmp_path, monkeypatch):
    out = _run(PublishFingerprintsInput(
        path=str(tmp_path / "does-not-exist.jsonl"), dataset_id="d", batch_id="b"),
        _Store(), monkeypatch)

    assert out.published == 0 and out.skipped == "no_rows"


def test_store_failure_degrades_and_reports_why(tmp_path, monkeypatch):
    """A dead index must not fail a run that already cleaned the data — but the
    reason has to be visible, not swallowed."""
    path = _dataset(tmp_path, [{"text": "x"}])

    out = _run(PublishFingerprintsInput(
        path=path, dataset_id="d", batch_id="b"), _Store(boom=True), monkeypatch)

    assert out.published == 0
    assert out.skipped == "store_unavailable"
    assert "postgres is down" in out.summary


def test_operator_is_registered_read_only():
    """It writes cairn's OWN index — no user data, no external system, no gate."""
    from cairn.core.operator import default_registry, load_all_operators

    default_registry.clear()
    load_all_operators(strict=False)
    md = default_registry.get_metadata("dataset/publish_fingerprints")
    assert md is not None, "operator must be discoverable via entry points"
    assert md.risk == "read_only"
    assert md.requires_approval is False


@pytest.mark.parametrize("rows", [1, 5, 50])
def test_publishes_one_hash_per_row(tmp_path, monkeypatch, rows):
    path = _dataset(tmp_path, [{"text": f"row-{i}"} for i in range(rows)])
    store = _Store()

    out = _run(PublishFingerprintsInput(
        path=path, dataset_id="d", batch_id="b"), store, monkeypatch)

    assert out.published == rows and len(store.calls[0][2]) == rows
