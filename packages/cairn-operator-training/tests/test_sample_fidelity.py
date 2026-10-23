"""The ``sample.fidelity`` kernel and the stub's declared curves (#2029).

Pack tests live beside the pack. What is asserted here is that a subsampled run is
comparable: the same seed keeps the same rows (the digest proves it), the record reaches the
``TrainSpec`` and so the run's ``parameters`` event, and an unset fidelity is a no-op. The
algorithm itself is ``cairn.core.dataflow.fidelity``'s and is tested there.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cairn.core.operator import OperationContext

from operator_training.flow_stages import sample_fidelity
from operator_training.operators.stub import StubTrainOp
from operator_training.schemas.train import TrainSpec


# ── the fidelity kernel ───────────────────────────────────────────────────


def _corpus(tmp_path: Path, n: int = 200) -> str:
    p = tmp_path / "corpus.jsonl"
    p.write_text("".join(json.dumps({"input": f"q{i}", "output": f"a{i}"}) + "\n"
                         for i in range(n)), encoding="utf-8")
    return str(p)


def test_sample_fidelity_is_a_no_op_without_a_fidelity(tmp_path):
    stats = sample_fidelity(tmp_path, {"dataset_uri": _corpus(tmp_path)}, [])
    assert stats["skipped"] is True and stats["fidelity_sampled"] is False
    assert not (tmp_path / "fidelity_sample.jsonl").exists()


def test_the_same_seed_reproduces_the_same_digest_and_a_different_one_does_not(tmp_path):
    src = _corpus(tmp_path)
    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    cfg = {"dataset_uri": src, "fidelity_fraction": 0.01, "fidelity_seed": 7}
    one = sample_fidelity(a, dict(cfg), [])
    two = sample_fidelity(b, dict(cfg), [])
    other = sample_fidelity(c, {**cfg, "fidelity_seed": 8}, [])
    assert one["fidelity_n"] == 2 and one["fidelity_sampled"] is True
    assert one["fidelity_digest"] == two["fidelity_digest"]
    assert one["fidelity_digest"] != other["fidelity_digest"]
    assert (a / "fidelity_sample.jsonl").read_text() == (b / "fidelity_sample.jsonl").read_text()


def test_the_fidelity_record_reaches_the_trainspec_so_it_lands_on_the_run(tmp_path):
    """The digest has to be on the RUN's record, not only in a stage's stats:
    ADR-0103 D3 writes ``TrainSpec.extra`` verbatim into the ``parameters``
    event, so this is what makes two subsampled runs comparable after the fact."""
    from operator_training.flow_stages import _spec_from_config

    src = _corpus(tmp_path)
    stats = sample_fidelity(tmp_path, {"dataset_uri": src, "fidelity_rows": 5}, [])
    spec = _spec_from_config({"dataset_uri": src, "base_model": "Qwen/Qwen2.5-0.5B"},
                             [], tmp_path)
    assert spec.dataset_uri.endswith("fidelity_sample.jsonl")
    assert spec.extra["fidelity_digest"] == stats["fidelity_digest"]
    assert spec.extra["fidelity_n"] == 5 and spec.extra["fidelity_seed"] == 13


# ── the stub's divergent curves (what makes a pruning round falsifiable) ──


@pytest.mark.asyncio
async def test_stub_curve_auto_differs_per_config_and_repeats_per_config():
    ctx = OperationContext(pipeline_id="p", node_name="n")
    base = dict(dataset_uri="d.jsonl", base_model="Qwen/Qwen2.5-0.5B",
                extra={"stub_curve": "auto"})
    a = await StubTrainOp().run(TrainSpec(**base, learning_rate=1e-4), ctx)
    b = await StubTrainOp().run(TrainSpec(**base, learning_rate=3e-4), ctx)
    again = await StubTrainOp().run(TrainSpec(**base, learning_rate=1e-4), ctx)
    assert a.metrics["eval_loss"] != b.metrics["eval_loss"], "trials must DIFFER"
    assert a.metrics["eval_loss"] == again.metrics["eval_loss"], "…deterministically"
    assert [p[0] for p in a.metrics["eval_curve"]] == [1, 2, 3]


@pytest.mark.asyncio
async def test_an_explicit_stub_curve_is_used_verbatim():
    ctx = OperationContext(pipeline_id="p", node_name="n")
    res = await StubTrainOp().run(TrainSpec(
        dataset_uri="d.jsonl", base_model="m", extra={"stub_curve": [0.9, 0.5, 0.25]}), ctx)
    assert res.metrics["eval_curve"] == [[1, 0.9], [2, 0.5], [3, 0.25]]
    assert res.metrics["eval_loss"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_a_stub_without_a_curve_is_unchanged():
    ctx = OperationContext(pipeline_id="p", node_name="n")
    res = await StubTrainOp().run(TrainSpec(dataset_uri="d.jsonl", base_model="m"), ctx)
    assert res.metrics["eval_loss"] == 0.0 and res.metrics["eval_curve"] == []
