"""The base model's evaluation loss before the first step (training.stage.train_qlora/1.2.0,
cairn-extensions#7). ``splits.baseline`` is what both trainers call between building the trainer
and ``train()``; these tests drive it with a fake evaluate, and the stage through the stub tier, so
no training stack is needed."""
from __future__ import annotations

import pytest
from cairn.core.dataflow.conformance_check import bundle, problems
from cairn.core.pack.contracts import _installed

from operator_training import splits
from operator_training.flow_stages import train_qlora


def _evaluate(calls: list):
    def evaluate():
        calls.append(1)
        return {"eval_loss": 2.5}
    return evaluate


def test_off_by_default_and_never_evaluates():
    calls: list = []
    assert splits.baseline({}, _evaluate(calls), has_eval=True, resumed=False) == {}
    assert calls == []


def test_measures_once_before_training():
    calls: list = []
    assert splits.baseline({"baseline_eval": True}, _evaluate(calls), has_eval=True, resumed=False) == {
        "baseline_eval_loss": 2.5}
    assert calls == [1]


@pytest.mark.parametrize("has_eval, resumed, reason", [
    (False, False, "no evaluation data"),
    (True, True, "resumed from a checkpoint"),
])
def test_skipped_with_the_reason_when_it_would_not_measure_the_base_model(has_eval, resumed, reason):
    calls: list = []
    assert splits.baseline({"baseline_eval": True}, _evaluate(calls), has_eval=has_eval, resumed=resumed) == {
        "baseline_skipped": reason}
    assert calls == []


def _schema(uri: str) -> dict:
    _resolved, bodies = _installed()
    return bundle({u: body for u, (_d, body) in bodies.items()}, uri)


def test_the_stage_reports_the_baseline_under_its_1_2_0_contract(tmp_path):
    config = {"stub": True, "output_name": "b", "train_uri": "file:///t.jsonl", "eval_uri": "file:///v.jsonl",
              "baseline_eval": True, "extra": {"stub_curve": "auto"}}
    stats = train_qlora(tmp_path, config, [])
    assert stats["baseline_eval_loss"] > stats["eval_loss"]  # the stub's curve falls from its first point
    _resolved, bodies = _installed()
    stage = bodies["contract://training.stage.train_qlora/1.2.0"][1]
    assert problems(_schema(stage["output"]), stats, "output") == []
    cfg = {k: v for k, v in config.items() if k != "extra"}
    assert problems(_schema(stage["configuration"]), cfg, "configuration") == []
    without = train_qlora(tmp_path / "b", {**config, "baseline_eval": False}, [])
    assert "baseline_eval_loss" not in without


def test_an_evaluation_without_a_loss_is_recorded_as_skipped_not_as_zero():
    out = splits.baseline({"baseline_eval": True}, lambda: {"eval_runtime": 1.0}, has_eval=True, resumed=False)
    assert out == {"baseline_skipped": "the evaluation reported no eval_loss"}
