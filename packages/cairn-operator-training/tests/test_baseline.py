"""The base model's evaluation loss, measured after training with the adapter disabled
(training.stage.train_qlora/1.2.0, cairn-extensions#7). ``splits.baseline`` is what both trainers
call after ``train()``; these tests drive it with a fake evaluate, and the stage through the stub tier, so
no training stack is needed."""
from __future__ import annotations

import pytest
from cairn.core.dataflow.conformance_check import bundle, problems
from cairn.core.pack.contracts import _installed

from operator_training import splits
from operator_training.flow_stages import train_qlora


def _evaluate(calls: list, key: str = "baseline_loss"):
    def evaluate():
        calls.append(1)
        return {key: 2.5}
    return evaluate


def test_off_by_default_and_never_evaluates():
    calls: list = []
    assert splits.baseline({}, _evaluate(calls), has_eval=True) == {}
    assert calls == []


@pytest.mark.parametrize("key", ["baseline_loss", "eval_loss"])
def test_measures_the_base_model_once(key):
    calls: list = []
    assert splits.baseline({"baseline_eval": True}, _evaluate(calls, key), has_eval=True) == {
        "baseline_eval_loss": 2.5}
    assert calls == [1]


def test_skipped_with_the_reason_without_evaluation_data():
    calls: list = []
    assert splits.baseline({"baseline_eval": True}, _evaluate(calls), has_eval=False) == {
        "baseline_skipped": "no evaluation data"}
    assert calls == []


def test_the_base_model_is_the_model_with_its_adapter_disabled():
    """Found on the first GPU run: evaluating BEFORE train() left the fp16 LoRA weights unscalable.
    The baseline is measured after training, inside PEFT's disable_adapter()."""
    seen: list = []

    class _Model:
        def disable_adapter(self):
            class _Ctx:
                def __enter__(self_inner):
                    seen.append("off")

                def __exit__(self_inner, *a):
                    seen.append("on")
            return _Ctx()

    class _Trainer:
        def evaluate(self, metric_key_prefix="eval"):
            seen.append(f"evaluate:{metric_key_prefix}")
            return {f"{metric_key_prefix}_loss": 3.0}

    assert splits.evaluate_with_adapter_disabled(_Model(), _Trainer()) == {"baseline_loss": 3.0}
    assert seen == ["off", "evaluate:baseline", "on"]


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
    out = splits.baseline({"baseline_eval": True}, lambda: {"eval_runtime": 1.0}, has_eval=True)
    assert out == {"baseline_skipped": "the evaluation reported no eval_loss"}
