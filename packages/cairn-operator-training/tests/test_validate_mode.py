"""train/local pipeline-validation mode (#536).

The validation POLICY is pure (model swap + step budget + throttle), so it's
tested here without the torch stack. The actual training run is exercised on the
box (the [local] extra isn't in the base test env)."""
from __future__ import annotations

import pytest

from operator_training.operators.local import _VALIDATE_MODEL, _validate_params, _validating
from operator_training.schemas.train import TrainSpec


def _spec(**kw) -> TrainSpec:
    return TrainSpec(base_model="meta-llama/Llama-3-8B", dataset_uri="d.jsonl", **kw)


def test_real_run_uses_requested_model_no_throttle(monkeypatch):
    monkeypatch.delenv("CAIRN_TRAIN_VALIDATE", raising=False)
    monkeypatch.delenv("CAIRN_TRAIN_VALIDATE_SECONDS", raising=False)
    spec = _spec(max_steps=42)
    assert _validating(spec) is False
    model, steps, throttle = _validate_params(spec)
    assert model == "meta-llama/Llama-3-8B" and steps == 42 and throttle == 0.0


def test_validate_flag_swaps_to_tiny_model(monkeypatch):
    monkeypatch.delenv("CAIRN_TRAIN_VALIDATE", raising=False)
    spec = _spec(validate_pipeline=True)
    assert _validating(spec) is True
    model, steps, throttle = _validate_params(spec)
    assert model == _VALIDATE_MODEL        # the 8B request is IGNORED on purpose
    assert steps == 30 and throttle == 0.0  # default budget, no throttle target set


def test_validate_env_toggles_without_changing_the_request(monkeypatch):
    monkeypatch.setenv("CAIRN_TRAIN_VALIDATE", "1")
    spec = _spec()  # a plain request — the env alone flips it to validation
    assert _validating(spec) is True
    model, _, _ = _validate_params(spec)
    assert model == _VALIDATE_MODEL


def test_throttle_stretches_wall_time(monkeypatch):
    monkeypatch.setenv("CAIRN_TRAIN_VALIDATE", "1")
    monkeypatch.setenv("CAIRN_TRAIN_VALIDATE_SECONDS", "600")
    _, steps, throttle = _validate_params(_spec(max_steps=20))
    assert steps == 20
    assert throttle == pytest.approx(30.0)  # 600s / 20 steps → ~10 min job


def test_env_off_values_do_not_validate(monkeypatch):
    for v in ("0", "false", "no", ""):
        monkeypatch.setenv("CAIRN_TRAIN_VALIDATE", v)
        assert _validating(_spec()) is False
