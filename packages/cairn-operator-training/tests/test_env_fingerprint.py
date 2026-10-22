"""#1622 — the per-run trainer-env fingerprint.

Hermetic: no GPU, and no trainer deps installed in CI. That is the interesting
case, not a limitation — the fingerprint's job is to be honest about a partial
env, and its hard rule is that it never raises, because a version probe must not
kill a paid GPU run. So the tests drive the real function on this bare CPU env
and then break torch on purpose.
"""
from __future__ import annotations

import platform
import sys
import types

from operator_training import env_fingerprint
from operator_training.env_fingerprint import trainer_env
from operator_training.images import TRAINER_PINS


def test_every_pinned_package_is_a_key():
    env = trainer_env()
    # A key per pin whether or not it's installed → diffing a run against the
    # pins is a plain dict comparison, with no "was it even probed?" ambiguity.
    for name in TRAINER_PINS:
        assert name in env
        assert isinstance(env[name], str)
    assert env["python"] == platform.python_version()
    assert env["platform"]


def test_missing_package_records_absent_not_an_error(monkeypatch):
    # unsloth is CUDA-only: absent on a CPU box, the honest answer is "absent", and it must not
    # look like a probe failure. Made absent here, so the test holds on a GPU box that has it.
    real = env_fingerprint.version

    def version(dist):
        if dist == "unsloth":
            raise env_fingerprint.PackageNotFoundError(dist)
        return real(dist)

    monkeypatch.setattr(env_fingerprint, "version", version)
    env = trainer_env()
    assert env["unsloth"] == env_fingerprint.ABSENT


def test_no_torch_still_returns_a_fingerprint(monkeypatch):
    # `None` in sys.modules makes `import torch` raise — the CPU-box shape.
    monkeypatch.setitem(sys.modules, "torch", None)
    env = trainer_env()
    assert "error" not in env
    # No torch ⇒ no device facts at all (their absence IS the signal that this
    # run did not train on a GPU).
    assert "cuda_available" not in env
    assert "gpu_name" not in env


def test_gpu_facts_recorded_when_torch_reports_cuda(monkeypatch):
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda i: "NVIDIA GeForce RTX 4090",
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    env = trainer_env()
    assert env["cuda_available"] is True
    assert env["gpu_name"] == "NVIDIA GeForce RTX 4090"


def test_device_probe_failure_is_recorded_not_raised(monkeypatch):
    def _boom():
        raise RuntimeError("CUDA driver version is insufficient")

    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=_boom, get_device_name=lambda i: "")
    monkeypatch.setitem(sys.modules, "torch", fake)
    env = trainer_env()
    assert "RuntimeError" in env["device_probe_error"]
    assert env["torch"]  # the version probe is independent of the device probe


def test_fingerprint_never_raises(monkeypatch):
    # Worst case: even the version lookup blows up. The run keeps going and the
    # failure is recorded as data.
    def _boom(_dist):
        raise SystemError("dist-info is corrupt")

    monkeypatch.setattr(env_fingerprint, "_installed", _boom)
    env = trainer_env()
    assert env == {"error": "SystemError: dist-info is corrupt"}


# ── it reaches the run: BOTH trainer entry paths put it in `metrics` ──────────
#
# `train_entry.main` (unsloth/GPU) and `_train_local` (plain LoRA, CPU/MPS) both
# assemble their result metrics through a `_base_metrics` helper that is pure
# over HF's TrainOutput — so the metrics contract is drivable without the
# trainer stack, which is the only way to test it hermetically.


class _Stats:
    """A stand-in for transformers' ``TrainOutput``."""
    training_loss = 0.42
    global_step = 7
    metrics = {"train_runtime": 12.5, "train_samples_per_second": 3.0}


def test_gpu_entry_metrics_carry_the_fingerprint():
    from operator_training.train_entry import _base_metrics

    metrics = _base_metrics(_Stats())
    assert metrics["train_loss"] == 0.42
    assert metrics["train_runtime"] == 12.5     # #1664 speed metrics still ride along
    env = metrics["trainer_env"]
    assert set(TRAINER_PINS) <= set(env)


def test_cpu_entry_metrics_carry_the_fingerprint():
    from operator_training.operators.local import _base_metrics

    metrics = _base_metrics(_Stats(), "cpu", 5)
    assert metrics == {
        "train_loss": 0.42, "steps": 7, "device": "cpu", "local": True,
        "trainer_env": metrics["trainer_env"],
    }
    assert set(TRAINER_PINS) <= set(metrics["trainer_env"])
