"""#2643 — `train.distill`: compress/distill through the ml-flow door.

A `task: compress/distill` leg ran the GKD trainer in the control plane (task legs run
in-process, #2182), which has no GPU. The kernel puts the same body (`distill_body`,
shared with the operator) on the ml engine's GPU target, and its stats are the flat
scalars a flow leg's findings and the gate read.
"""
from __future__ import annotations

import pytest

from operator_training.flow_stages import train_distill


def test_the_stub_tier_returns_flat_scalar_stats(tmp_path):
    out = train_distill(tmp_path, {"teacher_model": "big", "base_model": "small",
                                   "dataset_uri": "s3://d.jsonl", "output_name": "st",
                                   "max_steps": 3, "stub": True}, [])
    assert out["status"] == "succeeded" and out["backend"] == "stub"
    assert out["adapter_uri"] == "file:///tmp/st" and out["student_model"] == "small"
    assert out["train_loss"] == 0.5 and out["steps"] == 3
    assert all(isinstance(v, (str, int, float)) and not isinstance(v, (dict, list)) for v in out.values())


def test_self_distillation_is_refused_before_any_gpu_work(tmp_path):
    with pytest.raises(ValueError, match="same"):
        train_distill(tmp_path, {"teacher_model": "m", "base_model": "m", "dataset_uri": "d", "stub": True}, [])


def test_the_upstream_artifact_is_the_dataset_when_none_is_configured(tmp_path):
    out = train_distill(tmp_path, {"teacher_model": "big", "base_model": "small", "stub": True}, ["s3://up.jsonl"])
    assert out["status"] == "succeeded"


def test_the_stage_declares_external_write():
    # The flow leg's gate is decided on the aggregate risk of its stages (enforce_flow_gate),
    # so this declaration is a governance fact: a distill run writes an adapter.
    assert train_distill.risk_class == "external_write"
