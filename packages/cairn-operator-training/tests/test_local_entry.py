"""Tests for cairn-train-local — the on-box CLI for the train/remote path.

Hermetic: ``_train_local`` (the real torch trainer) is monkeypatched, so these run
without the ``[local]`` stack. They cover spec resolution (env / JSON / file),
and that the CLI publishes ``metrics.json`` + ``result.json`` + the adapter dir
into the workdir so ``remote/exec``'s poll/fetch find them.
"""
from __future__ import annotations

import sys

import json
import os

import pytest

from operator_training import local_entry
from operator_training.schemas.train import TrainResult, TrainSpec


def _clear_cairn_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("CAIRN_"):
            monkeypatch.delenv(k, raising=False)


def test_load_spec_from_individual_env(monkeypatch):
    _clear_cairn_env(monkeypatch)
    monkeypatch.setenv("CAIRN_DATASET_URI", "yahma/alpaca-cleaned")
    monkeypatch.setenv("CAIRN_BASE_MODEL", "sshleifer/tiny-gpt2")
    monkeypatch.setenv("CAIRN_MAX_STEPS", "3")
    monkeypatch.setenv("CAIRN_LORA_R", "4")
    monkeypatch.setenv("CAIRN_LEARNING_RATE", "0.001")
    spec = local_entry.load_spec(["cairn-train-local"])
    assert spec.dataset_uri == "yahma/alpaca-cleaned"
    assert spec.base_model == "sshleifer/tiny-gpt2"
    assert spec.max_steps == 3 and spec.lora_r == 4
    assert spec.learning_rate == pytest.approx(0.001)


def test_load_spec_from_json_blob_wins(monkeypatch):
    _clear_cairn_env(monkeypatch)
    monkeypatch.setenv("CAIRN_DATASET_URI", "ignored-because-blob-wins")
    blob = TrainSpec(dataset_uri="d", base_model="m", max_steps=7).model_dump_json()
    monkeypatch.setenv("CAIRN_TRAIN_SPEC", blob)
    spec = local_entry.load_spec(["cairn-train-local"])
    assert spec.dataset_uri == "d" and spec.max_steps == 7


def test_load_spec_from_file(tmp_path, monkeypatch):
    _clear_cairn_env(monkeypatch)
    p = tmp_path / "spec.json"
    p.write_text(TrainSpec(dataset_uri="from-file").model_dump_json(), encoding="utf-8")
    spec = local_entry.load_spec(["cairn-train-local", str(p)])
    assert spec.dataset_uri == "from-file"


def test_load_spec_missing_raises(monkeypatch):
    _clear_cairn_env(monkeypatch)
    with pytest.raises(SystemExit):
        local_entry.load_spec(["cairn-train-local"])


def test_publish_outputs_writes_metrics_and_copies_adapter(tmp_path, monkeypatch):
    # a fake trained adapter dir the CLI should copy into the workdir
    adapter_src = tmp_path / "src-adapter"
    adapter_src.mkdir()
    (adapter_src / "adapter_model.safetensors").write_bytes(b"weights")
    workdir = tmp_path / "work"
    workdir.mkdir()

    spec = TrainSpec(dataset_uri="d", output_name="myadapter")
    result = TrainResult(
        status="succeeded", backend="local", adapter_uri=f"file://{adapter_src}",
        metrics={"train_loss": 1.23, "eval_loss": 1.10, "steps": 3},
    )
    local_entry._publish_outputs(spec, result, str(workdir))

    metrics = json.loads((workdir / "metrics.json").read_text())
    assert metrics["train_loss"] == 1.23 and metrics["steps"] == 3
    assert (workdir / "result.json").exists()
    # adapter copied to <workdir>/<output_name>/
    assert (workdir / "myadapter" / "adapter_model.safetensors").read_bytes() == b"weights"


def test_cli_end_to_end_mocked(tmp_path, monkeypatch):
    _clear_cairn_env(monkeypatch)
    adapter_src = tmp_path / "out"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text("{}", encoding="utf-8")

    captured = {}

    async def fake_train(spec, *, progress_path=None, reporter=None):
        assert spec.dataset_uri == "ds"
        captured["progress_path"] = progress_path
        return TrainResult(
            status="succeeded", backend="local",
            adapter_uri=f"file://{adapter_src}", metrics={"train_loss": 0.9},
        )

    monkeypatch.setattr(local_entry, "_train_local", fake_train)
    monkeypatch.setenv("CAIRN_DATASET_URI", "ds")
    monkeypatch.setenv("CAIRN_OUTPUT_NAME", "out")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_entry.sys, "argv", ["cairn-train-local"])

    local_entry._cli()
    # #553: the CLI wires live-progress writes into the workdir's metrics.json.
    assert captured["progress_path"] == str(tmp_path / "metrics.json")
    assert json.loads((tmp_path / "metrics.json").read_text())["train_loss"] == 0.9


def test_tracking_env_threads_report_to_and_mlflow_uri(monkeypatch):
    """#729/#6: CAIRN_REPORT_TO (list-repr from remote/exec) + CAIRN_TRACKING_* land
    on the TrainSpec, and a non-empty CAIRN_MLFLOW_TRACKING_URI promotes to
    MLFLOW_TRACKING_URI so the on-box trainer's MLflowCallback logs."""
    from operator_training import local_entry

    monkeypatch.setenv("CAIRN_DATASET_URI", "ds")
    monkeypatch.setenv("CAIRN_REPORT_TO", "['mlflow']")  # shell list-repr
    monkeypatch.setenv("CAIRN_TRACKING_PROJECT", "cairn-ft")
    monkeypatch.setenv("CAIRN_TRACKING_RUN_NAME", "run-42")
    monkeypatch.setenv("CAIRN_MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    local_entry._apply_tracking_env()
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"

    spec = local_entry.load_spec(["cairn-train-local"])
    assert spec.report_to == ["mlflow"]
    assert spec.tracking_project == "cairn-ft"
    assert spec.tracking_run_name == "run-42"


def test_empty_mlflow_uri_does_not_clobber_box_env(monkeypatch):
    """An EMPTY CAIRN_MLFLOW_TRACKING_URI must leave the box's own MLFLOW_TRACKING_URI
    untouched (the binding always sets the CAIRN_ var, possibly to '')."""
    from operator_training import local_entry

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://box-own:5000")
    monkeypatch.setenv("CAIRN_MLFLOW_TRACKING_URI", "")
    local_entry._apply_tracking_env()
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://box-own:5000"


def test_train_local_refuses_without_its_extra_and_names_it(monkeypatch):
    """A base install carries no torch; train/local must say which extra brings it."""
    import pytest

    from operator_training.operators.local import _require_local

    monkeypatch.setitem(sys.modules, "torch", None)   # `import torch` now raises ImportError
    with pytest.raises(RuntimeError, match=r"cairn-operator-training\[local\]"):
        _require_local()

