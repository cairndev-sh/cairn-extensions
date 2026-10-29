"""#6 — cairn-train-local routes a real GPU run to the unsloth engine (train_entry).

Hermetic: no GPU, no torch/unsloth in CI. We test the real gate branches (no-torch and
validation ⇒ CPU path) and mock the gate to prove the dispatch wraps train_entry.main's
result with the LOCAL adapter dir so the SSH governed path can SFTP-fetch it.
"""
from __future__ import annotations

import asyncio
import sys

from operator_training.operators import local
from operator_training.schemas.train import TrainSpec


def _spec(**kw):
    return TrainSpec(**{"dataset_uri": "ds", "method": "qlora", "output_name": "out", **kw})


def test_no_torch_or_no_cuda_takes_cpu_path(monkeypatch):
    # No torch → the cuda probe raises → CPU path (never GPU). `None` in sys.modules makes
    # `import torch` raise, so the test holds on a CUDA box too.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert local._use_gpu_unsloth(_spec()) is False


def test_validation_never_uses_gpu(monkeypatch):
    # Validation is checked BEFORE the cuda probe, so it stays on the CPU tiny-model
    # path regardless of hardware.
    monkeypatch.setenv("CAIRN_TRAIN_VALIDATE", "1")
    assert local._use_gpu_unsloth(_spec()) is False


def test_gpu_dispatch_wraps_unsloth_result(monkeypatch):
    # Force the gate on (bypassing the hardware probe) and mock the unsloth engine.
    monkeypatch.setattr(local, "_use_gpu_unsloth", lambda spec: True)

    called = {}

    def fake_main(spec_dict, reporter=None, *, output_dir=None):
        called["spec"] = spec_dict
        called["output_dir"] = output_dir
        return {"adapter_uri": "hf://acme/out", "metrics": {"train_loss": 0.7, "eval_loss": 0.9}}

    import operator_training.train_entry as te
    monkeypatch.setattr(te, "main", fake_main)

    res = asyncio.run(local._train_local(_spec()))
    # The unsloth engine ran on the spec dict…
    assert called["spec"]["dataset_uri"] == "ds"
    assert called["spec"]["method"] == "qlora"
    # …metrics threaded through, backend tagged unsloth-gpu.
    assert res.backend == "unsloth-gpu"
    assert res.metrics["eval_loss"] == 0.9
    # adapter_uri forced to the LOCAL dir train_entry writes (the caller publishes it),
    # NOT the hf:// push target. No job workdir here, so /tmp/<output_name>.
    assert res.adapter_uri == called["output_dir"] == "/tmp/out"


class _CudaTorch:
    """A stand-in torch whose cuda probe says a GPU is present."""

    class cuda:  # noqa: N801 — mirrors torch.cuda
        @staticmethod
        def is_available():
            return True


def test_lora_on_cuda_without_unsloth_takes_plain_lora_on_the_gpu(monkeypatch):
    # The [local] extra on an NVIDIA machine (the laptop demo): no unsloth installed, so a
    # `lora` run must not be routed to the unsloth engine ("No module named 'unsloth'").
    import importlib.util as iu

    monkeypatch.setitem(sys.modules, "torch", _CudaTorch)
    monkeypatch.setattr(iu, "find_spec", lambda name, *a, **k: None)
    assert local._use_gpu_unsloth(_spec(method="lora")) is False


def test_lora_on_cuda_with_unsloth_keeps_the_unsloth_engine(monkeypatch):
    import importlib.util as iu

    monkeypatch.setitem(sys.modules, "torch", _CudaTorch)
    monkeypatch.setattr(iu, "find_spec", lambda name, *a, **k: object() if name == "unsloth" else None)
    assert local._use_gpu_unsloth(_spec(method="lora")) is True


def test_qlora_on_cuda_without_unsloth_still_fails_loud(monkeypatch):
    # 4-bit QLoRA is the [gpu] extra's job: no silent switch to plain LoRA.
    import importlib.util as iu

    monkeypatch.setitem(sys.modules, "torch", _CudaTorch)
    monkeypatch.setattr(iu, "find_spec", lambda name, *a, **k: None)
    assert local._use_gpu_unsloth(_spec(method="qlora")) is True
