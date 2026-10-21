"""compress/distill — train a small student to mimic a larger teacher (#1726).

Lives in ``cairn-operator-training`` rather than ``cairn-operator-compress`` because the
implementation must reuse ``train_entry``'s adapter-save, MLflow and
checkpoint-relay helpers, and cross-operator-pack imports are not allowed.
The CAPABILITY is compression, so the operator is named ``compress/distill`` —
the same split ``cairn-operator-eval`` already uses when it provides ``model/*`` and
``rag/*``. Its quality gate is ``eval/delta``, which already describes itself as
"the gate that unlocks the compression lane (quantize / distill / prune)".

The stub tier is deterministic and needs no torch, trl or GPU, so the whole
governed path — plan, gate, lineage, registry — is exercisable on a laptop.
"""
from __future__ import annotations

import os

from cairn.core.operator import Operation, OperationContext, register_operator
from cairn.core.operator.reports import ReportSpec

from operator_training.schemas.distill import DistillResult, DistillSpec


def _stub_selected(spec: DistillSpec) -> bool:
    if spec.stub:
        return True
    return os.environ.get("CAIRN_TRAIN_STUB", "").strip().lower() in ("1", "true", "yes", "on")


@register_operator(name="compress/distill", version="0.1.0", capabilities=["compress.distill"])
class CompressDistillOp(Operation[DistillSpec, DistillResult]):
    """Knowledge distillation: a smaller student, gated on the quality delta."""

    input_schema = DistillSpec
    output_schema = DistillResult
    # ADR-0103 — stages only (judgement: no series until distill_entry emits them; see model/merge)
    reports = ReportSpec(stages=("download", "train", "export"))
    supports_mock = True
    # Declared on the CLASS, which is what the runtime reads (#1539) — a
    # manifest-only declaration is decorative (CLAUDE.md rule 7). This one
    # holds a GPU and spends: a read_only default would let the bridge's
    # restart path silently re-buy a training run.
    risk = "paid_compute"
    long_running = True
    requires_gpu = True

    def mock(self, inp: DistillSpec, ctx: OperationContext) -> DistillResult:
        return DistillResult(
            status="succeeded", backend="mock", adapter_uri="",
            teacher_model=inp.teacher_model, student_model=inp.base_model,
            metrics={"mock": True},
        )

    async def run(self, inp: DistillSpec, ctx: OperationContext) -> DistillResult:
        return distill_body(inp)


def distill_body(inp: DistillSpec) -> DistillResult:
    """The distillation itself — ONE body for the ``compress/distill`` operator and
    the ``train.distill`` flow kernel (#2643), which is how ``compress/distill``
    pipelines reach it: on the ml engine's GPU target, never the control plane."""
    if inp.teacher_model.strip() == inp.base_model.strip():
        # trl would accept this (teacher defaults to the model itself), and
        # it would train happily against its own logits — a self-distillation
        # nobody asked for. Refuse rather than burn a GPU on a typo.
        raise ValueError(
            "compress/distill: teacher_model and base_model are the same "
            f"({inp.base_model!r}) — distillation needs a LARGER teacher to "
            "learn from, or there is nothing to compress")

    if _stub_selected(inp):
        # Deterministic: no torch, no trl, no GPU. Same convention as
        # train/stub, so the composed pipeline is runnable on a laptop.
        return DistillResult(
            status="succeeded", backend="stub",
            adapter_uri=f"file:///tmp/{inp.output_name}",
            teacher_model=inp.teacher_model, student_model=inp.base_model,
            metrics={
                "train_loss": 0.5, "steps": max(1, inp.max_steps or 1),
                "lmbda": inp.lmbda, "jsd_beta": inp.jsd_beta,
                "teacher_model": inp.teacher_model, "stub": True,
            },
        )

    from operator_training import distill_entry  # noqa: PLC0415 — trainer deps live on the GPU backend only (D5)

    res = distill_entry.main(inp.model_dump())
    return DistillResult(
        status="succeeded", backend="gkd-gpu",
        adapter_uri=str(res.get("adapter_uri") or ""),
        teacher_model=inp.teacher_model, student_model=inp.base_model,
        metrics=dict(res.get("metrics") or {}),
    )


__all__ = ["CompressDistillOp", "distill_body"]
