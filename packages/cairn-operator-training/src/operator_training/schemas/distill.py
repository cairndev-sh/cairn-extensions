"""Schemas for compress/distill — knowledge distillation (#1726, revives #467).

A DELIBERATELY separate spec from ``TrainSpec``, for one reason worth stating:
``TrainSpec.beta`` already means *DPO preference-loss strength vs the frozen
reference*, while GKD's ``beta`` interpolates the generalized Jensen-Shannon
divergence (0.0 = forward KL, 1.0 = reverse KL). Same name, unrelated meaning.
Inheriting it would produce a run that trains successfully with a quietly
different objective than the author asked for — no error, no crash. The field
here is ``jsd_beta`` and it is not interchangeable.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DistillSpec(BaseModel):
    """One distillation run: a small student learns from a larger teacher."""

    teacher_model: str = Field(
        ..., description="HF id (or local path) of the TEACHER — the larger model whose behaviour is copied.")
    base_model: str = Field(
        "Qwen/Qwen2.5-0.5B-Instruct",
        description="HF id of the STUDENT being trained. Smaller than the teacher, or the run has no point.")
    dataset_uri: str = Field(
        ..., description="Prompts/conversations to distil over: local path, HF id, or artifact uri.")
    output_name: str = Field("student-adapter", description="Name for the produced LoRA adapter.")

    # ── GKD-specific (trl GKDConfig). Names mirror trl to keep the mapping
    # obvious; `jsd_beta` is the one deliberate rename — see the module docstring.
    lmbda: float = Field(
        0.5, ge=0.0, le=1.0,
        description="On-policy student fraction. 0.0 = supervised JSD on teacher token probabilities; "
                    "1.0 = the student generates and the teacher grades its own output. >0 means the student "
                    "GENERATES during training — a materially heavier GPU profile than plain SFT.")
    jsd_beta: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Generalized JSD interpolation: 0.0 ≈ forward KL, 1.0 ≈ reverse KL. Maps to trl's "
                    "GKDConfig.beta. NOT the DPO beta on TrainSpec — different objective entirely.")
    seq_kd: bool = Field(
        False, description="Sequence-level KD: supervised fine-tuning on teacher-generated output. "
                           "With lmbda=0.0 this reduces to supervised JSD over teacher sequences.")
    temperature: float = Field(0.9, gt=0, description="Sampling temperature for generated completions.")
    max_new_tokens: int = Field(
        128, gt=0, description="Tokens generated per completion. With lmbda>0 this is the cost driver.")

    # ── the usual training surface ───────────────────────────────────────
    method: Literal["qlora", "lora"] = Field("qlora", description="PEFT method for the student adapter.")
    epochs: float = Field(1.0, ge=0)
    max_steps: int = Field(0, description="If >0, cap steps (overrides epochs).")
    learning_rate: float = Field(2e-5, gt=0, description="GKD default is an order below SFT's.")
    lora_r: int = Field(16, gt=0)
    lora_alpha: int = Field(16, gt=0)
    max_seq_len: int = Field(1024, gt=0)
    seed: int = Field(42)
    save_steps: int = Field(50, ge=0)
    save_total_limit: int = Field(2, ge=0)

    artifact_target: Literal["hf", "local", "s3"] = Field("s3")
    hf_repo: str = Field("")
    registered_model_name: str = Field("")
    report_to: list[Literal["mlflow", "wandb"]] = Field(default_factory=list)
    tracking_project: str = Field("")
    tracking_run_name: str = Field("")
    gpu: str = Field("A10G", description="GPU type for the backend.")
    stub: bool = Field(False, description="Deterministic stub tier: no torch, no trl, no GPU.")


class DistillResult(BaseModel):
    status: str = Field("succeeded")
    adapter_uri: str = Field("", description="Where the student adapter landed.")
    backend: str = Field("", description="Which tier produced it (stub | gkd-gpu).")
    teacher_model: str = ""
    student_model: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
