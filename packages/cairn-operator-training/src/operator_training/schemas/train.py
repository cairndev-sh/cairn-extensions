"""Schemas for the `train.run` capability."""
from __future__ import annotations

from typing import Optional, Any, Literal

from pydantic import BaseModel, Field


class TrainSpec(BaseModel):
    """Inputs for one fine-tuning job (provider-agnostic)."""

    base_model: str = Field(
        "Qwen/Qwen2.5-3B-Instruct",
        description="HF model id of the base to fine-tune (Apache-2.0 default).",
    )
    dataset_uri: str = Field(
        ...,
        description="Dataset of (input,output) pairs: local path, HF Hub id, or artifact uri.",
    )
    method: Literal["qlora", "lora"] = Field("qlora", description="PEFT method.")
    output_name: str = Field("adapter", description="Name for the produced LoRA adapter.")

    # QLoRA-sane hyperparameters.
    epochs: float = Field(1.0, ge=0)
    max_steps: int = Field(0, description="If >0, cap steps (overrides epochs).")
    learning_rate: float = Field(2e-4, gt=0)
    lora_r: int = Field(16, gt=0)
    lora_alpha: int = Field(16, gt=0)
    target_modules: list[str] = Field(
        default_factory=list,
        description="LoRA target modules (e.g. q_proj,k_proj,v_proj,o_proj for attention-only). "
                    "Empty = the trainer's default (unsloth: all attention + MLP projections).")
    #: DPO (#1463 round 4): the preference-loss temperature — how hard the pairs
    #: push relative to the frozen reference. Only train.dpo reads it.
    beta: float = Field(0.1, gt=0, description="DPO beta (preference-loss strength vs the reference model).")
    #: Two-stage recipes (#1463): stage B continues stage A's ADAPTER (same LoRA
    #: weights, fresh optimizer/schedule) instead of starting from random init.
    init_adapter_uri: str = Field(
        "", description="Adapter to CONTINUE training from: s3://…/adapter/ or a local dir with "
                        "adapter_config.json. Empty = fresh adapter. Overrides lora_r/alpha/target_modules "
                        "(the loaded adapter_config wins).")
    max_seq_len: int = Field(2048, gt=0)

    # Data split + reproducibility.
    eval_split: float = Field(0.1, ge=0, lt=1, description="Fraction held out for mid-training eval.")
    # cairn-extensions#4 (training.stage.train_qlora/1.1.0): the governed splits and the trainer
    # knobs. Every default keeps today's behaviour. train.run/1.0.0's input schema is open, so
    # these optional fields do not change that contract.
    train_uri: str = Field("", description="A governed train split; when set it is the training data, in place of dataset_uri.")
    eval_uri: str = Field("", description="A governed validation split; when set the trainer evaluates on it and eval_split is not used.")
    test_uri: str = Field("", description="The held-out test split; recorded on the result, never opened by the trainer.")
    per_device_batch_size: int = Field(0, ge=0, description="Rows per device per step; 0 keeps the trainer's default.")
    gradient_accumulation_steps: int = Field(0, ge=0, description="Steps accumulated before an optimizer step; 0 keeps the trainer's default.")
    gradient_checkpointing: Optional[bool] = Field(None, description="Trade compute for activation memory; None keeps the trainer's default.")
    lora_dropout: Optional[float] = Field(None, ge=0, le=1, description="LoRA dropout; None keeps the trainer's default.")
    seed: int = Field(42, description="Split/training seed for reproducibility.")

    #: Pipeline-validation mode (#536): run the ENTIRE training path on CPU with a
    #: tiny random model + a couple of steps, keeping the REAL dataset + prompt
    #: formatting — so dataset parsing, mapping/tokenization, trainer setup,
    #: metrics.json, and the downstream cost/learning/model gates all run in
    #: seconds with ZERO GPU spend before provisioning a real GPU. Also honored via
    #: the CAIRN_TRAIN_VALIDATE env var (set it on a CPU box, change no request).
    validate_pipeline: bool = Field(
        False, description="Validate the whole pipeline on CPU (tiny model, throttled) before GPU."
    )

    # Mid-training eval + job-resume checkpointing (session caps / spot GPUs).
    eval_steps: int = Field(50, ge=0, description="Evaluate every N steps (0 = epoch-end).")
    save_steps: int = Field(50, ge=0, description="Checkpoint every N steps; resume_from_checkpoint on restart.")
    #: ADR-0064 §3 (OOM safety): fixed per-step VRAM instead of ragged batches.
    packing: bool = Field(True, description="Pack multiple short sequences per step for predictable VRAM use.")
    #: ADR-0064 §3: bounds local + S3 checkpoint storage (older checkpoints pruned).
    save_total_limit: int = Field(2, ge=0, description="Keep only the N most-recent checkpoints (0 = unbounded).")

    # Where the trained adapter is saved.
    artifact_target: Literal["hf", "local", "s3"] = Field("hf", description="Adapter destination.")
    registered_model_name: str = Field(
        "", description="If set and MLflow is reporting, register the trained adapter as a new "
                        "version of this MLflow Registered Model (runs:/<run>/adapter). Empty = no registration.")
    hf_repo: str = Field("", description="HF Hub repo id (user/name) for artifact_target=hf; defaults from output_name.")

    # Backend hint (provider maps to its GPU types).
    gpu: str = Field("T4", description="GPU type for the backend (T4, A10G, A100, …).")

    # Experiment tracking (#136). Selects which tracker(s) the trainer reports to
    # (HF report_to). Credentials come from a tracking.* connection injected into
    # the run env by the operator — never in the spec. Empty = no tracking.
    report_to: list[Literal["mlflow", "wandb"]] = Field(
        default_factory=list,
        description="Experiment trackers: 'mlflow' and/or 'wandb' (empty = none).",
    )
    tracking_project: str = Field(
        "", description="MLflow experiment / W&B project name (defaults from output_name)."
    )
    tracking_run_name: str = Field(
        "", description="Tracking run name (defaults from output_name)."
    )

    #: Part of the published train.run/1.0.0 input (ADR-0104 D6, #2074): which compute
    #: connection a managed provider's job runs on. Read only by managed-provider operations,
    #: which this wheel does not ship; the stub and local trainers ignore it.
    compute_connection: str = Field(
        "", description="Name of the compute.<provider> connection this managed job "
                        "runs on (e.g. the `cairn connect runpod` connection). Empty = "
                        "the dispatch-bound compute target, if any.")

    extra: dict[str, Any] = Field(default_factory=dict, description="Backend/trainer passthrough.")


class TrainResult(BaseModel):
    """Outcome of a fine-tuning job."""

    status: Literal["succeeded", "failed", "running"] = "succeeded"
    backend: str = ""
    #: #2189 (T04) — where the adapter bytes are on the producing box, even after
    #: `adapter_uri` was replaced by a published (hf://, s3://) location: the
    #: candidate's content identity is computed from these bytes.
    local_adapter_dir: str = ""
    adapter_uri: str = Field("", description="Location of the produced adapter (safetensors).")
    run_url: str = Field("", description="Tracking/job URL, if any.")
    job_id: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
