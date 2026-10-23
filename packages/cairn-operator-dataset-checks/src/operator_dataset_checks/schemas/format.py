"""Schemas for dataset/format — fine-tuning dataset format + structure check."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class FormatFinding(BaseModel):
    row: int = Field(..., description="0-based row index in the JSONL file.")
    issue: str = Field(..., description="What is structurally wrong with this row.")


class FormatInput(BaseModel):
    path: str = Field(
        ..., description="Path to a JSONL fine-tuning dataset (one example per line).",
    )
    target_trainers: list[str] = Field(
        default_factory=lambda: ["trl", "unsloth", "axolotl"],
        description="Trainers to report format compatibility for.",
    )
    expected_format: Optional[str] = Field(
        None,
        description=(
            "Force a format (openai_chat | sharegpt | alpaca | completion) "
            "instead of auto-detecting from the rows."
        ),
    )
    sample_size: Optional[int] = Field(
        None,
        description=(
            "When set, reservoir-sample this many rows before checking — a "
            "statistically valid estimate for a readiness gate; omit to scan "
            "every row."
        ),
    )


class FormatConfig(BaseModel):
    """Deployment-tunable knobs of ``dataset/format`` (ADR-0070) — the trainer set
    + scan bound. **Operational** scope (the default): a workflow/run may tune
    these freely. A subset of ``FormatInput``; the safe defaults live on
    ``FormatInput``."""

    model_config = ConfigDict(extra="forbid")

    target_trainers: list[str] = Field(
        default_factory=lambda: ["trl", "unsloth", "axolotl"],
        title="Target trainers",
        description="Trainers to report format compatibility for.",
    )
    sample_size: Optional[int] = Field(
        None, title="Format scan sample size",
        description="Reservoir-sample this many rows before checking (null = scan every row).",
    )


class FormatOutput(BaseModel):
    detected_format: str = Field(
        ..., description="openai_chat | sharegpt | alpaca | completion | unknown.",
    )
    total: int = Field(..., description="Total non-empty rows read (sampled count when sampled).")
    conformant: int = Field(
        ..., description="Rows passing structural validation for the detected format.",
    )
    conformant_pct: float = Field(..., description="conformant / total * 100, rounded.")
    findings: list[FormatFinding] = Field(
        default_factory=list, description="Per-row structural issues (deterministic).",
    )
    trainer_compatibility: dict[str, bool] = Field(
        default_factory=dict,
        description="Per target trainer: is the detected format accepted?",
    )
    sampled: bool = Field(
        False,
        description="True when the check ran over a reservoir sample, not every row.",
    )
    sample_size: Optional[int] = Field(
        None,
        description="Rows sampled when `sampled` is true; None when the full dataset was scanned.",
    )
