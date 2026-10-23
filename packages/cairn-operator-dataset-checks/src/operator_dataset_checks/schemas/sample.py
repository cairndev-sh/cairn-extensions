"""Schemas for dataset/sample — deterministic (input, output) sampling that
feeds the LLM-as-judge (eval/judge). Deterministic: same dataset → same sample.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SamplePair(BaseModel):
    input: str
    output: str


class SampleInput(BaseModel):
    path: str = Field(..., description="Path to the JSONL fine-tuning dataset.")
    n: int = Field(20, description="How many (input, output) pairs to sample.")


class SampleConfig(BaseModel):
    """Deployment-tunable knobs of ``dataset/sample`` (ADR-0070) — the sample size.
    **Operational** scope (the default): a workflow/run may tune this freely. A
    subset of ``SampleInput``; the safe default lives on ``SampleInput``."""

    model_config = ConfigDict(extra="forbid")

    n: int = Field(
        20, title="Sample size", description="How many (input, output) pairs to sample.",
    )


class SampleOutput(BaseModel):
    samples: list[SamplePair] = Field(
        default_factory=list, description="Evenly-spaced, deterministic sample."
    )
    total_rows: int = Field(0, description="Rows read from the dataset.")
    extractable: int = Field(
        0, description="Rows with a clear (input, output) split."
    )
    sampled: int = Field(0, description="Number of pairs returned.")
