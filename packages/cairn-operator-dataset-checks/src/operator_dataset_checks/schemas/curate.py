"""Schemas for dataset/curate — LLM-judge quality selection (#384).

Prompts an LLM (through the governed gateway) to rate each example's quality
and keeps only the examples that clear the bar. Two industry-standard methods:

- **AlpaGasus** (arXiv:2307.08701): rate each (instruction, response) on a fixed
  0-5 rubric; keep examples scoring above a threshold (default 4.5) or the top
  ``keep_fraction``.
- **DEITA** (arXiv:2312.15685): score complexity AND quality separately (each
  0-5) and select by their product ``evol_score = complexity * quality``, again
  thresholded or top-fraction.

Non-deterministic (it calls an LLM); risk ``paid_compute``. Writes a curated
JSONL of the kept rows plus a report of the score distribution.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RowScore(BaseModel):
    """The judge's verdict for one example."""

    index: int = Field(..., description="Row position in the (capped) input stream.")
    score: float = Field(
        ..., description="Selection score: AlpaGasus quality, or DEITA complexity*quality."
    )
    quality: Optional[float] = Field(
        None, description="Quality sub-score (both methods populate this)."
    )
    complexity: Optional[float] = Field(
        None, description="Complexity sub-score (DEITA only)."
    )
    kept: bool = Field(..., description="Whether this row met the selection criterion.")
    input_preview: str = Field("", description="First 120 chars of the example's input.")


class CurateInput(BaseModel):
    path: str = Field(..., description="Path to the source JSONL fine-tuning dataset.")
    output_path: str = Field(
        "curated.jsonl", description="Where to write the kept (curated) rows as JSONL."
    )
    method: str = Field(
        "alpagasus",
        description=(
            "'alpagasus' (single 0-5 quality score, default) | "
            "'deita' (complexity x quality, each 0-5)."
        ),
    )
    model: str = Field(
        "gpt-4o-mini",
        description="Gateway model id used as the quality judge.",
    )
    threshold: Optional[float] = Field(
        None,
        description=(
            "Keep rows whose score is >= this value. AlpaGasus default is 4.5 "
            "(applied when neither threshold nor keep_fraction is set). Mutually "
            "exclusive with keep_fraction; keep_fraction wins if both are set."
        ),
    )
    keep_fraction: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Keep the top fraction of rows by score (0.0-1.0), overriding "
            "threshold. e.g. 0.2 keeps the best-scoring 20%."
        ),
    )
    rubric: Optional[str] = Field(
        None,
        description=(
            "Custom rubric/criteria text spliced into the scoring prompt. "
            "Falls back to the method's standard rubric when unset."
        ),
    )
    max_samples: int = Field(
        200,
        ge=0,
        description="Cost guard: cap the number of examples scored (paid LLM calls).",
    )
    concurrency: int = Field(
        8, ge=1, description="Max in-flight LLM scoring calls (asyncio.Semaphore)."
    )
    temperature: float = Field(
        0.0, description="Judge sampling temperature (0.0 = most deterministic)."
    )


class CurateOutput(BaseModel):
    method: str = Field(..., description="Curation method used ('alpagasus' | 'deita').")
    model: str = Field(..., description="Judge model id.")
    criterion: str = Field(
        ..., description="How rows were selected, e.g. 'threshold>=4.5' or 'keep_fraction=0.2'."
    )
    total_rows: int = Field(0, description="Rows read from the source dataset.")
    extractable: int = Field(0, description="Rows with a clear (input, output) split.")
    scored: int = Field(0, description="Rows actually scored by the judge.")
    kept: int = Field(0, description="Rows kept in the curated output.")
    dropped: int = Field(0, description="Scored rows that failed the criterion.")
    parse_failures: int = Field(
        0, description="Rows whose LLM score couldn't be parsed (skipped, not kept)."
    )
    mean_score: float = Field(0.0, description="Mean selection score over scored rows.")
    score_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Histogram of scores bucketed by integer floor (e.g. '4' -> count).",
    )
    scores: list[RowScore] = Field(
        default_factory=list, description="Per-row score records (capped sample)."
    )
    output_path: str = Field(..., description="Path the curated JSONL was written to.")
