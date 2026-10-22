"""Schemas for dataset/decontam_apply — the *remediation* half of dataset/decontam.

dataset/decontam DETECTS train↔eval contamination + reports (read_only);
decontam_apply DROPS the contaminated training rows and writes a cleaned JSONL
(external_write). #991 (ADR-0076 D3). Reuses decontam's detection so a row it
quarantines is exactly one the check flagged as leaking the eval set.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DecontamApplyInput(BaseModel):
    path: str = Field(..., description="Path to the TRAINING JSONL dataset.")
    eval_path: Optional[str] = Field(
        None, description="Eval/benchmark JSONL to decontaminate against. If omitted, nothing is dropped.",
    )
    output_path: str = Field(..., description="Where to write the decontaminated training JSONL.")
    normalize: bool = Field(True, description="Whitespace-collapse before hashing (match dedup's setting).")
    method: str = Field(
        "exact",
        description="'exact', 'ngram', 'both', or 'minhash' (needs [minhash]) — mirrors dataset/decontam.",
    )
    ngram: int = Field(13, description="Shingle window in words (ngram/both/minhash).")
    overlap_threshold: float = Field(
        0.8, description="Jaccard similarity at/above which a training row is contaminated.",
    )
    num_perm: int = Field(128, description="MinHash permutation count (minhash).")


class DecontamApplyOutput(BaseModel):
    output_path: str = Field(..., description="Path the decontaminated dataset was written to.")
    total: int = Field(..., description="Rows in the source training dataset.")
    kept: int = Field(..., description="Rows written (survivors).")
    dropped: int = Field(..., description="Contaminated rows dropped (kept - total).")
    checked: bool = Field(..., description="False when no eval_path was given (nothing dropped).")
    method_used: str = Field(..., description="The contamination method actually applied.")
