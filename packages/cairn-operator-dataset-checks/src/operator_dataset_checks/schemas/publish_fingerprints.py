"""Schemas for dataset/publish_fingerprints — the CONTROL-SIDE half of
cross-batch dedup (#1050, ADR-0081 P1c · ADR-0080 P2).

Publishing is deliberately not a dataflow stage: the data plane runs on a box
with no database access, so the row-hash index has to be written from the control
plane after the job returns its cleaned output. This operator is that step.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PublishFingerprintsInput(BaseModel):
    path: str = Field(
        ...,
        description="Path to the CLEANED JSONL a dataflow produced (control-side "
                    "copy — the SSH engine fetches export outputs back). Its rows "
                    "are what a later batch's cross-batch probe dedups against.",
    )
    dataset_id: str = Field(
        ...,
        description="Logical dataset this batch belongs to — the index is scoped to it. "
                    "Empty means 'no cross-batch identity', and publishing is skipped.",
    )
    batch_id: str = Field(
        ...,
        description="This batch's id; a probe excludes it so a batch never dedups "
                    "against itself.",
    )
    normalize: bool = Field(
        True,
        description="Collapse whitespace before hashing — MUST match the setting the "
                    "dedup stage used, or the published hashes won't line up with "
                    "what a future probe computes.",
    )


class PublishFingerprintsOutput(BaseModel):
    published: int = Field(
        0, description="Row hashes written to the index (0 when skipped — see `skipped`).",
    )
    dataset_id: str = ""
    batch_id: str = ""
    skipped: Optional[str] = Field(
        None,
        description="Why nothing was published: 'no_dataset_id' | 'no_rows' | "
                    "'store_unavailable'. None on a real publish.",
    )
    summary: str = ""
