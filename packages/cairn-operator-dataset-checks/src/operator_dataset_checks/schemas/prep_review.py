"""Schemas for dataset/prep_review — the HITL review of a dirty scrub (#883).

When ``dataset/prep_gate`` runs in ``mode=hitl`` and prep's re-scan still finds
PII, the prep template routes here. The operator declares ``requires_approval``
(pack.toml, ADR-0034), so the run PAUSES ``awaiting_approval`` BEFORE this body
runs — a human sees the residual-PII evidence and approves (ship the scrub) or
rejects (kill the run). This body executes ONLY on approve; it records the
sign-off.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PrepReviewInput(BaseModel):
    post_scrub_pii_rows: int = Field(
        0, ge=0, description="Residual PII rows the reviewer is signing off on."
    )
    post_scrub_pii_counts: dict[str, int] = Field(
        default_factory=dict, description="Per-kind residual counts, for the reviewer."
    )
    reasons: list[str] = Field(
        default_factory=list, description="The prep_gate cited reason(s) shown to the reviewer."
    )


class PrepReviewOutput(BaseModel):
    reviewed: bool = Field(
        ..., description="True — a human signed off (this body runs only after approval)."
    )
    residual_pii_rows: int = Field(0, description="Residual PII the human accepted.")
    note: str = Field("", description="Human-readable record of the sign-off.")
