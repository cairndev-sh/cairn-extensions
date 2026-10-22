"""Schemas for dataset/capture_to_pairs — captured signal → DPO preference pairs.

ADR-0095 §1.2 (#1629). Output rows carry the shape the DPO rung already
consumes — ``{prompt, chosen, rejected, kind}``, the same one
``operator_training.flow_stages.dataset_dpo_pairs`` emits — so the existing
``train.dpo`` leg reads capture-derived pairs with no adapter in between.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from operator_dataset_checks.schemas.capture_to_sft import AdapterCount


class CaptureToPairsInput(BaseModel):
    path: str = Field(
        ..., description="The exported capture JSONL (one CaptureEvent per line).",
    )
    out_path: str = Field(..., description="Where to write the DPO pairs JSONL.")
    only_adapter: str = Field(
        "", description="Restrict to signal about ONE adapter_uri (attribution).",
    )
    pair_rejects: bool = Field(
        True, description="Also mine `reject` / `retry` rows by pairing them with "
                          "a LATER accept/edit/trace on the same trace_id or "
                          "run_id. Rows with no such successor are dropped and "
                          "counted — a rejection with nothing better to point at "
                          "teaches a preference model nothing.",
    )


class CaptureToPairsOutput(BaseModel):
    out_path: str = Field(..., description="Path the pairs dataset was written to.")
    rows: int = Field(0, description="Pairs written.")
    from_edits: int = Field(
        0, description="Pairs where the human's edit is `chosen` and the original "
                       "completion is `rejected` — the cleanest preference signal "
                       "capture produces.",
    )
    from_rejects: int = Field(0, description="Pairs mined from a `reject` row.")
    from_retries: int = Field(0, description="Pairs mined from a `retry` row.")
    unpaired_dropped: int = Field(
        0, description="reject/retry rows with no later accepted text on the same "
                       "correlation id — dropped, never paired against a guess.",
    )
    duplicates_dropped: int = Field(0, description="Pairs identical to one already written.")
    skipped: int = Field(0, description="Signal rows this projection does not consume.")
    by_adapter: list[AdapterCount] = Field(
        default_factory=list, description="Attribution readout, sorted.",
    )
