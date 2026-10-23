"""Schemas for dataset/capture_to_sft — captured signal → gold SFT rows.

ADR-0095 §1.2 (#1629). The SFT rung wants examples of what GOOD looks like; the
capture store holds two kinds of evidence for that, and they are not equal. An
``accept`` says the completion was good enough to ship. An ``edit`` says what
good actually WAS — the human wrote it down. So an edit's ``edited_text`` is the
assistant turn, and the completion it replaced is discarded here (it becomes the
``rejected`` half over in ``dataset/capture_to_pairs``).
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AdapterCount(BaseModel):
    """Rows contributed by signal about ONE model/adapter — the attribution
    readout (ADR-0095 capture decision 1). Counts only; never row text."""

    adapter_uri: str = Field("", description="The adapter the rows are evidence about ('' = unattributed).")
    model: str = Field("", description="The model name captured alongside it.")
    rows: int = Field(0, description="Projected rows from signal about it.")


class CaptureToSftInput(BaseModel):
    path: str = Field(
        ..., description="The exported capture JSONL (one CaptureEvent per line) — "
                         "a local path or a resolved 'dataset:capture-signal@N' URI.",
    )
    out_path: str = Field(..., description="Where to write the SFT JSONL.")
    include_accepts: bool = Field(
        True, description="Include `accept` rows (prompt + the completion as "
                          "shipped). `edit` rows are ALWAYS included — they are "
                          "the richest signal in the store. Turning this off "
                          "trains only on corrections, which is the right call "
                          "when accepts are a passive default rather than a "
                          "deliberate act.",
    )
    only_adapter: str = Field(
        "", description="Restrict to signal about ONE adapter_uri. Without it a "
                        "cycle-9 training set silently absorbs complaints about "
                        "cycle-3 behaviour, which is the failure the attribution "
                        "field exists to prevent.",
    )
    system_prompt: str = Field(
        "", description="Optional system turn prepended to every example. Empty "
                        "(the default) writes user/assistant pairs only.",
    )


class CaptureToSftOutput(BaseModel):
    out_path: str = Field(..., description="Path the SFT dataset was written to.")
    rows: int = Field(0, description="Rows written.")
    from_edits: int = Field(0, description="Rows whose assistant turn is a human EDIT.")
    from_accepts: int = Field(0, description="Rows whose assistant turn was accepted as-is.")
    duplicates_dropped: int = Field(
        0, description="Rows identical to one already written (canonical hash).",
    )
    skipped: int = Field(
        0, description="Signal rows that could not become an example — no prompt, "
                       "no gold text, or a kind this projection does not consume.",
    )
    by_adapter: list[AdapterCount] = Field(
        default_factory=list,
        description="Attribution readout, sorted by adapter_uri then model.",
    )
