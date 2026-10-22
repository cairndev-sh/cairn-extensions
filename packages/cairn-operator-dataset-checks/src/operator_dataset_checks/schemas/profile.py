"""Schemas for dataset/profile — lightweight, deterministic dataset profile."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LengthStats(BaseModel):
    min: int = Field(..., description="Shortest response (chars).")
    max: int = Field(..., description="Longest response (chars).")
    mean: float = Field(..., description="Mean response length (chars), rounded.")
    p50: int = Field(..., description="Median response length (chars).")


class ProfileInput(BaseModel):
    path: str = Field(
        ..., description="Path to a JSONL fine-tuning dataset (one example per line).",
    )
    short_threshold: int = Field(
        10, description="Responses shorter than this many chars are flagged as short.",
    )
    refusal_markers: Optional[list[str]] = Field(
        None,
        description="Case-insensitive substrings that mark a refusal (default set used if None).",
    )
    token_backend: str = Field(
        "whitespace",
        description=(
            "Token-count backend: 'whitespace' (default, zero-dep word count), "
            "'tokenizers' (HF tokenizers subword count; requires the [tokenizers] "
            "extra), or 'auto' (tokenizers if importable, else whitespace)."
        ),
    )
    tokenizer_name: Optional[str] = Field(
        None,
        description="HF Hub tokenizer id to load (e.g. 'bert-base-uncased'). "
        "Used only when token_backend requests tokenizers; fetched via "
        "Tokenizer.from_pretrained (needs network/HF cache).",
    )
    tokenizer_path: Optional[str] = Field(
        None,
        description="Local tokenizer.json path (Tokenizer.from_file — fully "
        "offline). Takes precedence over tokenizer_name when both are set.",
    )
    sample_size: Optional[int] = Field(
        None,
        description=(
            "When set, reservoir-sample this many rows before checking — a "
            "statistically valid estimate for a readiness gate; omit to scan "
            "every row."
        ),
    )
    strict: bool = Field(
        False,
        description=(
            "Malformed-line handling (#1008): False (default) quarantines bad "
            "JSONL lines to a side file and reports the count; True restores "
            "fail-fast (raise on the first malformed line)."
        ),
    )


class ProfileConfig(BaseModel):
    """Deployment-tunable knobs of ``dataset/profile`` (ADR-0070) — the short-
    response threshold + scan bound. **Operational** scope (the default): a
    workflow/run may tune these freely. A subset of ``ProfileInput``; the safe
    defaults live on ``ProfileInput``."""

    model_config = ConfigDict(extra="forbid")

    short_threshold: int = Field(
        10, title="Short-response threshold",
        description="Responses shorter than this many chars are flagged as short.",
    )
    sample_size: Optional[int] = Field(
        None, title="Profile scan sample size",
        description="Reservoir-sample this many rows before checking (null = scan every row).",
    )


class ProfileOutput(BaseModel):
    total: int = Field(..., description="Total non-empty rows read (sampled count when sampled).")
    response_length_chars: LengthStats = Field(
        ..., description="Distribution of the assistant/output response length.",
    )
    refusals: int = Field(..., description="Responses matching a refusal marker.")
    refusal_pct: float = Field(..., description="refusals / total * 100, rounded.")
    short_responses: int = Field(
        ..., description="Responses below short_threshold chars (likely low-signal).",
    )
    response_length_tokens: Optional[LengthStats] = Field(
        None,
        description="Distribution of the assistant/output response length in "
        "tokens, per token_backend. None when token_backend='whitespace' was "
        "never requested to run (kept optional so existing consumers of the "
        "char-only output are unaffected).",
    )
    token_backend_used: Optional[str] = Field(
        None, description="The resolved token backend actually used ('whitespace' or 'tokenizers').",
    )
    sampled: bool = Field(
        False,
        description="True when the check ran over a reservoir sample, not every row.",
    )
    sample_size: Optional[int] = Field(
        None,
        description="Rows sampled when `sampled` is true; None when the full dataset was scanned.",
    )
    quarantined_rows: int = Field(
        0,
        description="Malformed source lines routed to the quarantine side file "
        "instead of crashing the run (#1008). 0 for a clean dataset.",
    )
    quarantine_path: Optional[str] = Field(
        None,
        description="Path of the quarantine side file (line number + parse error "
        "+ bounded raw prefix per bad line); None when nothing was quarantined.",
    )
