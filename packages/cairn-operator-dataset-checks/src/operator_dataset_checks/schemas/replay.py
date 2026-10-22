"""Schemas for dataset/replay — rehearsal sampling over governed dataset versions.

ADR-0095 §2.3 (#1625): anti-forgetting is a DATA operation, not an algorithm.
Cycle N's fresh train slice is mixed with rows sampled from prior cycles'
governed dataset versions (#1466), and the mix is recorded per row-GROUP —
which source contributed how many rows — so lineage survives the blend without
stamping unknown keys onto the training rows themselves.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ReplayMixEntry(BaseModel):
    """One replay source's contribution to the mix — the per-row-group
    provenance record. Counts and locations only; never row text."""

    source: str = Field(
        ..., description="The source as given (a resolved dataset URI keeps its "
                         "'#sha256=<hash>' pin, so the mix names an exact version).",
    )
    rows_taken: int = Field(0, description="Rows sampled from this source into the mix.")
    rows_available: int = Field(
        0, description="Rows this source could have contributed (after the "
                       "dedup_against_new filter, if enabled).",
    )


class ReplayInput(BaseModel):
    new_rows_path: str = Field(
        ..., description="The cycle's fresh train slice (local path or resolved URI).",
    )
    out_path: str = Field(..., description="Where to write the mixed JSONL dataset.")
    sources: list[str] = Field(
        default_factory=list,
        description="Prior-cycle replay sources, OLDEST first. Each entry is a "
                    "resolved dataset URI or a local path; a launch-resolved "
                    "'dataset:<name>@<v>' ref arrives here as 'uri#sha256=<hash>' "
                    "and the fragment is stripped before reading (#1466).",
    )
    ratio: float = Field(
        0.2, ge=0.0, lt=1.0,
        description="Replay share of the FINAL MIX — the denominator is the "
                    "OUTPUT row count, not the new-row count. 0.2 means replay "
                    "rows are ~20% of the rows written (new_rows * ratio/(1-ratio) "
                    "replay rows). When the sources cannot fill the quota the op "
                    "takes everything available and reports the true ratio_actual.",
    )
    strategy: str = Field(
        "uniform",
        description="How the replay quota is spread across `sources` — an OPEN "
                    "enum (ADR-0095 Amendment 1 item 2). Implemented: 'uniform' "
                    "(each source's share is proportional to its eligible row "
                    "count, i.e. every prior row is equally likely) and "
                    "'recency_weighted' (share proportional to position+1, so "
                    "LATER entries in `sources` — the more recent cycles — get "
                    "more of the quota). RESERVED for v2, rejected with a clear "
                    "error until #1626 lands the per-row/per-version eval "
                    "history they need: 'hard_example' (rows the current model "
                    "gets wrong) and 'forgetting_aware' (rows whose score "
                    "degraded between adapter versions).",
    )
    seed: int = Field(
        0, description="Seed for ALL sampling. Same inputs + same seed → "
                       "byte-identical output, so the mix is reproducible from "
                       "the receipt.",
    )
    dedup_against_new: bool = Field(
        True, description="Drop replay rows that exactly duplicate a new row "
                          "(the pack's canonical row hash — the same notion of "
                          "'the same example' dataset/dedup uses).",
    )
    normalize: bool = Field(
        True, description="Collapse whitespace before hashing for "
                          "dedup_against_new (matches dataset/dedup).",
    )
    eval_report: Optional[str] = Field(
        None,
        description="RESERVED (ADR-0095 Amendment 1 item 2): the eval/forgetting "
                    "(#1626) report the v2 'hard_example' / 'forgetting_aware' "
                    "strategies will score candidate rows against. Accepted now "
                    "so v2 is additive; ignored by the v1 strategies.",
    )


class ReplayConfig(BaseModel):
    """Deployment-tunable knobs of ``dataset/replay`` (ADR-0070) — the mix
    dial and its determinism seed. **Operational** scope (the default): a
    workflow/run may tune these freely (ADR-0095 Amendment 1 item 3 wants the
    replay ratio steerable by a human-approved suggestion). A subset of
    ``ReplayInput``; the safe defaults live on ``ReplayInput``."""

    model_config = ConfigDict(extra="forbid")

    ratio: float = Field(
        0.2, ge=0.0, lt=1.0, title="Replay ratio",
        description="Replay share of the final mix (denominator = output rows).",
    )
    strategy: str = Field(
        "uniform", title="Sampling strategy",
        description="'uniform' or 'recency_weighted' (v1).",
    )
    seed: int = Field(0, title="Seed", description="Sampling seed; same seed → same mix.")


class ReplayOutput(BaseModel):
    out_path: str = Field(..., description="Path the mixed dataset was written to.")
    total_rows: int = Field(0, description="Rows written (new + replay).")
    new_rows: int = Field(0, description="Rows carried over from new_rows_path.")
    replay_rows: int = Field(0, description="Rows sampled from the replay sources.")
    ratio_actual: float = Field(
        0.0, description="replay_rows / total_rows — the ratio actually achieved, "
                         "which is below `ratio` when the sources ran dry.",
    )
    strategy_used: str = Field("uniform", description="The strategy actually applied.")
    mix: list[ReplayMixEntry] = Field(
        default_factory=list,
        description="Per-row-group provenance: one entry per source, in `sources` "
                    "order — the same order their blocks appear in the output.",
    )
