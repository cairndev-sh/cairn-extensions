"""Schemas for dataset/decontam — train↔eval contamination (exact overlap)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DecontamInput(BaseModel):
    path: str = Field(..., description="Path to the TRAINING JSONL dataset.")
    eval_path: Optional[str] = Field(
        None,
        description="Eval / benchmark JSONL to check contamination against. "
        "If omitted, the check is skipped (checked=false) rather than failing.",
    )
    normalize: bool = Field(
        True,
        description="Whitespace-collapse before hashing (must match dedup's setting).",
    )
    method: str = Field(
        "exact",
        description=(
            "Contamination method: 'exact' (default — canonical_hash byte-"
            "identical match), 'ngram' (word n-gram/shingle Jaccard overlap — "
            "also catches reworded/near-duplicate leaks), 'both' (union of the "
            "exact and n-gram row sets), or 'minhash' (MinHash+LSH approximate "
            "Jaccard overlap via datasketch — the scalable near-dup path for "
            "large corpora, catching paraphrased leakage without the O(train x "
            "eval) all-pairs shingle comparison; needs the [minhash] extra)."
        ),
    )
    ngram: int = Field(
        13, description="Shingle window size in words, used when method is "
        "'ngram', 'both' or 'minhash'. 13 matches common near-dup-detection "
        "defaults (e.g. GPT-3/RedPajama-style decontamination window sizes).",
    )
    overlap_threshold: float = Field(
        0.8,
        description="Jaccard similarity (0-1) at/above which a training row is "
        "flagged as near-duplicate contamination, used when method is 'ngram', "
        "'both' or 'minhash' (the MinHash estimate for 'minhash').",
    )
    num_perm: int = Field(
        128,
        description="MinHash permutation count (signature length) — higher is "
        "more accurate but slower. Used only when method='minhash'.",
    )


class DecontamConfig(BaseModel):
    """Deployment-tunable knobs of ``dataset/decontam`` (ADR-0070) — the near-dup
    overlap parameters. **Operational** scope (the default): a workflow/run may
    tune these freely. A subset of ``DecontamInput``; the safe defaults live on
    ``DecontamInput``."""

    model_config = ConfigDict(extra="forbid")

    overlap_threshold: float = Field(
        0.8, title="Contamination overlap threshold",
        description="Jaccard similarity (0-1) at/above which a train row is flagged as near-dup contamination.",
        # #791 — author-recommended (optimal) value, distinct from the safe default:
        # 0.7 catches more borderline contamination in practice. Surfaced as a
        # one-click suggestion in the console; never auto-applied.
        json_schema_extra={"recommended": 0.7},
    )
    ngram: int = Field(
        13, title="Shingle window",
        description="Shingle window size in words (ngram/both/minhash).",
    )
    num_perm: int = Field(
        128, title="MinHash permutations",
        description="MinHash signature length — higher is more accurate but slower (minhash).",
    )


class DecontamOutput(BaseModel):
    checked: bool = Field(
        ..., description="False when no eval set was provided (contamination not assessed).",
    )
    train_total: int = Field(..., description="Rows in the training set.")
    eval_total: int = Field(..., description="Rows in the eval/benchmark set.")
    contaminated: int = Field(
        ..., description="Training rows that also appear in the eval set.",
    )
    overlap_pct: float = Field(
        ..., description="contaminated / train_total * 100, rounded.",
    )
    contaminated_rows: list[int] = Field(
        default_factory=list,
        description="0-based training-row indices found in the eval set.",
    )
    ngram_contaminated_rows: list[int] = Field(
        default_factory=list,
        description="0-based training-row indices flagged by n-gram/shingle "
        "overlap (>= overlap_threshold) rather than an exact hash match. Empty "
        "when method='exact' (the default). Rows already in contaminated_rows "
        "via exact match are not repeated here even under method='both'.",
    )
    minhash_contaminated_rows: list[int] = Field(
        default_factory=list,
        description="0-based training-row indices flagged by MinHash+LSH "
        "approximate Jaccard overlap (>= overlap_threshold) against the eval "
        "set. Populated only when method='minhash'; empty otherwise.",
    )
