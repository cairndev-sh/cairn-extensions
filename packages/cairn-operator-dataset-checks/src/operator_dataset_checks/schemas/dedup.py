"""Schemas for dataset/dedup — exact duplicate detection + effective size."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DuplicateGroup(BaseModel):
    hash: str = Field(..., description="Short content-hash prefix shared by the group.")
    rows: list[int] = Field(..., description="0-based indices of the identical rows.")


class NearDuplicateGroup(BaseModel):
    rows: list[int] = Field(
        ..., description="0-based indices of rows clustered as near-duplicates "
        "(MinHash Jaccard >= threshold, transitively grouped).",
    )


class DedupInput(BaseModel):
    path: str = Field(
        ..., description="Path to a JSONL fine-tuning dataset (one example per line).",
    )
    normalize: bool = Field(
        True,
        description=(
            "Collapse whitespace before hashing, so rows differing only in "
            "spacing count as duplicates. Set False for byte-exact dedup."
        ),
    )
    method: str = Field(
        "exact",
        description=(
            "Duplicate detection method: 'exact' (default — canonical_hash "
            "byte-identical grouping, zero-dep), 'minhash' (MinHash+LSH "
            "near-duplicate clustering via datasketch, catches reworded/"
            "paraphrased dups exact hashing misses), or 'both' (exact groups "
            "plus a separate near-duplicate report). 'minhash'/'both' need the "
            "[minhash] extra."
        ),
    )
    threshold: float = Field(
        0.8,
        description="Jaccard similarity (0-1) at/above which two rows are "
        "clustered as near-duplicates, used when method is 'minhash' or 'both'.",
    )
    num_perm: int = Field(
        128,
        description="MinHash permutation count — the signature length. Higher "
        "is more accurate (less LSH estimation error) but slower/larger. "
        "Used when method is 'minhash' or 'both'.",
    )
    ngram: int = Field(
        3,
        description="Word-shingle window size for MinHash. Shorter windows make "
        "near-dup matching more lenient; used when method is 'minhash' or "
        "'both'.",
    )


class DedupConfig(BaseModel):
    """Deployment-tunable knobs of ``dataset/dedup`` (ADR-0070) — the MinHash
    near-duplicate parameters. **Operational** scope (the default): a workflow/run
    may tune these freely. A subset of ``DedupInput``; the safe defaults live on
    ``DedupInput`` so an un-configured run is unchanged."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(
        0.8, title="Near-duplicate Jaccard threshold",
        description="Jaccard similarity (0-1) at/above which two rows cluster as near-duplicates (minhash/both).",
    )
    num_perm: int = Field(
        128, title="MinHash permutations",
        description="MinHash signature length — higher is more accurate but slower (minhash/both).",
    )
    ngram: int = Field(
        3, title="Shingle window",
        description="Word-shingle window size for MinHash; shorter is more lenient (minhash/both).",
    )


class DedupOutput(BaseModel):
    total: int = Field(..., description="Total non-empty rows read.")
    unique: int = Field(..., description="Distinct rows after dedup (= effective size).")
    exact_duplicates: int = Field(..., description="Rows that are duplicates of an earlier row.")
    effective_size: int = Field(..., description="Usable training rows after dedup (= unique).")
    duplicate_pct: float = Field(..., description="exact_duplicates / total * 100, rounded.")
    duplicate_groups: list[DuplicateGroup] = Field(
        default_factory=list,
        description="One entry per set of identical rows (size > 1).",
    )
    near_duplicate_groups: list[NearDuplicateGroup] = Field(
        default_factory=list,
        description="MinHash+LSH near-duplicate clusters (size > 1). Empty when "
        "method='exact' (the default). Includes exact-duplicate rows too, since "
        "they are also near-duplicates of each other.",
    )
    near_duplicates: int = Field(
        0,
        description="Total rows that are near-duplicates of an earlier row "
        "(sum over near_duplicate_groups of len-1). 0 when method='exact'.",
    )
