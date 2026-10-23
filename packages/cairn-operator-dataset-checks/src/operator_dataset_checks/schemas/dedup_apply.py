"""Schemas for dataset/dedup_apply — the *remediation* half of dataset/dedup.

dataset/dedup DETECTS + reports (read_only); dedup_apply DROPS the duplicate rows
and writes a cleaned JSONL (external_write). #991 (ADR-0076 D2/D3) — the repair
path a failed readiness gate needs. Reuses dedup's detection primitives so a
row dropped here is exactly a row the check flagged.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DedupApplyInput(BaseModel):
    path: str = Field(..., description="Path to the source JSONL dataset (one example per line).")
    output_path: str = Field(..., description="Where to write the deduplicated JSONL dataset.")
    normalize: bool = Field(
        True, description="Collapse whitespace before hashing (rows differing only in spacing are dups).",
    )
    method: str = Field(
        "exact",
        description="'exact' (canonical-hash), 'minhash' (near-dup clustering via [minhash]), "
                    "or 'both'. Non-exact methods need the [minhash] extra.",
    )
    threshold: float = Field(
        0.8, description="Jaccard similarity at/above which rows are near-duplicates (minhash/both).",
    )
    num_perm: int = Field(128, description="MinHash permutation count (minhash/both).")
    ngram: int = Field(3, description="Word-shingle window for MinHash (minhash/both).")


class DedupApplyOutput(BaseModel):
    output_path: str = Field(..., description="Path the deduplicated dataset was written to.")
    total: int = Field(..., description="Rows in the source dataset.")
    kept: int = Field(..., description="Rows written (survivors).")
    dropped: int = Field(..., description="Rows dropped (kept - total).")
    dropped_exact: int = Field(0, description="Rows dropped as exact duplicates.")
    dropped_near: int = Field(0, description="Additional rows dropped as near-duplicates (minhash/both).")
    method_used: str = Field(..., description="The dedup method actually applied.")
