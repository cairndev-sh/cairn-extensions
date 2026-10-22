"""Shared canonicalization for the dataset_checks operators (intra-bundle).

dataset/dedup (within-set duplicates) and dataset/decontam (across-set overlap
vs an eval/benchmark set) MUST agree on what "the same example" means, so the
hash lives here and both import it.
"""
from __future__ import annotations

import hashlib
import json


def canonical_hash(row: dict, normalize: bool = True) -> str:
    """Deterministic content hash of a dataset row: sorted-key JSON, optionally
    whitespace-collapsed, then sha256."""
    s = json.dumps(row, sort_keys=True, ensure_ascii=False)
    if normalize:
        s = " ".join(s.split())
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
