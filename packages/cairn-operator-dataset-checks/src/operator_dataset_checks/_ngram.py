"""N-gram (shingle) overlap for dataset/decontam (issue #381).

Exact ``canonical_hash`` matching (``_canon.py``) only catches byte-identical
rows. It misses NEAR-duplicate contamination — an eval question copied into
training with a reworded prefix, extra whitespace, or a paraphrased sentence
still leaks the answer, but its hash differs. Word-level n-gram shingling +
Jaccard similarity is the standard cheap way to catch that: split each text
into overlapping windows of ``n`` words (a "shingle"), and compare the two
shingle SETS.

Verify-upstream note: this is intentionally plain stdlib (``set`` intersection
/ union), not a library. The natural library upgrade path — MinHash + LSH via
``datasketch`` — approximates Jaccard similarity to scale to millions of rows
without an O(n*m) all-pairs comparison; that's flagged in ``_io.py`` for a
LATER issue (#382), not this one. For decontam's use case (one eval set vs one
train set, checked per-row) exact shingle sets are simpler, dependency-free,
and precise — no approximation error to reason about.
"""
from __future__ import annotations


def shingles(text: str, n: int) -> set[tuple[str, ...]]:
    """The set of overlapping ``n``-word windows ("shingles") in ``text``.

    Whitespace-tokenized (case-folded) — matches ``canonical_hash``'s
    normalize-then-compare spirit without needing a real tokenizer. Texts
    shorter than ``n`` words yield a single shingle of everything available so
    short rows are still comparable rather than contributing an empty set.
    """
    words = text.lower().split()
    if not words:
        return set()
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two sets; 0.0 when both are empty (no shared
    signal to compare, not "fully similar")."""
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def row_text(row: dict) -> str:
    """Flatten a dataset row to one string of its text content for shingling.

    Mirrors ``canonical_hash``'s row-agnostic approach — rather than assume a
    schema, join every string leaf in the row (works across chat/alpaca/
    completion formats without per-format branching).
    """
    parts: list[str] = []
    _collect_strings(row, parts)
    return " ".join(parts)


def _collect_strings(obj: object, out: list[str]) -> None:
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_strings(v, out)
