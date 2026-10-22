"""MinHash + LSH near-duplicate detection for dataset/dedup and decontam (#382).

Exact ``canonical_hash`` matching (``_canon.py``) only catches byte-identical
rows; ``_ngram.py`` catches NEAR-duplicates by comparing full shingle SETS with
Jaccard similarity, but that is an O(n·m) all-pairs comparison — fine for
decontam's one-eval-vs-one-train scale, quadratic-blowup at millions of rows.

MinHash + Locality-Sensitive Hashing (LSH) is the standard library answer:
approximate each row's shingle set by a fixed-length MinHash signature (an
unbiased Jaccard estimator), then bucket signatures into LSH bands so that only
rows whose bands collide are ever compared. Near-duplicates land in the same
bucket with high probability; unrelated rows almost never do — so we get
near-linear near-dup detection instead of the quadratic shingle-set scan.

Verify-upstream note: we do NOT hand-roll the MinHash permutations or the LSH
banding (both are subtly easy to get wrong — band/row split, hash mixing, the
false-negative/false-positive trade-off). We use `datasketch
<https://github.com/ekzhu/datasketch>`_ (``MinHash`` + ``MinHashLSH``), the
standard, maintained implementation, whose ``MinHashLSH(threshold, num_perm)``
constructor picks the band/row split that minimizes the probabilistic error at
the requested Jaccard threshold. It's behind the ``[minhash]`` extra so the base
install stays lean and hermetic (mirrors the ``_pii``/``_ngram`` module style);
``resolve()`` raises a clear ``RuntimeError`` when the extra is missing.

Determinism: ``datasketch``'s MinHash permutations are seeded, so with a FIXED
``seed`` (default :data:`SEED` below) the signatures — and therefore the LSH
buckets and the reported near-dup clusters — are byte-reproducible across
processes. Tests rely on this.
"""
from __future__ import annotations

import importlib.util

from operator_dataset_checks._ngram import row_text, shingles

# Fixed defaults so signatures/buckets are reproducible and the probabilistic
# error is small enough for the test fixtures to be deterministic.
SEED = 1
NUM_PERM = 128


def minhash_available() -> bool:
    """True when the ``[minhash]`` extra (datasketch) is importable."""
    return bool(importlib.util.find_spec("datasketch"))


# Method/backend values whose code path builds a MinHash and therefore needs
# the [minhash] extra. dedup's 'both' = exact groups + a near-dup report, so it
# needs datasketch too — not just the standalone 'minhash' value.
_NEEDS_MINHASH = frozenset({"minhash", "both"})


def needs_minhash(backend: str) -> bool:
    """True when ``backend`` (an operator method value) runs a MinHash path."""
    return backend in _NEEDS_MINHASH


def resolve(backend: str) -> str:
    """Validate a near-dup backend, raising a clear error when the extra a
    MinHash path needs is missing (mirrors ``_pii.resolve``).

    ``backend`` is the operator's own vocabulary ('exact'/'minhash'/'both' for
    dedup). Only MinHash paths (:func:`needs_minhash`) require the extra; every
    other value passes through untouched so the hermetic exact/ngram paths
    never require datasketch. NOTE: decontam's 'both' means exact+ngram (both
    hermetic), so decontam guards its 'minhash' value directly rather than via
    this helper — see decontam.py.
    """
    if needs_minhash(backend) and not minhash_available():
        raise RuntimeError(
            f"method/backend='{backend}' requires the [minhash] extra: "
            "pip install 'cairn-operator-dataset-checks[minhash]'."
        )
    return backend


def build_minhash(text: str, *, ngram: int, num_perm: int = NUM_PERM, seed: int = SEED):
    """A ``datasketch.MinHash`` signature over ``text``'s word-shingle set.

    Reuses ``_ngram.shingles`` so MinHash's shingling is identical to the exact
    n-gram path — the MinHash Jaccard estimate then approximates the SAME true
    shingle-set Jaccard the ngram method computes exactly, just scalably.
    """
    from datasketch import MinHash  # optional dependency (minhash extra)

    m = MinHash(num_perm=num_perm, seed=seed)
    for shingle in shingles(text, ngram):
        m.update(" ".join(shingle).encode("utf-8"))
    return m


def row_minhash(row: dict, *, ngram: int, num_perm: int = NUM_PERM, seed: int = SEED):
    """MinHash of a dataset row (flattened via ``_ngram.row_text``)."""
    return build_minhash(row_text(row), ngram=ngram, num_perm=num_perm, seed=seed)


def near_dup_groups(
    rows: list[dict],
    *,
    threshold: float,
    ngram: int,
    num_perm: int = NUM_PERM,
    seed: int = SEED,
) -> list[list[int]]:
    """Cluster ``rows`` into near-duplicate groups via MinHash + LSH.

    Returns one sorted list of row indices per cluster of size > 1 (first-seen
    order, so the output is stable). Two rows share a cluster when their MinHash
    Jaccard estimate is at/above ``threshold`` AND they collided in an LSH band.

    Clustering is transitive via union-find: A~B and B~C put A, B and C in one
    group even if A and C don't directly collide, matching how "duplicate
    groups" behave in exact dedup. LSH keeps this near-linear — each row is only
    compared against its bucket-mates, never all others.
    """
    signatures = [
        row_minhash(row, ngram=ngram, num_perm=num_perm, seed=seed) for row in rows
    ]
    return cluster_signatures(signatures, threshold=threshold, num_perm=num_perm)


def cluster_signatures(
    signatures: list, *, threshold: float, num_perm: int = NUM_PERM
) -> list[list[int]]:
    """Cluster PRECOMPUTED MinHash signatures into near-duplicate groups by
    position (0..len-1). The LSH + union-find REDUCE half of
    :func:`near_dup_groups`, separated so a sharded dedup (#962) can build
    signatures in parallel across shards, then cluster them **once, globally** —
    a near-dup pair that spans two shards is still caught, because clustering
    sees every signature. Deterministic (seeded signatures + first-seen union),
    so the sharded output equals the inline ``near_dup_groups`` exactly."""
    from datasketch import MinHashLSH  # optional dependency (minhash extra)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, sig in enumerate(signatures):
        lsh.insert(str(i), sig)

    parent = list(range(len(signatures)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Attach the higher index under the lower so roots stay first-seen.
            parent[max(ra, rb)] = min(ra, rb)

    for i, sig in enumerate(signatures):
        for match in lsh.query(sig):
            j = int(match)
            if j != i:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(len(signatures)):
        clusters.setdefault(find(i), []).append(i)

    return sorted(
        (sorted(idxs) for idxs in clusters.values() if len(idxs) > 1),
        key=lambda g: g[0],
    )


def contaminated_by_minhash(
    train: list[dict],
    eval_rows: list[dict],
    *,
    threshold: float,
    ngram: int,
    num_perm: int = NUM_PERM,
    seed: int = SEED,
) -> list[int]:
    """Training-row indices whose MinHash Jaccard vs ANY eval row is at/above
    ``threshold``, found via an LSH index over the eval set.

    Only the eval set is indexed; each train row is a single LSH query, so this
    is near-linear in train size (vs the ngram method's O(train·eval) all-pairs
    Jaccard) and scales to large corpora while still catching paraphrased leaks
    that exact-hash — and even fixed-n-gram-set overlap at scale — can miss.
    """
    from datasketch import MinHashLSH  # optional dependency (minhash extra)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for j, row in enumerate(eval_rows):
        lsh.insert(f"eval-{j}", row_minhash(row, ngram=ngram, num_perm=num_perm, seed=seed))

    matches: list[int] = []
    for i, row in enumerate(train):
        sig = row_minhash(row, ngram=ngram, num_perm=num_perm, seed=seed)
        if lsh.query(sig):
            matches.append(i)
    return matches
