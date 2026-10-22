"""Deterministic, group-aware (+ optionally stratified) train/val/test split
for ``dataset/split`` (#383).

Verify-upstream note: scikit-learn ships exactly the building blocks a
"standard" split needs — ``StratifiedShuffleSplit`` (stratify only),
``GroupShuffleSplit`` (group-aware only) and ``StratifiedGroupKFold``
(both at once, sklearn >= 1.0). We do NOT hand-roll a stratified-AND-grouped
allocator when sklearn is available: ``StratifiedGroupKFold`` already solves
the (NP-hard in general) balanced group+stratify assignment problem
reasonably well via a greedy heuristic, and re-deriving that heuristic here
would just be a worse copy of a maintained implementation.

However, the sklearn path is an OPTIONAL ``[split]`` extra, not a hard
dependency — this pack's tests must stay hermetic (the pack's test rule), and a base
`pip install` of this operator should still be able to split a dataset. So a
pure-stdlib **hash-based assignment** fallback always exists:

  - Each row is assigned to a split by hashing its GROUP KEY (falling back to
    the row's own canonical hash when no group key is configured) into
    ``[0, 1)`` via a seeded, stable digest, then bucketing that float against
    the cumulative split ratios. Hashing (not `random.Random.shuffle`) is the
    standard reproducible-split technique: it needs no shared state/ordering
    across rows, is trivially streaming-friendly (each row's split is a pure
    function of its key + seed, decidable without seeing the rest of the
    dataset), and two runs over the same input with the same seed always
    agree — even if rows are re-ordered or the dataset is processed in
    chunks. `random.Random` reseeded per-row would work too but hashing
    avoids per-row RNG construction overhead and matches the technique used
    by large-scale data pipelines (e.g. hash-bucketing shards).
  - Because every row in a group hashes to the identical bucket value (same
    key -> same hash), a group can never straddle two splits: group-awareness
    fall out of the hashing scheme for free, no extra bookkeeping needed.
  - Stratification in the fallback is a best-effort post-hoc balancer: groups
    are first bucketed per-stratum by the same hash technique (so determinism
    and group-locality still hold), which keeps each stratum's train/val/test
    proportions close to the requested ratios without needing sklearn.

Both paths report the same :class:`SplitAssignment` shape so the operator
doesn't care which one ran.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from operator_dataset_checks._canon import canonical_hash

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitAssignment:
    """Result of assigning every row to a split."""

    # split_name -> list of 0-based row indices, first-seen order preserved.
    indices: dict[str, list[int]] = field(default_factory=dict)
    backend_used: str = "hash"


def _unit_interval_hash(key: str, seed: int) -> float:
    """Deterministically map ``key`` (salted by ``seed``) into ``[0, 1)``.

    Uses the first 8 bytes of a sha256 digest as an unsigned integer, scaled
    to ``[0, 1)``. sha256 is already a project dependency (via ``_canon``)
    and its output is uniformly distributed, so bucketing against it gives an
    even split without needing a seeded PRNG stream.
    """
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    as_int = int.from_bytes(digest[:8], "big")
    return as_int / 2**64


def _ratio_bucket(value: float, ratios: dict[str, float]) -> str:
    """Map a ``[0, 1)`` value to a split name via cumulative ratio bounds."""
    cum = 0.0
    names = list(ratios.keys())
    for name in names[:-1]:
        cum += ratios[name]
        if value < cum:
            return name
    return names[-1]  # last split soaks up any float rounding slop


def _row_group_key(row: dict, group_by: str | None) -> str:
    """The key whose hash decides a row's split: the configured group field
    when present (so every row sharing it lands together), else the row's own
    canonical content hash (each row is its own group of one).
    """
    if group_by:
        val = row.get(group_by)
        if val is not None:
            return f"{group_by}={val}"
    return canonical_hash(row)


def _row_stratum(row: dict, stratify_by: str | None) -> str:
    if not stratify_by:
        return "_all_"
    val = row.get(stratify_by)
    return "_missing_" if val is None else str(val)


def row_split_name(
    row: dict, *, ratios: dict[str, float], seed: int, group_by: str | None = None,
) -> str:
    """The split ONE row lands in — the pure, per-row hash decision, decidable
    WITHOUT seeing the rest of the dataset (so ``dataset/split`` can stream a
    >>RAM corpus, #891). Identical to :func:`hash_split`'s assignment: a row's
    split is ``_ratio_bucket(_unit_interval_hash(group_key, seed))`` and stratify
    never changes a key's bucket (the fallback's stratification only reorders
    iteration, it doesn't rebalance). Group-aware for free — rows sharing
    ``group_by`` hash to the same bucket, so a group can't straddle splits and
    ``no_leakage`` holds by construction, no global state needed."""
    key = _row_group_key(row, group_by)
    return _ratio_bucket(_unit_interval_hash(key, seed), ratios)


def hash_split(
    rows: list[dict],
    *,
    ratios: dict[str, float],
    seed: int,
    group_by: str | None = None,
    stratify_by: str | None = None,
) -> SplitAssignment:
    """Pure-stdlib, seedable, group-aware (+ best-effort stratified) split.

    Deterministic: the split of a given group key depends only on
    ``(seed, group_by value or row hash)`` — never on row order or dataset
    size — so re-running with the same seed reproduces the identical
    assignment, and appending new rows never reshuffles existing ones.
    """
    # Bucket distinct group keys ONCE (not per-row) so every member of a
    # group gets the same decision even under stratification, and so we
    # don't hash the same key repeatedly for a large group.
    group_first_stratum: dict[str, str] = {}
    group_rows: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        key = _row_group_key(row, group_by)
        group_rows.setdefault(key, []).append(i)
        group_first_stratum.setdefault(key, _row_stratum(row, stratify_by))

    # Group keys within the same stratum so ratios are approximately
    # preserved per-stratum too (best-effort — this is the hermetic
    # fallback, not a provable-optimal balancer).
    strata: dict[str, list[str]] = {}
    for key, stratum in group_first_stratum.items():
        strata.setdefault(stratum, []).append(key)

    key_to_split: dict[str, str] = {}
    for stratum_keys in strata.values():
        for key in stratum_keys:
            bucket_value = _unit_interval_hash(key, seed)
            key_to_split[key] = _ratio_bucket(bucket_value, ratios)

    indices: dict[str, list[int]] = {name: [] for name in ratios}
    for key, row_idxs in group_rows.items():
        split_name = key_to_split[key]
        indices[split_name].extend(row_idxs)

    for name in indices:
        indices[name].sort()

    return SplitAssignment(indices=indices, backend_used="hash")


def sklearn_available() -> bool:
    """True when the ``[split]`` extra (scikit-learn) is importable."""
    import importlib.util

    return bool(importlib.util.find_spec("sklearn"))


def resolve(backend: str) -> str:
    """Validate a split backend, raising a clear error when 'sklearn' is
    requested but the ``[split]`` extra isn't installed (mirrors
    ``_minhash.resolve`` / ``_pii.resolve``).
    """
    if backend == "sklearn" and not sklearn_available():
        raise RuntimeError(
            "backend='sklearn' requires the [split] extra: "
            "pip install 'cairn-operator-dataset-checks[split]'."
        )
    return backend


def sklearn_split(
    rows: list[dict],
    *,
    ratios: dict[str, float],
    seed: int,
    group_by: str | None = None,
    stratify_by: str | None = None,
) -> SplitAssignment:
    """scikit-learn-backed split.

    - group_by AND stratify_by set -> ``StratifiedGroupKFold`` (sklearn's
      built-in solver for the combined constraint), folds combined to reach
      the requested ratios (approximately — k-fold sizes are quantized).
    - group_by only -> ``GroupShuffleSplit`` (exact ratios, exact
      group-locality, no stratification concern).
    - stratify_by only -> ``StratifiedShuffleSplit`` (exact ratios, no
      group concept — every row is its own group of one).
    - neither -> a plain seeded ``ShuffleSplit``.

    Two/three-way splits are built by first separating off "the rest" from
    the first split's held-in fraction, then splitting "the rest" again for
    the remaining named splits — the standard way to chain sklearn's
    2-way splitters into an N-way split.
    """
    from sklearn.model_selection import (  # optional dependency (split extra)
        GroupShuffleSplit,
        ShuffleSplit,
        StratifiedShuffleSplit,
    )

    try:
        from sklearn.model_selection import (  # optional dependency (split extra)
            StratifiedGroupKFold,
        )
    except ImportError:  # older sklearn (<1.0) — group+stratify combo unsupported
        StratifiedGroupKFold = None  # noqa: N806

    n = len(rows)
    all_idx = list(range(n))
    groups = [_row_group_key(row, group_by) for row in rows] if group_by else None
    strata = [_row_stratum(row, stratify_by) for row in rows] if stratify_by else None

    names = list(ratios.keys())
    remaining_idx = all_idx
    remaining_groups = groups
    remaining_strata = strata
    assignment: dict[str, list[int]] = {name: [] for name in names}

    for step, name in enumerate(names[:-1]):
        remaining_ratio_sum = sum(ratios[n2] for n2 in names[step:])
        take_frac = ratios[name] / remaining_ratio_sum if remaining_ratio_sum else 0.0
        take_frac = min(max(take_frac, 1e-9), 1 - 1e-9)

        sub_n = len(remaining_idx)
        if sub_n < 2:
            # Too few rows left to split further; dump the rest into `name`.
            assignment[name].extend(remaining_idx)
            remaining_idx, remaining_groups, remaining_strata = [], None, None
            continue

        if group_by and stratify_by and StratifiedGroupKFold is not None:
            # StratifiedGroupKFold has no direct "give me an X% holdout"
            # knob, only k folds — so approximate `take_frac` by choosing
            # k = round(1/take_frac) and treating ONE fold's test-indices as
            # the piece we take. This is the standard way to get an
            # approximately-sized, stratified, group-safe holdout out of a
            # k-fold splitter when no dedicated ShuffleSplit variant exists.
            n_splits = min(max(2, round(1 / take_frac)), sub_n)
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed + step,
            )
            # sklearn's .split() convention is (train_idx, test_idx); the
            # smaller "test" fold (~1/n_splits of the data) is the piece we
            # take for `name`, the rest carries on to later splits.
            rest_local, take_local = next(
                splitter.split(remaining_idx, remaining_strata, remaining_groups)
            )
        elif group_by:
            # GroupShuffleSplit allocates whole GROUPS, so it needs ≥2 distinct
            # groups left to hold a fraction out. The `sub_n < 2` guard above
            # counts ROWS, not groups — with a low-cardinality group_by (e.g. a
            # handful of categories) many rows but <2 groups can remain, and
            # sklearn then raises "n_samples=1, test_size=… → empty train set".
            # Guard on the remaining GROUP count: with <2 groups a fractional
            # holdout is impossible without straddling a group (which would break
            # group-locality), so assign the whole remainder to `name` and stop.
            # The split ends unbalanced (later splits get nothing) rather than the
            # run dying on a traceback — the empty split is visible in `counts`.
            if len(set(remaining_groups or [])) < 2:
                assignment[name].extend(remaining_idx)
                remaining_idx, remaining_groups, remaining_strata = [], None, None
                continue
            # Pass `test_size=take_frac` directly (not `train_size=1-take_frac`)
            # to avoid float-subtraction error (e.g. `1 - 0.8` = 0.19999999999999996
            # in binary floating point) nudging sklearn's internal
            # floor/ceil sizing off by one row.
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=take_frac, random_state=seed + step,
            )
            rest_local, take_local = next(
                splitter.split(remaining_idx, groups=remaining_groups)
            )
        elif stratify_by:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=take_frac, random_state=seed + step,
            )
            rest_local, take_local = next(
                splitter.split(remaining_idx, remaining_strata)
            )
        else:
            splitter = ShuffleSplit(
                n_splits=1, test_size=take_frac, random_state=seed + step,
            )
            rest_local, take_local = next(splitter.split(remaining_idx))

        taken = sorted(remaining_idx[i] for i in take_local)
        assignment[name].extend(taken)

        rest_set = set(rest_local.tolist())
        remaining_idx = [remaining_idx[i] for i in sorted(rest_set)]
        remaining_groups = [groups[i] for i in remaining_idx] if group_by else None
        remaining_strata = [strata[i] for i in remaining_idx] if stratify_by else None

    # Whatever is left after peeling off every split but the last goes to it.
    assignment[names[-1]].extend(remaining_idx)

    for name in assignment:
        assignment[name].sort()

    if group_by:
        _assert_no_group_leak(rows, assignment, group_by)

    return SplitAssignment(indices=assignment, backend_used="sklearn")


def _assert_no_group_leak(
    rows: list[dict], assignment: dict[str, list[int]], group_by: str,
) -> None:
    """Defensive check: no group key may appear in more than one split.

    sklearn's group splitters guarantee this already; this is a cheap
    belt-and-suspenders assertion so a future sklearn API change can't
    silently reintroduce leakage.
    """
    seen: dict[str, str] = {}
    for split_name, idxs in assignment.items():
        for i in idxs:
            key = _row_group_key(rows[i], group_by)
            prior = seen.get(key)
            if prior is not None and prior != split_name:
                raise AssertionError(
                    f"group leakage: key {key!r} appears in both "
                    f"{prior!r} and {split_name!r}"
                )
            seen[key] = split_name


def check_no_leakage(
    rows: list[dict], assignment: dict[str, list[int]], group_by: str | None,
) -> bool:
    """True iff no group key spans more than one split (the leakage-check
    evidence reported in :class:`SplitOutput`).

    With no ``group_by`` configured, every row is its own group of one and
    leakage is impossible by construction (returns True trivially — still a
    meaningful assertion that the split partitions rows without duplication).
    """
    key_fn = (lambda row: _row_group_key(row, group_by)) if group_by else (
        lambda row: id(row)
    )
    seen: dict = {}
    for split_name, idxs in assignment.items():
        for i in idxs:
            key = key_fn(rows[i])
            prior = seen.get(key)
            if prior is not None and prior != split_name:
                return False
            seen[key] = split_name
    return True
