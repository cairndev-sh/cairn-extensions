"""dataset/split — deterministic, group-aware (+ optionally stratified)
train/val[/test] split (#383).

Nothing splits datasets today (``dataset/sample`` is an evenly-spaced feeder
for the judge, not a split). This operator produces reproducible train/val
(+optional test) JSONL files:

- **Deterministic**: same input + seed -> byte-identical split (see
  ``_split.py`` for the hashing scheme).
- **Group-aware**: rows sharing a configurable group key (e.g. a doc/source
  id) always land in the SAME split — the operator asserts zero groups span
  splits and reports that as ``no_leakage``.
- **Stratified** (optional): balances a configurable label field's
  distribution across splits.
- **Test split is opt-in** (``test_ratio`` defaults to 0): SFT practice is
  train + a small held-out val for loss monitoring, not a classic
  70/15/15 split — final quality is measured via the finetune eval-gate's
  benchmarks, not a held-out test set here.

Split order per #378/#383: run this AFTER dedup/decontam so the group-aware
assignment can't be undermined by duplicates/near-dups the pipeline already
should have removed upstream.

**Frozen child versions (#2020, ADR-0102 D1).** With ``register_as`` set, each
split is not just a file but a registered CHILD VERSION of the parent dataset —
``<register_as>-train`` (role ``training``), ``-val`` (``validation``),
``-test`` (``test``) — carrying its own content hash, a ``parent_version``
pointer, and the DECLARED policy (ratios, ``stratify_by``, ``group_by``,
``seed``, backend) on the record. Items never move: the split is a pure
function of the rows and the policy, so re-running it with the same policy
re-registers the same hash (no new version), and re-running it with a DIFFERENT
policy mints new children while the old ones stay byte-identical at their old
version numbers. That is what makes an eval split citable months later.

Registration is CONTROL-SIDE, like ``dataset/publish_fingerprints``: the data
plane's job box has no database (ADR-0081 D3), so the store is imported lazily
and only touched when the caller asked for registration. When it was asked for
and cannot happen, the operator RAISES — the same strict posture the leg
executor's ``register:`` hook takes, because a requested registration that
silently vanishes leaves a training run pointing at nothing.

Risk: external_write (writes new dataset files).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks._io import ShardedJsonlWriter, iter_rows, load_rows
from operator_dataset_checks._split import (
    _row_group_key,
    _row_stratum,
    check_no_leakage,
    resolve,
    row_split_name,
    sklearn_available,
    sklearn_split,
)
from operator_dataset_checks.schemas.split import (
    SplitConfig,
    SplitInput,
    SplitOutput,
    StratumSummary,
)


@register_operator(
    name="dataset/split",
    version="0.1.0",
    capabilities=["dataset.split"],
    implements=["contract://dataset.split/1.0.0"],  # #2242 V1-D — cairn-contracts-dataset
)
class SplitOp(Operation[SplitInput, SplitOutput]):
    """Deterministic, group-aware, optionally-stratified train/val/test split."""

    input_schema = SplitInput
    output_schema = SplitOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    # #1539 — governance declarations; the [operator.*] table is generated from these
    risk = 'external_write'
    config_schema = SplitConfig

    async def run(self, inp: SplitInput, ctx: OperationContext) -> SplitOutput:
        out = self.compute(inp)
        if inp.register_as.strip():
            await _register_children(inp, out)
        return out

    def compute(self, inp: SplitInput) -> SplitOutput:
        """#2381 (ADR-0118 M7a) — the split itself, sync and context-free: the ``split``
        kernel calls it as ``run_split`` (no event loop, no stand-in context). Registration of
        the children (``register_as``) is the control plane's and stays in ``run``."""
        raw_ratios = {"train": inp.train_ratio, "val": inp.val_ratio}
        if inp.test_ratio > 0:
            raw_ratios["test"] = inp.test_ratio
        ratio_sum = sum(raw_ratios.values())
        ratios = {name: r / ratio_sum for name, r in raw_ratios.items()}

        backend = resolve(inp.backend)
        use_sklearn = backend == "sklearn" or (backend == "auto" and sklearn_available())

        out_dir = Path(inp.output_dir)
        if use_sklearn:
            # sklearn does a GLOBAL grouped/stratified allocation, so it needs every
            # row at once — the materialized (small-file / [split]-extra) path. It
            # never shards (one file per split); #891's streaming is the hash path.
            out = self._sklearn_split(inp, ratios, out_dir)
        else:
            out = self._streaming_hash_split(inp, ratios, out_dir)
        out.split_policy = _declared_policy(inp, ratios, out.backend_used)
        return out

    # ── streaming hash path (#891): one pass, bounded memory, sharded output ──
    def _streaming_hash_split(self, inp, ratios, out_dir: Path) -> SplitOutput:
        writers = {
            name: ShardedJsonlWriter(out_dir, name, shard_size=inp.shard_size)
            for name in ratios
        }
        counts = {name: 0 for name in ratios}
        group_keys: set[str] = set()
        per_stratum: dict[str, dict[str, int]] = {}
        total = 0
        try:
            for row in iter_rows(inp.path):
                total += 1
                name = row_split_name(
                    row, ratios=ratios, seed=inp.seed, group_by=inp.group_by,
                )
                writers[name].write(row)
                counts[name] += 1
                if inp.group_by:
                    group_keys.add(_row_group_key(row, inp.group_by))
                if inp.stratify_by:
                    label = _row_stratum(row, inp.stratify_by)
                    per_stratum.setdefault(label, {}).setdefault(name, 0)
                    per_stratum[label][name] += 1
        finally:
            for w in writers.values():
                w.close()

        output_paths = {name: w.paths[0] for name, w in writers.items()}
        shard_paths = {name: list(w.paths) for name, w in writers.items()}
        ratios_achieved = {
            name: round(counts[name] / total, 4) if total else 0.0 for name in ratios
        }
        strat_summary = [
            StratumSummary(label=label, counts=split_counts)
            for label, split_counts in sorted(per_stratum.items())
        ]
        return SplitOutput(
            backend_used="hash",
            total=total,
            counts=counts,
            ratios_achieved=ratios_achieved,
            ratios_requested={k: round(v, 4) for k, v in ratios.items()},
            # A group's split is a pure function of its key hash, so no group can
            # straddle splits — leakage-free by construction, no full-pass check.
            no_leakage=True,
            group_by=inp.group_by,
            group_count=len(group_keys) if inp.group_by else None,
            stratify_by=inp.stratify_by,
            stratification_summary=strat_summary,
            output_paths=output_paths,
            shard_paths=shard_paths,
        )

    # ── sklearn path: global allocation → needs all rows in memory (unchanged) ──
    def _sklearn_split(self, inp, ratios, out_dir: Path) -> SplitOutput:
        rows = load_rows(inp.path)
        total = len(rows)
        assignment = sklearn_split(
            rows, ratios=ratios, seed=inp.seed,
            group_by=inp.group_by, stratify_by=inp.stratify_by,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, str] = {}
        counts: dict[str, int] = {}
        for name, idxs in assignment.indices.items():
            path = out_dir / f"{name}.jsonl"
            path.write_text(
                "\n".join(json.dumps(rows[i], ensure_ascii=False) for i in idxs),
                encoding="utf-8",
            )
            output_paths[name] = str(path)
            counts[name] = len(idxs)

        ratios_achieved = {
            name: round(counts[name] / total, 4) if total else 0.0 for name in ratios
        }
        no_leakage = check_no_leakage(rows, assignment.indices, inp.group_by)
        group_count = None
        if inp.group_by:
            keys = {
                (row.get(inp.group_by) if row.get(inp.group_by) is not None else i)
                for i, row in enumerate(rows)
            }
            group_count = len(keys)
        strat_summary: list[StratumSummary] = []
        if inp.stratify_by:
            per_stratum: dict[str, dict[str, int]] = {}
            for name, idxs in assignment.indices.items():
                for i in idxs:
                    val = rows[i].get(inp.stratify_by)
                    label = "_missing_" if val is None else str(val)
                    per_stratum.setdefault(label, {}).setdefault(name, 0)
                    per_stratum[label][name] += 1
            strat_summary = [
                StratumSummary(label=label, counts=split_counts)
                for label, split_counts in sorted(per_stratum.items())
            ]
        return SplitOutput(
            backend_used=assignment.backend_used,
            total=total,
            counts=counts,
            ratios_achieved=ratios_achieved,
            ratios_requested={k: round(v, 4) for k, v in ratios.items()},
            no_leakage=no_leakage,
            group_by=inp.group_by,
            group_count=group_count,
            stratify_by=inp.stratify_by,
            stratification_summary=strat_summary,
            output_paths=output_paths,
            shard_paths={name: [p] for name, p in output_paths.items()},
        )


# ─── frozen child versions (#2020, ADR-0102 D1) ──────────────────────────────

#: Split name -> the registry ROLE its child version carries. A split never
#: produces a ``golden`` child: golden is human-rated by definition (ADR-0102
#: D1) and no split can manufacture a human's ruling.
_SPLIT_ROLES = {"train": "training", "val": "validation", "test": "test"}


def _declared_policy(inp: SplitInput, ratios: dict, backend_used: str) -> dict:
    """The policy the children are frozen under — declared, not inferred.

    ADR-0102 D1: "the default is declared and stratified, never a random tail".
    Written onto every child version so a split can be reproduced, and audited,
    long after the run that made it.
    """
    return {
        "ratios": {name: round(value, 6) for name, value in sorted(ratios.items())},
        "seed": inp.seed,
        "group_by": inp.group_by,
        "stratify_by": inp.stratify_by,
        "backend": backend_used,
    }


def _content_hash(paths: list[str]) -> str:
    """sha256 over a split's shards, in order — the child version's identity.

    Shard boundaries are part of the bytes, so the same rows written as one file
    and as three shards are different artifacts. That is correct: a consumer
    reads the shard list, and a re-shard is a new version, not a silent
    substitution.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


async def _register_children(inp: SplitInput, out: SplitOutput) -> None:
    """Register each split as a frozen child version of ``inp.register_as``.

    Raises on any failure. ``register_as`` is an explicit request; the leg
    executor's ``register:`` hook takes the same line (#1466) — a registration
    that was asked for and did not happen must fail the step loudly rather than
    leave a downstream train run pointing at a name that does not exist.
    """
    # lazy: control-plane storage stays off module load — the data plane imports
    # this pack too, and there it has no database (ADR-0081 D3).
    from cairn.core.dataflow.contract import ArtifactDescriptor
    from cairn.core.operator import build_artifact_store

    parent = inp.register_as.strip()
    store = build_artifact_store()
    try:
        parent_meta = await _parent_meta(store, parent, inp.parent_ref)
        parent_hash = str(parent_meta.get("artifact_id") or "") if parent_meta else ""
        parent_version = (
            f"dataset:{parent}@{parent_meta['version']}" if parent_meta else "")
        for split_name, paths in sorted(out.shard_paths.items()):
            role = _SPLIT_ROLES.get(split_name)
            if role is None:
                raise ValueError(
                    f"split {split_name!r} has no registry role — "
                    f"expected one of {sorted(_SPLIT_ROLES)}")
            child = f"{parent}-{split_name}"
            digest = _content_hash(paths)
            await store.register(
                ArtifactDescriptor(
                    content_hash=digest, kind="dataset", uri=out.output_paths[split_name],
                    rows=out.counts.get(split_name, 0),
                    parents=[parent_hash] if parent_hash else [],
                ),
            )
            version = await store.register_dataset_version(
                child, digest, actor="dataset/split", role=role,
                schema=(parent_meta or {}).get("schema"),
                column_mapping=(parent_meta or {}).get("column_mapping"),
                parent_version=parent_version, split_policy=out.split_policy,
            )
            out.child_versions[split_name] = f"dataset:{child}@{version}"
            out.child_hashes[split_name] = digest
    finally:
        closer = getattr(store, "close", None)
        if callable(closer):
            await closer()


async def _parent_meta(store, parent: str, parent_ref: str) -> dict | None:
    """The parent version's registry record, from ``parent_ref`` when given and
    from the name's head otherwise. ``None`` when the parent is not registered —
    a split of an unregistered file still produces children, they simply have no
    lineage to point at."""
    version = None
    if parent_ref.strip():
        from cairn.core.operator import parse_dataset_ref

        parsed = parse_dataset_ref(parent_ref.strip())
        if parsed is None:
            raise ValueError(f"parent_ref {parent_ref!r} is not a dataset: ref")
        name, version = parsed
        if name != parent:
            raise ValueError(
                f"parent_ref {parent_ref!r} names {name!r} but register_as is {parent!r}")
    return await store.get_dataset_version_meta(parent, version)


def run_split(inp: SplitInput) -> SplitOutput:
    return SplitOp().compute(inp)
