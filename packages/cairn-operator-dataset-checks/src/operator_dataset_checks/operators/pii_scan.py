"""dataset/pii_scan — count PII / secret occurrences across a dataset.

Detection runs through a pluggable backend (:mod:`operator_dataset_checks._pii`):
Microsoft **Presidio** in production (the ``[pii]`` extra), the single-pass
**regexset** re2 accelerant (the ``[fastpii]`` extra) for multi-million-row
corpora, or a zero-dependency **regex** set for local testing / no-extra
installs. Reports per-kind counts + which rows carry PII. Deterministic for
the regex and regexset backends (byte-for-byte count parity between them).
"""
from __future__ import annotations

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec
from cairn.core.operator.sharded import Shardable, ShardSpec

from collections.abc import Iterator
from itertools import islice

from operator_dataset_checks import _pii
from operator_dataset_checks._io import iter_rows, reservoir_sample_stream
from operator_dataset_checks.schemas.pii_scan import (
    PiiFinding,
    PiiInput,
    PiiOutput,
    PiiScanConfig,
)


def _scan_batched(
    rows: Iterator[dict], *, backend: str, kinds, cap: int, offset: int = 0,
    text_fields=None, settings=None,
) -> dict:
    """Scan an iterator of rows in backend batches (#964); return raw accumulators
    (picklable dict) so both the inline ``run`` and a sharded ``run_shard`` share
    one implementation. Findings carry GLOBAL row indices (``offset + local``) and
    are capped at ``cap`` per call — enough for the merge to reconstruct the exact
    global first-``cap`` in shard order. ``cap <= 0`` = uncapped. ``settings`` (a
    ``_pii.PiiSettings``) selects the cascade model / n_process / analyze-chunk."""
    st = settings or _pii.resolve_pii_settings()
    counts: dict[str, int] = {}
    findings: list[tuple[int, list[str]]] = []
    total = 0
    rows_with_pii = 0
    chunk = st.analyze_chunk
    buf: list[tuple[int, str]] = []

    def _flush() -> None:
        nonlocal total, rows_with_pii
        if not buf:
            return
        per = _pii.scan_many(
            [t for _, t in buf], backend=backend, kinds=kinds,
            model=st.model, n_process=st.n_process,
        )
        for (idx, _text), row_counts in zip(buf, per):
            total += 1
            if row_counts:
                rows_with_pii += 1
                for kind, n in row_counts.items():
                    counts[kind] = counts.get(kind, 0) + n
                if cap <= 0 or len(findings) < cap:
                    findings.append((idx, sorted(row_counts)))
        buf.clear()

    for i, row in enumerate(rows):
        # route to the training-text leaves only — a row collapses to ONE doc of
        # its actual text, not source_text + every token + every label (#1021 F4).
        buf.append((offset + i, " ".join(_pii.iter_text_leaves(row, text_fields))))
        if len(buf) >= chunk:
            _flush()
    _flush()
    return {"counts": counts, "findings": findings, "total": total, "rows_with_pii": rows_with_pii}


@register_operator(
    name="dataset/pii_scan",
    version="0.1.0",
    capabilities=["dataset.pii_scan"],
    implements=["contract://dataset.pii_scan/1.0.0"],  # #2242 V1-D — cairn-contracts-dataset
)
class PiiScanOp(Operation[PiiInput, PiiOutput], Shardable):
    """Count PII / secret matches per kind and flag the rows that carry them.

    #962 — Shardable: a FULL scan over a large corpus is sharded across a budgeted
    process pool (row ranges), so it doesn't monopolize the interactive worker.
    Sampled scans stay inline (the reservoir is over the whole stream)."""

    input_schema = PiiInput
    output_schema = PiiOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    config_schema = PiiScanConfig  # ADR-0070 — OPERATIONAL knobs (sample_size, backend)
    # ADR-0080 P0 (#1028) — a full-corpus Presidio scan is CPU-bound; declare a
    # generous activity budget so a large scan finishes instead of hitting a
    # blanket timeout. Sampled scans (the gate default) return in seconds.
    activity_timeout_seconds = 3600.0

    async def run(self, inp: PiiInput, ctx: OperationContext) -> PiiOutput:
        backend = _pii.resolve(inp.backend)
        settings = _pii.resolve_pii_settings(
            preset=inp.preset, ner=inp.ner, model=inp.model,
            n_process=inp.n_process, analyze_chunk=inp.analyze_chunk,
        )

        # STREAM the source (#891): the FULL-scan gate (#889, sample_size=null)
        # scans every row without ever materializing the corpus; the sampled path
        # draws a reservoir off the stream (O(k) memory). Either way `counts` /
        # `rows_with_pii` are exact — only `findings` is capped (bounded report).
        seen = [0]

        def _counting(it: Iterator[dict]) -> Iterator[dict]:
            for row in it:
                seen[0] += 1
                yield row

        # ── ADR-0080 P1 (R4): deterministic SECRET tier scans EVERY row ──────
        # The NER density tier is sampled (below), but a rare leaked key would be
        # missed by a 1000-row sample ~90% of the time. So run cheap secret regex
        # over the WHOLE stream as it passes (single pass — the reservoir already
        # reads every row) and merge the authoritative counts in the gate's vocab.
        secret_acc: dict[str, int] = {}
        secret_rows = [0]

        def _secret_scanning(it: Iterator[dict]) -> Iterator[dict]:
            for row in it:
                text = " ".join(_pii.iter_text_leaves(row, inp.text_fields))
                hits = _pii.secret_counts(text)
                if hits:
                    secret_rows[0] += 1
                    for k, v in hits.items():
                        secret_acc[k] = secret_acc.get(k, 0) + v
                yield row

        base = _counting(iter_rows(inp.path))
        if inp.secrets_full_scan:
            base = _secret_scanning(base)

        if inp.sample_size is not None:
            rows: Iterator[dict] = iter(reservoir_sample_stream(base, inp.sample_size))
            sampled = seen[0] > inp.sample_size
        else:
            rows = base  # full scan, streamed
            sampled = False

        cap = inp.max_findings
        acc = _scan_batched(
            rows, backend=backend, kinds=inp.kinds, cap=cap,
            text_fields=inp.text_fields, settings=settings,
        )
        total = acc["total"]
        rows_with_pii = acc["rows_with_pii"]
        counts = acc["counts"]
        # merge the full-corpus secrets over the (possibly sampled) main scan, in
        # the gate's lowercase vocab — a secret is now caught 100% of the time and
        # lands in the block list (was warn-only when cascade emitted UPPERCASE).
        if inp.secrets_full_scan:
            counts = _pii.merge_secret_counts(counts, secret_acc)
        findings = [PiiFinding(row=r, kinds=k) for r, k in acc["findings"]]
        truncated = cap > 0 and rows_with_pii > cap

        density = round(rows_with_pii / total * 100, 1) if total else 0.0
        return PiiOutput(
            backend_used=backend,
            total=total,
            rows_with_pii=rows_with_pii,
            density_pct=density,
            secret_rows=secret_rows[0] if inp.secrets_full_scan else None,
            counts=counts,
            findings=findings,
            findings_truncated=truncated,
            sampled=sampled,
            sample_size=inp.sample_size if sampled else None,
        )

    # ── #962 Shardable: the core sharded executor calls these for a large FULL
    #    scan; below CAIRN_BATCH_THRESHOLD_ROWS (or sampled) it takes run() above.
    def shard_total(self, inp: PiiInput) -> int:
        # sampled scans draw a reservoir over the WHOLE stream — not shardable;
        # return 0 so the runtime keeps them inline. Full scan → exact row count.
        if inp.sample_size is not None:
            return 0
        return sum(1 for _ in iter_rows(inp.path))

    def run_shard(self, inp: PiiInput, spec: ShardSpec) -> dict:
        # scan just this shard's row range; findings carry GLOBAL indices.
        backend = _pii.resolve(inp.backend)
        settings = _pii.resolve_pii_settings(
            preset=inp.preset, ner=inp.ner, model=inp.model,
            n_process=inp.n_process, analyze_chunk=inp.analyze_chunk,
        )
        rows = islice(iter_rows(inp.path), spec.start, spec.end)
        return _scan_batched(
            rows, backend=backend, kinds=inp.kinds, cap=inp.max_findings,
            offset=spec.start, text_fields=inp.text_fields, settings=settings,
        )

    def merge(self, inp: PiiInput, parts: list[dict]) -> PiiOutput:
        backend = _pii.resolve(inp.backend)
        cap = inp.max_findings
        counts: dict[str, int] = {}
        total = 0
        rows_with_pii = 0
        merged: list[tuple[int, list[str]]] = []
        for p in parts:                       # parts are IN SHARD ORDER (row order)
            total += p["total"]
            rows_with_pii += p["rows_with_pii"]
            for kind, n in p["counts"].items():
                counts[kind] = counts.get(kind, 0) + n
            merged.extend(p["findings"])
        findings = merged if cap <= 0 else merged[:cap]  # global first-cap in row order
        truncated = cap > 0 and rows_with_pii > cap
        density = round(rows_with_pii / total * 100, 1) if total else 0.0
        return PiiOutput(
            backend_used=backend,
            total=total,
            rows_with_pii=rows_with_pii,
            density_pct=density,
            counts=counts,
            findings=[PiiFinding(row=r, kinds=k) for r, k in findings],
            findings_truncated=truncated,
            sampled=False,
            sample_size=None,
        )
