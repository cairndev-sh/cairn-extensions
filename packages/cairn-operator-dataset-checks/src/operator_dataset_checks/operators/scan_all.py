"""dataset/scan_all — the fused single-pass readiness check (ADR-0080 P1, #1029).

Four readiness checks used to each re-stream the whole dataset (format · dedup ·
profile · pii_scan) — four reads, four CPU passes, on the shared control worker.
At 1M rows that is tens of minutes and it saturates the box (F2). This operator
folds them into ONE streaming read:

  full corpus (every row, cheap)  → dedup canonical hashes · format-detection
                                     vote · deterministic SECRET regex
  reservoir sample (sample_size)  → format conformance · profile stats · NER density

The per-row logic is IMPORTED verbatim from the four operators (``_detect_format``,
``_validate_row``, ``_response_text``, ``canonical_hash``, ``_pii.*``), so the
findings this emits are byte-identical to the four-operator wave and the
deterministic gate is unchanged — only the number of reads drops from ~4 to 1.

Rejected designs it closes: **R4** (a sampled-only PII gate misses a rare leaked
key ~90% of the time — secrets scan the full corpus) and **R7** (a parallel
fan-out of five CPU-bound scans over the SAME rows just multiplies reads and
contends — fusion + one read is strictly better; #1021 measured the fan-out never
helped). Provenance / decontam / sample stay separate ops (they read other inputs
or are metadata) and run in a cheap parallel wave alongside this one.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from typing import Any

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec
from cairn.core.operator.sharded import Shardable, ShardSpec

from operator_dataset_checks import _pii
from operator_dataset_checks._canon import canonical_hash
from operator_dataset_checks._io import RowErrors, iter_rows, reservoir_sample_stream
from operator_dataset_checks.operators.dedup import DedupOp
from operator_dataset_checks.operators.format import (
    _TRAINER_FORMATS,
    _detect_format,
    _validate_row,
)
from operator_dataset_checks.operators.profile import (
    _DEFAULT_REFUSAL_MARKERS,
    _response_text,
    _stats,
)
from operator_dataset_checks.schemas.dedup import DedupInput
from operator_dataset_checks.schemas.format import FormatFinding
from operator_dataset_checks.schemas.scan_all import (
    ScanAllConfig,
    ScanAllInput,
    ScanAllOutput,
)


@register_operator(
    name="dataset/scan_all",
    version="0.1.0",
    capabilities=["dataset.scan_all"],
)
class ScanAllOp(Operation[ScanAllInput, ScanAllOutput], Shardable):
    """One streaming read → format + dedup + profile + PII findings for the gate.

    Shardable ONLY so the compiler routes it to the ``cairn-data`` fast-check queue
    (ADR-0080 D1); it runs INLINE (``supports_sharding = False``) as a single fused
    pass — sharding the reservoir across workers is a job-machine (P4) concern, and
    the fusion already delivers the "one read" win the ADR targets."""

    input_schema = ScanAllInput
    output_schema = ScanAllOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    config_schema = ScanAllConfig
    # routed to cairn-data, but never fanned out — one inline fused pass.
    supports_sharding = False
    # a full-corpus fused pass over a multi-million-row dataset is CPU-bound;
    # inherit the same generous budget the standalone PII scan declares (P0 #1028).
    activity_timeout_seconds = 3600.0

    async def run(self, inp: ScanAllInput, ctx: OperationContext) -> ScanAllOutput:
        return self.sync_run(inp)

    @staticmethod
    def sync_run(inp: ScanAllInput) -> ScanAllOutput:
        # #2242 V1-B (M7a) — the body, sync and context-free: the dataflow kernels call it
        # as `run_scan_all` (no event loop, no stand-in context); `run` is the operator's door.
        # 'duckdb' as a PII backend means the regex tier; the sample's density
        # scan below is per-row Python, so it runs the same tier there.
        backend = "regex" if inp.backend == "duckdb" else _pii.resolve(inp.backend)
        settings = _pii.resolve_pii_settings(
            preset=inp.preset, ner=inp.ner, model=inp.model,
            n_process=inp.n_process, analyze_chunk=inp.analyze_chunk,
        )
        cap = inp.max_findings

        # ── full-corpus accumulators (every row, cheap) ───────────────────────
        hashes: list[str] = []            # dedup — O(n) hashes, exact grouping
        fmt_votes: Counter = Counter()    # format detection vote over the whole corpus
        secret_acc: dict[str, int] = {}   # ADR-0080 P1: secrets can't hide from a sample
        secret_rows = [0]
        row_errors = RowErrors()          # #1008 — malformed lines quarantined, not fatal

        def _accumulate(it: Iterator[dict]) -> Iterator[dict]:
            for row in it:
                hashes.append(canonical_hash(row, inp.normalize))
                fmt_votes[_detect_format(row)] += 1
                if inp.secrets_full_scan:
                    text = " ".join(_pii.iter_text_leaves(row, inp.text_fields))
                    hits = _pii.secret_counts(text)
                    if hits:
                        secret_rows[0] += 1
                        for k, v in hits.items():
                            secret_acc[k] = secret_acc.get(k, 0) + v
                yield row

        duck: dict | None = None
        if inp.engine == "duckdb":
            # #1436 leg C — the full-corpus tiers as SQL aggregates + a reservoir
            # sample pulled by DuckDB; only the sample crosses into Python.
            duck = _duckdb_corpus_pass(inp)
            sample = duck["sample"]
            for row in sample:
                fmt_votes[_detect_format(row)] += 1   # vote over the SAMPLE under duckdb
            total = duck["total"]
            sampled = inp.sample_size is not None and total > inp.sample_size
            secret_acc.update(duck["secret_counts"])
            secret_rows[0] = duck["secret_rows"]
        else:
            # ONE read: the full-corpus tiers accumulate as rows pass, and the same
            # stream feeds the reservoir for the statistical tiers (O(k) extra memory).
            stream = iter_rows(inp.path, strict=inp.strict, errors=row_errors)
            if inp.sample_size is not None:
                sample = reservoir_sample_stream(_accumulate(stream), inp.sample_size)
                sampled = len(hashes) > inp.sample_size
            else:
                sample = list(_accumulate(stream))  # explicit full-corpus statistical scan
                sampled = False
            total = len(hashes)
        sample_n = len(sample)

        # ── format winner from the FULL-corpus vote (cheap, more robust than a
        #    sampled vote); conformance validated over the sample against it ─────
        if inp.expected_format:
            fmt = inp.expected_format
        else:
            fmt = fmt_votes.most_common(1)[0][0] if fmt_votes else "unknown"
        conformant = 0
        fmt_findings: list[FormatFinding] = []
        for i, row in enumerate(sample):
            issue = _validate_row(row, fmt)
            if issue is None:
                conformant += 1
            elif cap <= 0 or len(fmt_findings) < cap:
                fmt_findings.append(FormatFinding(row=i, issue=issue))
        format_pct = round(conformant / sample_n * 100, 1) if sample_n else 0.0
        compat = {t: (fmt in _TRAINER_FORMATS.get(t, set())) for t in inp.target_trainers}

        # ── dedup over the FULL corpus (reuse the exact reduce → parity) ───────
        if duck is not None:
            unique = duck["distinct"]
            dd = _DedupCounts(
                total=total, exact_duplicates=total - unique, effective_size=unique,
                duplicate_pct=round((total - unique) / total * 100, 1) if total else 0.0)
        else:
            dd = DedupOp._report(hashes, None, DedupInput(path=inp.path, normalize=inp.normalize))

        # ── profile over the sample ───────────────────────────────────────────
        markers = [m.lower() for m in (inp.refusal_markers or _DEFAULT_REFUSAL_MARKERS)]
        lengths: list[int] = []
        refusals = 0
        short = 0
        for row in sample:
            resp = _response_text(row)
            lengths.append(len(resp))
            if len(resp) < inp.short_threshold:
                short += 1
            low = resp.lower()
            if any(m in low for m in markers):
                refusals += 1
        refusal_pct = round(refusals / sample_n * 100, 1) if sample_n else 0.0

        # ── PII: NER density over the sample (guard early-exit), secrets full ──
        # If format is already catastrophic the dataset FAILs the format gate, so
        # the CPU-heavy NER pass is pointless — skip it. Full-corpus secrets were
        # already scanned above and still block; only WARN-level sampled PII is
        # skipped, so the verdict is unchanged.
        ner_skipped = (
            inp.ner_skip_below_conformant_pct > 0
            and format_pct < inp.ner_skip_below_conformant_pct
        )
        pii_counts: dict[str, int] = {}
        pii_rows = 0
        pii_findings: list[dict] = []
        if not ner_skipped and sample_n:
            texts = [" ".join(_pii.iter_text_leaves(r, inp.text_fields)) for r in sample]
            chunk = settings.analyze_chunk or len(texts)
            per: list[dict[str, int]] = []
            for i in range(0, len(texts), chunk):
                per.extend(_pii.scan_many(
                    texts[i:i + chunk], backend=backend, kinds=None,
                    model=settings.model, n_process=settings.n_process,
                ))
            for idx, row_counts in enumerate(per):
                if row_counts:
                    pii_rows += 1
                    for k, v in row_counts.items():
                        pii_counts[k] = pii_counts.get(k, 0) + v
                    if cap <= 0 or len(pii_findings) < cap:
                        pii_findings.append({"row": idx, "kinds": sorted(row_counts)})
        # merge the full-corpus secrets into the gate's LOWERCASE blocking vocab —
        # a leaked key is caught 100% of the time and blocks (ADR-0080 P1).
        if inp.secrets_full_scan:
            pii_counts = _pii.merge_secret_counts(pii_counts, secret_acc)
        pii_density = round(pii_rows / sample_n * 100, 1) if sample_n else 0.0

        return ScanAllOutput(
            scanned=total,
            sampled=sampled,
            sample_size=inp.sample_size if sampled else None,
            ner_skipped=ner_skipped,
            quarantined_rows=duck["quarantined"] if duck is not None else row_errors.count,
            quarantine_path=None if duck is not None else row_errors.quarantine_path,
            format_detected=fmt,
            format_conformant_pct=format_pct,
            format_findings=fmt_findings,
            format_trainer_compat=compat,
            dedup_total=dd.total,
            dedup_exact=dd.exact_duplicates,
            dedup_effective_size=dd.effective_size,
            dedup_pct=dd.duplicate_pct,
            profile_total=sample_n,
            profile_lengths=_stats(lengths),
            profile_refusal_pct=refusal_pct,
            profile_short=short,
            pii_backend=backend,
            pii_rows=pii_rows,
            pii_density=pii_density,
            pii_counts=pii_counts,
            secret_rows=secret_rows[0] if inp.secrets_full_scan else None,
            pii_findings=pii_findings,
        )

    # ── Shardable ABC — satisfied for the cairn-data routing, never invoked
    #    (supports_sharding=False → should_shard() short-circuits to inline). ──
    def shard_total(self, inp: ScanAllInput) -> int:
        return 0

    def run_shard(self, inp: ScanAllInput, spec: ShardSpec) -> Any:  # pragma: no cover
        raise NotImplementedError("dataset/scan_all runs inline (supports_sharding=False)")

    def merge(self, inp: ScanAllInput, parts: list[Any]) -> Any:  # pragma: no cover
        raise NotImplementedError("dataset/scan_all runs inline (supports_sharding=False)")


class _DedupCounts:
    """The four dedup numbers the report needs, when DuckDB (not
    ``DedupOp._report``) produced them."""

    def __init__(self, *, total: int, exact_duplicates: int, effective_size: int, duplicate_pct: float):
        self.total, self.exact_duplicates = total, exact_duplicates
        self.effective_size, self.duplicate_pct = effective_size, duplicate_pct


def _duckdb_corpus_pass(inp: ScanAllInput) -> dict:
    """#1436 leg C: the every-row tiers of scan.all as ONE DuckDB aggregate —
    row count, distinct canonical keys (dedup), per-kind secret hits and
    ``secret_rows`` (the gate's tier, ADR-0080 P1) — plus a repeatable
    reservoir sample of ``sample_size`` rows for the statistical tiers.
    Malformed lines are skipped by the reader (``ignore_errors``) and counted
    as quarantined (line count − parsed rows); no side file is written."""
    from operator_dataset_checks import _duck  # noqa: PLC0415 - optional dependency: the [duckdb] extra

    secrets = [(k, pat) for k, pat in _pii.REGEX_KINDS if k in _pii._SECRET_KINDS]
    # ignore_errors turns a malformed line into a NULL row — drop those here so
    # count(*) is PARSED rows and the sample never carries a None.
    READ = ("(SELECT json FROM read_json_auto(?, format='newline_delimited', records=false, "
            "ignore_errors=true) WHERE json IS NOT NULL)")
    con = _duck.connect()
    try:
        agg = f"""
            SELECT count(*) AS total,
                   count(DISTINCT {_duck.key_expr(inp.normalize)}) AS distinct_keys,
                   count(*) FILTER (WHERE {_duck.any_hit_expr(secrets)}) AS secret_rows
                   {", " + _duck.hit_exprs(secrets) if inp.secrets_full_scan and secrets else ""}
            FROM {READ}
        """
        row = con.execute(agg, [inp.path]).fetchone()
        stats = dict(zip([d[0] for d in con.description], row))
        if inp.sample_size is not None:
            cur = con.execute(
                f"SELECT to_json(json) FROM {READ} USING SAMPLE reservoir({int(inp.sample_size)} ROWS) REPEATABLE (13)",
                [inp.path])
        else:
            cur = con.execute(f"SELECT to_json(json) FROM {READ}", [inp.path])
        sample = [json.loads(line) for (line,) in cur.fetchall()]
    finally:
        con.close()
    with open(inp.path, "rb") as f:
        lines = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))
    total = int(stats["total"])
    return {
        "total": total,
        "distinct": int(stats["distinct_keys"]),
        "secret_rows": int(stats["secret_rows"]) if inp.secrets_full_scan else 0,
        "secret_counts": {k: int(stats[k]) for k, _ in secrets
                          if inp.secrets_full_scan and stats.get(k)},
        "sample": sample,
        "quarantined": max(0, lines - total),
    }


#: #2242 V1-B (M7a) — the shared helper the `scan.all` kernel calls directly.
run_scan_all = ScanAllOp.sync_run
