"""dataset/anonymize — redact PII from a dataset, writing a cleaned copy.

Walks every string value in every example and replaces detected PII via the
pluggable backend (:mod:`operator_dataset_checks._pii`): Microsoft **Presidio**
in production (``[pii]`` extra), the single-pass **regexset** re2 accelerant
(``[fastpii]`` extra) for large corpora, or the zero-dependency **regex**
redactor for local testing. Writes a redacted JSONL to ``output_path`` and
reports what was removed. Risk: external_write (it produces a new dataset
file).
"""
from __future__ import annotations

import os
import shutil
from itertools import islice
from pathlib import Path

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec
from cairn.core.operator.sharded import Shardable, ShardSpec

from operator_dataset_checks import _pii
from operator_dataset_checks._io import JsonlWriter, iter_rows
from operator_dataset_checks.schemas.anonymize import (
    AnonymizeConfig,
    AnonymizeInput,
    AnonymizeOutput,
)


def _redact_chunked(rows, writer, *, backend, kinds, text_fields=None, settings=None) -> tuple[dict, int, int]:
    """Redact ``rows`` in backend batches (#964) into ``writer``; return
    ``(counts, rows_redacted, total)``. Shared by the inline ``run`` and a sharded
    ``run_shard`` so both produce byte-identical output. iter_text_leaves/
    map_text_leaves visit the SAME leaves in the same order, so batched results
    zip back by position. ``text_fields`` routes NER to the training-text fields
    only — token/label/id leaves are never scanned (#1021 F4). ``settings`` (a
    ``_pii.PiiSettings``) selects the cascade model / n_process / analyze-chunk."""
    st = settings or _pii.resolve_pii_settings()
    counts: dict[str, int] = {}
    rows_redacted = 0
    total = 0
    chunk_size = st.analyze_chunk

    def _flush(buf: list) -> None:
        nonlocal rows_redacted, total
        if not buf:
            return
        leaves = [s for row in buf for s in _pii.iter_text_leaves(row, text_fields)]
        results = _pii.anonymize_many(
            leaves, backend=backend, kinds=kinds,
            model=st.model, n_process=st.n_process,
        )
        it = iter(results)
        for row in buf:
            row_counts: dict[str, int] = {}

            def _redact(_s: str) -> str:
                red, c = next(it)
                for kind, n in c.items():
                    row_counts[kind] = row_counts.get(kind, 0) + n
                return red

            writer.write(_pii.map_text_leaves(row, _redact, text_fields))
            total += 1
            if row_counts:
                rows_redacted += 1
                for kind, n in row_counts.items():
                    counts[kind] = counts.get(kind, 0) + n

    buf: list = []
    for row in rows:
        buf.append(row)
        if len(buf) >= chunk_size:
            _flush(buf)
            buf = []
    _flush(buf)
    return counts, rows_redacted, total


@register_operator(
    name="dataset/anonymize",
    version="0.1.0",
    capabilities=["dataset.anonymize"],
    implements=["contract://dataset.anonymize/1.0.0"],  # #2242 V1-D — cairn-contracts-dataset
)
class AnonymizeOp(Operation[AnonymizeInput, AnonymizeOutput], Shardable):
    """Redact PII from every example and write a cleaned dataset.

    #962 — Shardable: a large corpus is redacted across a budgeted process pool
    (row ranges → per-shard ``shard_i.jsonl`` → concatenated into ``output_path``),
    so it never monopolizes the interactive worker."""

    input_schema = AnonymizeInput
    output_schema = AnonymizeOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    # #1539 — governance declarations; the [operator.*] table is generated from these
    risk = 'external_write'
    config_schema = AnonymizeConfig  # ADR-0070 — GOVERNANCE: which PII kinds must be redacted (env-owned)
    # ADR-0080 P0 (#1028) — redaction CANNOT sample (every row is rewritten), so
    # a full 1M-row cascade pass is inherently minutes-to-an-hour on a modest box.
    # Declare a generous budget; the P3 batch job gives it a dedicated machine.
    activity_timeout_seconds = 7200.0

    async def run(self, inp: AnonymizeInput, ctx: OperationContext) -> AnonymizeOutput:
        return self.sync_run(inp)

    @staticmethod
    def sync_run(inp: AnonymizeInput) -> AnonymizeOutput:
        # #2242 V1-B (M7a) — the body, sync and context-free: the dataflow kernels call it
        # as `run_anonymize` (no event loop, no stand-in context); `run` is the operator's door.
        backend = _pii.resolve(inp.backend)
        settings = _pii.resolve_pii_settings(
            preset=inp.preset, ner=inp.ner, model=inp.model,
            n_process=inp.n_process, analyze_chunk=inp.analyze_chunk,
        )

        # STREAM the source in CHUNKS (#891 stream + #964 batch): a bounded window
        # of rows redacted in one backend batch, then written — bounded memory,
        # no per-string NLP. Shared with the sharded path via _redact_chunked.
        with JsonlWriter(Path(inp.output_path)) as writer:
            counts, rows_redacted, total = _redact_chunked(
                iter_rows(inp.path), writer, backend=backend, kinds=inp.kinds,
                text_fields=inp.text_fields, settings=settings,
            )

        return AnonymizeOutput(
            backend_used=backend,
            total=total,
            rows_redacted=rows_redacted,
            counts=counts,
            output_path=inp.output_path,
        )

    # ── #962 Shardable: large corpora redact across a budgeted process pool ──
    def shard_total(self, inp: AnonymizeInput) -> int:
        return sum(1 for _ in iter_rows(inp.path))

    def run_shard(self, inp: AnonymizeInput, spec: ShardSpec) -> dict:
        backend = _pii.resolve(inp.backend)
        settings = _pii.resolve_pii_settings(
            preset=inp.preset, ner=inp.ner, model=inp.model,
            n_process=inp.n_process, analyze_chunk=inp.analyze_chunk,
        )
        shard_path = str(Path(spec.output_dir) / f"shard_{spec.index:05d}.jsonl")
        rows = islice(iter_rows(inp.path), spec.start, spec.end)
        with JsonlWriter(Path(shard_path)) as writer:
            counts, rows_redacted, total = _redact_chunked(
                rows, writer, backend=backend, kinds=inp.kinds,
                text_fields=inp.text_fields, settings=settings,
            )
        return {"counts": counts, "rows_redacted": rows_redacted, "total": total, "shard_path": shard_path}

    def merge(self, inp: AnonymizeInput, parts: list[dict]) -> AnonymizeOutput:
        backend = _pii.resolve(inp.backend)
        counts: dict[str, int] = {}
        rows_redacted = 0
        total = 0
        # concatenate shard files in order into output_path. Each JsonlWriter shard
        # has no trailing newline, so a "\n" between shards reproduces exactly the
        # single-writer "\n".join(all rows) — byte-identical to the inline path.
        # mkdir the parent like JsonlWriter does (the inline path) — the sharded
        # merge writes with a raw open() and must create the dir itself.
        # ATOMIC (#1007): concatenate into a tmp sibling, then os.replace — a crash
        # mid-merge never leaves a truncated file at the real output_path.
        final = Path(inp.output_path)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_name(f"{final.name}.tmp-{os.getpid()}")
        try:
            with open(tmp, "w", encoding="utf-8") as out:
                first = True
                for p in parts:
                    total += p["total"]
                    rows_redacted += p["rows_redacted"]
                    for kind, n in p["counts"].items():
                        counts[kind] = counts.get(kind, 0) + n
                    if not first:
                        out.write("\n")
                    with open(p["shard_path"], encoding="utf-8") as f:
                        shutil.copyfileobj(f, out)  # streamed, one shard at a time
                    first = False
            os.replace(tmp, final)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return AnonymizeOutput(
            backend_used=backend,
            total=total,
            rows_redacted=rows_redacted,
            counts=counts,
            output_path=inp.output_path,
        )


#: #2242 V1-B (M7a) — the shared helper the `pii.redact` kernel calls directly.
run_anonymize = AnonymizeOp.sync_run
