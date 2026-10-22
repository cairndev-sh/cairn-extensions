"""Dataflow stage kernels (ADR-0081 §5, #1049) — the dataset pack's
contributions to the ``cairn.dataflow_stages`` registry.

Stage packs ARE operator packs: kernels live NEXT TO the proven operator
bodies they wrap (v1 = whole-dataset adapters around ``*Apply``/``Anonymize``
— zero rewrite, the ``run_fix_pipeline`` reuse pattern; v2 exposes per-batch
kernels for distributed engines, #1054). The generic ``cairn-operator-dataflow``
wheel never learns dataset logic — the registry is the seam.

Stage contract (P1a): ``stage(workdir, config, inputs) -> scalar stats``.
Chaining convention: the working dataset is ``workdir/current.jsonl`` — the
first stage materializes it from ``config.path`` (or ``inputs[0]``), each
transform rewrites it, ``export.*`` emits the final artifact. Heavy imports
stay INSIDE the functions (D5: the registry must be enumerable without the
data deps installed).
"""
from __future__ import annotations
from cairn.core.operator import getenv

import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path

# #1382 — survivors stream out of DuckDB in batches of this many rows; peak
# Python-side memory is one batch, independent of corpus size.
_DEDUP_FETCH_ROWS = 10_000


class _Ctx:
    """Minimal OperationContext stand-in for reused operator bodies (the
    run_fix_pipeline pattern — stages get no engine context by design)."""
    pipeline_id = "dataflow-stage"
    node_name = "kernel"


def _current(workdir: Path) -> Path:
    return workdir / "current.jsonl"


def _ingest_record(workdir: Path) -> Path:
    """What ``ingest.normalize`` read, for ``dataset.manifest`` later in the same leg."""
    return workdir / "ingest.json"


def _resolve_submit_relative(source: str) -> str:
    """A RELATIVE local path resolves against the submitter's cwd (the engine
    runs kernels in a temp workdir) — caught live on the first clean run."""
    submit_cwd = os.environ.get("CAIRN_SUBMIT_CWD", "")
    if submit_cwd and "://" not in source and not os.path.isabs(source) and not source.startswith("hf:"):
        candidate = Path(submit_cwd) / source
        if candidate.exists():
            return str(candidate)
    return source


def ingest_normalize(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Materialize the working dataset: stream the source (jsonl/csv/parquet/
    HF id) to ``current.jsonl`` and count rows — one pass, bounded memory.

    cairn-extensions#5 — ``config.revision`` loads a Hub id at that commit, tag or branch; the
    stats record it with the commit it resolved to, the fetch time and ``snapshot_hash`` (the
    materialised bytes), and the same record goes to ``ingest.json`` for ``dataset.manifest``.
    ``source_hash`` keeps its meaning: the bytes of a local file, which the runner names as the
    upstream export's registry id (#1432); any other source keeps the URI parent."""
    from datetime import datetime, timezone  # noqa: PLC0415 - stage bodies keep their imports (D5)

    from operator_dataset_checks._io import JsonlWriter, RowErrors, _normalize_source, hub_commit, is_hub_id, iter_rows  # noqa: PLC0415 - data deps live in stage bodies (D5)

    source = str(config.get("path") or (inputs[0] if inputs else ""))
    if not source:
        raise ValueError("ingest.normalize needs config.path or an input artifact")
    source = _resolve_submit_relative(source)
    # #1466 — a dataset-registry ref resolved at launch carries the expected
    # hash as a fragment (s3://…#sha256=<hash>): a NAME must guarantee BYTES.
    source, _, _frag = source.partition("#")
    expected_hash = _frag[len("sha256="):] if _frag.startswith("sha256=") else ""
    origin = source  # what was READ — the lineage parent (#1432), even when fetched
    # #2171 (C1) — a registered version's URI is what the runner described
    # (`file://<abs>`, cairn/core/dataflow/runner.py). The hash check below
    # opened it as ``Path(source)``, which collapses ``file:///x`` into
    # ``file:/x`` and never exists. Resolve the local form ONCE, here, so the
    # verification, the streaming read and the lineage hash all see one path.
    source = _normalize_source(source)
    if source.startswith("s3://"):
        # #1442 — an upstream leg's export in the artifact store: fetch it once
        # into the workdir (the DuckDB stages need a local file anyway), then
        # stream as usual. boto3 = the s3/put contract's client (IAM/env creds).
        source = _s3_download(source, workdir / "ingest-source.jsonl")
    if expected_hash:
        got = _sha256_file(Path(source))[0]
        if got != expected_hash:
            raise RuntimeError(
                f"dataset verification FAILED for {origin}: fetched bytes hash "
                f"{got[:12]}… but the registry pinned {expected_hash[:12]}… — "
                "the object changed since registration (overwrite/corruption); "
                "refusing to run on unverified data (#1466)")
    # A file or URL keeps its `#sha256=` pin and ignores `revision`, so none is recorded for it.
    revision = (str(config.get("revision") or "").strip() or None) if is_hub_id(source) else None
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = 0
    # #2396 (ADR-0102 D1) — malformed lines are quarantined, not fatal (#1008); their count is the
    # `quarantined_rows` the intake verdict reads: any nonzero count is `invalid_format`.
    errors = RowErrors()
    with JsonlWriter(_current(workdir)) as writer:
        for row in iter_rows(source, errors=errors, revision=revision):
            writer.write(row)
            total += 1
    out = {"rows": total, "source": origin, "quarantined_rows": errors.count}
    # #1432 — the registry is CONTENT-addressed (an artifact's id is its
    # sha256; a re-run that exports identical bytes keeps the first record and
    # its URI), so lineage must name the parent by CONTENT, not location: hash
    # what was read (a local file after any fetch) and report it as the
    # source's registry id. ~4 s per GB, one streaming pass.
    if source and os.path.isfile(source):
        out["source_hash"], out["source_bytes"] = _sha256_file(Path(source))
    # cairn-extensions#5 — what was materialised, for every source: one Hub revision always names
    # the same bytes and a moved branch a new hash. Never a lineage parent (`source_hash` is).
    out["snapshot_hash"] = _sha256_file(_current(workdir))[0]
    out["revision"] = revision
    # Looked up after the load: a branch that moves in between records the newer commit, which
    # `snapshot_hash` would then contradict. Pin a commit sha as `revision` to rule that out.
    # Resolved for an unpinned Hub load too (the head it got); None for any other source.
    out["revision_sha"] = hub_commit(source, revision)
    out["fetched_at"] = fetched_at
    _ingest_record(workdir).write_text(json.dumps(out, sort_keys=True), encoding="utf-8")
    return out


def dataset_manifest(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """cairn-extensions#5 — write ``manifest.json`` for the working dataset: source, revision,
    hashes, row count, field names, licence and citation from the Hub card (null when absent),
    fetch time. Never a row. Runs after ``export.*`` in the same leg, so the report's parent is
    the export and ``data_content_hash`` is the export's registry id.

    ``config.store`` as in ``export.jsonl``: ``local`` keeps the file in this workdir, ``s3``
    uploads it beside the export."""
    from operator_dataset_checks._io import dataset_info, hub_card_license, is_hub_id  # noqa: PLC0415 - data deps live in stage bodies (D5)

    record_path = _ingest_record(workdir)
    if not record_path.is_file():
        raise ValueError("dataset.manifest needs an ingest.normalize stage earlier in the same leg")
    ingest = json.loads(record_path.read_text(encoding="utf-8"))
    digest, _size, rows = _sha256_lines(_current(workdir))
    fields: set[str] = set()
    with _current(workdir).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    fields.update(row)
    source = str(ingest.get("source") or "")
    license_, citation = None, None
    if is_hub_id(source):
        try:
            info = dataset_info(source, ingest.get("revision"))
            license_, citation = (info.license or None), (info.citation or None)
        except Exception:  # noqa: BLE001 — a card that cannot be read is recorded as absent, not guessed
            pass
        # the card's own `license:` first: the builder info is often empty for a repo of files
        license_ = hub_card_license(source, ingest.get("revision")) or license_
    manifest = {
        "format": "cairn.dataset-manifest/1",
        "source": source,
        "revision": ingest.get("revision"),
        "revision_sha": ingest.get("revision_sha"),
        "source_hash": ingest.get("source_hash"),
        "snapshot_hash": ingest.get("snapshot_hash") or digest,
        "data_content_hash": digest,
        "rows": rows,
        "quarantined_rows": int(ingest.get("quarantined_rows") or 0),
        "fields": sorted(fields),
        "license": license_,
        "citation": citation,
        "fetched_at": ingest.get("fetched_at"),
    }
    name = str(config.get("name", "manifest.json"))
    path = workdir / name
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out = {"kind": "report", "content_hash": _sha256_file(path)[0], "data_content_hash": digest,
           "dataset_rows": rows, "fields": len(fields)}
    store = str(config.get("store") or _artifact_store_backend()).lower()
    if store == "s3":
        return {**out, "output": _s3_upload(path, _s3_export_uri(workdir, name, root=config.get("root"))),
                "store": "s3"}
    return {**out, "output": str(path), "store": "local"}


def dedup_exact(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Drop exact duplicates. ``config.backend``:

    - ``python`` (default) — the proven ``DedupApplyOp`` body, verbatim.
    - ``duckdb`` — the dedup runs as ONE SQL window query over the jsonl
      (row-scale work in the columnar engine; explicit opt-in, so the
      default readiness flow is byte-identical to before). Requires the
      ``duckdb`` package and fails LOUDLY without it — an opted-in backend
      silently downgrading would fake the very evidence a run reads from
      the ``backend`` stat.
    """
    if str(config.get("backend", "python")).lower() == "duckdb":
        return _dedup_exact_duckdb(workdir, normalize=bool(config.get("normalize", True)))
    from operator_dataset_checks.operators.dedup_apply import run_dedup_apply  # noqa: PLC0415 - data deps live in stage bodies (D5)
    from operator_dataset_checks.schemas.dedup_apply import DedupApplyInput  # noqa: PLC0415

    out = workdir / "deduped.jsonl"
    result = run_dedup_apply(DedupApplyInput(
        path=str(_current(workdir)), output_path=str(out),
        method=str(config.get("method", "exact")),
        normalize=bool(config.get("normalize", True)),
    ))
    shutil.move(str(out), _current(workdir))  # shutil.move: workdir may span filesystems (the EXDEV lesson)
    return {"rows": result.kept, "dropped": result.dropped,
            "dropped_exact": result.dropped_exact, "dropped_near": result.dropped_near,
            "backend": "python"}


def _dedup_exact_duckdb(workdir: Path, *, normalize: bool) -> dict:
    """Exact dedup in DuckDB: hash each row's canonical JSON (whitespace
    collapsed + lowercased when ``normalize``), keep the FIRST occurrence in
    file order, write the survivors back. The SQL does the row-scale work;
    Python only streams the surviving lines out (a one-JSON-column COPY
    would wrap every row in ``{"json": …}``)."""
    try:
        import duckdb  # noqa: PLC0415 - optional dependency: the [duckdb] extra
    except ImportError as e:
        raise RuntimeError(
            "dedup.exact backend=duckdb needs the duckdb package: pip install duckdb"
        ) from e

    src = _current(workdir)
    # to_json(), not CAST AS VARCHAR: auto-detect infers a STRUCT whose
    # VARCHAR form is a python-ish repr, not JSON (caught by the pack test).
    key_expr = ("md5(lower(regexp_replace(to_json(json), '\\s+', ' ', 'g')))"
                if normalize else "md5(to_json(json))")
    con = duckdb.connect()
    out = workdir / "deduped.jsonl"
    kept = 0
    try:
        # total FIRST: a later execute() on the same connection would
        # invalidate the streaming result below.
        total = int(con.execute(
            "SELECT count(*) FROM read_json_auto(?, format='newline_delimited', records=false)",
            [str(src)]).fetchone()[0])
        cur = con.execute(f"""
            SELECT to_json(json)
            FROM (
                SELECT json, rid,
                       row_number() OVER (PARTITION BY {key_expr} ORDER BY rid) AS rn
                FROM (
                    SELECT json, row_number() OVER () AS rid
                    FROM read_json_auto(?, format='newline_delimited', records=false)
                )
            )
            WHERE rn = 1 ORDER BY rid
        """, [str(src)])
        # #1382: stream survivors out in bounded batches — fetchall() held the
        # whole corpus as a Python list, O(dataset) RAM (OOM at 1M rows).
        with out.open("w", encoding="utf-8") as f:
            while True:
                batch = cur.fetchmany(_DEDUP_FETCH_ROWS)
                if not batch:
                    break
                kept += len(batch)
                for (line,) in batch:
                    f.write(line + "\n")
    finally:
        con.close()
    shutil.move(str(out), src)  # shutil.move: workdir may span filesystems (the EXDEV lesson)
    return {"rows": kept, "dropped": total - kept, "dropped_exact": total - kept,
            "dropped_near": 0, "backend": "duckdb"}


def pii_redact(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Redact secrets + PII — the ``AnonymizeOp`` cascade, every row (redaction
    cannot sample). n_process resolves to the machine's cores: a stage runs on
    a box that OWNS them (ADR-0080 R2 — the reason this is a dataflow)."""
    from operator_dataset_checks.operators.anonymize import run_anonymize  # noqa: PLC0415 - data deps live in stage bodies (D5)
    from operator_dataset_checks.schemas.anonymize import AnonymizeInput  # noqa: PLC0415

    backend = str(config.get("backend", "auto")).lower()
    # `none` = a DECLARED pass-through, for corpora whose entities ARE the
    # training signal (synthetic news facts: dates, locations, tickers). The
    # cascade rewrote such a window into <DATE_TIME>/<LOCATION> placeholders
    # and a real GPU cycle trained on it (2026-08-28). Never a silent default:
    # absent/auto still redacts, and the stats say `none` on the record so a
    # gate or reviewer can see redaction was declined, not forgotten.
    if backend == "none":
        n = sum(1 for line in _current(workdir).open(encoding="utf-8") if line.strip())
        return {"rows": n, "redacted_rows": 0, "pii_backend": "none"}
    if backend == "duckdb":
        return _pii_redact_duckdb(workdir, kinds=config.get("kinds"))
    out = workdir / "redacted.jsonl"
    # Forward the cascade tuning knobs (ADR-0079 presets) from the flow YAML.
    # Before this only `backend` crossed the stage boundary, so a flow could
    # never pin n_process / preset — and the env fallback is unreachable under
    # any preset (the preset always supplies an explicit n_process). Absent
    # keys stay None → the preset's own value, byte-identical to before.
    tuning = {k: config[k] for k in ("preset", "ner", "model", "n_process", "analyze_chunk")
              if config.get(k) is not None}
    result = run_anonymize(AnonymizeInput(
        path=str(_current(workdir)), output_path=str(out),
        backend=str(config.get("backend", "auto")), **tuning,
    ))
    shutil.move(str(out), _current(workdir))
    return {"rows": result.total, "redacted_rows": result.rows_redacted,
            "pii_backend": result.backend_used}


def _sql_str(s: str) -> str:
    """A SQL single-quoted literal (patterns carry backslashes: DuckDB string
    literals are NOT escape-processed, so only the quote needs doubling)."""
    return "'" + s.replace("'", "''") + "'"


def _merge_eval_probes(workdir: Path, sources: list[str]) -> tuple[str, list[str]]:
    """Concatenate several eval/holdout versions into ONE probe file.

    #2021 (ADR-0102 D1): ``decontam.apply``'s probe defaults to **every** golden
    and test version in the workspace, not the single path an author remembered
    to pass — the control plane resolves that list (it owns the registry) and
    hands the URIs down as ``eval_paths``. ``DecontamApplyOp`` indexes one eval
    side, so the honest way to check against N sets is to build the union once
    and index that: the per-row decision is identical to running the op N times,
    at one pass instead of N.

    Returns ``(probe path, sources actually read)``. A source that will not read
    is skipped and left out of the returned list, so ``eval_sets`` on the record
    names what was really compared — never what was merely intended.
    """
    from operator_dataset_checks._io import JsonlWriter, iter_rows  # noqa: PLC0415 - data deps live in stage bodies (D5)

    probe = workdir / "eval-probe.jsonl"
    read: list[str] = []
    with JsonlWriter(probe) as writer:
        for source in sources:
            resolved = _resolve_submit_relative(source)
            if resolved.startswith("s3://"):
                resolved = _s3_download(
                    resolved, workdir / f"eval-{len(read):02d}.jsonl")
            try:
                for row in iter_rows(resolved):
                    writer.write(row)
            except Exception:  # noqa: BLE001 — an unreadable probe is reported, never fatal
                _log_skipped_probe(source)
                continue
            read.append(source)
    return str(probe), read


def _log_skipped_probe(source: str) -> None:
    import logging  # noqa: PLC0415 - leaf concern of this one helper

    logging.getLogger("cairn.dataset_checks").warning(
        "decontam.apply: eval probe %s could not be read — it is NOT in eval_sets", source)


def decontam_apply(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Drop train↔eval contamination — the proven ``DecontamApplyOp`` body.
    No ``eval_path``/``eval_paths`` → nothing dropped, reported honestly as
    ``checked=False`` (the run-decontam convention: skip loudly, never fail
    silently).

    ``eval_paths`` (#2021) is the list form: the control plane fills it with
    every golden and test version in the workspace, so an import is
    decontaminated against the holdouts that exist rather than the one an
    author named. An explicit ``eval_path`` still wins — a pipeline that says
    which set it means knows more than the default does.
    """
    from operator_dataset_checks.operators.decontam_apply import run_decontam_apply  # noqa: PLC0415 - data deps live in stage bodies (D5)
    from operator_dataset_checks.schemas.decontam_apply import DecontamApplyInput  # noqa: PLC0415

    eval_path = str(config.get("eval_path") or "")
    eval_sets: list[str] = [eval_path] if eval_path else []
    if not eval_path:
        wanted = config.get("eval_paths") or []
        if isinstance(wanted, str):
            wanted = [p for p in wanted.split(",") if p.strip()]
        wanted = [str(p) for p in wanted if str(p).strip()]
        if wanted:
            eval_path, eval_sets = _merge_eval_probes(workdir, wanted)
            if not eval_sets:
                eval_path = ""
    method = str(config.get("method", "exact"))
    if str(config.get("backend", "python")).lower() == "duckdb" and method == "exact" and eval_path:
        out = _decontam_exact_duckdb(
            workdir, _resolve_submit_relative(eval_path),
            normalize=bool(config.get("normalize", True)))
        out["eval_sets"] = ",".join(eval_sets)
        return out
    out = workdir / "decontaminated.jsonl"
    result = run_decontam_apply(DecontamApplyInput(
        path=str(_current(workdir)), output_path=str(out),
        eval_path=_resolve_submit_relative(eval_path) if eval_path else None,
        method=str(config.get("method", "exact")),
        normalize=bool(config.get("normalize", True)),
        ngram=int(config.get("ngram", 13)),
        overlap_threshold=float(config.get("overlap_threshold", 0.8)),
    ))
    shutil.move(str(out), _current(workdir))
    return {"rows": result.kept, "dropped": result.dropped,
            "checked": result.checked, "method": result.method_used,
            # FLAT scalar (the boundary rule): the probe versions this run was
            # actually compared against, so "checked" can be audited.
            "eval_sets": ",".join(eval_sets),
            "backend": "python"}


def _stream_out(cur, out: Path) -> int:
    """Stream a one-column (json text) DuckDB result to ``out`` in bounded
    batches (#1382 — never fetchall a corpus). Returns rows written."""
    kept = 0
    with out.open("w", encoding="utf-8") as f:
        while True:
            batch = cur.fetchmany(_DEDUP_FETCH_ROWS)
            if not batch:
                break
            kept += len(batch)
            for (line,) in batch:
                f.write(line + "\n")
    return kept


def _decontam_exact_duckdb(workdir: Path, eval_path: str, *, normalize: bool) -> dict:
    """Exact decontamination as ONE anti-join in DuckDB (#1436 leg A): drop
    every train row whose canonical JSON text equals a probe row's. Same
    outcome as ``DecontamApplyOp(method=exact)`` for rows written by one
    serializer (``_duck.key_expr``); ngram/minhash stay on the Python op."""
    from operator_dataset_checks import _duck  # noqa: PLC0415 - data deps live in stage bodies (D5)

    src = _current(workdir)
    out = workdir / "decontaminated.jsonl"
    key = _duck.key_expr(normalize)
    con = _duck.connect()
    try:
        total = int(con.execute(
            f"SELECT count(*) FROM {_duck.READ_JSONL}", [str(src)]).fetchone()[0])
        con.execute(f"CREATE TEMP TABLE probe_keys AS SELECT DISTINCT {key} AS k FROM {_duck.READ_JSONL}",
                    [str(eval_path)])
        cur = con.execute(f"""
            SELECT to_json(json)
            FROM (SELECT json, row_number() OVER () AS rid FROM {_duck.READ_JSONL}) t
            WHERE {key} NOT IN (SELECT k FROM probe_keys)
            ORDER BY rid
        """, [str(src)])
        kept = _stream_out(cur, out)
    finally:
        con.close()
    shutil.move(str(out), src)
    return {"rows": kept, "dropped": total - kept, "checked": True,
            "method": "exact", "backend": "duckdb"}


def _pii_redact_duckdb(workdir: Path, *, kinds=None) -> dict:
    """The regex tier as ONE vectorized SQL pass (#1436 leg B): every
    ``_pii.REGEX_KINDS`` pattern chained as ``regexp_replace(…, '<KIND>', 'g')``
    over the row's JSON text, survivors streamed out; per-kind hit counts and
    ``redacted_rows`` from an aggregate pass. Same patterns and replacement
    tokens as ``backend=regex`` (byte-identical on the 2k real-row smoke) —
    differences: it sees the WHOLE row text (keys/short enums too, where the
    Python path skips leaves under 16 chars) and never NER. Secrets included
    (the data gate's tier)."""
    from operator_dataset_checks import _duck, _pii  # noqa: PLC0415 - data deps live in stage bodies (D5)

    src = _current(workdir)
    out = workdir / "redacted.jsonl"
    active = [(k, pat) for k, pat in _pii.REGEX_KINDS if kinds is None or k in kinds]
    expr = _duck.redact_expr(active)
    con = _duck.connect()
    try:
        row = con.execute(f"""
            SELECT count(*) AS total,
                   count(*) FILTER (WHERE {expr} <> to_json(json)::VARCHAR) AS redacted_rows,
                   {_duck.hit_exprs(active) or "0 AS none"}
            FROM {_duck.READ_JSONL}
        """, [str(src)]).fetchone()
        stats = dict(zip([d[0] for d in con.description], row))
        cur = con.execute(f"""
            SELECT {expr}
            FROM (SELECT json, row_number() OVER () AS rid FROM {_duck.READ_JSONL}) t
            ORDER BY rid
        """, [str(src)])
        kept = _stream_out(cur, out)
    finally:
        con.close()
    shutil.move(str(out), src)
    # Per-kind hits as FLAT scalars: stage stats cross the dataflow boundary
    # as an ArtifactDescriptor and only scalars may (ADR-0081 D3 / ADR-0080
    # R5) — a nested `counts` dict failed the 300k live run at the boundary.
    out_stats = {"rows": kept, "redacted_rows": int(stats["redacted_rows"]),
                 "pii_backend": "duckdb"}
    out_stats.update({f"hits_{k}": int(stats[k]) for k, _ in active if stats.get(k)})
    return out_stats


def split(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Deterministic group-aware train/val[/test] split — the ``SplitOp`` body.
    Terminal stage: writes ``workdir/splits/<name>.jsonl`` per split (the prep
    convention); ``current.jsonl`` is left as-is for any sibling export stage.

    cairn-extensions#6 — ``config.export`` returns each split as a product of the stage: its
    ``outputs.<name>.*`` stats (file, hash, rows; uploaded beside the export with ``store: s3``)
    become one artifact descriptor per split in the runner. ``config.register`` names the
    dataset versions ``<register>-train`` / ``-val`` / ``-test`` with roles training /
    validation / test — the names and roles ``dataset/split`` registers — so the holdout door
    finds the test version the trainer must not meet."""
    from operator_dataset_checks.operators.split import run_split  # noqa: PLC0415 - data deps live in stage bodies (D5)
    from operator_dataset_checks.schemas.split import SplitInput  # noqa: PLC0415

    export = bool(config.get("export", False))
    register = str(config.get("register") or "").strip()
    if register and not export:
        raise ValueError("split: `register` needs `export: true` (only exported splits are registered)")
    if export and int(config.get("shard_size", 0)):
        raise ValueError("split: `export` needs shard_size 0 (one file per split is one product)")
    out_dir = workdir / "splits"
    result = run_split(SplitInput(
        path=str(_current(workdir)), output_dir=str(out_dir),
        seed=int(config.get("seed", 13)),
        train_ratio=float(config.get("train_ratio", 0.9)),
        val_ratio=float(config.get("val_ratio", 0.1)),
        test_ratio=float(config.get("test_ratio", 0.0)),
        group_by=config.get("group_by") or None,
        stratify_by=config.get("stratify_by") or None,
        backend=str(config.get("backend", "auto")),
        shard_size=int(config.get("shard_size", 0)),
    ))
    stats: dict = {"rows": result.total, "backend": result.backend_used,
                   "no_leakage": result.no_leakage}
    for name, n in result.counts.items():
        stats[f"rows_{name}"] = n
    for name, p in result.output_paths.items():
        stats[f"{name}_path"] = p
    if export:
        stats.update(_split_products(workdir, config, result.output_paths, register))
    return stats


#: The role each split is registered for — the names and roles `dataset/split` uses (#2020).
_SPLIT_ROLES = {"train": "training", "val": "validation", "test": "test"}


def _split_products(workdir: Path, config: dict, paths: dict, register: str) -> dict:
    """The ``outputs`` stats of an exported split: per non-empty split its file (local, or
    uploaded with ``store: s3``), hash, size, rows and, with ``register``, the version name and
    role. An empty split is no product: registering a 0-row version would only be refused."""
    store = str(config.get("store") or _artifact_store_backend()).lower()
    out: dict = {"outputs": []}
    for name, path in paths.items():
        digest, size, rows = _sha256_lines(Path(path))
        if not rows:
            continue
        uri = (_s3_upload(Path(path), _s3_export_uri(workdir, f"{name}.jsonl", root=config.get("root")))
               if store == "s3" else str(path))
        out["outputs"].append(name)
        out.update({f"outputs.{name}.output": uri, f"outputs.{name}.content_hash": digest,
                    f"outputs.{name}.bytes": size, f"outputs.{name}.rows": rows,
                    f"outputs.{name}.kind": "dataset"})
        if register:
            out[f"outputs.{name}.register_as"] = f"{register}-{name}"
            out[f"outputs.{name}.register_role"] = _SPLIT_ROLES[name]
    return out


def scan_all(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """The CHECK kernel — the fused single-pass readiness scan (``ScanAllOp``,
    ADR-0080 P1) as a dataflow stage. READ-ONLY: never rewrites ``current.jsonl``.
    Reads ``config.path`` directly when given (one read, no ingest copy) else the
    chain's working file. Scalar stats cross the boundary (D6/R5); the full
    findings payload (pii_counts, trainer_compat, per-row findings) lands as the
    ``scan_findings.json`` artifact for the control-side hoist + console."""
    from operator_dataset_checks.operators.scan_all import run_scan_all  # noqa: PLC0415 - data deps live in stage bodies (D5)
    from operator_dataset_checks.schemas.scan_all import ScanAllInput  # noqa: PLC0415

    source = str(config.get("path") or (inputs[0] if inputs else ""))
    path = _resolve_submit_relative(source) if source else str(_current(workdir))
    allowed = set(ScanAllInput.model_fields) - {"path"}
    kwargs = {k: v for k, v in config.items() if k in allowed}
    result = run_scan_all(ScanAllInput(path=path, **kwargs))

    findings_path = workdir / "scan_findings.json"
    findings_path.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")
    return {
        "scanned": result.scanned, "sampled": result.sampled,
        "sample_size": result.sample_size, "ner_skipped": result.ner_skipped,
        "quarantined_rows": result.quarantined_rows,
        "format_detected": result.format_detected,
        "format_conformant_pct": result.format_conformant_pct,
        "dedup_total": result.dedup_total, "dedup_exact": result.dedup_exact,
        "dedup_effective_size": result.dedup_effective_size, "dedup_pct": result.dedup_pct,
        "profile_total": result.profile_total,
        "profile_refusal_pct": result.profile_refusal_pct, "profile_short": result.profile_short,
        "pii_backend": result.pii_backend, "pii_rows": result.pii_rows,
        "pii_density": result.pii_density, "secret_rows": result.secret_rows,
        "findings_path": str(findings_path),
    }


def decontam_scan(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Detect-only contamination check (``DecontamOp``) — the check-side sibling
    of ``decontam.apply``. Read-only; row indices land in the findings artifact."""
    from operator_dataset_checks.operators.decontam import run_decontam  # noqa: PLC0415 - data deps live in stage bodies (D5)
    from operator_dataset_checks.schemas.decontam import DecontamInput  # noqa: PLC0415

    source = str(config.get("path") or "")
    path = _resolve_submit_relative(source) if source else str(_current(workdir))
    eval_path = str(config.get("eval_path") or "")
    result = run_decontam(DecontamInput(
        path=path,
        eval_path=_resolve_submit_relative(eval_path) if eval_path else None,
        method=str(config.get("method", "exact")),
        normalize=bool(config.get("normalize", True)),
        ngram=int(config.get("ngram", 13)),
        overlap_threshold=float(config.get("overlap_threshold", 0.8)),
    ))
    findings_path = workdir / "decontam_findings.json"
    findings_path.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")
    return {"checked": result.checked, "train_total": result.train_total,
            "eval_total": result.eval_total, "contaminated": result.contaminated,
            "overlap_pct": result.overlap_pct, "findings_path": str(findings_path)}


def filter_rows(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Keep only rows whose ``field`` equals ``value`` (or is in ``values``) —
    #1456: ONE handoff artifact (all splits of a synthesized QA set) feeds the
    train leg (``split == train``) and the eval legs (eval/paraphrase/control)
    without re-synthesizing. DuckDB one-pass when available; a streaming Python
    fallback otherwise. The field is read as TEXT so it works on any JSON key."""
    field = str(config.get("field") or "split")
    values = config.get("values")
    if values is None:
        values = [config.get("value", "train")]
    values = [str(v) for v in values]
    src = _current(workdir)
    out = workdir / "filtered.jsonl"
    total = sum(1 for _ in src.open("rb"))
    try:
        from operator_dataset_checks._duck import READ_JSONL, connect, sql_str  # noqa: PLC0415 - optional duckdb
        con = connect()
        in_list = ", ".join(sql_str(v) for v in values)
        cur = con.execute(
            f"SELECT to_json(json) AS j FROM {READ_JSONL} "
            f"WHERE CAST(json_extract_string(json, {sql_str('$.' + field)}) AS VARCHAR) IN ({in_list})",
            [str(src)])
        kept = _stream_out(cur, out)
    except ImportError:
        kept = 0
        with src.open(encoding="utf-8") as fi, out.open("w", encoding="utf-8") as fo:
            for line in fi:
                if not line.strip():
                    continue
                if str(json.loads(line).get(field, "")) in values:
                    fo.write(line if line.endswith("\n") else line + "\n")
                    kept += 1
    shutil.move(str(out), str(src))
    return {"rows": kept, "total": total, "dropped": total - kept, "field": field,
            "values": ",".join(values)}


def sample_rows(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Reduce the working file to a repeatable reservoir sample of ``config.n``
    ROWS (C1b, #1439): the cheap deterministic front door of a curation leg,
    so an LLM judge scores thousands of rows, not the whole corpus. DuckDB
    ``USING SAMPLE reservoir(N ROWS) REPEATABLE (seed)`` — one pass, streamed
    out; a corpus at or under ``n`` passes through untouched (``sampled=False``).
    ``method: stride`` picks every k-th row instead (no duckdb needed).

    Two additions from T10 (#2029), because a sampled corpus that cannot be
    identified later is a corpus nobody can compare two runs on:

    * ``fraction`` — ``n`` as a share of the corpus rather than an absolute
      count, so "1 %" survives a corpus that grew (``n`` still wins when both
      are given: an absolute cap is what someone types when they mean it);
    * ``digest`` — an order-independent SHA-256 over the KEPT rows' content
      hashes (:func:`cairn.core.dataflow.fidelity.digest_rows`), published in the stats
      beside ``seed`` and ``n``. Two stages that kept the same rows agree on it
      whatever order they streamed them in; two that did not, cannot.

    ``method: hash`` is the third option and the one a **proxy run** should use:
    the kept set is the ``n`` rows with the smallest ``H(seed ‖ row_hash)``, so
    it depends on the corpus and the seed and on nothing else — not on row
    order, not on the engine, not on duckdb's version. The reservoir default is
    unchanged for every existing caller.
    """
    from cairn.core.dataflow.fidelity import Fidelity, digest_rows, plan, row_hash  # noqa: PLC0415 - core helper, kept out of registry enumeration (D5)

    seed = int(config.get("seed", 13))
    method = str(config.get("method", "reservoir")).lower()
    src = _current(workdir)
    out = workdir / "sampled.jsonl"
    total = sum(1 for _ in src.open("rb"))
    # `n` stays what was ASKED FOR, never clamped to the corpus: the stats say
    # "500 wanted, 50 present, not sampled", which is the honest reading and the
    # one every existing caller already gets. A `fraction` only DERIVES an n.
    requested = int(config.get("n", 0) or 0)
    fraction = float(config.get("fraction", 0.0) or 0.0)
    n = requested or (max(1, int(total * fraction)) if fraction > 0 else 5000)
    if total <= n:
        with src.open(encoding="utf-8") as fh:
            kept_rows = [line for line in fh if line.strip()]
        return {"rows": total, "total": total, "sampled": False, "method": method, "n": n,
                "seed": seed, "digest": digest_rows(kept_rows)}
    if method == "hash":
        with src.open(encoding="utf-8") as fh:
            rows = [line for line in fh if line.strip()]
        result = plan(rows, Fidelity(rows=n, seed=seed))
        with out.open("w", encoding="utf-8") as g:
            for line in result.kept:
                g.write(line if line.endswith("\n") else line + "\n")
        shutil.move(str(out), src)
        return {"rows": result.n, "total": total, "sampled": True, "method": method,
                "n": n, "seed": seed, "digest": result.digest}
    if method == "stride":
        stride = total / n
        picks = {int(k * stride) for k in range(n)}
        kept = 0
        with src.open("rb") as f, out.open("wb") as g:
            for i, line in enumerate(f):
                if i in picks:
                    g.write(line)
                    kept += 1
    else:
        from operator_dataset_checks import _duck  # noqa: PLC0415 - data deps live in stage bodies (D5)
        con = _duck.connect()
        try:
            cur = con.execute(
                f"SELECT to_json(json) FROM {_duck.READ_JSONL} "
                f"USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})", [str(src)])
            kept = _stream_out(cur, out)
        finally:
            con.close()
    # The digest is taken from the file that WILL be the working file, before the
    # move — the kept rows are the record, not the ones that were asked for.
    with out.open(encoding="utf-8") as fh:
        digest = digest_rows([row_hash(line) for line in fh if line.strip()], hashed=True)
    shutil.move(str(out), src)
    return {"rows": kept, "total": total, "sampled": True, "method": method, "n": n,
            "seed": seed, "digest": digest}


def curate_score(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """LLM-judge curation of the working file (C1b, #1439) — the proven
    ``CurateOp`` body (AlpaGasus quality, or DEITA quality×complexity) through
    the governed ``llm_complete`` chokepoint: rows are scored, the survivors
    (``keep_fraction`` top share, or ``threshold``) become the working file.
    ``max_samples`` is the cost guard (paid calls) — the flow puts
    ``sample.rows`` in front so it never sees the whole corpus. Stats are FLAT
    scalars (boundary rule); the per-row scores land in ``curate_scores.json``."""
    from operator_dataset_checks.operators.curate import CurateOp  # noqa: PLC0415 - data deps live in stage bodies (D5)
    from operator_dataset_checks.schemas.curate import CurateInput  # noqa: PLC0415

    out = workdir / "curated.jsonl"
    allowed = set(CurateInput.model_fields) - {"path", "output_path"}
    kwargs = {k: v for k, v in config.items() if k in allowed and v is not None}
    result = asyncio.run(CurateOp().run(CurateInput(
        path=str(_current(workdir)), output_path=str(out), **kwargs), _Ctx()))
    (workdir / "curate_scores.json").write_text(
        json.dumps({"scores": [r.model_dump() for r in result.scores],
                    "score_distribution": result.score_distribution}, indent=2),
        encoding="utf-8")
    shutil.move(str(out), _current(workdir))
    stats: dict = {
        "rows": result.kept, "total": result.total_rows, "extractable": result.extractable,
        "scored": result.scored, "kept": result.kept, "dropped": result.dropped,
        "parse_failures": result.parse_failures, "mean_score": result.mean_score,
        "method": result.method, "model": result.model, "criterion": result.criterion,
    }
    for bucket, count in (result.score_distribution or {}).items():
        stats[f"bucket_{str(bucket).replace('.', '_').replace('-', '_')}"] = int(count)
    return stats


def export_parquet(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Emit the final artifact as parquet (columnar, ADR-0080 D4)."""
    from operator_dataset_checks._io import iter_rows, write_parquet  # noqa: PLC0415 - data deps live in stage bodies (D5)

    dest = workdir / str(config.get("name", "output.parquet"))
    rows = write_parquet(iter_rows(str(_current(workdir))), str(dest))
    return {"rows": rows, "output": str(dest), "format": "parquet"}


def export_jsonl(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Emit the final artifact as JSONL — the leg's PRODUCT, what the next leg
    ingests. ``config.store``: ``local`` (a path in this workdir; one box only)
    or ``s3`` (#1442: uploaded under ``CAIRN_ARTIFACT_S3_BUCKET/PREFIX/dataflow/
    <workdir-id>/<name>``, ``output`` = the ``s3://`` URI). Unset → the
    environment's ``artifact_store`` backend (profile → CAIRN_ARTIFACT_STORE →
    local). Always returns ``content_hash`` + ``bytes`` so the engine registers
    a REAL descriptor, not a placeholder."""
    name = str(config.get("name", "output.jsonl"))
    src = _current(workdir)
    # #2396 (ADR-0102 D1) — `rows` is the `rows_out` the intake verdict reads (0 → invalid_format),
    # counted in the same pass as the hash.
    digest, size, rows = _sha256_lines(src)
    # #1466 — `register: <dataset-name>` asks the CONTROL PLANE to add this
    # export as a new dataset version once the leg lands (the box has no DB
    # access, D4) — the kernel only echoes the request into its stats, which
    # ride the descriptor to leg_executor's registration block.
    register_as = str(config.get("register") or "").strip()
    store = str(config.get("store") or _artifact_store_backend()).lower()
    if store == "s3":
        uri = _s3_upload(src, _s3_export_uri(workdir, name, root=config.get("root")))
        out = {"output": uri, "format": "jsonl", "store": "s3",
               "content_hash": digest, "bytes": size, "rows": rows}
        if register_as:
            out["register_as"] = register_as
        return out
    dest = workdir / name
    shutil.copyfile(src, dest)
    out = {"output": str(dest), "format": "jsonl", "store": "local",
           "content_hash": digest, "bytes": size, "rows": rows}
    if register_as:  # was s3-branch-only — a local export silently never registered
        out["register_as"] = register_as
    return out


def _artifact_store_backend() -> str:
    """The environment's ``artifact_store`` selection (profile → env → local)
    through the ONE config plane; ``local`` when core is unavailable."""
    try:
        from cairn.core.backend_selection import selected  # noqa: PLC0415 - control-plane import inside a kernel; optional at run time
        return str(selected("artifact_store", default="local") or "local")
    except Exception:  # noqa: BLE001 — a kernel must run without the control plane installed
        return getenv("CAIRN_ARTIFACT_STORE", "local") or "local"


def _sha256_file(path: Path) -> tuple[str, int]:
    digest, size, _ = _sha256_lines(path)
    return digest, size


def _sha256_lines(path: Path) -> tuple[str, int, int]:
    """sha256, size and line count in ONE pass (a JSONL file: one record per line)."""
    h = hashlib.sha256()
    n = lines = 0
    last = b"\n"
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    return h.hexdigest(), n, lines + (last != b"\n")


def _boto3():
    try:
        import boto3  # noqa: PLC0415 - optional dependency: the s3 store
    except ImportError as e:
        raise RuntimeError("artifact_store=s3 needs boto3: pip install boto3") from e
    return boto3


def _s3_client():
    region = (getenv("CAIRN_ARTIFACT_S3_REGION") or os.environ.get("AWS_DEFAULT_REGION") or None)
    endpoint = getenv("CAIRN_ARTIFACT_S3_ENDPOINT") or None
    return _boto3().client("s3", region_name=region, endpoint_url=endpoint)


def _s3_export_uri(workdir: Path, name: str, *, root: str | None = None) -> str:
    """``<root or s3://bucket/prefix>/dataflow/<workdir-id>/<name>`` — the
    workdir id (the leg's temp dir name) keeps legs from overwriting each other."""
    if root:
        base = str(root).rstrip("/")
    else:
        bucket = getenv("CAIRN_ARTIFACT_S3_BUCKET", "").strip()
        if not bucket:
            raise RuntimeError("artifact_store=s3 needs CAIRN_ARTIFACT_S3_BUCKET (the s3/put contract)")
        prefix = getenv("CAIRN_ARTIFACT_S3_PREFIX", "").strip("/")
        base = f"s3://{bucket}" + (f"/{prefix}" if prefix else "")
    return f"{base}/dataflow/{workdir.name}/{name}"


def _split_s3(uri: str) -> tuple[str, str]:
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def _s3_upload(src: Path, uri: str) -> str:
    bucket, key = _split_s3(uri)
    _s3_client().upload_file(str(src), bucket, key)  # streams, multipart when large
    return uri


def _s3_download(uri: str, dest: Path) -> str:
    bucket, key = _split_s3(uri)
    _s3_client().download_file(bucket, key, str(dest))
    return str(dest)


# ─── #2021 (ADR-0102 D1) — the intake chain's two new kernels ─────────────
#
# The admission chain a readiness pipeline runs is a dataflow of these kernels, and D1 says what it
# must contain: exact AND near-duplicate dedup, decontamination against every
# golden/test version in the workspace, PII redaction, declared expectations,
# and an automatic token-length profile with the base model's tokenizer. The
# first three already had kernels; these two are what was missing.


def dedup_near(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Drop NEAR-duplicates — the ``DedupApplyOp`` body at ``method=minhash``.

    Runs AFTER ``dedup.exact`` in the admission chain, which is why it reports
    ``dropped_near`` separately: "3 exact and 2 near-duplicate rows" is two
    facts a reader needs, and one merged count hides which pass found what.

    Near-duplicate detection needs datasketch (the pack's ``[minhash]`` extra).
    Without it this kernel is an HONEST SKIP — ``checked: false`` plus a note,
    the working file untouched — and NOT a failure, because it now sits in the
    DEFAULT chain that every import runs: a lean install must still be able to
    import a dataset. The intake verdict carries the note (never a degrade
    reason: a missing optional dependency is the deployment's business, not the
    dataset's fault). ``dedup.exact``'s opted-in ``backend=duckdb`` keeps the
    opposite posture — an explicit opt-in that silently downgraded would fake
    the evidence a gate reads.
    """
    from operator_dataset_checks._minhash import minhash_available  # noqa: PLC0415 - data deps live in stage bodies (D5)

    src = _current(workdir)
    if not minhash_available():
        rows = sum(1 for line in src.open(encoding="utf-8") if line.strip())
        return {"rows": rows, "dropped_near": 0, "dropped": 0, "checked": False,
                "method": "skipped",
                "note": "near-duplicate dedup needs the [minhash] extra "
                        "(pip install 'cairn-operator-dataset-checks[minhash]') — only exact "
                        "duplicates were removed"}
    from operator_dataset_checks.operators.dedup_apply import run_dedup_apply  # noqa: PLC0415 - data deps live in stage bodies (D5)
    from operator_dataset_checks.schemas.dedup_apply import DedupApplyInput  # noqa: PLC0415

    out = workdir / "deduped-near.jsonl"
    result = run_dedup_apply(DedupApplyInput(
        path=str(src), output_path=str(out), method="minhash",
        normalize=bool(config.get("normalize", True)),
        threshold=float(config.get("threshold", 0.8)),
        num_perm=int(config.get("num_perm", 128)),
        ngram=int(config.get("ngram", 3)),
    ))
    shutil.move(str(out), src)  # shutil.move: workdir may span filesystems (the EXDEV lesson)
    return {"rows": result.kept, "dropped": result.dropped,
            "dropped_near": result.dropped_near, "checked": True,
            "method": result.method_used,
            "threshold": float(config.get("threshold", 0.8))}


#: Histogram edges for the token-length profile, in tokens. Powers of two
#: because that is what ``max_seq_len`` is chosen from (D2.2: "next power of
#: two >= p95"), so the bucket a corpus lands in IS the recommendation.
_TOKEN_BUCKETS = (0, 128, 256, 512, 1024, 2048, 4096, 8192)

#: What to do with a row longer than ``max_seq_len``. Never ``truncate``:
#: OpenAI truncates over-length examples "from the end" with no error, and
#: ADR-0102 D1 names that as the anti-pattern — "counted and dropped or
#: refused, never silently truncated. There is no third behaviour."
_OVER_LENGTH_POLICIES = ("count", "drop", "refuse")


def _percentile(sorted_values: list[int], q: float) -> int:
    """The ``q``-quantile by nearest-rank — no numpy, and exact on the small
    integer counts a token profile produces."""
    if not sorted_values:
        return 0
    idx = max(0, min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1)))))
    return int(sorted_values[idx])


def _resolve_tokenizer(config: dict) -> tuple[str, str, str]:
    """``(backend, tokenizer, note)`` for the token profile.

    ADR-0102 D1: profile "with the base model chosen at import or later at
    launch", and **fall back to whitespace when no tokenizer is available,
    saying so**. Three ways to name one — an explicit local ``tokenizer_path``,
    an explicit ``tokenizer_name``, or the flow's ``base_model`` (a HF id, which
    is what a base model already is) — and every failure to load one degrades
    to the zero-dep word count WITH a note rather than failing the import.
    """
    from operator_dataset_checks import _tokenizer  # noqa: PLC0415 - data deps live in stage bodies (D5)

    path = str(config.get("tokenizer_path") or "")
    name = str(config.get("tokenizer_name") or config.get("base_model") or "")
    if not (path or name):
        return ("whitespace", "",
                "no base model or tokenizer was named at import — token counts are "
                "whitespace WORD counts, not the target model's subword tokens")
    if not _tokenizer.tokenizers_available():
        return ("whitespace", path or name,
                f"tokenizer {path or name!r} was requested but the [tokenizers] extra is "
                "not installed — token counts are whitespace WORD counts")
    try:
        _tokenizer.count("probe", backend="tokenizers", tokenizer_name=name or None,
                         tokenizer_path=path or None)
    except Exception as exc:  # noqa: BLE001 — an unreachable Hub must not fail an import
        return ("whitespace", path or name,
                f"tokenizer {path or name!r} could not be loaded "
                f"({type(exc).__name__}: {exc}) — token counts are whitespace WORD counts")
    return ("tokenizers", path or name, "")


def profile_tokens(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Token-length profile of the working dataset, and the over-length policy.

    One streaming pass: per-row token counts (the target model's tokenizer when
    one is available, whitespace words otherwise — always said out loud in
    ``token_backend_note``), the total, p50/p95/max, a power-of-two histogram,
    and the count of rows longer than ``max_seq_len``.

    ``over_length`` decides what happens to those rows: ``count`` (default —
    they stay, the report names them), ``drop`` (they are removed and the count
    reported), or ``refuse`` (they stay, and the intake verdict comes back
    ``blocked`` so the launch door refuses the version). Truncation is not one
    of the options.

    The token total is the number the cost estimate reads (``tokens_total`` on
    the verdict report), and p95 is what the configuration recommender turns
    into ``max_seq_len`` (ADR-0102 D2.2).
    """
    max_seq_len = int(config.get("max_seq_len", 0) or 0)
    policy = str(config.get("over_length", "count")).lower()
    if policy not in _OVER_LENGTH_POLICIES:
        raise ValueError(
            f"over_length must be one of {_OVER_LENGTH_POLICIES} — got {policy!r}. "
            "Truncation is deliberately not an option (ADR-0102 D1).")
    # cairn-extensions#16 — `example` measures prompt + response, which is what a training
    # sequence holds (the chat template's own tokens aside); `response` (default) as before.
    measure = str(config.get("measure", "response")).lower()
    if measure not in ("response", "example"):
        raise ValueError(f"measure must be 'response' or 'example' — got {measure!r}")
    backend, tokenizer, note = _resolve_tokenizer(config)

    from operator_dataset_checks import _tokenizer  # noqa: PLC0415 - data deps live in stage bodies (D5)
    from operator_dataset_checks._io import JsonlWriter  # noqa: PLC0415
    from operator_dataset_checks.operators.profile import _prompt_text, _response_text  # noqa: PLC0415 - ONE definition of "the model's answer"

    src = _current(workdir)
    within = workdir / "within-length.jsonl"
    tok_path = str(config.get("tokenizer_path") or "") or None
    tok_name = (tokenizer or None) if (backend == "tokenizers" and not tok_path) else None
    lengths: list[int] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    over = 0
    kept = 0
    quarantined = 0

    def _scan(keep) -> None:
        """One streaming pass; ``keep`` is called with every row that survives
        the over-length policy (a no-op unless the policy is ``drop``)."""
        nonlocal over, kept, quarantined
        with src.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    quarantined += 1
                    continue
                if not isinstance(row, dict):
                    quarantined += 1
                    continue
                n = _tokenizer.count(_response_text(row), backend=backend,
                                     tokenizer_name=tok_name, tokenizer_path=tok_path)
                if measure == "example":
                    p = _tokenizer.count(_prompt_text(row), backend=backend,
                                         tokenizer_name=tok_name, tokenizer_path=tok_path)
                    prompt_lengths.append(p)
                    response_lengths.append(n)
                    n += p
                lengths.append(n)
                if max_seq_len and n > max_seq_len:
                    over += 1
                    if policy == "drop":
                        continue
                keep(row)
                kept += 1

    if policy == "drop":
        with JsonlWriter(within) as writer:
            _scan(writer.write)
        shutil.move(str(within), src)  # shutil.move: workdir may span filesystems
    else:
        _scan(lambda _row: None)

    ordered = sorted(lengths)
    stats: dict = {
        "rows": kept,
        "tokens_total": sum(lengths),
        "tokens_p50": _percentile(ordered, 0.50),
        "tokens_p95": _percentile(ordered, 0.95),
        "tokens_max": ordered[-1] if ordered else 0,
        "token_backend": backend,
        "tokenizer": tokenizer,
        "max_seq_len": max_seq_len,
        "over_length_rows": over,
        "over_length_policy": policy,
        "quarantined_rows": quarantined,
    }
    if note:
        stats["token_backend_note"] = note
    stats["measure"] = measure
    if measure == "example":
        for side, values in (("prompt", sorted(prompt_lengths)), ("response", sorted(response_lengths))):
            stats.update({f"{side}_tokens_p50": _percentile(values, 0.50),
                          f"{side}_tokens_p95": _percentile(values, 0.95),
                          f"{side}_tokens_max": values[-1] if values else 0})
    # Histogram as FLAT scalars — only scalars cross the dataflow boundary
    # (ADR-0081 D3 / ADR-0080 R5); cairn.core.intake reassembles the buckets.
    edges = list(_TOKEN_BUCKETS)
    for i, low in enumerate(edges):
        high = edges[i + 1] if i + 1 < len(edges) else None
        count = sum(1 for n in lengths if n >= low and (high is None or n < high))
        stats[f"bucket_{low}_{high if high is not None else 'plus'}"] = count
    return stats


_DERIVE_MODES = ("extract", "normalize")
_DERIVE_CASES = ("keep", "lower", "upper")


def _derive_rules(config: dict) -> list[dict]:
    """Validated ``fields.derive`` rules, each with its compiled pattern."""
    import re  # noqa: PLC0415 - stage bodies keep their imports (D5)

    rules = config.get("rules") or []
    if not isinstance(rules, list):
        raise ValueError("fields.derive: `rules` is a list of {field, source, mode, pattern}")
    out, fields = [], set()
    for rule in rules:
        field, source = str(rule.get("field") or ""), str(rule.get("source") or "")
        mode, case = str(rule.get("mode", "extract")), str(rule.get("case", "keep"))
        if not field or not source or not rule.get("pattern"):
            raise ValueError(f"fields.derive: a rule needs field, source and pattern — got {rule!r}")
        if mode not in _DERIVE_MODES or case not in _DERIVE_CASES:
            raise ValueError(f"fields.derive: mode must be one of {_DERIVE_MODES} and case one of "
                             f"{_DERIVE_CASES} — got {mode!r}, {case!r}")
        if field in fields:
            raise ValueError(f"fields.derive: two rules write {field!r}")
        fields.add(field)
        flags = re.IGNORECASE if rule.get("ignore_case") else 0
        out.append({"field": field, "source": source, "mode": mode, "case": case,
                    "rx": re.compile(str(rule["pattern"]), flags),
                    "replace": str(rule.get("replace", "#")), "default": str(rule.get("default", "other"))})
    return out


def fields_derive(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """cairn-extensions#16 — add fields to every row from a regular expression over another
    field, for a dataset that has no topic or template column of its own.

    ``extract`` keeps the first match (its first group when the pattern has one) and
    ``default`` (``other``) when nothing matches — a topic label. ``normalize`` replaces every
    match with ``replace`` (``#``) and collapses whitespace — a template family, so rows that
    differ only in numbers or names share one value. A rule never overwrites a field the row
    already has: the source data stays as it was, and a trainer that formats the Alpaca or chat
    fields never sees the added ones. Stats are counts only. No rules (a pipeline that bound
    none) leaves the working dataset as it is."""
    from operator_dataset_checks._io import JsonlWriter  # noqa: PLC0415 - data deps live in stage bodies (D5)

    rules = _derive_rules(config)
    src, tmp = _current(workdir), workdir / "derived.jsonl"
    if not rules:
        return {"rows": _sha256_lines(src)[2], "fields": []}
    matched = {r["field"]: 0 for r in rules}
    rows = 0
    with JsonlWriter(tmp) as writer, src.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            for r in rules:
                if r["field"] in row:
                    raise ValueError(f"fields.derive: rows already carry {r['field']!r}; "
                                     "a derived field never overwrites data")
                text = str(row.get(r["source"]) or "")
                if r["mode"] == "extract":
                    m = r["rx"].search(text)
                    found = (m.group(1) if m.re.groups else m.group(0)) if m else None
                    # An optional group that did not take part is no match either.
                    value = found if found is not None else r["default"]
                    matched[r["field"]] += found is not None
                else:
                    value = " ".join(r["rx"].sub(r["replace"], text).split())
                    matched[r["field"]] += value != " ".join(text.split())
                row[r["field"]] = value.lower() if r["case"] == "lower" else value.upper() if r["case"] == "upper" else value
            writer.write(row)
            rows += 1
    shutil.move(str(tmp), src)
    stats: dict = {"rows": rows, "fields": [r["field"] for r in rules]}
    for field, n in matched.items():
        stats[f"{field}_matched"] = n
    return stats


def profile_fields(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """cairn-extensions#16 — the distribution of row fields, as counts. READ-ONLY.

    ``count`` fields (labels, e.g. a derived topic) report their ``top`` values (default 20,
    each cut to ``value_chars``, default 64) and counts as two flat lists, the rest summed as
    ``other``. ``groups`` fields (e.g. a template family, whose values are prompt text) report
    sizes only — distinct groups, the largest, singletons, the median — never a value, so no
    row content reaches the record. Neither set reports the row count only."""
    from collections import Counter  # noqa: PLC0415 - stage bodies keep their imports (D5)

    count_fields = [str(f) for f in (config.get("count") or [])]
    group_fields = [str(f) for f in (config.get("groups") or [])]
    top, chars = int(config.get("top", 20)), int(config.get("value_chars", 64))
    counters = {f: Counter() for f in {*count_fields, *group_fields}}
    missing = dict.fromkeys(counters, 0)
    rows = 0
    with _current(workdir).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            for f, c in counters.items():
                if row.get(f) in (None, ""):
                    missing[f] += 1
                else:
                    c[str(row[f])] += 1
    stats: dict = {"rows": rows}
    for f in count_fields:
        common = counters[f].most_common(top)
        stats.update({f"{f}_values": [" ".join(v.split())[:chars] for v, _ in common],
                      f"{f}_counts": [n for _, n in common],
                      f"{f}_other": sum(counters[f].values()) - sum(n for _, n in common),
                      f"{f}_distinct": len(counters[f]), f"{f}_missing": missing[f]})
    for f in group_fields:
        sizes = sorted(counters[f].values())
        stats.update({f"{f}_groups": len(sizes), f"{f}_largest": sizes[-1] if sizes else 0,
                      f"{f}_singletons": sum(1 for n in sizes if n == 1),
                      f"{f}_median": _percentile(sizes, 0.50), f"{f}_missing": missing[f]})
    return stats


# ─── risk classes (ADR-0081 D2, #1051) ────────────────────────────────────
# The compiler aggregates these per dataflow: any external_write stage makes
# the binding template REQUIRE an approval gate at compile. Undeclared kernels
# count as external_write (fail-closed, #744) — declaring read_only is the
# opt-out, reserved for genuinely non-mutating scans.
for _fn in (ingest_normalize, dedup_exact, dedup_near, decontam_apply, pii_redact,
            profile_tokens, split, export_parquet, export_jsonl, sample_rows,
            curate_score, filter_rows):
    _fn.risk_class = "external_write"
for _fn in (scan_all, decontam_scan):
    _fn.risk_class = "read_only"
# cairn-extensions#5 — the manifest describes the dataset; with store=s3 it uploads a report.
dataset_manifest.risk_class = "external_write"
# cairn-extensions#16 — fields.derive rewrites the working dataset; profile.fields only reads it.
fields_derive.risk_class = "external_write"
profile_fields.risk_class = "read_only"

# ─── stage contracts (ADR-0118 D1/M3a, #2242 V1-C) ──────────────────────────
# The readiness stages implement their shared stage contract (cairn-contracts-dataset-stages):
# `contract://dataset.stage.<kernel>/1.0.0`, recorded at the launch door and checked on the
# control plane and in the runner. The other kernels stay legacy.
for _fn in (ingest_normalize, dedup_exact, dedup_near, decontam_apply, pii_redact,
            profile_tokens, export_jsonl, scan_all):
    _fn.implements = (f"contract://dataset.stage.{_fn.__name__}/1.0.0",)
# #2396 — the two kernels whose output gained the stats the intake verdict reads implement the
# 1.1.0 revision, which requires them (1.0.0 stays published, unchanged).
for _fn in (ingest_normalize, export_jsonl):
    _fn.implements = (f"contract://dataset.stage.{_fn.__name__}/1.1.0",)
# cairn-extensions#5 — ingest takes a Hub revision (configuration 1.1.0, output 1.2.0); the
# manifest stage is new.
ingest_normalize.implements = ("contract://dataset.stage.ingest_normalize/1.2.0",)
dataset_manifest.implements = ("contract://dataset.stage.dataset_manifest/1.0.0",)
# cairn-extensions#6 — the split kernel joins the stage contracts, with `export` and `register`.
split.implements = ("contract://dataset.stage.split/1.0.0",)
# cairn-extensions#16 — profile.tokens measures prompt + response on request (1.1.0); the two
# field kernels are new.
profile_tokens.implements = ("contract://dataset.stage.profile_tokens/1.1.0",)
fields_derive.implements = ("contract://dataset.stage.fields_derive/1.0.0",)
profile_fields.implements = ("contract://dataset.stage.profile_fields/1.0.0",)
