"""Dataset loading for the dataset_checks operators.

Three entry points, chosen by memory profile (issue #380 — the memory keystone):

- :func:`load_rows` — materialize the whole dataset into a ``list[dict]``. The
  convenience path for small datasets and operators that need random access or
  two passes. HF ``datasets`` backed (local JSONL/JSON/CSV/parquet + Hub ids).
- :func:`iter_rows` — STREAM rows one at a time, never holding the full dataset
  in memory. Local ``.jsonl`` streams line-by-line with the stdlib reader (O(1)
  memory, no schema inference, nested ``messages``/``conversations`` preserved
  verbatim); other local formats and Hub ids stream through ``datasets``'
  ``IterableDataset``. Single-pass operators (dedup/pii_scan/profile/…) consume
  this so a multi-million-row corpus never blows RAM. Needs NO extra — the
  memory fix runs on the base ``datasets`` dep.
- :func:`scan_frame` — a polars ``LazyFrame`` over a local flat file for the
  COLUMNAR fast path: vectorized token counts (#381) and MinHash (#382) over
  Arrow. Behind the ``[fast]`` extra; returns ``None`` when polars is absent or
  the source isn't a lazily-scannable local file (Hub id / nested chat) so
  callers fall back to :func:`iter_rows`.
"""
from __future__ import annotations

import json
import os
import random
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

import boto3
from datasets import load_dataset, load_dataset_builder

if TYPE_CHECKING:  # type-only; polars is an optional [fast] extra, never at runtime
    import polars as pl

try:  # keep test/CLI output clean
    from datasets import disable_progress_bars

    disable_progress_bars()
except Exception:  # noqa: BLE001 — older datasets without the helper
    pass

_BUILDER_BY_SUFFIX = {
    ".jsonl": "json",
    ".json": "json",
    ".csv": "csv",
    ".parquet": "parquet",
}


def _normalize_source(source: str) -> str:
    """Resolve a ``file://`` URI to a plain local path; pass everything else
    through unchanged (issue #584).

    ``FinetuneRequest`` (#571) accepts ``scheme://`` dataset paths, and an
    ADR-0066 dataset pack can emit a ``file://`` ``dataset_uri``. Without this,
    ``Path("file:///tmp/x.jsonl").exists()`` is False → the loader mis-classifies
    it as an HF Hub id and errors on a mangled ``/cwd/file:/tmp/x.jsonl``. A
    ``file://`` URI is a local path; ``urlparse`` handles ``file:///abs`` and
    ``file://localhost/abs`` and ``unquote`` decodes percent-escapes. Non-``file``
    schemes (``s3://``/``https://``/Hub ids) pass through — the loaders route
    remote URLs via ``data_files=`` (see :func:`_remote_builder`, #913).

    #2171: the single-slash ``file:/abs`` form is accepted too — it is what
    ``str(Path("file:///abs"))`` produces, and a source that was already
    mangled that way once must still name the same bytes.
    """
    if source.startswith("file:"):
        return unquote(urlparse(source).path)
    return source


_REMOTE_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _is_s3_jsonl(source: str) -> bool:
    """``s3://bucket/key.jsonl`` — the one remote form that must NOT go through
    the Arrow builder. Arrow's JSON reader infers column types, so an ISO date
    string comes back as a ``datetime`` object — which breaks byte-level
    contracts downstream (``canonical_hash`` dedup crashed on it, found live on
    a production replay leg, 2026-08-28). Local ``.jsonl`` already has a
    verbatim fast path; this predicate routes the s3 twin to the same reader."""
    return source.startswith("s3://") and urlparse(source).path.lower().endswith(".jsonl")


def _fetch_s3_jsonl(source: str) -> Path:
    """Download ``s3://bucket/key.jsonl`` to a private temp file via boto3 and
    return the local path, so the stdlib line reader (:func:`_iter_jsonl`)
    serves s3 sources with VERBATIM strings. boto3 deliberately, not s3fs:
    boto3 is a base dep, while s3fs drags aiobotocore's botocore pin into
    conflict with it (#1716 dropped that dep for exactly this reason).

    The temp file lands in the leg's tmp and is cleaned with it."""
    u = urlparse(source)
    fd, name = tempfile.mkstemp(suffix=".jsonl", prefix="s3-jsonl-")
    os.close(fd)
    _s3_client().download_file(u.netloc, u.path.lstrip("/"), name)
    return Path(name)


def _s3_client():
    """Same contract as ``operator_training.checkpoint_sync.s3_client`` (packs
    own their impl, so the 6 lines are copied, not imported): honor
    ``AWS_ENDPOINT_URL``/``AWS_DEFAULT_REGION`` from the operator env injection
    so MinIO/R2-compatible stores resolve identically to checkpoints."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or None
    region = os.environ.get("AWS_DEFAULT_REGION") or None
    return boto3.client("s3", endpoint_url=endpoint, region_name=region)


def upload_s3(local: Path, dest: str) -> None:
    """Upload ``local`` to ``dest`` (``s3://bucket/key``) via boto3 — the write
    twin of :func:`_fetch_s3_jsonl`, for writers whose declared output path is
    an s3 URI (``dataset/replay``'s ``out_path``: the train leg may run on a
    different box, so a local file at a path SPELLED ``s3://…`` is a ghost —
    exactly what :class:`JsonlWriter` produced before this existed)."""
    u = urlparse(dest)
    _s3_client().upload_file(str(local), u.netloc, u.path.lstrip("/"))


def _remote_builder(source: str) -> str | None:
    """Builder name for a REMOTE ``scheme://`` source, else ``None`` (#913).

    ``load_dataset(path=url)`` does NOT fetch URLs — it mangles them into a
    relative local path (``/app/http:/…`` → FileNotFoundError), which broke every
    readiness/prep op on the exact dataset forms `FinetuneRequest` (#571) REQUIRES
    with a ``compute_target``. Remote files must go through ``data_files=`` with
    an explicit builder; ``datasets``' bundled fsspec then fetches http(s) natively
    (s3/gs need their fsspec extra installed, same as anywhere else). The builder
    comes from the URL path's suffix, defaulting to json — training corpora are
    JSONL. Called AFTER ``_normalize_source`` (file:// already resolved to local).
    """
    if not _REMOTE_URL.match(source):
        return None
    suffix = Path(urlparse(source).path).suffix.lower()
    return _BUILDER_BY_SUFFIX.get(suffix, "json")


def _drop_nulls(row: dict) -> dict:
    """Keep only the keys a row actually carried.

    ``datasets`` normalizes to a columnar schema and null-pads absent keys; the
    checks expect per-row key presence to match the source JSON, so we restore
    it. The stdlib JSONL reader has no null-padding, but explicit ``null`` values
    are dropped here too for parity with the ``datasets`` path.
    """
    return {k: v for k, v in row.items() if v is not None}


def _reservoir_sample(rows: list[dict], k: int, seed: int = 1) -> list[dict]:
    """Deterministically down-sample ``rows`` to ``k`` via reservoir sampling.

    Algorithm R with a FIXED-seed RNG, so the selected subset is reproducible
    and replayable — a hard requirement for the readiness gate, where the same
    dataset must always yield the same verdict (#656). For a go/no-go gate a
    uniform sample is a statistically valid estimate of the aggregates the
    statistical checks compute (format conformance, PII density, length
    distributions), so scanning a sample gives the same verdict as scanning
    every row without materializing 10M rows of work.

    Returns all rows (a shallow copy) when ``k >= len(rows)``.
    """
    if k >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    reservoir = list(rows[:k])
    for i in range(k, len(rows)):
        j = rng.randint(0, i)
        if j < k:
            reservoir[j] = rows[i]
    return reservoir


def reservoir_sample_stream(rows: "Iterator[dict]", k: int, seed: int = 1) -> list[dict]:
    """Reservoir-sample a STREAM to ``k`` rows in O(k) memory (#891).

    The streaming twin of :func:`_reservoir_sample`: consumes an ``iter_rows``
    stream so the whole dataset never lands in RAM, yet selects the SAME uniform
    k-row subset — same Algorithm R, same fixed-seed RNG, same ``randint(0, i)``
    call sequence — so a sampled gate stays byte-for-byte reproducible whether it
    read the file into a list or streamed it (#656). Returns every row when the
    stream is shorter than ``k`` (no RNG consumed, matching the list version)."""
    rng = random.Random(seed)
    reservoir: list[dict] = []
    for i, row in enumerate(rows):
        if i < k:
            reservoir.append(row)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = row
    return reservoir


def _tmp_path(final: Path) -> Path:
    """The private sibling a writer streams into before an atomic commit (#1007).
    Pid-suffixed so concurrent writers (shard children) never collide."""
    return final.with_name(f"{final.name}.tmp-{os.getpid()}")


class RowErrors:
    """Running tally of malformed source lines routed to quarantine (#1008).

    Pass one to :func:`iter_rows`/:func:`load_rows` to observe how many lines
    were quarantined and where the side file landed. Mutated in place while the
    stream is consumed, so it reflects the running state mid-iteration and the
    final state after exhaustion."""

    def __init__(self) -> None:
        self.count = 0
        self.quarantine_path: str | None = None

    def as_dict(self) -> dict:
        return {"quarantined_rows": self.count, "quarantine_path": self.quarantine_path}


def write_parquet(rows: "Iterator[dict]", path: str, *, batch_size: int = 10000) -> int:
    """Stream ``rows`` to a Parquet file at ``path``, atomically (ADR-0080 D4).

    The ADR-0080 P3 fix job's columnar output. Uses ``pyarrow`` — a BASE transitive
    dep (via ``datasets``), so no extra is needed to WRITE parquet (unlike the
    polars/duckdb read accelerants). Streams in ``batch_size`` record batches so a
    million-row dataset never fully materializes; the schema is fixed from the first
    batch and later batches are coerced to it (fine for the homogeneous cleaned data
    the fix produces). Commits via tmp + ``os.replace`` (#1007), matching
    :class:`JsonlWriter`. Returns the row count written."""
    import pyarrow as pa  # noqa: PLC0415 — base transitive dep (datasets), not an extra
    import pyarrow.parquet as pq  # noqa: PLC0415

    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(final)
    writer: "pq.ParquetWriter | None" = None
    schema = None
    total = 0
    buf: list[dict] = []

    def _flush() -> None:
        nonlocal writer, schema
        if not buf:
            return
        table = (
            pa.Table.from_pylist(buf)
            if schema is None
            else pa.Table.from_pylist(buf, schema=schema)
        )
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(str(tmp), schema)
        writer.write_table(table)
        buf.clear()

    try:
        for row in rows:
            buf.append(row)
            total += 1
            if len(buf) >= batch_size:
                _flush()
        _flush()
        if writer is None:  # empty dataset — still emit a valid (empty) parquet file
            pq.write_table(pa.table({"record": pa.array([], type=pa.string())}), str(tmp))
        else:
            writer.close()
            writer = None
        os.replace(tmp, final)
    finally:
        if writer is not None:
            writer.close()
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return total


class JsonlWriter:
    """Incrementally write rows to ONE JSONL file at an exact path, byte-compatible
    with ``"\\n".join(json.dumps(r) for r in rows)`` (no trailing newline) — so a
    streaming write reproduces the old materialized output exactly (#891). Use as a
    context manager; an empty stream still writes an empty file.

    ATOMIC (#1007): rows stream into a private ``<path>.tmp-<pid>`` sibling which
    is ``os.replace``d onto the real path only on clean exit — same-filesystem
    rename, so the final path holds either its previous content or the complete
    new content, never a truncated prefix. On an exception the tmp is removed and
    the real path is untouched."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._tmp = _tmp_path(self._path)
        self._fh = None
        self._wrote = False

    def write(self, row: dict) -> None:
        if self._wrote:
            self._fh.write("\n")
        self._fh.write(json.dumps(row, ensure_ascii=False))
        self._wrote = True

    def __enter__(self) -> "JsonlWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._tmp.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if exc_type is None:
            os.replace(self._tmp, self._path)
        else:
            self._tmp.unlink(missing_ok=True)


class ShardedJsonlWriter:
    """Incrementally write rows to one or more JSONL shards WITHOUT holding them
    in memory (#891) — the write half of the streaming refactor.

    Each shard is byte-compatible with ``"\\n".join(json.dumps(r) for r in rows)``
    (no trailing newline), so a single-shard write reproduces the old output
    exactly. ``shard_size=0`` (the default) writes ONE file — the back-compatible
    shape — at ``<dir>/<name>.jsonl``; ``shard_size>0`` rolls to
    ``<dir>/<name>.00001.jsonl`` … every ``shard_size`` rows so a >>RAM split
    lands as many bounded files a downstream reader can stream/parallelize.
    ``paths`` lists every shard actually written (a single-element list when
    unsharded), and ``count`` the rows. Use as a context manager.

    ATOMIC (#1007): each shard streams into a ``.tmp-<pid>`` sibling and is
    ``os.replace``d into place when it rolls (it is then a COMPLETE shard) or on
    ``close()``. On an exception the in-flight shard's tmp is removed and dropped
    from ``paths`` — already-committed shards remain, each one complete; the
    final path set never contains a truncated file."""

    def __init__(self, out_dir: Path, name: str, *, shard_size: int = 0) -> None:
        self._dir = out_dir
        self._name = name
        self._shard_size = max(0, int(shard_size))
        self.paths: list[str] = []
        self.count = 0
        self._fh = None
        self._cur_final: Path | None = None  # the shard being streamed (uncommitted)
        self._cur_tmp: Path | None = None
        self._in_shard = 0  # rows written to the current shard
        self._shard_idx = 0

    def _shard_path(self, idx: int) -> Path:
        # shard 0 keeps the bare <name>.jsonl so output_paths[name] is unchanged.
        return self._dir / (f"{self._name}.jsonl" if idx == 0 else f"{self._name}.{idx:05d}.jsonl")

    def _commit(self) -> None:
        """Close the in-flight shard and atomically publish it at its final path."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            os.replace(self._cur_tmp, self._cur_final)
            self._cur_final = self._cur_tmp = None

    def _roll(self) -> None:
        self._commit()  # the previous shard is complete → publish it
        self._cur_final = self._shard_path(self._shard_idx)
        self._cur_tmp = _tmp_path(self._cur_final)
        self._fh = self._cur_tmp.open("w", encoding="utf-8")
        self.paths.append(str(self._cur_final))
        self._in_shard = 0

    def write(self, row: dict) -> None:
        if self._fh is None:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._roll()
        elif self._shard_size and self._in_shard >= self._shard_size:
            self._shard_idx += 1
            self._roll()
        elif self._in_shard:
            self._fh.write("\n")  # no leading newline on a shard's first row
        self._fh.write(json.dumps(row, ensure_ascii=False))
        self._in_shard += 1
        self.count += 1

    def close(self) -> None:
        self._commit()
        if not self.paths:  # never wrote a row → still emit an empty primary shard
            self._dir.mkdir(parents=True, exist_ok=True)
            p = self._shard_path(0)
            tmp = _tmp_path(p)
            tmp.write_text("", encoding="utf-8")
            os.replace(tmp, p)
            self.paths.append(str(p))

    def _abort(self) -> None:
        """Error exit: discard the in-flight (incomplete) shard; keep committed ones."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._cur_tmp.unlink(missing_ok=True)
            if self.paths and self.paths[-1] == str(self._cur_final):
                self.paths.pop()
            self._cur_final = self._cur_tmp = None

    def __enter__(self) -> "ShardedJsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self._abort()


def load_rows(
    source: str,
    *,
    split: str = "train",
    strict: bool = False,
    errors: "RowErrors | None" = None,
) -> list[dict]:
    """Load a dataset into a list of row dicts.

    ``source`` is a local file path (builder inferred from the extension) or an
    HF Hub dataset id (``org/name``). Null-padded columns are dropped so each
    row carries only the keys it actually had in the source.

    ``strict``/``errors`` control malformed-line handling for local ``.jsonl``
    (#1008 — see :func:`_iter_jsonl`); other builders parse inside ``datasets``
    and keep their own error behavior.
    """
    source = _normalize_source(source)
    p = Path(source)
    # Local .jsonl: read with the stdlib line reader, ~12x faster than routing
    # through the ``datasets`` Arrow builder (which infers a columnar schema and
    # materializes an Arrow table just to hand back Python dicts). Same rows,
    # same null-drop; nested messages/conversations preserved verbatim. #321.
    if p.exists() and p.suffix.lower() == ".jsonl":
        return list(_iter_jsonl(p, strict=strict, errors=errors))
    # s3 .jsonl takes the SAME verbatim reader (see _is_s3_jsonl for why Arrow
    # must not see it): fetched once via boto3, then read line-by-line.
    if _is_s3_jsonl(source):
        return list(_iter_jsonl(_fetch_s3_jsonl(source), strict=strict, errors=errors))
    if p.exists():
        builder = _BUILDER_BY_SUFFIX.get(p.suffix.lower(), "json")
        ds = load_dataset(builder, data_files=source, split=split)
    elif (rb := _remote_builder(source)) is not None:  # scheme:// URL (#913)
        ds = load_dataset(rb, data_files=source, split=split)
    else:
        ds = load_dataset(source, split=split)
    return [_drop_nulls(row) for row in ds]


def iter_rows(
    source: str,
    *,
    split: str = "train",
    strict: bool = False,
    errors: "RowErrors | None" = None,
) -> Iterator[dict]:
    """Stream a dataset row-by-row WITHOUT materializing it (issue #380).

    A local ``.jsonl`` file streams line-by-line via the stdlib JSON reader —
    O(1) memory and nested ``messages``/``conversations`` preserved exactly as
    written. Every other source (local ``.json``/``.csv``/``.parquet`` and HF
    Hub ids) streams through ``datasets``' ``IterableDataset``
    (``streaming=True``), which yields examples lazily off Arrow rather than
    building one giant list. Null-valued keys are dropped per row to match
    :func:`load_rows`.

    ``strict``/``errors`` control malformed-line handling for local ``.jsonl``
    (#1008 — see :func:`_iter_jsonl`).

    This is the memory-safe primitive single-pass operators should consume in
    place of :func:`load_rows`.
    """
    source = _normalize_source(source)
    p = Path(source)
    if p.exists() and p.suffix.lower() == ".jsonl":
        yield from _iter_jsonl(p, strict=strict, errors=errors)
        return
    # s3 .jsonl: verbatim reader, same rationale as load_rows (_is_s3_jsonl).
    if _is_s3_jsonl(source):
        yield from _iter_jsonl(_fetch_s3_jsonl(source), strict=strict, errors=errors)
        return
    if p.exists():
        builder = _BUILDER_BY_SUFFIX.get(p.suffix.lower(), "json")
        ds = load_dataset(builder, data_files=source, split=split, streaming=True)
    elif (rb := _remote_builder(source)) is not None:  # scheme:// URL (#913)
        ds = load_dataset(rb, data_files=source, split=split, streaming=True)
    else:
        ds = load_dataset(source, split=split, streaming=True)
    for row in ds:
        yield _drop_nulls(row)


def _iter_jsonl(
    path: Path, *, strict: bool = False, errors: "RowErrors | None" = None,
) -> Iterator[dict]:
    """Yield each row of a line-delimited JSON file, one line at a time. Blank
    lines are skipped; the file handle only ever holds a single line in memory.

    QUARANTINE (#1008): a malformed line (bad JSON, or valid JSON that isn't an
    object) is no longer fatal. By default it is routed to a side file
    ``<path>.quarantine.jsonl`` — one entry per bad line with the line number,
    the parse error, and a bounded raw prefix — and skipped; the stream keeps
    going. Never a silent drop: the side file always records it, and a caller
    that passes a :class:`RowErrors` sees the running count. ``strict=True``
    restores the old fail-fast (raise on the first bad line). The quarantine
    file is written via tmp + ``os.replace`` (#1007) when the stream finishes
    or the generator is closed, so it is never left truncated."""
    qfh = None
    qtmp: Path | None = None
    qfinal: Path | None = None
    try:
        with path.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise json.JSONDecodeError("line is not a JSON object", line, 0)
                except json.JSONDecodeError as e:
                    if strict:
                        raise
                    if qfh is None:
                        qfinal = path.with_name(f"{path.name}.quarantine.jsonl")
                        qtmp = _tmp_path(qfinal)
                        qfh = qtmp.open("w", encoding="utf-8")
                    # bounded raw prefix — enough to locate/diagnose, never the
                    # unbounded (possibly sensitive) full payload in a log-ish file
                    entry = {"line": lineno, "error": str(e), "raw": line[:500]}
                    qfh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    if errors is not None:
                        errors.count += 1
                        errors.quarantine_path = str(qfinal)
                    continue
                yield _drop_nulls(row)
    finally:
        if qfh is not None:
            qfh.close()
            os.replace(qtmp, qfinal)


def scan_frame(source: str) -> "pl.LazyFrame | None":
    """Return a polars ``LazyFrame`` over a local flat file for the COLUMNAR
    fast path (#381 token counts, #382 MinHash), or ``None`` when the columnar
    path isn't available.

    ``None`` is returned when polars (the ``[fast]`` extra) isn't installed, the
    source is an HF Hub id (not lazily scannable — use :func:`iter_rows`), or the
    file suffix isn't a flat columnar format. The frame is lazy: nothing is read
    until the caller ``.collect()``s (ideally ``streaming=True``), so building it
    costs no memory.
    """
    try:
        import polars as pl  # optional [fast] extra — absent in the base install
    except ImportError:
        return None
    source = _normalize_source(source)
    p = Path(source)
    if not p.exists():
        return None  # Hub ids aren't lazily scannable — use iter_rows/load_rows
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(source)
    if suffix == ".csv":
        return pl.scan_csv(source)
    if suffix == ".jsonl":
        return pl.scan_ndjson(source)
    return None


def dataset_info(source: str):
    """Cheaply read a dataset's :class:`DatasetInfo` (license, homepage,
    citation, …) without downloading/generating the data.

    Populated for Hub datasets (from the card); empty fields for a raw local
    JSONL. ``dataset/provenance`` harvests this so the standard metadata HF
    already carries isn't re-declared by hand.
    """
    source = _normalize_source(source)
    p = Path(source)
    if p.exists():
        builder = _BUILDER_BY_SUFFIX.get(p.suffix.lower(), "json")
        return load_dataset_builder(builder, data_files=source).info
    if (rb := _remote_builder(source)) is not None:  # scheme:// URL (#913)
        return load_dataset_builder(rb, data_files=source).info
    return load_dataset_builder(source).info
