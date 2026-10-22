"""DuckDB helpers shared by the columnar stage/operator backends (#1436).

The dataset_checks kernels are Python row streams by default; the DuckDB
backends are OPT-IN (``backend: duckdb`` / ``engine: duckdb``) and fail LOUDLY
when the package is absent — an opted-in accelerant must never degrade
silently (the ``dedup.exact backend=duckdb`` contract). Everything here works
on the row's JSON text as DuckDB serializes it (``to_json(json)``: compact
separators, raw UTF-8, keys in file order).

Two facts every caller relies on:

* **row identity** = ``md5`` of the whitespace-collapsed JSON text (lowercased
  for dedup). Order-preserving where Python's ``canonical_hash`` sorts keys —
  identical for any one serializer, which every corpus this pipeline sees is.
* **regex over JSON text** must never eat an escape: a value's newline is the
  two characters ``\\n``, so ``\\n@app.route(...)`` reads as the email
  ``n@app.route`` and an unguarded replace leaves ``\\<EMAIL>`` — invalid JSON
  (caught live on a real smoltalk row). RE2 has no lookbehind, so every
  pattern is guarded by capturing the preceding non-backslash character and
  putting it back (``\\1``). ``to_json`` emits raw UTF-8, so ``\\uXXXX`` never
  appears in this text; the only escapes are ``\\" \\\\ \\/ \\b \\f \\n \\r \\t``.
"""
from __future__ import annotations

from collections.abc import Sequence

READ_JSONL = "read_json_auto(?, format='newline_delimited', records=false)"


def connect():
    """A fresh in-memory DuckDB connection, or a LOUD RuntimeError."""
    try:
        import duckdb  # noqa: PLC0415 - optional dependency: the [duckdb] extra
    except ImportError as e:
        raise RuntimeError(
            "backend=duckdb needs the duckdb package: pip install duckdb") from e
    return duckdb.connect()


def sql_str(s: str) -> str:
    """A SQL single-quoted literal. DuckDB string literals are NOT
    escape-processed, so patterns keep their backslashes verbatim; only the
    quote needs doubling."""
    return "'" + s.replace("'", "''") + "'"


def key_expr(normalize: bool, *, lower: bool = False) -> str:
    """The row-identity expression (see module doc)."""
    text = "to_json(json)::VARCHAR"
    if normalize:
        text = f"regexp_replace({text}, '\\s+', ' ', 'g')"
        if lower:
            text = f"lower({text})"
    return f"md5({text})"


def guarded(pattern: str) -> str:
    """``pattern`` wrapped so a match can never start on the letter of a JSON
    escape (``\\n@app`` must not read as ``n@app``): the preceding text must be
    start-of-string, a non-backslash, or ``\\\\`` — an ESCAPED backslash, i.e. a
    real backslash in the value, after which the next character is real too
    (``C:\\\\temp\\\\rk_live_…`` keeps its Stripe key). Group 1 is that prefix,
    restored by ``\\1``. Residual blind spot: a literal backslash immediately
    followed by an escape (``\\\\\\n@…``) — three backslashes — is read as allowed."""
    return "(^|[^\\\\]|\\\\\\\\)(?:" + pattern + ")"


def redact_expr(kinds: Sequence[tuple[str, str]], text: str = "to_json(json)::VARCHAR") -> str:
    """Chained ``regexp_replace`` — one per (kind, pattern), in order — that
    rewrites every match to ``<KIND>`` exactly like the Python regex tier."""
    expr = text
    for k, pat in kinds:
        expr = f"regexp_replace({expr}, {sql_str(guarded(pat))}, '\\1<{k.upper()}>', 'g')"
    return expr


def hit_exprs(kinds: Sequence[tuple[str, str]], text: str = "to_json(json)::VARCHAR") -> str:
    """Per-kind match counts (``sum(len(regexp_extract_all(...)))``) as select-list
    entries aliased by kind — the aggregate half of a redact/scan."""
    return ", ".join(
        f'sum(len(regexp_extract_all({text}, {sql_str(guarded(pat))}))) AS "{k}"'
        for k, pat in kinds)


def any_hit_expr(kinds: Sequence[tuple[str, str]], text: str = "to_json(json)::VARCHAR") -> str:
    """Boolean: the row matches at least one of ``kinds`` (secret_rows)."""
    return " OR ".join(f"regexp_matches({text}, {sql_str(guarded(pat))})" for k, pat in kinds) or "false"
