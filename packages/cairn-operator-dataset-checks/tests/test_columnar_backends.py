"""#1436 — DuckDB backends for decontam.apply / pii.redact / scan.all must
produce the SAME numbers as the Python paths on the same rows, and the regex
tier must be RE2-safe over JSON text (never eat an escape).

The 300k real-row campaign corpus made the Python paths the wall (PII ~3–11 h,
decontam ~10 min); these pin the accelerated paths to the reference ones.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from operator_dataset_checks import _duck, _pii  # noqa: E402
from operator_dataset_checks import dataflow_stages as S  # noqa: E402


def _row(user: str, assistant: str = "ok") -> dict:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


ROWS = [
    _row("plain question about the weather today"),
    _row("mail me at ana.marsh@example-corp.com or call (415) 555-2671 tomorrow"),
    _row("SSN 412-55-1234 and card 4111 1111 1111 9876 on file, ip 10.0.0.1"),
    _row("export GH_TOKEN=ghp_" + "a" * 36 + " and sk-" + "b" * 24 + " in the shell"),
    _row("flask app:\n@app.route(\"/products\", methods=[\"GET\"])\ndef products(): pass"),
    _row("windows path C:\\temp\\rk_live_" + "c" * 20 + " leaked stripe key"),
    _row("bare number 1234567890123456 must survive, version 1.2.3.4 too"),
    _row("plain question about the weather today"),   # exact dupe of row 0
    _row("plain  question about the   weather today"),  # normalized dupe of row 0
    _row("held out probe row copied into train"),
]
PROBE = [_row("held out probe row copied into train"), _row("another probe row")]


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    src = tmp_path / "rows.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    (tmp_path / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in PROBE), encoding="utf-8")
    wd = tmp_path / "wd"
    wd.mkdir()
    S.ingest_normalize(wd, {"path": str(src)}, [])
    return wd


def _lines(wd: Path) -> list[dict]:
    return [json.loads(l) for l in S._current(wd).read_text().splitlines()]


# ── regex tier is RE2-safe and JSON-safe ────────────────────────────────────

def test_regex_kinds_compile_in_re2_and_never_break_json():
    con = duckdb.connect()
    for k, pat in _pii.REGEX_KINDS:
        con.execute("SELECT regexp_replace('x', ?, 'y', 'g')", [_duck.guarded(pat)])
    # every ROW's redacted JSON text must still parse — including the Flask
    # `\n@app.route` row and the `\rk_live_` row that broke the naive version.
    for row in ROWS:
        text = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        red = text
        for k, pat in _pii.REGEX_KINDS:
            red = re.sub(_duck.guarded(pat), lambda m, k=k: m.group(1) + f"<{k.upper()}>", red)
        json.loads(red)


def test_new_regex_kinds_require_separators():
    """A bare 16-digit / 10-digit number is NOT a card / phone / SSN — the
    patterns also run over raw JSON text, where an unquoted token would corrupt
    the row."""
    red, counts = _pii._regex_anonymize("num 1234567890123456 tel 4155552671 ssn 412551234", None)
    assert counts == {} and "1234567890123456" in red
    red, counts = _pii._regex_anonymize(
        "SSN 412-55-1234, card 4111 1111 1111 9876, phone (415) 555-2671, +44 7700 900123", None)
    assert counts == {"us_ssn": 1, "credit_card": 1, "phone": 2}
    assert "<US_SSN>" in red and "<CREDIT_CARD>" in red and red.count("<PHONE>") == 2


# ── pii.redact: duckdb ≡ regex ──────────────────────────────────────────────

def test_pii_redact_duckdb_matches_regex_backend(workdir: Path, tmp_path: Path):
    import shutil
    keep = tmp_path / "keep.jsonl"
    shutil.copy(S._current(workdir), keep)
    r_py = S.pii_redact(workdir, {"backend": "regex"}, [])
    rows_py = _lines(workdir)
    shutil.copy(keep, S._current(workdir))
    r_db = S.pii_redact(workdir, {"backend": "duckdb"}, [])
    rows_db = _lines(workdir)
    assert r_db["pii_backend"] == "duckdb" and r_py["pii_backend"] == "regex"
    assert r_db["redacted_rows"] == r_py["redacted_rows"] == 5
    assert rows_db == rows_py
    # the Flask row is untouched (no email there once escapes are respected)
    assert rows_db[4] == ROWS[4]
    # the bare number survives; the secrets, SSN, card, phone, email, ip are gone
    joined = json.dumps(rows_db)
    assert "1234567890123456" in joined
    for tok in ("<EMAIL>", "<PHONE>", "<US_SSN>", "<CREDIT_CARD>", "<IPV4>",
                "<GITHUB_TOKEN>", "<OPENAI_KEY>", "<STRIPE_KEY>"):
        assert tok in joined, tok
    assert r_db["hits_stripe_key"] == 1 and r_db["hits_phone"] == 1


def test_duckdb_stage_stats_cross_the_dataflow_boundary(workdir: Path, tmp_path: Path):
    """Every DuckDB stage's stats dict must be accepted by ArtifactDescriptor —
    the 300k live run died at this boundary on a nested `counts` dict."""
    from cairn.core.dataflow.contract import ArtifactDescriptor
    probe = str(tmp_path / "probe.jsonl")
    for stats in (
        S.decontam_apply(workdir, {"eval_path": probe, "backend": "duckdb"}, []),
        S.pii_redact(workdir, {"backend": "duckdb"}, []),
        S.scan_all(workdir, {"engine": "duckdb", "backend": "duckdb"}, []),
    ):
        ArtifactDescriptor(uri="file:///x", kind="dataset", content_hash="0" * 64, stats=stats)


# ── decontam.apply: duckdb ≡ python (exact) ─────────────────────────────────

def test_decontam_duckdb_matches_python(workdir: Path, tmp_path: Path):
    import shutil
    probe = str(tmp_path / "probe.jsonl")
    keep = tmp_path / "keep.jsonl"
    shutil.copy(S._current(workdir), keep)
    r_py = S.decontam_apply(workdir, {"eval_path": probe}, [])
    rows_py = _lines(workdir)
    shutil.copy(keep, S._current(workdir))
    r_db = S.decontam_apply(workdir, {"eval_path": probe, "backend": "duckdb"}, [])
    rows_db = _lines(workdir)
    assert r_py["backend"] == "python" and r_db["backend"] == "duckdb"
    assert r_py["dropped"] == r_db["dropped"] == 1
    assert rows_py == rows_db and len(rows_db) == len(ROWS) - 1


def test_decontam_duckdb_falls_back_to_python_for_ngram(workdir: Path, tmp_path: Path):
    r = S.decontam_apply(workdir, {"eval_path": str(tmp_path / "probe.jsonl"),
                                   "backend": "duckdb", "method": "ngram"}, [])
    assert r["backend"] == "python" and r["method"] == "ngram"


# ── scan.all: engine=duckdb corpus tiers ≡ python ───────────────────────────

def test_scan_all_duckdb_engine_matches_python_corpus_tiers(workdir: Path):
    py = S.scan_all(workdir, {"backend": "regex", "n_process": 1}, [])
    db = S.scan_all(workdir, {"engine": "duckdb", "backend": "duckdb"}, [])
    for k in ("scanned", "dedup_total", "dedup_exact", "secret_rows", "quarantined_rows"):
        assert py[k] == db[k], k
    assert db["scanned"] == len(ROWS) and db["dedup_exact"] == 2   # exact + normalized dupe
    assert db["secret_rows"] == 2                                   # ghp_/sk- row + rk_live_ row
    assert db["pii_backend"] == "regex"                            # sample density = python regex tier
    # full corpus fits the sample → sample-tier numbers agree too
    assert py["pii_rows"] == db["pii_rows"] and py["format_detected"] == db["format_detected"]


def test_scan_all_duckdb_engine_counts_malformed_lines_as_quarantined(tmp_path: Path):
    src = tmp_path / "rows.jsonl"
    src.write_text(json.dumps(ROWS[0]) + "\n{not json\n" + json.dumps(ROWS[1]) + "\n", encoding="utf-8")
    wd = tmp_path / "wd"; wd.mkdir()
    S.ingest_normalize(wd, {"path": str(src)}, [])
    r = S.scan_all(wd, {"engine": "duckdb", "backend": "duckdb", "path": str(src)}, [])
    assert r["scanned"] == 2 and r["quarantined_rows"] == 1


def test_duckdb_backends_fail_loud_without_duckdb(monkeypatch, workdir: Path, tmp_path: Path):
    import builtins
    real = builtins.__import__

    def no_duck(name, *a, **k):
        if name == "duckdb":
            raise ImportError("no duckdb")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_duck)
    with pytest.raises(RuntimeError, match="duckdb"):
        S.pii_redact(workdir, {"backend": "duckdb"}, [])
    with pytest.raises(RuntimeError, match="duckdb"):
        S.decontam_apply(workdir, {"eval_path": str(tmp_path / "probe.jsonl"), "backend": "duckdb"}, [])


# ── #1456: filter.rows — one handoff artifact, many splits ─────────────────

def test_filter_rows_keeps_only_matching_split(tmp_path):
    import json
    from operator_dataset_checks import dataflow_stages as S
    rows = [{"split": "train", "input": "q1", "output": "a1"}, {"split": "eval", "question": "q2"},
            {"split": "train", "input": "q3", "output": "a3"}, {"split": "control", "question": "q4"}]
    src = tmp_path / "all.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    wd = tmp_path / "wd"; wd.mkdir()
    S.ingest_normalize(wd, {"path": str(src)}, [])
    st = S.filter_rows(wd, {"field": "split", "value": "train"}, [])
    kept = [json.loads(l) for l in S._current(wd).read_text().splitlines()]
    assert st == {"rows": 2, "total": 4, "dropped": 2, "field": "split", "values": "train"}
    assert [r["input"] for r in kept] == ["q1", "q3"]
    # values list form
    S.ingest_normalize(wd, {"path": str(src)}, [])
    st = S.filter_rows(wd, {"field": "split", "values": ["eval", "control"]}, [])
    assert st["rows"] == 2 and st["values"] == "eval,control"
    assert S.filter_rows.risk_class == "external_write"
