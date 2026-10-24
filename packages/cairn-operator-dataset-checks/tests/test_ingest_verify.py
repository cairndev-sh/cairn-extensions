"""#1466 — ingest.normalize verifies registry-pinned hashes (fragment form)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from operator_dataset_checks.dataflow_stages import ingest_normalize


def _write(p: Path, rows: list[dict]) -> str:
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_matching_hash_passes(tmp_path: Path):
    src = tmp_path / "data.jsonl"
    digest = _write(src, [{"a": 1}, {"a": 2}])
    out = ingest_normalize(tmp_path, {"path": f"{src}#sha256={digest}"}, [])
    assert out["rows"] == 2
    assert out["source"] == str(src)  # fragment stripped from lineage origin


def test_hash_mismatch_fails_closed(tmp_path: Path):
    src = tmp_path / "data.jsonl"
    _write(src, [{"a": 1}])
    with pytest.raises(RuntimeError, match="dataset verification FAILED"):
        ingest_normalize(tmp_path, {"path": f"{src}#sha256={'0' * 64}"}, [])


def test_no_fragment_unchanged(tmp_path: Path):
    src = tmp_path / "data.jsonl"
    _write(src, [{"a": 1}])
    out = ingest_normalize(tmp_path, {"path": str(src)}, [])
    assert out["rows"] == 1


def test_export_echoes_register_request(tmp_path: Path, monkeypatch):
    """#1466 — `register:` config rides the export stats to the control plane."""
    from operator_dataset_checks import dataflow_stages as D

    (tmp_path / "current.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    monkeypatch.setenv("OBS_ARTIFACT_S3_BUCKET", "bkt")
    monkeypatch.setattr(D, "_s3_upload", lambda src, uri: uri)
    out = D.export_jsonl(tmp_path, {"name": "x.jsonl", "store": "s3", "register": "facts-train"}, [])
    assert out["register_as"] == "facts-train"
    out2 = D.export_jsonl(tmp_path, {"name": "x.jsonl", "store": "s3"}, [])
    assert "register_as" not in out2
    # the LOCAL store refuses the request (it once dropped it silently, caught 2026-08-20; since
    # cairn-extensions#6 a local export is never registered: test_export_s3.py)
    with pytest.raises(ValueError, match="`register` needs `store: s3`"):
        D.export_jsonl(tmp_path, {"name": "x.jsonl", "store": "local", "register": "facts-train"}, [])


def test_malformed_lines_and_an_empty_export_reach_the_intake_verdict_as_invalid_format(tmp_path: Path):
    """#2396 (ADR-0102 D1) — `cairn/core/intake.py` reads `ingest.normalize.quarantined_rows`
    and `export.jsonl.rows`; the kernels now emit both, so a corpus with any malformed line, or an
    export that ends empty, is `invalid_format` (which the launch door refuses), and a clean one
    is not."""
    from cairn.core.intake import INVALID_FORMAT, verdict_from_flow_stats
    from operator_dataset_checks import dataflow_stages as D

    def _verdict(text: str, work: Path):
        work.mkdir()
        src = work / "data.jsonl"
        src.write_text(text, encoding="utf-8")
        ingest = D.ingest_normalize(work, {"path": str(src)}, [])
        export = D.export_jsonl(work, {"name": "x.jsonl", "store": "local"}, [])
        flat = {**{f"ingest.normalize.{k}": v for k, v in ingest.items()},
                **{f"export.jsonl.{k}": v for k, v in export.items()}}
        return ingest, export, verdict_from_flow_stats(flat)

    ingest, export, v = _verdict('{"a": 1}\nnot json\n[1]\n\n{"a": 2}\n', tmp_path / "malformed")
    assert (ingest["rows"], ingest["quarantined_rows"], export["rows"]) == (2, 2, 2)
    assert v.verdict == INVALID_FORMAT and v.report["quarantined_rows"] == 2 and v.report["rows_out"] == 2
    ingest, export, v = _verdict("", tmp_path / "empty")
    assert (ingest["quarantined_rows"], export["rows"]) == (0, 0) and v.verdict == INVALID_FORMAT
    ingest, export, v = _verdict('{"a": 1}\n{"a": 2}', tmp_path / "clean")   # no trailing newline
    assert (ingest["quarantined_rows"], export["rows"]) == (0, 2) and v.verdict != INVALID_FORMAT


def test_the_export_row_count_comes_from_the_hash_pass(tmp_path: Path):
    """The count is taken in the same pass as the digest — no second read of the export."""
    from operator_dataset_checks import dataflow_stages as D

    f = tmp_path / "x.jsonl"
    for text, lines in (("", 0), ("a\n", 1), ("a\nb", 2), ("a\nb\n", 2)):
        f.write_text(text)
        digest, size, rows = D._sha256_lines(f)
        assert (rows, size) == (lines, len(text)) and (digest, size) == D._sha256_file(f)
