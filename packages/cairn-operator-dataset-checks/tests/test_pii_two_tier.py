"""ADR-0080 P1 (#1029 / rejected design R4) — two-tier PII: full-corpus secrets.

A sampled PII scan (1000 rows) misses a rare leaked key ~90% of the time — a
governance hole for a gate whose pitch is "no secrets before a GPU spins up". The
SECRET tier now scans EVERY row (cheap regex) and merges into `counts` in the
gate's LOWERCASE blocking vocabulary, so a secret is caught 100% of the time and
actually blocks (cascade used to emit UPPERCASE → the gate warned instead).
"""
from __future__ import annotations

import asyncio
import json

from operator_dataset_checks import _pii
from operator_dataset_checks.operators.pii_scan import PiiScanOp
from operator_dataset_checks.schemas.pii_scan import PiiInput


class _Ctx:
    pass


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def _corpus(tmp_path, n=5000, n_secrets=3):
    # n_secrets rows carry a leaked AWS key (AKIA + 16 upper-alnum); rest is clean.
    rows = [{"text": f"ticket {i}: everything nominal, no issues here."} for i in range(n)]
    for j in range(n_secrets):
        idx = (j + 1) * (n // (n_secrets + 1))
        rows[idx]["text"] = f"urgent: leaked key AKIA{j:016X} found in the logs"
    return _write(tmp_path / "corpus.jsonl", rows)


def test_full_corpus_secret_count_is_exact_under_sampling(tmp_path):
    src = _corpus(tmp_path, n=5000, n_secrets=3)
    out = asyncio.run(PiiScanOp().run(
        PiiInput(path=src, backend="regex", sample_size=1000), _Ctx()))
    # the NER/density tier is over a 1000-row sample, but the secret tier is FULL:
    # exactly 3 keys, regardless of what the sample happened to draw.
    assert out.counts.get("aws_access_key") == 3, out.counts
    assert out.secret_rows == 3
    assert out.sampled is True  # density really was sampled


def test_secret_kind_is_gate_blocking_vocab(tmp_path):
    """The merged secret kind must be one the gate blocks on (lowercase cairn),
    not the cascade's UPPERCASE presidio type that slipped past the block list."""
    from operator_dataset_checks.operators.gate import _HARD_SECRET_PII

    src = _corpus(tmp_path, n=2000, n_secrets=2)
    out = asyncio.run(PiiScanOp().run(
        PiiInput(path=src, backend="regex", sample_size=500), _Ctx()))
    secret_keys = [k for k in out.counts if k in _HARD_SECRET_PII]
    assert secret_keys, f"no gate-blocking secret kind in {out.counts}"


def test_merge_secret_counts_strips_all_secret_spellings():
    # main scan (cascade) reported an UPPERCASE secret + a real NER kind; the merge
    # drops every secret spelling and re-adds the authoritative lowercase count.
    main = {"AWS_ACCESS_KEY": 1, "EMAIL_ADDRESS": 12, "PERSON": 40}
    merged = _pii.merge_secret_counts(main, {"aws_access_key": 3})
    assert merged == {"EMAIL_ADDRESS": 12, "PERSON": 40, "aws_access_key": 3}


def test_secrets_full_scan_off_disables_the_tier(tmp_path):
    src = _corpus(tmp_path, n=1500, n_secrets=2)
    out = asyncio.run(PiiScanOp().run(
        PiiInput(path=src, backend="regex", sample_size=500, secrets_full_scan=False), _Ctx()))
    assert out.secret_rows is None


def test_gate_blocks_on_full_corpus_secret(tmp_path):
    """End-to-end: the two-tier scan output makes the deterministic gate FAIL,
    which a sampled-only scan would have passed ~90% of the time."""
    from operator_dataset_checks.operators.gate import GateOp
    from operator_dataset_checks.schemas.gate import GateInput

    src = _corpus(tmp_path, n=5000, n_secrets=3)
    scan = asyncio.run(PiiScanOp().run(
        PiiInput(path=src, backend="regex", sample_size=1000), _Ctx()))
    verdict = asyncio.run(GateOp().run(
        GateInput(pii_counts=scan.counts, dataset_uri=src), _Ctx()))
    assert verdict.verdict == "fail", verdict
    assert any("secret" in b.lower() for b in verdict.blocking), verdict.blocking
