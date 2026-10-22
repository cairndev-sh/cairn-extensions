"""ADR-0080 P1 (#1029) — dataset/scan_all: the fused single-pass check.

The fusion's contract is PARITY: one streaming read must emit the same
state.findings the four re-streaming operators (format · dedup · profile ·
pii_scan) did, so the deterministic gate is unchanged. These tests pin that
parity against the standalone operators, then the two ADR acceptance criteria:
full-corpus secrets survive sampling (R4), and the fused output drives the gate
to the same verdict.
"""
from __future__ import annotations

import asyncio
import json

from operator_dataset_checks.operators.dedup import DedupOp
from operator_dataset_checks.operators.format import FormatOp
from operator_dataset_checks.operators.gate import GateOp, _HARD_SECRET_PII
from operator_dataset_checks.operators.pii_scan import PiiScanOp
from operator_dataset_checks.operators.profile import ProfileOp
from operator_dataset_checks.operators.scan_all import ScanAllOp
from operator_dataset_checks.schemas.dedup import DedupInput
from operator_dataset_checks.schemas.format import FormatInput
from operator_dataset_checks.schemas.gate import GateInput
from operator_dataset_checks.schemas.pii_scan import PiiInput
from operator_dataset_checks.schemas.profile import ProfileInput
from operator_dataset_checks.schemas.scan_all import ScanAllInput


class _Ctx:
    pass


def _run(coro):
    return asyncio.run(coro)


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def _chat(user, assistant):
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


def _corpus(tmp_path, n=300, dups=20, refusals=15, shorts=10, secrets=3):
    """A mixed but SMALL (< default 1000 sample) corpus so scan_all's reservoir
    returns every row — apples-to-apples with the full-scan standalone ops."""
    rows = [_chat(f"question number {i} about the topic at hand",
                  f"Here is a thorough answer to question {i} with real content.")
            for i in range(n)]
    for i in range(dups):                       # exact duplicates of row 0
        rows[i] = _chat("question number 0 about the topic at hand",
                        "Here is a thorough answer to question 0 with real content.")
    for i in range(dups, dups + refusals):      # refusals
        rows[i] = _chat("do the thing", "I'm sorry, I cannot help with that.")
    for i in range(dups + refusals, dups + refusals + shorts):  # short responses
        rows[i] = _chat("hi", "ok")
    for j in range(secrets):                    # leaked AWS keys, spread out
        rows[n - 1 - j] = _chat("deploy it", f"use key AKIA{j:016X} to authenticate")
    return _write(tmp_path / "corpus.jsonl", rows)


# ── parity against the standalone operators ───────────────────────────────────

def test_scan_all_matches_standalone_operators(tmp_path):
    src = _corpus(tmp_path)
    scan = _run(ScanAllOp().run(ScanAllInput(path=src, backend="regex"), _Ctx()))

    fmt = _run(FormatOp().run(FormatInput(path=src), _Ctx()))
    ded = _run(DedupOp().run(DedupInput(path=src), _Ctx()))
    prof = _run(ProfileOp().run(ProfileInput(path=src), _Ctx()))

    # format
    assert scan.format_detected == fmt.detected_format
    assert scan.format_conformant_pct == fmt.conformant_pct
    assert scan.format_trainer_compat == fmt.trainer_compatibility
    # dedup (full corpus)
    assert scan.dedup_total == ded.total
    assert scan.dedup_exact == ded.exact_duplicates
    assert scan.dedup_pct == ded.duplicate_pct
    assert scan.dedup_effective_size == ded.effective_size
    # profile
    assert scan.profile_total == prof.total
    assert scan.profile_refusal_pct == prof.refusal_pct
    assert scan.profile_short == prof.short_responses
    assert scan.profile_lengths == prof.response_length_chars


def test_scan_all_pii_secret_counts_match_standalone(tmp_path):
    src = _corpus(tmp_path)
    scan = _run(ScanAllOp().run(ScanAllInput(path=src, backend="regex"), _Ctx()))
    pii = _run(PiiScanOp().run(PiiInput(path=src, backend="regex", sample_size=None), _Ctx()))
    # both scan every row here (small corpus) → identical aws key counts.
    assert scan.pii_counts.get("aws_access_key") == pii.counts.get("aws_access_key") == 3
    assert scan.secret_rows == 3


# ── ADR acceptance: full-corpus secrets survive sampling (R4) ─────────────────

def test_planted_secrets_survive_sampling(tmp_path):
    # 5000 clean rows, 3 leaked keys — a 1000-row sample would miss them ~90% of
    # the time; the full-corpus secret tier catches all three, every time.
    rows = [_chat(f"q{i}", f"clean answer {i}") for i in range(5000)]
    for j in range(3):
        rows[(j + 1) * 1200] = _chat("deploy", f"key AKIA{j:016X} here")
    src = _write(tmp_path / "big.jsonl", rows)
    scan = _run(ScanAllOp().run(ScanAllInput(path=src, backend="regex", sample_size=1000), _Ctx()))
    assert scan.sampled is True
    assert scan.scanned == 5000
    assert scan.pii_counts.get("aws_access_key") == 3
    assert scan.secret_rows == 3
    # the secret kind is in the gate's LOWERCASE blocking vocab (P1a).
    assert any(k in _HARD_SECRET_PII for k in scan.pii_counts)


def test_secrets_full_scan_off_disables_tier(tmp_path):
    src = _corpus(tmp_path)
    scan = _run(ScanAllOp().run(
        ScanAllInput(path=src, backend="regex", secrets_full_scan=False), _Ctx()))
    assert scan.secret_rows is None


# ── guard early-exit ──────────────────────────────────────────────────────────

def test_guard_skips_ner_but_still_blocks_secrets(tmp_path):
    # every row is an unrecognized format → 0% conformant → NER skipped. A planted
    # key (in a free-text leaf) must STILL be caught by the full-corpus secret tier.
    rows = [{"data": f"row {i} content that is not any known training format"} for i in range(200)]
    rows[100] = {"data": "deploy with key AKIA0000000000000042 please"}
    src = _write(tmp_path / "broken.jsonl", rows)
    scan = _run(ScanAllOp().run(ScanAllInput(path=src, backend="regex"), _Ctx()))
    assert scan.ner_skipped is True
    assert scan.format_conformant_pct < 40.0
    # full-corpus secret scan still ran → key present in blocking vocab.
    assert scan.pii_counts.get("aws_access_key") == 1
    assert scan.secret_rows == 1


# ── end-to-end: the fused output drives the gate to the same verdict ──────────

def test_gate_verdict_parity_scan_all_vs_standalone(tmp_path):
    src = _corpus(tmp_path)
    scan = _run(ScanAllOp().run(ScanAllInput(path=src, backend="regex"), _Ctx()))
    fmt = _run(FormatOp().run(FormatInput(path=src), _Ctx()))
    ded = _run(DedupOp().run(DedupInput(path=src), _Ctx()))
    prof = _run(ProfileOp().run(ProfileInput(path=src), _Ctx()))
    pii = _run(PiiScanOp().run(PiiInput(path=src, backend="regex", sample_size=None), _Ctx()))

    fused = _run(GateOp().run(GateInput(
        format_conformant_pct=scan.format_conformant_pct,
        format_trainer_compat=scan.format_trainer_compat,
        dedup_pct=scan.dedup_pct, dedup_effective_size=scan.dedup_effective_size,
        pii_counts=scan.pii_counts, pii_rows=scan.pii_rows,
        profile_refusal_pct=scan.profile_refusal_pct,
        profile_short=scan.profile_short, profile_total=scan.profile_total,
        dataset_uri=src,
    ), _Ctx()))
    legacy = _run(GateOp().run(GateInput(
        format_conformant_pct=fmt.conformant_pct,
        format_trainer_compat=fmt.trainer_compatibility,
        dedup_pct=ded.duplicate_pct, dedup_effective_size=ded.effective_size,
        pii_counts=pii.counts, pii_rows=pii.rows_with_pii,
        profile_refusal_pct=prof.refusal_pct,
        profile_short=prof.short_responses, profile_total=prof.total,
        dataset_uri=src,
    ), _Ctx()))
    assert fused.verdict == legacy.verdict
    assert set(fused.blocking) == set(legacy.blocking)


def test_full_corpus_statistical_scan_when_sample_none(tmp_path):
    src = _corpus(tmp_path, n=300)
    scan = _run(ScanAllOp().run(ScanAllInput(path=src, backend="regex", sample_size=None), _Ctx()))
    assert scan.sampled is False
    assert scan.sample_size is None
    assert scan.profile_total == 300      # every row profiled, not a sample
    assert scan.scanned == 300
