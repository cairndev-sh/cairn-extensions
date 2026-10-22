"""Secret regex covers MODERN credential formats (pre-deploy audit fix).

The full-corpus secret tier is the gate's absolute blocker ("no secrets before a
GPU spins up"). The original patterns only knew legacy formats — a current-gen
OpenAI key (`sk-proj-…`), a fine-grained GitHub PAT (`github_pat_…`), or a Slack/
Stripe token slipped straight through. These tests pin the widened coverage AND
guard against false positives on benign text.
"""
from __future__ import annotations

import asyncio
import json

from operator_dataset_checks import _pii
from operator_dataset_checks.operators.gate import GateOp, _HARD_SECRET_PII
from operator_dataset_checks.operators.pii_scan import PiiScanOp
from operator_dataset_checks.schemas.gate import GateInput
from operator_dataset_checks.schemas.pii_scan import PiiInput


class _Ctx:
    pass


# (label, text, expected cairn secret kind)
_MODERN = [
    ("aws AKIA", "deploy AKIAIOSFODNN7EXAMPLE now", "aws_access_key"),
    ("aws ASIA temp", "creds ASIAY34FZKBOKMKB2N7Q here", "aws_access_key"),
    ("openai legacy", "token sk-abcdefghijklmnopqrstuvwx1234", "openai_key"),
    ("openai sk-proj", "use sk-proj-abcdef1234567890ABCDEFxyz today", "openai_key"),
    ("openai sk-svcacct", "t sk-svcacct-abcd1234EFGH5678ijkl90 x", "openai_key"),
    ("github classic", "pat ghp_" + "a" * 36, "github_token"),
    ("github fine-grained", "pat github_pat_11ABCDEFG0abcdefghij_" + "x" * 40, "github_token"),
    ("github oauth gho_", "tok gho_" + "b" * 36, "github_token"),
    ("slack xoxb", "tok xoxb-123456789012-abcdefghijkl", "slack_token"),
    ("stripe sk_live", "key sk_live_" + "a" * 24, "stripe_key"),
]

_BENIGN = [
    "Please deploy the service to Berlin next week.",
    "commit a1b2c3d4e5f6 and open a PR",
    "upgrade to v1.2.3 build 4567",
    "see [docs](https://x.io/guide) for the skill list",
]


def test_modern_formats_detected_in_blocking_vocab():
    for label, text, kind in _MODERN:
        hits = _pii.secret_counts(text)
        assert hits.get(kind) == 1, f"{label}: {hits}"
        assert kind in _HARD_SECRET_PII, f"{label}: {kind} not a gate blocker"


def test_no_false_positives_on_benign_text():
    for text in _BENIGN:
        assert _pii.secret_counts(text) == {}, text


def test_new_secret_kinds_block_the_gate():
    assert "slack_token" in _HARD_SECRET_PII
    assert "stripe_key" in _HARD_SECRET_PII


def test_modern_key_fails_the_gate_end_to_end(tmp_path):
    # a corpus whose only defect is one current-gen OpenAI key must FAIL — the
    # exact case the old pattern passed.
    rows = [{"text": f"clean row {i}"} for i in range(2000)]
    rows[999]["text"] = "deploy with sk-proj-abcdef1234567890ABCDEFghij please"
    src = tmp_path / "c.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    scan = asyncio.run(PiiScanOp().run(
        PiiInput(path=str(src), backend="regex", sample_size=500), _Ctx()))
    assert scan.counts.get("openai_key") == 1
    assert scan.secret_rows == 1
    verdict = asyncio.run(GateOp().run(
        GateInput(pii_counts=scan.counts, dataset_uri=str(src)), _Ctx()))
    assert verdict.verdict == "fail"
    assert any("secret" in b.lower() for b in verdict.blocking)


def test_merge_keeps_new_secret_kinds_lowercase(tmp_path):
    # cascade would emit SLACK_TOKEN/STRIPE_KEY (uppercase); the merge must land
    # them in the gate's lowercase blocking vocab.
    merged = _pii.merge_secret_counts(
        {"SLACK_TOKEN": 1, "PERSON": 5}, {"slack_token": 2, "stripe_key": 1})
    assert merged == {"PERSON": 5, "slack_token": 2, "stripe_key": 1}
