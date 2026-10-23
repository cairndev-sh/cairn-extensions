"""Run-tunable PII config (ADR-0079): presets + NER/model/n_process knobs.

The pure resolution (resolve_pii_settings, n_process clamp) is hermetic; the
NER-off *behaviour* (regex fires, PERSON doesn't) is guarded on presidio.
"""
from __future__ import annotations

import os
from unittest import mock

import pytest

from operator_dataset_checks import _pii


# ── preset + override resolution (pure) ───────────────────────────────
def test_default_preset_is_governed_sm():
    st = _pii.resolve_pii_settings()
    assert st.model == "sm"                   # governed → sm NER (short name)
    assert _pii._resolve_model_name(st.model) == "en_core_web_sm"
    assert st.analyze_chunk == 2048
    assert st.n_process >= 1


def test_logs_preset_disables_ner():
    st = _pii.resolve_pii_settings(preset="logs")
    assert st.model == "blank"               # NER off → tokenizer-only
    assert st.n_process == 1
    assert st.analyze_chunk == 5000


def test_ner_false_override_forces_blank():
    st = _pii.resolve_pii_settings(preset="governed", ner=False)
    assert st.model == "blank"


def test_model_override_when_ner_on():
    st = _pii.resolve_pii_settings(model="lg")
    assert st.model == "lg"
    assert _pii._resolve_model_name(st.model) == "en_core_web_lg"


def test_explicit_knobs_override_preset():
    st = _pii.resolve_pii_settings(preset="logs", analyze_chunk=999)
    assert st.analyze_chunk == 999


# ── n_process clamp: cpu_count is the ceiling (experiment finding) ─────
def test_n_process_clamped_to_cpu_count():
    with mock.patch("os.cpu_count", return_value=2):
        assert _pii._pii_n_process(2) == 2       # 2 cores allow 2 (the +40% win)
        assert _pii._pii_n_process(8) == 2       # oversubscription capped
        assert _pii._pii_n_process(0) == 2       # 0 = auto = all cores
        assert _pii._pii_n_process(1) == 1       # explicit single process


def test_n_process_env_fallback():
    with mock.patch("os.cpu_count", return_value=4), \
         mock.patch.dict(os.environ, {"CAIRN_PII_N_PROCESS": "3"}):
        assert _pii._pii_n_process() == 3        # env used when no explicit
        assert _pii._pii_n_process(1) == 1       # explicit still wins


# ── NER-off behaviour: regex/secrets fire, PERSON does not ────────────
@pytest.mark.skipif(not _pii.cascade_available(), reason="needs the [pii] extra")
def test_cascade_ner_off_catches_patterns_not_names():
    os.environ.setdefault("CAIRN_PII_SPACY_MODEL", "en_core_web_sm")
    text = "Email John Smith at john@acme.com; AWS key AKIA1234567890ABCDEF."
    # NER ON (governed): should find the person AND the email/secret
    on = _pii.scan(text, backend="cascade", model="sm")
    assert "EMAIL_ADDRESS" in on
    # NER OFF (logs): still finds email + secret, but NOT the person name
    off = _pii.scan(text, backend="cascade", model="blank")
    assert "EMAIL_ADDRESS" in off
    assert "PERSON" not in off
