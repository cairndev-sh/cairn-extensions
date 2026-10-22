"""Tests for the 'regexset' PII backend (issue #379).

``regexset`` combines the same patterns as the zero-dep ``regex`` backend into
one named-group alternation and matches them in a single linear pass via
``google-re2`` (the ``[fastpii]`` extra). These tests must stay hermetic when
the extra isn't installed:

  - fast-path tests (parity, anonymize) use ``pytest.importorskip("re2")`` and
    SKIP without the extra;
  - a dedicated test hides ``re2`` via ``sys.modules`` and asserts
    ``resolve("regexset")`` raises a clear ``RuntimeError`` — mirroring the
    existing presidio-absent behavior — so the failure mode is proven even in
    a hermetic CI run that never installs google-re2.
"""
from __future__ import annotations

import sys

import pytest

import operator_dataset_checks._pii as pii_mod


# ── hermetic: regexset absent → clear error, not an ImportError deep in a call ──
def test_resolve_regexset_raises_without_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "re2", None)
    pii_mod._combined_pattern.cache_clear()
    with pytest.raises(RuntimeError, match=r"\[fastpii\]"):
        pii_mod.resolve("regexset")


def test_regexset_available_false_without_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "re2", None)
    assert pii_mod.regexset_available() is False


# ── fast path: SKIP entirely if google-re2 isn't installed ────────────────
def test_regexset_available_true_with_extra():
    pytest.importorskip("re2")
    assert pii_mod.regexset_available() is True


def test_resolve_regexset_passthrough():
    pytest.importorskip("re2")
    assert pii_mod.resolve("regexset") == "regexset"


TEXT = (
    "mail me at bob@example.com or alice@test.org, server is 10.0.0.1, "
    "creds: AKIAIOSFODNN7EXAMPLE, key sk-abcdefghijklmnopqrstuvwxyz123456, "
    "token ghp_" + "a" * 36 + ", "
    "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dGVzdHNpZw, "
    "-----BEGIN RSA PRIVATE KEY-----"
)


def test_regexset_counts_match_regex_backend_exactly():
    """Parity: regexset must report byte-for-byte the same counts as regex."""
    pytest.importorskip("re2")
    regex_counts = pii_mod.scan(TEXT, backend="regex")
    regexset_counts = pii_mod.scan(TEXT, backend="regexset")
    assert regexset_counts == regex_counts
    # Sanity: this fixture actually exercises every recognizer kind.
    assert set(regex_counts) == {
        "private_key", "jwt", "aws_access_key", "openai_key",
        "github_token", "email", "ipv4",
    }


def test_regexset_counts_match_regex_backend_with_kinds_filter():
    pytest.importorskip("re2")
    regex_counts = pii_mod.scan(TEXT, backend="regex", kinds=["email", "ipv4"])
    regexset_counts = pii_mod.scan(TEXT, backend="regexset", kinds=["email", "ipv4"])
    assert regexset_counts == regex_counts == {"email": 2, "ipv4": 1}


def test_regexset_anonymize_matches_regex_backend():
    pytest.importorskip("re2")
    regex_text, regex_counts = pii_mod.anonymize(TEXT, backend="regex")
    regexset_text, regexset_counts = pii_mod.anonymize(TEXT, backend="regexset")
    assert regexset_counts == regex_counts
    assert regexset_text == regex_text


def test_regexset_anonymize_clean_text_is_unchanged():
    pytest.importorskip("re2")
    text = "just some harmless training text, nothing to see"
    red, counts = pii_mod.anonymize(text, backend="regexset")
    assert red == text
    assert counts == {}


def test_regexset_scan_no_matches_returns_empty_dict():
    pytest.importorskip("re2")
    assert pii_mod.scan("nothing sensitive here", backend="regexset") == {}


def test_auto_backend_unaffected_by_regexset(monkeypatch):
    """'auto' must keep its pre-existing behavior: presidio > regex, never regexset."""
    monkeypatch.setattr(pii_mod, "presidio_available", lambda: False)
    assert pii_mod.resolve("auto") == "regex"
