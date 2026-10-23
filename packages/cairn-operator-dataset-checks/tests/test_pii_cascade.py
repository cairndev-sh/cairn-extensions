"""Governed tiered PII cascade (#1021, ADR-0079).

The cascade registers cairn's secret patterns INSIDE one Presidio AnalyzerEngine,
so a single pass runs regex + NER + context and Presidio merges overlaps. These
tests pin the two correctness fixes over the bare-presidio backend — secrets are
caught, and the `kinds` filter works across vocabularies — plus the content-hash
cache. Skipped unless Presidio + a spaCy model are installed (the [pii] extra).
"""
from __future__ import annotations


import pytest

presidio = pytest.importorskip("presidio_analyzer")


def _pick_model() -> str | None:
    import spacy
    for m in ("en_core_web_sm", "en_core_web_lg"):
        try:
            spacy.load(m)
            return m
        except OSError:
            continue
    return None


_MODEL = _pick_model()
pytestmark = pytest.mark.skipif(_MODEL is None, reason="no spaCy model installed")


@pytest.fixture(autouse=True)
def _model_env(monkeypatch):
    monkeypatch.setenv("CAIRN_PII_SPACY_MODEL", _MODEL)
    from operator_dataset_checks import _pii
    # engines are lru_cached on the model — clear so the env takes effect
    _pii._cascade_analyzer.cache_clear()
    _pii._cascade_batch_analyzer.cache_clear()
    _pii._cascade_cache.clear()
    yield


SECRETS = "AWS key AKIA0123456789ABCDEF and token sk-abcdefghij0123456789xyz"
NAMED = "Please email Alice Johnson at alice.johnson@corp.com about the invoice."


def test_auto_resolves_to_cascade_when_presidio_present():
    from operator_dataset_checks import _pii
    assert _pii.resolve("auto") == "cascade"


def test_cascade_catches_secrets_the_bare_path_drops():
    # THE correctness fix: cairn's AKIA…/sk-… recognizers are inside Presidio now.
    from operator_dataset_checks import _pii
    c = _pii.scan(SECRETS, backend="cascade")
    assert c.get("AWS_ACCESS_KEY") == 1
    assert c.get("OPENAI_KEY") == 1


def test_cascade_adds_ner_and_presidio_builtins():
    from operator_dataset_checks import _pii
    c = _pii.scan(NAMED, backend="cascade")
    assert c.get("PERSON", 0) >= 1        # spaCy NER
    assert c.get("EMAIL_ADDRESS", 0) >= 1  # Presidio predefined


def test_cascade_anonymize_replaces_entities():
    from operator_dataset_checks import _pii
    red, counts = _pii.anonymize(SECRETS, backend="cascade")
    assert "AKIA0123456789ABCDEF" not in red
    assert "sk-abcdefghij0123456789xyz" not in red
    assert "<AWS_ACCESS_KEY>" in red and "<OPENAI_KEY>" in red


def test_kinds_filter_accepts_cairn_vocabulary():
    # bare presidio matched NOTHING when given cairn kinds (entity-type mismatch);
    # the cascade maps cairn kinds → Presidio entity types.
    from operator_dataset_checks import _pii
    c = _pii.scan(NAMED, backend="cascade", kinds=["email"])
    assert "EMAIL_ADDRESS" in c
    assert "PERSON" not in c  # filtered out


def test_content_hash_cache_dedups_identical_strings():
    from operator_dataset_checks import _pii
    _pii._cascade_cache.clear()
    texts = [NAMED] * 50 + [SECRETS]
    out = _pii.scan_many(texts, backend="cascade")
    assert len(out) == 51
    assert len(_pii._cascade_cache) == 2  # only the 2 unique strings analyzed
    # every duplicate returned the same detection as the first
    assert all(o == out[0] for o in out[:50])


def test_cascade_requires_presidio_error_is_clear(monkeypatch):
    from operator_dataset_checks import _pii
    monkeypatch.setattr(_pii, "presidio_available", lambda: False)
    with pytest.raises(RuntimeError, match=r"\[pii\] extra"):
        _pii.resolve("cascade")
