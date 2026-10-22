"""PII recognition + anonymization backends for the dataset_checks bundle.

Three backends, one interface:

  - **regex** (default, zero-dep): the recognizers used for local testing and
    any deployment that hasn't installed an extra. Loops over each pattern
    calling ``.findall``/``.subn`` — O(patterns × text). Fast enough for small
    corpora; hermetic fallback for everything else.
  - **regexset** (``[fastpii]`` extra): the SAME regex patterns as ``regex``,
    but combined into one named-group alternation and matched in a SINGLE
    linear pass via Google's `re2 <https://github.com/google/re2>`_
    (``google-re2`` on PyPI). On multi-million-row corpora the per-pattern
    Python-``re`` loop is the bottleneck; re2's DFA/NFA matches all
    alternatives simultaneously in one O(text) scan instead of
    O(patterns × text), and (being RE2) is immune to catastrophic
    backtracking. Reports the SAME kind vocabulary and counts as ``regex``
    (verified byte-for-byte parity in tests) — it's a drop-in accelerant, not
    a different recognizer set.
  - **presidio** (production, ``[pii]`` extra): Microsoft Presidio analyzer +
    anonymizer — NER + context, far higher recall. Needs a spaCy model
    (``python -m spacy download en_core_web_lg``).

Why re2 over the other candidates considered (see #379): `pyahocorasick
<https://github.com/WojciechMula/pyahocorasick>`_ only matches literal
strings, not regexes — several of our patterns (``sk-[A-Za-z0-9]{20,}``,
``AKIA[0-9A-Z]{16}``, the email/JWT/ipv4 patterns) are genuine regexes, so
Aho-Corasick alone can't express them without hand-splitting literal
prefixes from variable suffixes. `hyperscan <https://github.com/darvid/
python-hyperscan>`_ is purpose-built for simultaneous multi-regex matching
and is also a strong option, but its match-callback API reports match
*spans* per pattern rather than a single ordered stream of non-overlapping
matches, which would need extra bookkeeping to reproduce the anonymize
in-order-replacement semantics; re2 already returns an ordered
``finditer`` stream with rewritable text, at the cost of not supporting a
handful of exotic regex features (backreferences, lookaround) that none of
our current patterns use anyway.

``backend="auto"`` uses presidio when importable, else regex (unchanged from
before regexset existed — regexset is opt-in only, since it trades a small
amount of behavioral surface, e.g. Unicode class support nuances, for raw
throughput and callers should choose it deliberately for big-corpus runs).
The three backends report different entity vocabularies only in the
presidio case (regex kinds vs Presidio entity types) — regex and regexset
always agree.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Optional, Sequence

# #964 — presidio runs a spaCy NLP pipeline per text. Called once-per-string it
# dominates runtime and blows the per-activity timeout on large corpora. The
# batch API (`scan_many`/`anonymize_many`) feeds a whole chunk of strings through
# presidio's BatchAnalyzerEngine (one `nlp.pipe` pass over the batch) instead of
# N separate calls — the single biggest lever for 50k+ row scans. regex/regexset are
# already O(text) per string, so their batch path is just a list comprehension.
#
# #1021 — this is the number of rows an operator buffers before ONE
# `analyze_iterator` call. It must be LARGE, not the old 256: when `n_process>1`
# spaCy forks a worker pool and COPIES the model per `analyze_iterator` call, so a
# small chunk pays that fork cost every 256 rows and `n_process=2` measured ~2.3×
# SLOWER than single-process. A large chunk amortizes the fork over thousands of
# rows (and is a free ~9% win even at n_process=1). Tune via CAIRN_PII_ANALYZE_CHUNK.
_ANALYZE_CHUNK_DEFAULT = 2048
_PRESIDIO_BATCH_SIZE = _ANALYZE_CHUNK_DEFAULT  # back-compat alias (was 256)


def _analyze_chunk_size() -> int:
    """Rows buffered before one batched NER pass (``CAIRN_PII_ANALYZE_CHUNK``,
    default 2048). Decoupled from any write-batch so the per-call spaCy fork cost
    (n_process>1) amortizes over a big chunk instead of every few hundred rows."""
    raw = os.environ.get("CAIRN_PII_ANALYZE_CHUNK")
    try:
        return max(1, int(raw)) if raw else _ANALYZE_CHUNK_DEFAULT
    except ValueError:
        return _ANALYZE_CHUNK_DEFAULT

# ── regex backend (default; local testing) ───────────────────────────
_REGEX: dict[str, re.Pattern] = {
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    # AWS access-key id: AKIA (long-term) + ASIA (temporary STS creds).
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    # OpenAI: legacy `sk-<48>` AND the current prefixed families
    # (`sk-proj-`, `sk-svcacct-`, `sk-admin-`) whose bodies carry `-`/`_` — the
    # old `sk-[A-Za-z0-9]{20,}` broke on the hyphen and MISSED every modern key.
    "openai_key": re.compile(r"\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}|\bsk-[A-Za-z0-9]{20,}\b"),
    # GitHub: classic + all server/oauth/user/refresh PATs (ghp_/gho_/ghs_/ghu_/
    # ghr_ + 36) AND fine-grained tokens (`github_pat_…`, which the old pattern
    # missed entirely).
    "github_token": re.compile(r"\b(?:gh[posur]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})\b"),
    # Slack bot/user/app/refresh/config tokens (xoxb-/xoxp-/xoxa-/xoxr-/xoxs-).
    "slack_token": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}"),
    # Stripe live/test secret + restricted keys (sk_live_/rk_live_/sk_test_/rk_test_).
    "stripe_key": re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # #1436 — the three PII kinds the regex tier lacked (Presidio supplied them
    # via NER-free pattern recognizers, which is why `logs` still crawled at
    # ~100 KB/s). Every one REQUIRES separators so a bare JSON number can
    # never match: these patterns also run over raw JSON text in the DuckDB
    # backend, and an unquoted `<CREDIT_CARD>` would corrupt the row.
    "us_ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b"),
    # +CC and (NXX) forms, US 3-3-4 with a separator, UK-style +44 7700 900123.
    "phone": re.compile(
        r"(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]\d{3}[ -]?\d{4}\b"
        r"|\+\d{2}[ -]\d{4}[ -]\d{6}\b"),
}

# The regex tier as (kind, RE2-safe pattern) pairs, in application order — the
# DuckDB backend chains one `regexp_replace(…, 'g')` per kind over the row's
# JSON text (#1436 leg B). No lookarounds/backrefs anywhere in `_REGEX`, so the
# Python `re` source strings ARE the RE2 patterns (pinned by the pack tests).
REGEX_KINDS: tuple[tuple[str, str], ...] = tuple((k, p.pattern) for k, p in _REGEX.items())


def iter_strings(obj: object) -> Iterator[str]:
    """Yield every string leaf in a nested JSON value."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_strings(v)


def map_strings(obj: object, fn) -> object:
    """Rebuild a nested JSON value with ``fn`` applied to every string leaf."""
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: map_strings(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [map_strings(v, fn) for v in obj]
    return obj


# ── schema-aware text routing (#1021 F4) ─────────────────────────────────────
# The PII operators used to scan EVERY string leaf, so a row carrying a
# tokenization (``mbert_text_tokens: ["The","patient",…]``) or a per-token label
# list ran the NER once PER TOKEN — the training-data blow-up. The fix skips
# exactly that: a list whose elements are all short, whitespace-free strings is a
# tokenization/label sequence and is not scanned. SCALAR string fields are always
# scanned (a bare ``email`` / ``id`` field may still be PII, even if short).
# ``text_fields`` restricts the walk to named top-level fields (e.g.
# ``["source_text"]`` / ``["messages"]``); unset = auto (all fields, token-lists
# excluded).
_MIN_TEXT_LEN = 16


def _looks_texty(s: str) -> bool:
    """Free text vs a token/label/enum: whitespace or length ≥ 16."""
    return " " in s or len(s) >= _MIN_TEXT_LEN


def _is_token_list(v: object) -> bool:
    """A tokenization / label sequence — a non-empty list whose every element is a
    short, whitespace-free string. Scanning it would NER once per token (#1021 F4).
    A list of dicts (chat ``messages``) or of long strings is NOT a token list."""
    return (
        isinstance(v, list) and len(v) > 0
        and all(isinstance(e, str) and not _looks_texty(e) for e in v)
    )


def _scoped(obj: object, text_fields: Optional[Sequence[str]]) -> object:
    """The sub-structure to walk: the named top-level fields, or the whole row."""
    if text_fields and isinstance(obj, dict):
        return {k: obj[k] for k in text_fields if k in obj}
    return obj


def _iter_text(obj: object) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_text(v)
    elif isinstance(obj, list) and not _is_token_list(obj):
        for v in obj:
            yield from _iter_text(v)


def iter_text_leaves(obj: object, text_fields: Optional[Sequence[str]] = None) -> Iterator[str]:
    """Yield the text string leaves of ``obj`` (optionally only within
    ``text_fields``), skipping tokenization/label lists — the routed twin of
    :func:`iter_strings`."""
    yield from _iter_text(_scoped(obj, text_fields))


def _map_text(obj: object, fn) -> object:
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _map_text(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return obj if _is_token_list(obj) else [_map_text(v, fn) for v in obj]
    return obj


def map_text_leaves(
    obj: object, fn, text_fields: Optional[Sequence[str]] = None
) -> object:
    """Rebuild ``obj`` applying ``fn`` to exactly the leaves
    :func:`iter_text_leaves` yields (same order), leaving every other value —
    non-selected fields and token/label lists — untouched."""
    if text_fields and isinstance(obj, dict):
        return {
            k: (_map_text(v, fn) if k in text_fields else v)
            for k, v in obj.items()
        }
    return _map_text(obj, fn)


def _regex_active(kinds: Optional[list[str]]) -> dict[str, re.Pattern]:
    return {k: p for k, p in _REGEX.items() if kinds is None or k in kinds}


def _regex_counts(text: str, kinds: Optional[list[str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, p in _regex_active(kinds).items():
        n = len(p.findall(text))
        if n:
            out[k] = n
    return out


def _regex_anonymize(text: str, kinds: Optional[list[str]]) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    red = text
    for k, p in _regex_active(kinds).items():
        red, n = p.subn(f"<{k.upper()}>", red)
        if n:
            counts[k] = n
    return red, counts


# ── regexset backend (fast; [fastpii] extra) ───────────────────────────
def regexset_available() -> bool:
    return bool(importlib.util.find_spec("re2"))


@lru_cache(maxsize=1)
def _combined_pattern():
    import re2  # optional dependency (fastpii extra)

    combined = "|".join(f"(?P<{k}>{p.pattern})" for k, p in _REGEX.items())
    return re2.compile(combined)


def _regexset_pattern(kinds: Optional[list[str]]):
    if kinds is None:
        return _combined_pattern()
    import re2  # optional dependency (fastpii extra)

    active = _regex_active(kinds)
    combined = "|".join(f"(?P<{k}>{p.pattern})" for k, p in active.items())
    return re2.compile(combined)


def _regexset_counts(text: str, kinds: Optional[list[str]]) -> dict[str, int]:
    pattern = _regexset_pattern(kinds)
    counts: dict[str, int] = {}
    for m in pattern.finditer(text):
        k = m.lastgroup
        counts[k] = counts.get(k, 0) + 1
    return counts


def _regexset_anonymize(text: str, kinds: Optional[list[str]]) -> tuple[str, dict[str, int]]:
    pattern = _regexset_pattern(kinds)
    counts: dict[str, int] = {}
    pieces: list[str] = []
    last = 0
    for m in pattern.finditer(text):
        k = m.lastgroup
        pieces.append(text[last:m.start()])
        pieces.append(f"<{k.upper()}>")
        last = m.end()
        counts[k] = counts.get(k, 0) + 1
    pieces.append(text[last:])
    return "".join(pieces), counts


# ── presidio backend (production; [pii] extra) ────────────────────────
def presidio_available() -> bool:
    return bool(
        importlib.util.find_spec("presidio_analyzer")
        and importlib.util.find_spec("presidio_anonymizer")
    )


@lru_cache(maxsize=1)
def _engines():
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    return AnalyzerEngine(), AnonymizerEngine()


def _presidio_counts(text: str, kinds: Optional[list[str]]) -> dict[str, int]:
    analyzer, _ = _engines()
    counts: dict[str, int] = {}
    for r in analyzer.analyze(text=text, language="en", entities=kinds):
        counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
    return counts


def _presidio_anonymize(text: str, kinds: Optional[list[str]]) -> tuple[str, dict[str, int]]:
    analyzer, anonymizer = _engines()
    results = analyzer.analyze(text=text, language="en", entities=kinds)
    red = anonymizer.anonymize(text=text, analyzer_results=results).text
    counts: dict[str, int] = {}
    for r in results:
        counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
    return red, counts


@lru_cache(maxsize=1)
def _batch_analyzer():
    # BatchAnalyzerEngine wraps the same AnalyzerEngine but analyzes an ITERABLE
    # of texts through spaCy's nlp.pipe in one pass — the batch speedup (#964).
    from presidio_analyzer import BatchAnalyzerEngine

    analyzer, _ = _engines()
    return BatchAnalyzerEngine(analyzer_engine=analyzer)


def _pii_n_process(explicit: Optional[int] = None) -> int:
    """spaCy multiprocessing degree for the NER pass (``analyze_iterator``).
    ``explicit`` (from run config) wins; else ``CAIRN_PII_N_PROCESS``; else 1.
    ``0`` (or any ≤0) = AUTO = every core.

    **Clamped to ``cpu_count``** (not cpu_count-1): the 200k-on-2-CPU experiment
    (ADR-0079, Linux fork + a large CAIRN_PII_ANALYZE_CHUNK) measured
    ``n_process=2`` on a 2-core box as ~**40% FASTER** than single-process — fork
    is copy-on-write cheap and the big chunk amortizes it. So all cores is the
    ceiling. (The earlier "2× slower" reading was an artifact of a tiny 256-row
    chunk forking per batch, since fixed.) Oversubscription above the core count is
    still capped. Pair any n_process>1 with a large CAIRN_PII_ANALYZE_CHUNK."""
    cpu = max(1, os.cpu_count() or 1)
    if explicit is None:
        raw = os.environ.get("CAIRN_PII_N_PROCESS")
        try:
            want = max(1, int(raw)) if raw else 1
        except ValueError:
            want = 1
    elif explicit <= 0:
        want = cpu                      # 0 = auto = all cores
    else:
        want = explicit
    return min(want, cpu)


def _pii_batch_size() -> int:
    """nlp.pipe batch size for ``analyze_iterator`` (``CAIRN_PII_BATCH_SIZE``,
    default 128). Larger amortizes per-batch overhead; smaller bounds memory."""
    raw = os.environ.get("CAIRN_PII_BATCH_SIZE")
    try:
        return max(1, int(raw)) if raw else 128
    except ValueError:
        return 128


def _analyze_iterator(batch, texts, entities, n_process=None):
    """``analyze_iterator`` with the tuned batch_size + n_process (#1021 P1c).
    ``n_process`` (from run config) overrides the env; None → env/default."""
    return list(batch.analyze_iterator(
        list(texts), language="en", entities=entities,
        batch_size=_pii_batch_size(), n_process=_pii_n_process(n_process),
    ))


def _presidio_analyze_batch(texts: Sequence[str], kinds: Optional[list[str]]):
    """List[List[RecognizerResult]] for ``texts`` — one nlp.pipe pass, not N."""
    return _analyze_iterator(_batch_analyzer(), texts, kinds)


def _presidio_counts_many(texts: Sequence[str], kinds: Optional[list[str]]) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for results in _presidio_analyze_batch(texts, kinds):
        counts: dict[str, int] = {}
        for r in results:
            counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
        out.append(counts)
    return out


def _presidio_anonymize_many(
    texts: Sequence[str], kinds: Optional[list[str]]
) -> list[tuple[str, dict[str, int]]]:
    _, anonymizer = _engines()
    per = _presidio_analyze_batch(texts, kinds)
    out: list[tuple[str, dict[str, int]]] = []
    for text, results in zip(texts, per):
        # anonymize itself is a cheap string splice; only the analyze (NLP) needed
        # batching, so it stays per-text.
        red = anonymizer.anonymize(text=text, analyzer_results=results).text
        counts: dict[str, int] = {}
        for r in results:
            counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
        out.append((red, counts))
    return out


# ── cascade backend (production; [pii] extra) — ADR-0079 ──────────────
# The tiered pipeline: cairn's deterministic SECRET recognizers registered INSIDE
# one presidio AnalyzerEngine, so a single pass runs regex + NER + context and
# presidio MERGES overlapping detections itself. This fixes the bare-presidio
# backend's defects: (1) it caught no secrets (default AnalyzerEngine has no
# custom recognizers); (2) its `kinds` filter used cairn's vocabulary against
# presidio's entity types → matched nothing. email/ipv4/credit-card come from
# presidio's OWN predefined recognizers, so we contribute only what it lacks:
# the API-key / token / private-key patterns.
_SECRET_KINDS = (
    "private_key", "jwt", "aws_access_key", "openai_key", "github_token",
    "slack_token", "stripe_key",
)

# cairn kind ↔ presidio entity type, so a `kinds` filter written in EITHER
# vocabulary resolves to the presidio entity types `analyze(entities=)` expects.
_KIND_TO_ENTITY = {
    "email": "EMAIL_ADDRESS",
    "ipv4": "IP_ADDRESS",
    "aws_access_key": "AWS_ACCESS_KEY",
    "openai_key": "OPENAI_KEY",
    "github_token": "GITHUB_TOKEN",
    "slack_token": "SLACK_TOKEN",
    "stripe_key": "STRIPE_KEY",
    "jwt": "JWT",
    "private_key": "PRIVATE_KEY",
    "credit_card": "CREDIT_CARD",
    "person": "PERSON",
    "location": "LOCATION",
    "phone": "PHONE_NUMBER",
    "us_ssn": "US_SSN",
}


# ── two-tier secret detection (ADR-0080 P1 / R4) ─────────────────────────────
# A rare secret (100 leaked keys in 1M rows = 0.01%) is MISSED by a 1000-row
# sample ~90% of the time — so the deterministic tier must scan EVERY row, cheap
# regex, while the NER density tier stays sampled. And it must emit the gate's
# vocabulary: the gate blocks on the lowercase cairn kinds below, but the cascade
# backend emits PRESIDIO types (AWS_ACCESS_KEY), so cascade-found secrets slipped
# past the block list (surfaced as a warn, not a fail). `secret_counts` returns
# the gate-recognized LOWERCASE kinds; `_SECRET_ENTITY_ALL` is every spelling to
# strip from the main scan so a secret is reported ONCE, in the blocking vocab.
_SECRET_ENTITY_ALL = frozenset(
    list(_SECRET_KINDS)
    + [k.upper() for k in _SECRET_KINDS]
    + [_KIND_TO_ENTITY[k] for k in _SECRET_KINDS if k in _KIND_TO_ENTITY]
)


def secret_counts(text: str) -> dict[str, int]:
    """Deterministic secret matches for ONE text, in the gate's lowercase cairn
    vocabulary (aws_access_key / openai_key / github_token / jwt / private_key).
    Regex — O(text), no NLP — so it runs over the full corpus cheaply."""
    return _regex_counts(text, list(_SECRET_KINDS))


def merge_secret_counts(main: dict[str, int], full_secret: dict[str, int]) -> dict[str, int]:
    """Fold authoritative full-corpus secret counts into a (possibly sampled) main
    scan: drop EVERY secret spelling from ``main`` (its counts are sample-scoped
    and may be in the wrong vocab), then add the full-corpus lowercase secrets."""
    out = {k: v for k, v in main.items() if k not in _SECRET_ENTITY_ALL}
    for k, v in full_secret.items():
        if v:
            out[k] = out.get(k, 0) + v
    return out


def cascade_available() -> bool:
    return presidio_available()


def _spacy_model() -> str:
    """spaCy model for the cascade NER tier. ``en_core_web_lg`` = highest recall
    (default, prod); set ``CAIRN_PII_SPACY_MODEL=en_core_web_sm`` for a ~3-5×
    faster tier where recall of the smaller model is acceptable (ADR-0079)."""
    return os.environ.get("CAIRN_PII_SPACY_MODEL", "en_core_web_lg")


# short model names (run/UI config) → full spaCy package name. 'blank' = a
# tokenizer-only pipeline (NER OFF) — the fast "logs"/regex-only path (ADR-0079).
_MODEL_FULL = {"sm": "en_core_web_sm", "lg": "en_core_web_lg", "trf": "en_core_web_trf"}


def _resolve_model_name(model: Optional[str]) -> str:
    """Normalize a config model to a spaCy package name or ``"blank"``.
    ``None`` → the ``CAIRN_PII_SPACY_MODEL`` default (legacy env behaviour)."""
    if model is None:
        return _spacy_model()
    if model == "blank":
        return "blank"
    return _MODEL_FULL.get(model, model)  # accept full names too


def _blank_nlp_engine():
    """A spaCy TOKENIZER-only NLP engine (no statistical NER) — regex + secret
    recognizers still fire, PERSON/LOCATION don't. Turns an hours-long NER scan
    into minutes on a small box (the 200k experiment's fastest, lightest path)."""
    import spacy
    from presidio_analyzer.nlp_engine import SpacyNlpEngine

    class _BlankSpacy(SpacyNlpEngine):
        def __init__(self):
            try:  # presidio needs a valid engine shell; we then swap in a blank pipe
                super().__init__(models=[{"lang_code": "en", "model_name": "en_core_web_sm"}])
            except Exception:
                pass
            self.nlp = {"en": spacy.blank("en")}

    return _BlankSpacy()


def _cascade_entities(kinds: Optional[list[str]]) -> Optional[list[str]]:
    """Normalize a `kinds` filter (cairn OR presidio vocabulary) to presidio
    entity types. ``None`` → all entities (fixes the old vocab-mismatch bug)."""
    if kinds is None:
        return None
    return [_KIND_TO_ENTITY.get(k, k.upper()) for k in kinds]


@lru_cache(maxsize=4)
def _cascade_analyzer(name: str):
    """Build the cascade AnalyzerEngine for a RESOLVED model name (full spaCy name
    or ``"blank"``). ``blank`` = NER OFF (tokenizer only, drop SpacyRecognizer) →
    regex + secrets only; else full NER + regex + secrets. Cached per name so
    switching model/NER at run time doesn't rebuild every call."""
    from presidio_analyzer import (
        AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry,
    )
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    if name == "blank":
        nlp_engine = _blank_nlp_engine()
        registry = RecognizerRegistry()
        try:
            registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en"])
        except TypeError:
            registry.load_predefined_recognizers(languages=["en"])
        # NER OFF: drop the model-based recognizers, keep the regex ones
        for rc in list(registry.recognizers):
            if rc.__class__.__name__ in ("SpacyRecognizer", "TransformersRecognizer"):
                try:
                    registry.remove_recognizer(rc.name)
                except Exception:
                    pass
        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, registry=registry, supported_languages=["en"],
        )
    else:
        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": name}],
        }).create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    # deterministic tier: cairn's secret patterns become presidio recognizers, so
    # the cascade catches AKIA…/sk-…/ghp_…/JWT/private-keys that presidio's own
    # defaults miss — the credentials the bare-presidio backend silently dropped.
    for kind in _SECRET_KINDS:
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity=kind.upper(),
            patterns=[Pattern(name=kind, regex=_REGEX[kind].pattern, score=0.9)],
        ))
    return analyzer


@lru_cache(maxsize=4)
def _cascade_batch_analyzer(name: str):
    from presidio_analyzer import BatchAnalyzerEngine

    return BatchAnalyzerEngine(analyzer_engine=_cascade_analyzer(name))


@lru_cache(maxsize=1)
def _cascade_anonymizer():
    from presidio_anonymizer import AnonymizerEngine

    return AnonymizerEngine()


# Content-hash cache (ADR-0079): identical strings — ubiquitous in templated
# training data, and the whole reason a dup-heavy corpus is slow — are analyzed
# ONCE and reused. Bounded LRU keyed by (entities, blake2b(text)) so a
# long-running shard worker never grows unbounded.
_CASCADE_CACHE_MAX = int(os.environ.get("CAIRN_PII_CACHE_MAX", "50000") or "50000")
_cascade_cache: "OrderedDict[tuple, list]" = OrderedDict()


def _cache_key(text: str, ekey, name: str) -> tuple:
    # keyed by (entities, MODEL, text) — sm/lg/blank give different spans for the
    # same text, so the model must be part of the key or a model switch reads stale.
    return (ekey, name, hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest())


def _cascade_analyze_texts(
    texts: Sequence[str], kinds: Optional[list[str]],
    model: Optional[str] = None, n_process: Optional[int] = None,
) -> list[list]:
    """Detection spans ``[(entity, start, end, score), …]`` per text, with
    content-hash dedup (cross-batch LRU + within-batch) and one nlp.pipe pass over
    the UNIQUE misses. Aligned to ``texts``. ``model`` selects the NER tier
    (``"blank"`` = NER off); ``n_process`` fans the pass across cores."""
    entities = _cascade_entities(kinds)
    ekey = tuple(entities) if entities is not None else None
    name = _resolve_model_name(model)
    by_text: "dict[str, Optional[list]]" = {}
    to_analyze: list[str] = []
    for t in texts:
        if t in by_text:
            continue
        cached = _cascade_cache.get(_cache_key(t, ekey, name))
        if cached is not None:
            _cascade_cache.move_to_end(_cache_key(t, ekey, name))
            by_text[t] = cached
        else:
            by_text[t] = None
            to_analyze.append(t)
    if to_analyze:
        per = _analyze_iterator(_cascade_batch_analyzer(name), to_analyze, entities, n_process)
        for t, results in zip(to_analyze, per):
            spans = [(r.entity_type, r.start, r.end, r.score) for r in results]
            by_text[t] = spans
            k = _cache_key(t, ekey, name)
            _cascade_cache[k] = spans
            _cascade_cache.move_to_end(k)
            while len(_cascade_cache) > _CASCADE_CACHE_MAX:
                _cascade_cache.popitem(last=False)
    return [by_text[t] for t in texts]


def _counts_from_spans(spans: list) -> dict[str, int]:
    out: dict[str, int] = {}
    for entity, _s, _e, _sc in spans:
        out[entity] = out.get(entity, 0) + 1
    return out


def _cascade_counts_many(
    texts: Sequence[str], kinds: Optional[list[str]],
    model: Optional[str] = None, n_process: Optional[int] = None,
) -> list[dict[str, int]]:
    return [_counts_from_spans(s) for s in _cascade_analyze_texts(texts, kinds, model, n_process)]


def _cascade_anonymize_many(
    texts: Sequence[str], kinds: Optional[list[str]],
    model: Optional[str] = None, n_process: Optional[int] = None,
) -> list[tuple[str, dict[str, int]]]:
    from presidio_analyzer import RecognizerResult

    anonymizer = _cascade_anonymizer()
    out: list[tuple[str, dict[str, int]]] = []
    for text, spans in zip(texts, _cascade_analyze_texts(texts, kinds, model, n_process)):
        if spans:
            results = [
                RecognizerResult(entity_type=e, start=s, end=en, score=sc)
                for e, s, en, sc in spans
            ]
            red = anonymizer.anonymize(text=text, analyzer_results=results).text
        else:
            red = text
        out.append((red, _counts_from_spans(spans)))
    return out


# ── resolver + public interface ───────────────────────────────────────
def resolve(backend: str) -> str:
    if backend == "auto":
        # presidio installed → the CASCADE (ADR-0079): strictly more correct than
        # the bare 'presidio' backend (same NER cost + the secret recognizers the
        # bare path dropped), so 'auto' now upgrades to it. Else the zero-dep regex
        # fallback. regexset stays opt-in (raw-speed accelerant, same recall).
        return "cascade" if cascade_available() else "regex"
    if backend in ("presidio", "cascade") and not presidio_available():
        raise RuntimeError(
            f"backend={backend!r} requires the [pii] extra: "
            "pip install 'cairn-operator-dataset-checks[pii]' (+ a spaCy model)."
        )
    if backend == "regexset" and not regexset_available():
        raise RuntimeError(
            "backend='regexset' requires the [fastpii] extra: "
            "pip install 'cairn-operator-dataset-checks[fastpii]'."
        )
    return backend


def scan(
    text: str, *, backend: str = "auto", kinds: Optional[list[str]] = None,
    model: Optional[str] = None, n_process: Optional[int] = None,
) -> dict[str, int]:
    resolved = resolve(backend)
    if resolved == "cascade":
        return _cascade_counts_many([text], kinds, model, n_process)[0]
    if resolved == "presidio":
        return _presidio_counts(text, kinds)
    if resolved == "regexset":
        return _regexset_counts(text, kinds)
    return _regex_counts(text, kinds)


def anonymize(
    text: str, *, backend: str = "auto", kinds: Optional[list[str]] = None,
    model: Optional[str] = None, n_process: Optional[int] = None,
) -> tuple[str, dict[str, int]]:
    resolved = resolve(backend)
    if resolved == "cascade":
        return _cascade_anonymize_many([text], kinds, model, n_process)[0]
    if resolved == "presidio":
        return _presidio_anonymize(text, kinds)
    if resolved == "regexset":
        return _regexset_anonymize(text, kinds)
    return _regex_anonymize(text, kinds)


def scan_many(
    texts: Sequence[str], *, backend: str = "auto", kinds: Optional[list[str]] = None,
    model: Optional[str] = None, n_process: Optional[int] = None,
) -> list[dict[str, int]]:
    """Per-text PII counts for a batch of strings. Same result as calling
    :func:`scan` on each, but presidio runs ONE batched nlp.pipe pass (#964).
    ``model``/``n_process`` tune the cascade NER tier (ignored by regex backends)."""
    resolved = resolve(backend)
    if resolved == "cascade":
        return _cascade_counts_many(texts, kinds, model, n_process)
    if resolved == "presidio":
        return _presidio_counts_many(texts, kinds)
    # regex/regexset are already O(text) per string — no NLP to batch.
    fn = _regexset_counts if resolved == "regexset" else _regex_counts
    return [fn(t, kinds) for t in texts]


def anonymize_many(
    texts: Sequence[str], *, backend: str = "auto", kinds: Optional[list[str]] = None,
    model: Optional[str] = None, n_process: Optional[int] = None,
) -> list[tuple[str, dict[str, int]]]:
    """Batch counterpart of :func:`anonymize` — presidio analyzes the whole batch
    in one nlp.pipe pass, then anonymizes each text (a cheap string splice)."""
    resolved = resolve(backend)
    if resolved == "cascade":
        return _cascade_anonymize_many(texts, kinds, model, n_process)
    if resolved == "presidio":
        return _presidio_anonymize_many(texts, kinds)
    fn = _regexset_anonymize if resolved == "regexset" else _regex_anonymize
    return [fn(t, kinds) for t in texts]


# ── presets + config resolution (ADR-0079; run/UI-tunable) ────────────────────
# Two ready-made bundles from the 200k-on-2-CPU experiment, then any knob overrides.
PII_PRESETS: dict[str, dict] = {
    # governed (default): full NER + regex + secrets — best recall (~7min/200k, 555MB sm)
    "governed": {"ner": True, "model": "sm", "n_process": 0, "analyze_chunk": 2048},
    # logs: regex + secrets only, NER OFF — faster/lighter, no names/locations
    "logs":     {"ner": False, "model": "sm", "n_process": 1, "analyze_chunk": 5000},
}


@dataclass(frozen=True)
class PiiSettings:
    """Resolved cascade tuning for one run. ``model`` is a spaCy short name or
    ``"blank"`` (NER off); ``n_process`` already clamped to the core count."""
    model: str
    n_process: int
    analyze_chunk: int


def resolve_pii_settings(
    *, preset: Optional[str] = None, ner: Optional[bool] = None,
    model: Optional[str] = None, n_process: Optional[int] = None,
    analyze_chunk: Optional[int] = None,
) -> PiiSettings:
    """Expand a preset then apply per-run overrides. ``ner=False`` (or model
    ``"blank"``) selects the tokenizer-only path (no NER). Env vars remain the
    fallback when a field is unset."""
    base = dict(PII_PRESETS.get(preset or "governed", PII_PRESETS["governed"]))
    if ner is not None:
        base["ner"] = ner
    if model is not None:
        base["model"] = model
    if n_process is not None:
        base["n_process"] = n_process
    if analyze_chunk is not None:
        base["analyze_chunk"] = analyze_chunk
    model_used = "blank" if (not base["ner"] or base["model"] == "blank") else base["model"]
    chunk = int(base["analyze_chunk"]) if base["analyze_chunk"] else _analyze_chunk_size()
    return PiiSettings(
        model=model_used,
        n_process=_pii_n_process(base["n_process"]),
        analyze_chunk=chunk,
    )
