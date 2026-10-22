"""Schema-aware text routing (#1021 P1b, ADR-0079 F4).

The PII operators used to scan EVERY string leaf, so a tokenized/labelled row
(ai4privacy-shaped) ran the NER once per token. A "text leaf" is now a string
that looks like free text; tokens/labels/ids/roles are skipped, and an explicit
`text_fields` restricts the walk to the training-text field(s). These are the
hermetic (regex-backend) tests of the routing; cascade behaviour is in
test_pii_cascade.
"""
from __future__ import annotations

import asyncio
import json

from operator_dataset_checks import _pii
from operator_dataset_checks.operators.anonymize import AnonymizeOp
from operator_dataset_checks.operators.pii_scan import PiiScanOp
from operator_dataset_checks.schemas.anonymize import AnonymizeInput
from operator_dataset_checks.schemas.pii_scan import PiiInput


class _Ctx:
    pass


# an ai4privacy-shaped row: the real text + a tokenization + BIO labels + id
DIRTY_ROW = {
    "id": "42",
    "source_text": "Email John Smith at john@acme.com about invoice 5567.",
    "mbert_text_tokens": ["Email", "John", "Smith", "at", "john", "@", "acme", ".", "com"],
    "mbert_bio_labels": ["O", "B-PER", "I-PER", "O", "B-EMAIL", "I-EMAIL", "O", "O", "O"],
}


def test_iter_text_leaves_skips_token_and_label_lists():
    leaves = list(_pii.iter_text_leaves(DIRTY_ROW))
    # the real sentence is scanned; the 9-token list and 9-label list are NOT
    assert DIRTY_ROW["source_text"] in leaves
    assert "Email" not in leaves and "B-PER" not in leaves  # no per-token scanning
    # scalar fields (source_text + id) are kept; the two lists are dropped
    assert set(leaves) == {DIRTY_ROW["source_text"], "42"}


def test_iter_text_leaves_honors_explicit_fields():
    assert list(_pii.iter_text_leaves(DIRTY_ROW, ["source_text"])) == [DIRTY_ROW["source_text"]]
    # a token-list field named explicitly still yields nothing (it's a token list)
    assert list(_pii.iter_text_leaves(DIRTY_ROW, ["mbert_text_tokens"])) == []


def test_chat_row_keeps_content():
    chat = {"messages": [
        {"role": "user", "content": "My email is jane@corp.com, please help."},
        {"role": "assistant", "content": "Sure, I have noted your request."},
    ]}
    leaves = list(_pii.iter_text_leaves(chat))
    # message contents are scanned (messages is a list of DICTS, not a token list)
    assert chat["messages"][0]["content"] in leaves
    assert chat["messages"][1]["content"] in leaves


def test_map_text_leaves_skips_token_lists(tmp_path):
    # realistic redactor: replaces the email wherever it appears
    out = _pii.map_text_leaves(DIRTY_ROW, lambda s: s.replace("john@acme.com", "<EMAIL>"))
    assert out["source_text"] == DIRTY_ROW["source_text"].replace("john@acme.com", "<EMAIL>")
    # the token + label LISTS are the blow-up — they must be passed through as-is
    assert out["mbert_text_tokens"] == DIRTY_ROW["mbert_text_tokens"]
    assert out["mbert_bio_labels"] == DIRTY_ROW["mbert_bio_labels"]
    assert out["id"] == "42"  # scalar scanned, but the redactor left it unchanged


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def test_anonymize_routes_and_preserves_metadata(tmp_path):
    src = _write(tmp_path / "in.jsonl", [DIRTY_ROW, DIRTY_ROW])
    out = str(tmp_path / "out.jsonl")
    o = asyncio.run(AnonymizeOp().run(
        AnonymizeInput(path=src, output_path=out, backend="regex"), _Ctx()))
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    # email redacted IN the text field; token/label/id fields byte-identical
    assert "john@acme.com" not in rows[0]["source_text"]
    assert "<EMAIL>" in rows[0]["source_text"]
    assert rows[0]["mbert_text_tokens"] == DIRTY_ROW["mbert_text_tokens"]
    assert rows[0]["mbert_bio_labels"] == DIRTY_ROW["mbert_bio_labels"]
    assert rows[0]["id"] == "42"
    assert o.rows_redacted == 2


def test_pii_scan_counts_text_only(tmp_path):
    # the id "5567"-like token / labels must not inflate the PII count
    src = _write(tmp_path / "in.jsonl", [DIRTY_ROW] * 3)
    o = asyncio.run(PiiScanOp().run(
        PiiInput(path=src, backend="regex", sample_size=None), _Ctx()))
    # one email per row's source_text; tokens/labels contribute nothing
    assert o.counts.get("email") == 3
    assert o.rows_with_pii == 3
