"""Tests for dataset/profile's tokenizers-backed token counts (issue #381).

The default (`token_backend='whitespace'`) path is covered by the pre-existing
golden tests in `tests/operators/dataset_checks/test_profile.py`, which must
stay green unchanged. These tests exercise the NEW optional `tokenizers`
backend and prove the fallback is hermetic when `tokenizers` isn't installed.
"""
from __future__ import annotations

import json
import sys

import pytest

from cairn.core.operator import OperationContext
from operator_dataset_checks._tokenizer import (
    count_whitespace,
    resolve,
    tokenizers_available,
)
from operator_dataset_checks.operators.profile import ProfileOp
from operator_dataset_checks.schemas.profile import ProfileInput


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="profile")


def _write(tmp_path, rows) -> str:
    p = tmp_path / "ds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def _chat(user, asst):
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": asst},
    ]}


def _local_tokenizer_json(tmp_path) -> str:
    """Train a tiny BPE tokenizer fully offline and save it, so tests never
    hit the network (Tokenizer.from_pretrained would fetch from the HF Hub)."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(special_tokens=["[UNK]"], vocab_size=200)
    tok.train_from_iterator(
        ["hello world this is a test", "another line of plain text here",
         "paris is the capital of france"],
        trainer=trainer,
    )
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    return str(path)


def test_resolve_auto_prefers_tokenizers_when_available():
    resolved = resolve("auto")
    assert resolved == ("tokenizers" if tokenizers_available() else "whitespace")


def test_resolve_explicit_tokenizers_without_extra_raises(monkeypatch):
    monkeypatch.setattr(
        "operator_dataset_checks._tokenizer.tokenizers_available", lambda: False,
    )
    with pytest.raises(RuntimeError, match=r"\[tokenizers\] extra"):
        resolve("tokenizers")


async def test_default_backend_leaves_token_fields_none(tmp_path):
    """token_backend defaults to 'whitespace', which must NOT invoke the
    tokenizers path — response_length_tokens stays None (additive design)."""
    rows = [_chat("q", "a short reply here")]
    out = await ProfileOp().run(ProfileInput(path=_write(tmp_path, rows)), _ctx())
    assert out.response_length_tokens is None
    assert out.token_backend_used is None


async def test_tokenizers_backend_reports_subword_counts(tmp_path):
    tok_path = _local_tokenizer_json(tmp_path)
    rows = [
        _chat("q0", "hello world"),
        _chat("q1", "paris is the capital of france"),
    ]
    out = await ProfileOp().run(
        ProfileInput(
            path=_write(tmp_path, rows),
            token_backend="tokenizers",
            tokenizer_path=tok_path,
        ),
        _ctx(),
    )
    assert out.token_backend_used == "tokenizers"
    assert out.response_length_tokens is not None
    # Token counts must be positive and reported independently of char counts.
    assert out.response_length_tokens.min > 0
    assert out.response_length_tokens.max >= out.response_length_tokens.min


async def test_tokenizers_backend_requires_extra_when_forced_absent(tmp_path, monkeypatch):
    """Simulate a base install without the [tokenizers] extra: hide the
    'tokenizers' module and confirm the operator fails closed with a clear
    message rather than silently falling back."""
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    rows = [_chat("q", "hello there")]
    with pytest.raises(RuntimeError, match=r"\[tokenizers\] extra"):
        await ProfileOp().run(
            ProfileInput(path=_write(tmp_path, rows), token_backend="tokenizers"),
            _ctx(),
        )


async def test_auto_backend_falls_back_to_whitespace_hermetically(tmp_path, monkeypatch):
    """With tokenizers hidden, backend='auto' must resolve to whitespace and
    run without ever importing tokenizers."""
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    rows = [_chat("q", "hello there friend")]
    out = await ProfileOp().run(
        ProfileInput(path=_write(tmp_path, rows), token_backend="auto"),
        _ctx(),
    )
    # 'auto' with tokenizers hidden must NOT run the token-count path at all
    # (whitespace is the resolved backend, and want_tokens only triggers for
    # 'tokenizers') — same additive contract as the default.
    assert out.token_backend_used is None
    assert out.response_length_tokens is None


def test_count_whitespace_is_plain_word_split():
    assert count_whitespace("hello world  foo") == 3
    assert count_whitespace("") == 0
