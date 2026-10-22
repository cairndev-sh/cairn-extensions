"""A base model's tokenizer has no chat template (cairn-extensions#7, found on the first GPU run
of Qwen/Qwen3-4B-Base): the run renders with ChatML and records it; a model's own is kept."""
from __future__ import annotations

import pytest

from operator_training.chat_template import CHATML, ensure_chat_template


class _Tok:
    def __init__(self, template=None):
        self.chat_template = template


def test_a_tokenizer_without_a_template_gets_chatml():
    tok = _Tok()
    assert ensure_chat_template(tok) == "chatml" and tok.chat_template == CHATML


def test_a_models_own_template_is_kept():
    tok = _Tok("{{ messages }}")
    assert ensure_chat_template(tok) == "model" and tok.chat_template == "{{ messages }}"


def test_chatml_renders_turns_and_the_generation_prompt():
    jinja2 = pytest.importorskip("jinja2")
    t = jinja2.Template(CHATML)
    msgs = [{"role": "user", "content": "What is an IfcWall?"}, {"role": "assistant", "content": "A wall."}]
    assert t.render(messages=msgs, add_generation_prompt=False) == (
        "<|im_start|>user\nWhat is an IfcWall?<|im_end|>\n<|im_start|>assistant\nA wall.<|im_end|>\n")
    assert t.render(messages=msgs[:1], add_generation_prompt=True).endswith("<|im_start|>assistant\n")
