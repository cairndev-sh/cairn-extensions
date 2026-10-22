"""The chat template a run renders its rows with (cairn-extensions#7).

The trainers map every row to chat ``messages`` and render them with the tokenizer's own chat
template. A *base* model's tokenizer has none (found on the first GPU run of Qwen/Qwen3-4B-Base:
``tokenizer.chat_template is not set``), so the run had nothing to render with. Such a tokenizer
gets ChatML — the turn format Qwen's vocabulary carries special tokens for and the one most
instruction-tuned models use — and the run records which template it trained with. A tokenizer
that has its own template keeps it.

Stdlib only, duck-typed over a tokenizer, so the tests need no training stack.
"""
from __future__ import annotations

from typing import Any

#: ChatML: ``<|im_start|>{role}\n{content}<|im_end|>\n`` per turn, and the assistant header when
#: a generation prompt is asked for.
CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def ensure_chat_template(tokenizer: Any) -> str:
    """Give ``tokenizer`` a chat template when it has none; return what the run renders with:
    ``model`` (its own) or ``chatml`` (the default for a base model)."""
    if getattr(tokenizer, "chat_template", None):
        return "model"
    tokenizer.chat_template = CHATML
    return "chatml"


__all__ = ["CHATML", "ensure_chat_template"]
