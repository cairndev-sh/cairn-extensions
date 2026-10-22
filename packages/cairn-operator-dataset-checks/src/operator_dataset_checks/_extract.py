"""Best-effort (instruction, response) extraction across the common fine-tuning
formats. Used by dataset/sample to feed the LLM-as-judge (eval/judge). Returns
None for rows with no clear input/output split (e.g. a bare ``{"text": ...}``
completion), which the sampler then skips.
"""
from __future__ import annotations

from typing import Optional

_USER = {"user", "human"}
_ASSISTANT = {"assistant", "gpt"}


def _last(turns: list[tuple], roles: set[str]) -> Optional[str]:
    for role, content in reversed(turns):
        if str(role).lower() in roles and str(content).strip():
            return str(content)
    return None


def extract_io(row: dict) -> Optional[tuple[str, str]]:
    """Return ``(input, output)`` for a row, or None if it can't be split."""
    if isinstance(row.get("messages"), list):  # openai_chat
        turns = [(m.get("role", ""), m.get("content", "")) for m in row["messages"]]
        i, o = _last(turns, _USER), _last(turns, _ASSISTANT)
        return (i, o) if i and o else None
    if isinstance(row.get("conversations"), list):  # sharegpt
        turns = [(t.get("from", ""), t.get("value", "")) for t in row["conversations"]]
        i, o = _last(turns, _USER), _last(turns, _ASSISTANT)
        return (i, o) if i and o else None
    if "instruction" in row and "output" in row:  # alpaca
        instr = str(row.get("instruction", "")).strip()
        extra = str(row.get("input", "")).strip()
        i = f"{instr}\n\n{extra}".strip() if extra else instr
        o = str(row.get("output", "")).strip()
        return (i, o) if i and o else None
    if "prompt" in row and "completion" in row:  # completion pair
        i, o = str(row["prompt"]).strip(), str(row["completion"]).strip()
        return (i, o) if i and o else None
    return None
