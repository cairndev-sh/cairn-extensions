"""dataset/format — detect a fine-tuning dataset's chat format and validate
its per-example structure, deterministically (no model download, no network).

Supported formats:
  - ``openai_chat`` : ``{"messages": [{"role", "content"}, ...]}``
  - ``sharegpt``    : ``{"conversations": [{"from", "value"}, ...]}``
  - ``alpaca``      : ``{"instruction", "output"[, "input"]}``
  - ``completion``  : ``{"text"}`` or ``{"prompt", "completion"}``

Conversational rules: at least one non-system turn, roles alternate
user→assistant, the last turn is the assistant's, no empty content. The
output (detected format, per-row findings, per-trainer compatibility) is the
deterministic evidence the readiness report cites.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks._io import load_rows, _reservoir_sample
from operator_dataset_checks.schemas.format import (
    FormatConfig,
    FormatFinding,
    FormatInput,
    FormatOutput,
)

# Declarative trainer profiles — which canonical formats each trainer accepts.
# We never import trl/unsloth/axolotl (GPU-heavy); this is knowledge, not code.
_TRAINER_FORMATS = {
    "trl": {"openai_chat", "alpaca", "completion"},
    "unsloth": {"openai_chat", "sharegpt", "alpaca"},
    "axolotl": {"openai_chat", "sharegpt", "alpaca", "completion"},
}
_USER_ROLES = {"user", "human"}
_ASSISTANT_ROLES = {"assistant", "gpt"}


def _detect_format(row: dict) -> str:
    if isinstance(row.get("messages"), list):
        return "openai_chat"
    if isinstance(row.get("conversations"), list):
        return "sharegpt"
    if "instruction" in row and "output" in row:
        return "alpaca"
    if "text" in row or ("prompt" in row and "completion" in row):
        return "completion"
    return "unknown"


def _turns(row: dict, fmt: str) -> list[tuple[str, object]]:
    """Normalize a conversational row to ``[(role, content), ...]``."""
    if fmt == "openai_chat":
        return [(m.get("role", ""), m.get("content", "")) for m in row.get("messages", [])]
    if fmt == "sharegpt":
        return [(t.get("from", ""), t.get("value", "")) for t in row.get("conversations", [])]
    return []


def _validate_conversational(turns: list[tuple[str, object]]) -> Optional[str]:
    if not turns:
        return "no turns"
    body = [t for t in turns if str(t[0]).lower() != "system"]
    if not body:
        return "only system turns"
    for role, content in body:
        if not str(content).strip():
            return "empty turn content"
        if str(role).lower() not in _USER_ROLES | _ASSISTANT_ROLES:
            return f"unknown role {role!r}"
    kinds = ["u" if str(r).lower() in _USER_ROLES else "a" for r, _ in body]
    for a, b in zip(kinds, kinds[1:]):
        if a == b:
            return "consecutive same-role turns (roles must alternate)"
    if kinds[-1] != "a":
        return "conversation does not end with the assistant"
    return None


def _validate_row(row: dict, fmt: str) -> Optional[str]:
    if fmt in ("openai_chat", "sharegpt"):
        return _validate_conversational(_turns(row, fmt))
    if fmt == "alpaca":
        if not str(row.get("instruction", "")).strip():
            return "empty instruction"
        if not str(row.get("output", "")).strip():
            return "empty output"
        return None
    if fmt == "completion":
        if "text" in row:
            return None if str(row["text"]).strip() else "empty text"
        return None if str(row.get("completion", "")).strip() else "empty completion"
    return "unrecognized format (not openai_chat/sharegpt/alpaca/completion)"


@register_operator(
    name="dataset/format",
    version="0.1.0",
    capabilities=["dataset.format_check"],
)
class FormatOp(Operation[FormatInput, FormatOutput]):
    """Detect the dataset's chat format and validate per-example structure."""

    input_schema = FormatInput
    output_schema = FormatOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    config_schema = FormatConfig

    async def run(self, inp: FormatInput, ctx: OperationContext) -> FormatOutput:
        rows = load_rows(inp.path)

        sampled = inp.sample_size is not None and len(rows) > inp.sample_size
        if sampled:
            rows = _reservoir_sample(rows, inp.sample_size)

        total = len(rows)
        if inp.expected_format:
            fmt = inp.expected_format
        else:
            detected = [_detect_format(r) for r in rows]
            # Most common detected format wins; ties resolve to first-seen order.
            # Counter is O(n); the old ``max(detected, key=detected.count)`` was
            # O(n²) (a full ``list.count`` scan per row) — ~300s on a 200k-row
            # dataset, the dominant cost of the readiness check (#321 benchmark).
            # ``Counter.most_common`` preserves first-seen order on ties (3.7+).
            fmt = Counter(detected).most_common(1)[0][0] if detected else "unknown"

        findings: list[FormatFinding] = []
        conformant = 0
        for i, row in enumerate(rows):
            issue = _validate_row(row, fmt)
            if issue is None:
                conformant += 1
            else:
                findings.append(FormatFinding(row=i, issue=issue))

        compat = {t: (fmt in _TRAINER_FORMATS.get(t, set())) for t in inp.target_trainers}
        pct = round(conformant / total * 100, 1) if total else 0.0
        return FormatOutput(
            detected_format=fmt,
            total=total,
            conformant=conformant,
            conformant_pct=pct,
            findings=findings,
            trainer_compatibility=compat,
            sampled=sampled,
            sample_size=inp.sample_size if sampled else None,
        )
