"""dataset/profile — lightweight, deterministic dataset profile.

Reports the assistant/output **response-length** distribution, the **refusal
ratio** (a tunable substring heuristic — too many refusals skews fine-tuning),
and the count of **short** responses (likely low-signal). Pure Python; a rich
statistical profile (distributions, correlations, representativeness) via
ydata-profiling is the deferred ``[profile]`` extra.

Issue #381 adds an OPTIONAL second length dimension: real subword **token**
counts via HF ``tokenizers`` (``token_backend='tokenizers'``/``'auto'``),
alongside the default zero-dep whitespace word count — char length alone
under/over-estimates what a model's context window actually consumes.
"""
from __future__ import annotations

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks._io import RowErrors, load_rows, _reservoir_sample
from operator_dataset_checks._tokenizer import count as token_count
from operator_dataset_checks._tokenizer import resolve as resolve_token_backend
from operator_dataset_checks.schemas.profile import (
    LengthStats,
    ProfileConfig,
    ProfileInput,
    ProfileOutput,
)

_DEFAULT_REFUSAL_MARKERS = [
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "as an ai",
]
_ASSISTANT_ROLES = {"assistant", "gpt"}


def _response_text(row: dict) -> str:
    """The model's answer for this example, across supported formats."""
    if isinstance(row.get("messages"), list):
        replies = [
            m.get("content", "") for m in row["messages"]
            if str(m.get("role", "")).lower() in _ASSISTANT_ROLES
        ]
        return str(replies[-1]) if replies else ""
    if isinstance(row.get("conversations"), list):
        replies = [
            t.get("value", "") for t in row["conversations"]
            if str(t.get("from", "")).lower() in _ASSISTANT_ROLES
        ]
        return str(replies[-1]) if replies else ""
    if "output" in row:
        return str(row["output"])
    if "completion" in row:
        return str(row["completion"])
    if "text" in row:
        return str(row["text"])
    return ""


def _stats(values: list[int]) -> LengthStats:
    if not values:
        return LengthStats(min=0, max=0, mean=0.0, p50=0)
    s = sorted(values)
    n = len(s)
    return LengthStats(min=s[0], max=s[-1], mean=round(sum(s) / n, 1), p50=s[n // 2])


@register_operator(
    name="dataset/profile",
    version="0.2.0",
    capabilities=["dataset.profile"],
)
class ProfileOp(Operation[ProfileInput, ProfileOutput]):
    """Profile response lengths, refusal ratio, and short-response count."""

    input_schema = ProfileInput
    output_schema = ProfileOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    config_schema = ProfileConfig

    async def run(self, inp: ProfileInput, ctx: OperationContext) -> ProfileOutput:
        markers = [m.lower() for m in (inp.refusal_markers or _DEFAULT_REFUSAL_MARKERS)]
        backend_used = resolve_token_backend(inp.token_backend)
        # Only pay the tokenizer-load cost when the caller actually asked for
        # it — 'whitespace' (the default) never imports tokenizers.
        want_tokens = backend_used == "tokenizers"

        # #1008 — a malformed line is a readiness FINDING, not a crash: it lands
        # in the quarantine side file and is surfaced in the profile as a count.
        row_errors = RowErrors()
        rows = load_rows(inp.path, strict=inp.strict, errors=row_errors)

        sampled = inp.sample_size is not None and len(rows) > inp.sample_size
        if sampled:
            rows = _reservoir_sample(rows, inp.sample_size)

        lengths: list[int] = []
        token_lengths: list[int] = []
        refusals = 0
        short = 0
        for row in rows:
            resp = _response_text(row)
            lengths.append(len(resp))
            if want_tokens:
                token_lengths.append(
                    token_count(
                        resp,
                        backend=backend_used,
                        tokenizer_name=inp.tokenizer_name,
                        tokenizer_path=inp.tokenizer_path,
                    )
                )
            if len(resp) < inp.short_threshold:
                short += 1
            low = resp.lower()
            if any(m in low for m in markers):
                refusals += 1

        total = len(rows)
        refusal_pct = round(refusals / total * 100, 1) if total else 0.0
        return ProfileOutput(
            total=total,
            response_length_chars=_stats(lengths),
            refusals=refusals,
            refusal_pct=refusal_pct,
            short_responses=short,
            response_length_tokens=_stats(token_lengths) if want_tokens else None,
            token_backend_used=backend_used if want_tokens else None,
            sampled=sampled,
            sample_size=inp.sample_size if sampled else None,
            quarantined_rows=row_errors.count,
            quarantine_path=row_errors.quarantine_path,
        )
