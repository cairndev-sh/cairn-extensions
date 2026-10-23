"""dataset/curate — LLM-judge-driven quality selection (#384).

Curates a fine-tuning dataset by asking an LLM to rate each example's quality
and keeping only the examples that clear the bar. Two industry-standard methods:

- **AlpaGasus** (arXiv:2307.08701): prompt the judge for a single 0-5 quality
  score per (instruction, response); keep rows scoring >= a threshold (paper
  default 4.5) or the top ``keep_fraction``.
- **DEITA** (arXiv:2312.15685): score complexity AND quality (each 0-5) and
  select by their product ``complexity * quality``.

The judge runs through the governed chokepoint ``cairn.core.llm.client.llm_complete``
(ADR-0031) — the SAME governed-gateway judging mechanism eval/judge uses, but
implemented in-pack (operator packs own their impl; they never import another
operator pack). Redaction, budget, cost and audit apply ambiently to every call.

Non-deterministic and paid: risk ``paid_compute``. Writes a curated JSONL of the
kept rows and a report of the score distribution / kept-vs-dropped counts.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from cairn.core.llm.client import llm_complete
from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks._extract import extract_io
from operator_dataset_checks._io import load_rows
from operator_dataset_checks.schemas.curate import (
    CurateInput,
    CurateOutput,
    RowScore,
)

# One clamped 0-5 score. Robust to LLM prose: pull the FIRST number out of the
# reply and clamp to [0, 5]. Handles "Score: 4.5", "4", "I'd rate this 3.5/5".
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

_ALPAGASUS_RUBRIC = (
    "Rate the QUALITY of the AI response to the given instruction on a scale of "
    "0 to 5, where 0 is unhelpful/incorrect and 5 is maximally helpful, accurate "
    "and well-formed. Consider helpfulness, accuracy, and clarity."
)

_DEITA_QUALITY_RUBRIC = (
    "Rate the QUALITY of the AI response to the given instruction on a scale of "
    "0 to 5, where 0 is unhelpful/incorrect and 5 is maximally helpful, accurate "
    "and well-formed."
)
_DEITA_COMPLEXITY_RUBRIC = (
    "Rate the COMPLEXITY (difficulty / reasoning depth) of the instruction on a "
    "scale of 0 to 5, where 0 is trivial and 5 is highly complex."
)


def _clamp(value: float, lo: float = 0.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, value))


def _parse_score(text: str) -> float | None:
    """Extract the first numeric score from an LLM reply, clamped to [0, 5].

    Returns ``None`` when no number is present (counted as a parse failure).
    """
    m = _NUMBER.search(text or "")
    if not m:
        return None
    try:
        return _clamp(float(m.group()))
    except ValueError:  # pragma: no cover — regex guarantees a float-parseable match
        return None


def _prompt(rubric: str, instruction: str, response: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"{rubric}\n\nReply with ONLY the numeric score (e.g. '4' or "
                "'3.5'). Do not explain."
            ),
        },
        {
            "role": "user",
            "content": f"### Instruction:\n{instruction}\n\n### Response:\n{response}",
        },
    ]


@register_operator(
    name="dataset/curate",
    version="0.1.0",
    capabilities=["dataset.curate"],
)
class CurateOp(Operation[CurateInput, CurateOutput]):
    """LLM-judge quality selection over a fine-tuning dataset (AlpaGasus / DEITA)."""

    input_schema = CurateInput
    output_schema = CurateOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    # #1539 — governance declarations; the [operator.*] table is generated from these
    risk = 'paid_compute'

    async def run(self, inp: CurateInput, ctx: OperationContext) -> CurateOutput:
        method = (inp.method or "alpagasus").lower()
        if method not in ("alpagasus", "deita"):
            raise ValueError(
                f"unknown curate method {inp.method!r} (expected 'alpagasus' | 'deita')"
            )

        rows = load_rows(inp.path)
        # Keep the original row alongside its extracted (input, output) so kept
        # rows are written back verbatim. Rows with no clear split are skipped.
        pairs: list[tuple[dict, str, str]] = []
        for row in rows:
            io = extract_io(row)
            if io is not None:
                pairs.append((row, io[0], io[1]))

        capped = pairs[: inp.max_samples] if inp.max_samples else pairs

        sem = asyncio.Semaphore(inp.concurrency)

        async def _score(instruction: str, response: str) -> tuple[float | None, float | None, float | None]:
            """Return (select_score, quality, complexity); Nones on parse failure."""
            async with sem:
                if method == "alpagasus":
                    rubric = inp.rubric or _ALPAGASUS_RUBRIC
                    resp = await llm_complete(
                        messages=_prompt(rubric, instruction, response),
                        model=inp.model,
                        temperature=inp.temperature,
                    )
                    q = _parse_score(_content(resp))
                    return (q, q, None)
                # DEITA: two independent judge calls (quality + complexity).
                q_resp, c_resp = await asyncio.gather(
                    llm_complete(
                        messages=_prompt(
                            inp.rubric or _DEITA_QUALITY_RUBRIC, instruction, response
                        ),
                        model=inp.model,
                        temperature=inp.temperature,
                    ),
                    llm_complete(
                        messages=_prompt(
                            _DEITA_COMPLEXITY_RUBRIC, instruction, response
                        ),
                        model=inp.model,
                        temperature=inp.temperature,
                    ),
                )
                quality = _parse_score(_content(q_resp))
                complexity = _parse_score(_content(c_resp))
                if quality is None or complexity is None:
                    return (None, quality, complexity)
                return (quality * complexity, quality, complexity)

        results = await asyncio.gather(
            *(_score(i, o) for _, i, o in capped)
        )

        parse_failures = 0
        scored: list[tuple[int, dict, float, float | None, float | None]] = []
        for idx, ((row, instruction, _out), (sel, quality, complexity)) in enumerate(
            zip(capped, results)
        ):
            if sel is None:
                parse_failures += 1
                ctx.logger.warning(
                    "dataset/curate: row %d score unparseable, skipping", idx
                )
                continue
            scored.append((idx, row, sel, quality, complexity))

        # Selection criterion: keep_fraction wins over threshold; else default.
        vals = sorted((s for _, _, s, _, _ in scored), reverse=True)
        if inp.keep_fraction is not None:
            n_keep = int(len(vals) * inp.keep_fraction + 0.5)
            # Cutoff = score of the n_keep-th best row; ties above cutoff kept.
            cutoff = vals[n_keep - 1] if 0 < n_keep <= len(vals) else float("inf")
            keep = (lambda s: s >= cutoff) if n_keep > 0 else (lambda s: False)
            criterion = f"keep_fraction={inp.keep_fraction}"
        else:
            thr = inp.threshold if inp.threshold is not None else 4.5
            keep = lambda s: s >= thr  # noqa: E731 — tiny local predicate
            criterion = f"threshold>={thr}"

        kept_rows: list[dict] = []
        records: list[RowScore] = []
        distribution: dict[str, int] = {}
        kept = 0
        for idx, row, sel, quality, complexity in scored:
            is_kept = keep(sel)
            if is_kept:
                kept += 1
                kept_rows.append(row)
            bucket = str(int(sel))
            distribution[bucket] = distribution.get(bucket, 0) + 1
            io = extract_io(row)
            preview = (io[0][:120] if io else "")
            records.append(
                RowScore(
                    index=idx,
                    score=round(sel, 4),
                    quality=round(quality, 4) if quality is not None else None,
                    complexity=round(complexity, 4) if complexity is not None else None,
                    kept=is_kept,
                    input_preview=preview,
                )
            )

        out_path = Path(inp.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in kept_rows),
            encoding="utf-8",
        )

        n = len(scored)
        mean = sum(s for _, _, s, _, _ in scored) / n if n else 0.0
        return CurateOutput(
            method=method,
            model=inp.model,
            criterion=criterion,
            total_rows=len(rows),
            extractable=len(pairs),
            scored=n,
            kept=kept,
            dropped=n - kept,
            parse_failures=parse_failures,
            mean_score=round(mean, 4),
            score_distribution=distribution,
            scores=records,
            output_path=str(out_path),
        )


def _content(resp) -> str:
    """Pull the assistant text out of a litellm/openai-shaped response."""
    try:
        return resp.choices[0].message.content or ""
    except Exception:  # noqa: BLE001 — be liberal about response shape
        return str(resp)
