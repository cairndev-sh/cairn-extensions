"""dataset/capture_to_sft — accepted/edited production signal → gold SFT rows.

ADR-0095 §1.2 (#1629), the first of three projections off the capture export.
Gold → SFT is the shortest rung of the ladder, and capture makes it self-serve:
what a human edited IS the target, written by the person who knows.

Two rules the shape enforces:

* **An edit outranks the completion it corrected.** ``edited_text`` becomes the
  assistant turn; the original is not written here at all (it earns its keep as
  the ``rejected`` half in ``dataset/capture_to_pairs``).
* **Nothing but ``messages`` lands on a training row.** The attribution
  (model/adapter) is reported in the OUTPUT, not stamped onto rows — an unknown
  key can break a trainer's chat formatter, the same reason ``dataset/replay``
  keeps its provenance in ``mix``.

D4: this reads the EXPORTED file, never the capture store. The export is where
the consent gate and the deterministic holdout split already ran, so a job box
with no database credentials still gets governed data.
"""
from __future__ import annotations

from pathlib import Path

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks._capture import (
    attribution,
    context_by_correlation,
    correlation,
    iter_capture,
    row_hash,
    text,
)
from operator_dataset_checks._io import JsonlWriter
from operator_dataset_checks.schemas.capture_to_sft import (
    AdapterCount,
    CaptureToSftInput,
    CaptureToSftOutput,
)

# D6 payload budget: findings carry COUNTS and adapter URIs, never row text, so
# the output stays a handful of small records however large the corpus is.

#: The kinds that endorse a completion. `trace` is not one of them — an
#: unreviewed production completion is not gold, it is just output.
_GOLD_KINDS = ("edit", "accept")


def adapter_counts(tally: dict[tuple[str, str], int]) -> list[AdapterCount]:
    """Deterministic attribution readout, shared by all three projections."""
    return [AdapterCount(adapter_uri=a, model=m, rows=n)
            for (a, m), n in sorted(tally.items())]


@register_operator(
    name="dataset/capture_to_sft",
    version="0.1.0",
    capabilities=["dataset.capture_to_sft"],
)
class CaptureToSftOp(Operation[CaptureToSftInput, CaptureToSftOutput]):
    """Project captured accepts/edits into chat-formatted SFT rows."""

    input_schema = CaptureToSftInput
    output_schema = CaptureToSftOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    # #1539 — governance declarations; the [operator.*] table is generated from these
    risk = 'external_write'

    async def run(self, inp: CaptureToSftInput, ctx: OperationContext) -> CaptureToSftOutput:
        wanted = _GOLD_KINDS if inp.include_accepts else ("edit",)
        # a feedback row usually carries only a correlation id — the prompt (and
        # the completion an `accept` endorsed) live on the trace it points at.
        context = context_by_correlation(inp.path)
        seen: set[str] = set()
        tally: dict[tuple[str, str], int] = {}
        from_edits = from_accepts = duplicates = skipped = 0

        with JsonlWriter(Path(inp.out_path)) as writer:
            # source order is the export's order (created_at, event_id), so the
            # output is deterministic without a sort — and it streams.
            for row in iter_capture(inp.path):
                kind = row.get("kind")
                attr = attribution(row)
                if kind not in wanted or (inp.only_adapter
                                          and attr["adapter_uri"] != inp.only_adapter):
                    skipped += 1
                    continue
                ctx_row = context.get(correlation(row), {})
                prompt = text(row, "prompt") or ctx_row.get("prompt", "")
                gold = (text(row, "edited_text") if kind == "edit"
                        else text(row, "completion") or ctx_row.get("completion", ""))
                if not prompt or not gold:
                    skipped += 1
                    continue

                messages = []
                if inp.system_prompt:
                    messages.append({"role": "system", "content": inp.system_prompt})
                messages += [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": gold}]
                example = {"messages": messages}

                digest = row_hash(example)
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                writer.write(example)
                if kind == "edit":
                    from_edits += 1
                else:
                    from_accepts += 1
                key = (attr["adapter_uri"], attr["model"])
                tally[key] = tally.get(key, 0) + 1

        return CaptureToSftOutput(
            out_path=inp.out_path, rows=from_edits + from_accepts,
            from_edits=from_edits, from_accepts=from_accepts,
            duplicates_dropped=duplicates, skipped=skipped,
            by_adapter=adapter_counts(tally),
        )
