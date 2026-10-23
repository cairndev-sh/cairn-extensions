"""dataset/capture_to_pairs — captured signal → DPO preference pairs.

ADR-0095 §1.2 (#1629). Emits the shape the DPO rung already eats —
``{prompt, chosen, rejected, kind}``, matching
``operator_training.flow_stages.dataset_dpo_pairs`` — so ``train.dpo`` consumes
capture-derived preferences with nothing in between.

Two sources, in descending order of how much they are worth:

* **edit** — the human wrote the better answer next to the worse one. Nothing
  else in the store is this clean: ``chosen`` and ``rejected`` come from the
  same row, so there is no matching to get wrong.
* **reject / retry** — a verdict with no alternative attached. It becomes a pair
  only when a LATER row on the same ``trace_id``/``run_id`` carries text a human
  endorsed. With no successor the row is DROPPED and counted, never paired
  against a guess: a preference pair whose ``chosen`` was invented teaches the
  model to prefer the invention.

Correlation is per-group and order-sensitive, so rows are grouped in file order
(the export's ``created_at`` order) and scanned forward. Output order is
group-then-position, which makes the file a deterministic function of its input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks._capture import (
    attribution,
    correlation,
    gold_of,
    iter_capture,
    row_hash,
    text,
)
from operator_dataset_checks._io import JsonlWriter
from operator_dataset_checks.operators.capture_to_sft import adapter_counts
from operator_dataset_checks.schemas.capture_to_pairs import (
    CaptureToPairsInput,
    CaptureToPairsOutput,
)

# D6 payload budget: counts and adapter URIs only — never a prompt or a
# completion in the findings.

#: Kinds whose text a later row may supply as `chosen`. A `trace` counts because
#: a retry that ended in a normal completion IS the better outcome; an unendorsed
#: trace only ever becomes `chosen` when something before it was rejected.
_ENDORSING = ("edit", "accept", "trace")


def _prompt_for(row: dict[str, Any], group: list[dict[str, Any]]) -> str:
    """The prompt this row is about — its own, else the group's first one.

    A feedback event often carries only a correlation id (the SDK helper does not
    ask an application to re-send the prompt it already traced), so the trace row
    in the same group is where the text lives.
    """
    own = text(row, "prompt")
    if own:
        return own
    for other in group:
        got = text(other, "prompt")
        if got:
            return got
    return ""


@register_operator(
    name="dataset/capture_to_pairs",
    version="0.1.0",
    capabilities=["dataset.capture_to_pairs"],
)
class CaptureToPairsOp(Operation[CaptureToPairsInput, CaptureToPairsOutput]):
    """Mine ``{prompt, chosen, rejected}`` preference pairs from capture."""

    input_schema = CaptureToPairsInput
    output_schema = CaptureToPairsOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    # #1539 — governance declarations; the [operator.*] table is generated from these
    risk = 'external_write'

    async def run(self, inp: CaptureToPairsInput, ctx: OperationContext) -> CaptureToPairsOutput:
        # group in file order; a dict preserves first-seen order, which is what
        # makes the output deterministic without a sort key of its own.
        groups: dict[str, list[dict[str, Any]]] = {}
        skipped = 0
        for row in iter_capture(inp.path):
            if inp.only_adapter and attribution(row)["adapter_uri"] != inp.only_adapter:
                skipped += 1
                continue
            corr = correlation(row)
            if not corr:
                skipped += 1
                continue
            groups.setdefault(corr, []).append(row)

        seen: set[str] = set()
        tally: dict[tuple[str, str], int] = {}
        from_edits = from_rejects = from_retries = unpaired = duplicates = 0

        with JsonlWriter(Path(inp.out_path)) as writer:
            for group in groups.values():
                for i, row in enumerate(group):
                    kind = row.get("kind")
                    if kind == "edit":
                        chosen = text(row, "edited_text")
                        rejected = text(row, "completion") or _earlier_completion(group, i)
                    elif kind in ("reject", "retry") and inp.pair_rejects:
                        rejected = text(row, "completion") or _earlier_completion(group, i)
                        chosen = _later_endorsed(group, i)
                        if rejected and not chosen:
                            unpaired += 1
                            continue
                    else:
                        skipped += 1
                        continue

                    prompt = _prompt_for(row, group)
                    if not prompt or not chosen or not rejected or chosen == rejected:
                        skipped += 1
                        continue

                    pair = {"prompt": prompt, "chosen": chosen, "rejected": rejected,
                            "kind": str(kind)}
                    digest = row_hash(pair)
                    if digest in seen:
                        duplicates += 1
                        continue
                    seen.add(digest)
                    writer.write(pair)
                    if kind == "edit":
                        from_edits += 1
                    elif kind == "reject":
                        from_rejects += 1
                    else:
                        from_retries += 1
                    attr = attribution(row)
                    key = (attr["adapter_uri"], attr["model"])
                    tally[key] = tally.get(key, 0) + 1

        return CaptureToPairsOutput(
            out_path=inp.out_path, rows=from_edits + from_rejects + from_retries,
            from_edits=from_edits, from_rejects=from_rejects, from_retries=from_retries,
            unpaired_dropped=unpaired, duplicates_dropped=duplicates, skipped=skipped,
            by_adapter=adapter_counts(tally),
        )


def _earlier_completion(group: list[dict[str, Any]], i: int) -> str:
    """The completion this feedback is about, when the feedback row itself does
    not repeat it: the nearest preceding row that carries one."""
    for row in reversed(group[:i]):
        got = text(row, "completion")
        if got:
            return got
    return ""


def _later_endorsed(group: list[dict[str, Any]], i: int) -> str:
    """The first human-endorsed text AFTER position ``i`` — what the rejection
    was eventually resolved with. Forward-only: text that existed BEFORE the
    rejection is what got rejected."""
    for row in group[i + 1:]:
        if row.get("kind") in _ENDORSING:
            got = gold_of(row)
            if got:
                return got
    return ""
