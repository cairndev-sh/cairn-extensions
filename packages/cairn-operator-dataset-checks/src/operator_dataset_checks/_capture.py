"""Shared reading of exported capture rows (#1629, ADR-0095).

The three ``dataset/capture_to_*`` projections all consume the SAME artifact:
the JSONL ``POST /capture/export`` registers as ``dataset:capture-signal@N``,
one serialized ``CaptureEvent`` per line. They must agree on what a row means —
which field is the prompt, when an ``edit`` supersedes the completion it
corrects — so the vocabulary lives here and all three import it, exactly as
``dataset/dedup`` and ``dataset/decontam`` share ``_canon``.

**D4: an operator never touches the capture STORE.** These are pure functions
over a file. The store is control-plane-only (a job box has no database
credentials), and the export is also where the consent gate and the deterministic
holdout split already happened — so reading the file is not a workaround, it is
the governed path.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator

from operator_dataset_checks._io import iter_rows

#: Kinds a projection may see. Anything else in the file is skipped and counted
#: rather than guessed at — a projection that invents a meaning for an unknown
#: kind would put rows nobody sanctioned into a training set.
KNOWN_KINDS = ("trace", "accept", "reject", "edit", "retry", "score")


def iter_capture(path: str) -> Iterator[dict[str, Any]]:
    """Stream exported capture rows. Non-dict lines are skipped."""
    for row in iter_rows(strip_pin(path)):
        if isinstance(row, dict):
            yield row


def strip_pin(source: str) -> str:
    """Drop the ``#sha256=<hash>`` pin a launch-resolved ``dataset:`` ref carries
    (#1466), the way the fetching kernels do. The pin is a RETRIEVAL rule for
    whoever fetches the bytes; a reader just needs the location."""
    return source.partition("#")[0]


def text(row: dict[str, Any], field: str) -> str:
    v = row.get(field)
    return v.strip() if isinstance(v, str) else ""


def correlation(row: dict[str, Any]) -> str:
    """The id a row's siblings share. ``trace_id`` wins because it identifies
    ONE completion; ``run_id`` can cover many."""
    return text(row, "trace_id") or text(row, "run_id")


def gold_of(row: dict[str, Any]) -> str:
    """What the human was willing to ship for this prompt.

    An ``edit`` supersedes the completion it corrects — that is the whole point
    of capturing edits, and it is why an edit is the richest row in the store.
    An ``accept`` endorses the completion as it stands.
    """
    if row.get("kind") == "edit":
        return text(row, "edited_text")
    return text(row, "completion")


def context_by_correlation(path: str) -> dict[str, dict[str, str]]:
    """First non-empty ``prompt``/``completion`` per correlation id.

    A feedback event usually carries ONLY a correlation id: the SDK helper does
    not ask an application to re-send the prompt it already traced, and an
    ``accept`` is often a bare thumbs-up. The trace row in the same group is
    where the text lives, so a projection resolves through this map rather than
    dropping the richest half of the corpus for want of a field.

    First-wins rather than last-wins: within one correlation the trace comes
    first (the export is ordered by ``created_at``), and it is the trace that
    holds the ORIGINAL prompt and completion — which is what an ``accept``
    endorsed and what an ``edit`` corrected.
    """
    out: dict[str, dict[str, str]] = {}
    for row in iter_capture(path):
        corr = correlation(row)
        if not corr:
            continue
        got = out.setdefault(corr, {"prompt": "", "completion": ""})
        for field in ("prompt", "completion"):
            if not got[field]:
                got[field] = text(row, field)
    return out


def attribution(row: dict[str, Any]) -> dict[str, str]:
    """Which model/adapter the row is evidence ABOUT. Carried onto every
    projected row so a training set can be traced back to the cycle it is
    complaining about (ADR-0095 capture decision 1)."""
    return {"model": text(row, "model"), "adapter_uri": text(row, "adapter_uri")}


def row_hash(payload: dict[str, Any]) -> str:
    """Canonical hash of a PROJECTED row — the dedup key inside one output.
    Separate from ``_canon.canonical_hash`` only in that it never normalizes
    whitespace: an edit that fixed spacing is a real correction, not a
    duplicate."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


__all__ = ["KNOWN_KINDS", "attribution", "context_by_correlation", "correlation",
           "gold_of", "iter_capture", "row_hash", "strip_pin", "text"]
