"""Schemas for dataset/anonymize — redact PII from a dataset."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class AnonymizeInput(BaseModel):
    path: str = Field(..., description="Path to the source JSONL dataset.")
    output_path: str = Field(..., description="Where to write the redacted JSONL dataset.")
    backend: str = Field(
        "auto",
        description=(
            "'auto' (presidio if installed, else regex) | 'regex' | "
            "'regexset' (single-pass re2, [fastpii] extra) | 'presidio' | 'cascade'."
        ),
    )
    preset: Literal["governed", "logs"] = Field(
        "governed",
        description=(
            "Ready-made redaction configuration; settings below override it. "
            "'governed' (default) = NER + regex + secrets (redacts names, locations, "
            "emails, phones, cards, API keys). 'logs' = regex + secrets only (no "
            "name/location redaction) — faster/lighter, for log/trace data. Redaction "
            "always processes every row (it cannot sample)."
        ),
    )
    ner: Optional[bool] = Field(
        None,
        description=(
            "Redact names & locations (PERSON/LOCATION/ORGANIZATION) via the language "
            "model. OFF = redact only pattern PII (emails, phones, cards, secrets) — "
            "faster/lighter but names are left in. Keep ON for training data. null = "
            "preset's value."
        ),
    )
    model: Optional[Literal["blank", "sm", "lg"]] = Field(
        None,
        description=(
            "spaCy model used to find names/locations to redact (ignored when NER is "
            "off). 'sm' = fast, ~550 MB; 'lg' = highest recall, ~1.2-1.7 GB; 'blank' = "
            "no NER. null = preset's model."
        ),
    )
    n_process: Optional[int] = Field(
        None,
        description=(
            "CPU workers sharing the redaction pass. 0 = auto (all cores; ~40% faster "
            "on 2 CPUs). 1 = single process (lowest RAM). Capped at the core count. "
            "null = preset's value."
        ),
    )
    analyze_chunk: Optional[int] = Field(
        None,
        description=(
            "Rows fed to the detector at once. Larger = parallel workers stay busy; "
            "smaller = less memory. ~2048 is a good default. null = preset's value."
        ),
    )
    kinds: Optional[list[str]] = Field(
        None, description="Subset of recognizers/entities to redact (default: all).",
    )
    text_fields: Optional[list[str]] = Field(
        None,
        description=(
            "Top-level field name(s) holding the training TEXT to redact (e.g. "
            "['source_text'] or ['messages']). null = auto: redact free-text leaves "
            "in every field, skipping short whitespace-free strings (tokens, labels, "
            "ids, chat roles) so a tokenized/labelled dataset isn't NER'd per token "
            "(#1021). Non-selected fields are copied through untouched."
        ),
    )


class AnonymizeConfig(BaseModel):
    """The deployment-tunable surface of ``dataset/anonymize`` (ADR-0068/0070).

    A subset of ``AnonymizeInput`` — just ``kinds``, tagged **governance** (ADR-0070):
    *which* PII entity kinds must be redacted is a compliance/privacy policy, not a
    per-run tuning knob. Like a gate threshold it is **env-owned and never weakenable**
    by a workflow or run — a run cannot NARROW the redaction set (redact fewer kinds =
    weaker privacy). The strict floor is the default ``None`` = *redact every recognized
    kind* (on ``AnonymizeInput``); env config may pin an explicit subset per environment.

    NOTE (the ADR-0070 clobber rule): ``kinds`` is intentionally left UNBOUND in the
    ``run-prep-anonymize`` operation YAML so the config layer supplies it — a baked
    ``$default($input.anon_kinds, …)`` binding would win over config and defeat the
    governance guarantee. ``backend`` stays a plain operational input (not governed)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"scope": "governance"})

    kinds: Optional[list[str]] = Field(
        None,
        title="PII kinds to redact",
        description=(
            "The recognizer/entity kinds that must be redacted (null = every recognized "
            "kind — the strict default). Env-owned governance: a workflow/run cannot narrow it."
        ),
    )
    preset: Literal["governed", "logs"] = Field(
        "governed", title="Redaction mode",
        description=(
            "Ready-made redaction configuration.\n"
            "• governed (default) — NER + regex + secrets: redacts names, locations, "
            "emails, phones, cards, and API keys. Best coverage.\n"
            "• logs — regex + secrets only (no name/location redaction). Faster and "
            "lighter; use for log/trace data with no personal names.\n"
            "Redaction always processes every row (it cannot sample), so mode here "
            "mainly trades coverage for speed. Env-owned (governance): a run cannot "
            "silently weaken it."
        ),
    )
    ner: bool = Field(
        True, title="Redact names & locations (NER)",
        description=(
            "Redact PERSON / LOCATION / ORGANIZATION detected by the language model. "
            "Turn OFF to redact only pattern-based PII (emails, phones, cards, "
            "secrets) — faster and lighter, but names and places are left in the data. "
            "Keep ON for training datasets. Env-owned (governance)."
        ),
    )
    model: Literal["blank", "sm", "lg"] = Field(
        "sm", title="Language model",
        description=(
            "spaCy model used to find names/locations to redact (ignored when NER is "
            "off). sm = fast, ~550 MB, small miss rate; lg = highest recall, "
            "~1.2-1.7 GB; blank = no NER. Env-owned (governance)."
        ),
    )
    n_process: int = Field(
        0, title="Parallel workers",
        description=(
            "CPU workers sharing the redaction pass. 0 = auto (all cores; ~40% faster "
            "on 2 CPUs). Set 1 for lowest RAM when the box runs other services. "
            "Capped at the core count."
        ),
    )
    analyze_chunk: int = Field(
        2048, title="Rows per batch",
        description=(
            "Rows fed to the detector at once. Larger = parallel workers stay busy; "
            "smaller = less memory. 2048 is a good default."
        ),
    )
    backend: str = Field(
        "auto", title="PII backend",
        description=(
            "Detector engine: 'auto' (cascade when presidio installed, else regex) "
            "fits almost every case. Advanced: 'cascade', 'presidio', 'regex', "
            "'regexset'."
        ),
    )


class AnonymizeOutput(BaseModel):
    backend_used: str = Field(
        ..., description="The backend actually used (regex | regexset | presidio).",
    )
    total: int = Field(..., description="Total rows processed.")
    rows_redacted: int = Field(..., description="Rows where at least one value was redacted.")
    counts: dict[str, int] = Field(
        default_factory=dict, description="Per-kind total redactions across the dataset.",
    )
    output_path: str = Field(..., description="Path the redacted dataset was written to.")
