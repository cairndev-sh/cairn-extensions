"""Schemas for dataset/pii_scan — PII / secret density over a dataset."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PiiFinding(BaseModel):
    row: int = Field(..., description="0-based row index.")
    kinds: list[str] = Field(..., description="PII/secret kinds detected in this row.")


class PiiInput(BaseModel):
    path: str = Field(
        ..., description="Path to a JSONL fine-tuning dataset (one example per line).",
    )
    kinds: Optional[list[str]] = Field(
        None,
        description="Subset of recognizers/entities to run (default: all available).",
    )
    text_fields: Optional[list[str]] = Field(
        None,
        description=(
            "Top-level field name(s) that hold the training TEXT to scan (e.g. "
            "['source_text'] or ['messages']). null = auto: scan every field but "
            "keep only free-text leaves — short whitespace-free strings (tokens, "
            "BIO labels, ids, chat roles) are skipped so a tokenized/labelled "
            "dataset doesn't run the NER once per token (#1021)."
        ),
    )
    backend: str = Field(
        "auto",
        description=(
            "'auto' (presidio if installed, else regex) | 'regex' | "
            "'regexset' (single-pass re2, [fastpii] extra) | 'presidio' | "
            "'cascade' (regex-in-presidio, ADR-0079)."
        ),
    )
    preset: Literal["governed", "logs"] = Field(
        "governed",
        description=(
            "Ready-made PII-scan configuration; individual settings below override "
            "it. 'governed' (default) = spaCy NER + regex + secrets — finds names, "
            "locations, emails, phones, cards, API keys (best recall; ~7 min/200k "
            "rows on a 2-CPU box, ~550 MB). 'logs' = regex + secrets only, NO "
            "name/location detection (NER off) — faster and ~200 MB; use for logs, "
            "traces, and incident data with no personal names."
        ),
    )
    ner: Optional[bool] = Field(
        None,
        description=(
            "Detect names & locations (PERSON/LOCATION/ORGANIZATION) with the spaCy "
            "model — the PII that has no fixed pattern. Turn OFF for regex + secrets "
            "only: much faster and lighter, but names and places are NOT found. Leave "
            "ON for training data; off is only safe for structured/log data. null = "
            "use the preset's value."
        ),
    )
    model: Optional[Literal["blank", "sm", "lg"]] = Field(
        None,
        description=(
            "Language model powering name/location detection (ignored when NER is "
            "off). 'sm' = fast, ~550 MB, misses ~1-2% of names (best on a small box); "
            "'lg' = highest recall, ~1.2-1.7 GB, same speed; 'blank' = tokenizer only "
            "(no NER). null = use the preset's model."
        ),
    )
    n_process: Optional[int] = Field(
        None,
        description=(
            "CPU workers sharing the detection pass. 0 = auto (all cores) — on a "
            "2-CPU box ~40% faster than a single worker. 1 = single process (lowest "
            "RAM, e.g. when the box runs other heavy services). Values above the core "
            "count are capped automatically. null = use the preset's value."
        ),
    )
    analyze_chunk: Optional[int] = Field(
        None,
        description=(
            "Rows fed to the detector at once. Larger batches keep the parallel "
            "workers busy (needed for 'n_process' to actually help); smaller batches "
            "use less memory. ~2048 is a good balance — you rarely need to change it. "
            "null = use the preset's value."
        ),
    )
    sample_size: Optional[int] = Field(
        1000,
        description=(
            "Reservoir-sample this many rows before checking — a statistically "
            "valid estimate for a readiness gate that keeps the scan O(1) in "
            "dataset size (the #737 safe cap; presidio-per-row on an unbounded HF "
            "dataset can wedge a small box). Set null to scan every row. "
            "Deployment/workflow-tunable via PiiScanConfig (ADR-0070)."
        ),
    )
    max_findings: int = Field(
        1000,
        description=(
            "Cap the enumerated `findings` list at this many rows so the REPORT "
            "stays bounded on a heavily-PII, >>RAM corpus (#891) — `counts` and "
            "`rows_with_pii` remain EXACT (every row is still scanned), only the "
            "per-row detail list is capped (`findings_truncated` flags it). 0 = "
            "unbounded (small-file convenience)."
        ),
    )
    secrets_full_scan: bool = Field(
        True,
        description=(
            "ADR-0080 P1 — the SECRET tier (AWS/OpenAI/GitHub keys, JWTs, private "
            "keys) scans EVERY row with cheap regex, even when NER density is "
            "sampled: a rare leaked key would slip past a 1000-row sample ~90% of "
            "the time. Merged into `counts` in the gate's blocking vocabulary. Set "
            "false only for a pure density estimate with no absolute-blocker guarantee."
        ),
    )


class PiiScanConfig(BaseModel):
    """The deployment-tunable surface of ``dataset/pii_scan`` (ADR-0068/0070) — the
    scan bound + backend. **Operational** scope (the default): a workflow or run may
    tune these freely (unlike a governance gate threshold). A subset of ``PiiInput``;
    the safe defaults live on ``PiiInput`` so an un-configured run is still bounded."""

    model_config = ConfigDict(extra="forbid")

    preset: Literal["governed", "logs"] = Field(
        "governed", title="PII scan mode",
        description=(
            "Pick a ready-made configuration, then override individual settings "
            "below if needed.\n"
            "• governed (default) — spaCy NER + regex + secret detection. Finds "
            "names, locations, emails, phones, cards, API keys. Best recall; "
            "~7 min / 200k rows on a 2-CPU box, ~550 MB RAM.\n"
            "• logs — regex + secrets only, NO name/location detection (NER off). "
            "~30% faster and ~200 MB RAM. Use for logs, traces, and incident data "
            "where there are no personal names to catch."
        ),
    )
    ner: bool = Field(
        True, title="Detect names & locations (NER)",
        description=(
            "Run the spaCy language model to catch PERSON, LOCATION, and "
            "ORGANIZATION — the PII that has no fixed pattern. Turn OFF for regex + "
            "secrets only: much faster and lighter, but names and places will NOT be "
            "found. Leave ON for training data; turning it off is only safe for "
            "structured/log data."
        ),
    )
    model: Literal["blank", "sm", "lg"] = Field(
        "sm", title="Language model",
        description=(
            "Which spaCy model powers name/location detection (ignored when NER is "
            "off).\n"
            "• sm (small, default) — fast, ~550 MB RAM, misses ~1-2% of names. Best "
            "fit for a small (2-CPU) box.\n"
            "• lg (large) — highest recall, ~1.2-1.7 GB RAM, same speed. Use where "
            "RAM is plentiful and you need to catch every name.\n"
            "• blank — tokenizer only, no NER at all (same as turning NER off)."
        ),
    )
    n_process: int = Field(
        0, title="Parallel workers",
        description=(
            "How many CPU workers share the detection pass. 0 = auto (use all "
            "available cores) — recommended. On a 2-CPU box this is ~40% faster than "
            "a single worker. Set to 1 to force single-process (lowest RAM, e.g. when "
            "the box also runs other heavy services). Values above the core count are "
            "capped automatically."
        ),
    )
    analyze_chunk: int = Field(
        2048, title="Rows per batch",
        description=(
            "How many rows are fed to the detector at once. Larger batches let the "
            "parallel workers stay busy (needed for 'Parallel workers' to help); "
            "smaller batches use less memory. 2048 is a good balance — you rarely "
            "need to change this."
        ),
    )
    sample_size: Optional[int] = Field(
        1000, title="Rows to sample",
        description=(
            "For a go/no-go readiness verdict the scan checks a random sample rather "
            "than every row, so cost stays flat no matter how large the dataset. 1000 "
            "is a statistically sound default. Raise it for a tighter estimate; set "
            "to blank (null) to scan every single row (slower on big datasets). Note: "
            "redaction (the fix step) always processes every row — sampling applies "
            "to the CHECK only."
        ),
    )
    backend: str = Field(
        "auto", title="PII backend",
        description=(
            "Detector engine: 'auto' (cascade when presidio is installed, else "
            "regex) covers almost every case. Advanced: 'cascade' (regex-in-presidio, "
            "ADR-0079), 'presidio', 'regex' (zero-dep), 'regexset' ([fastpii] re2)."
        ),
    )
    secrets_full_scan: bool = Field(
        True, title="Full-corpus secret scan",
        description=(
            "Scan every row for leaked credentials (API keys, tokens, private "
            "keys) even when name/PII detection is sampled — a rare secret would "
            "slip past a sample. Recommended on; turn off only for a pure sampled "
            "density estimate."
        ),
    )


class PiiOutput(BaseModel):
    backend_used: str = Field(
        ..., description="The recognizer backend actually used (regex | regexset | presidio).",
    )
    total: int = Field(..., description="Total non-empty rows read (sampled count when sampled).")
    rows_with_pii: int = Field(..., description="Rows containing at least one match.")
    density_pct: float = Field(..., description="rows_with_pii / total * 100, rounded.")
    secret_rows: Optional[int] = Field(
        None,
        description=(
            "ADR-0080 P1 — rows carrying at least one leaked secret, counted over "
            "the FULL corpus (not the sample). None when secrets_full_scan is off."
        ),
    )
    counts: dict[str, int] = Field(
        default_factory=dict,
        description="Per-kind total match count across the dataset.",
    )
    findings: list[PiiFinding] = Field(
        default_factory=list, description="Per-row kinds detected (rows with ≥1 match).",
    )
    findings_truncated: bool = Field(
        False,
        description="True when `findings` hit the max_findings cap (#891) — `counts` "
        "and `rows_with_pii` are still exact; only the enumerated detail was capped.",
    )
    sampled: bool = Field(
        False,
        description="True when the check ran over a reservoir sample, not every row.",
    )
    sample_size: Optional[int] = Field(
        None,
        description="Rows sampled when `sampled` is true; None when the full dataset was scanned.",
    )
