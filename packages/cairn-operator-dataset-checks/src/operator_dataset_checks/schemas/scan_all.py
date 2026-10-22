"""Schemas for dataset/scan_all — the fused single-pass readiness check (ADR-0080
P1, #1029).

One streaming read of the dataset emits the evidence four separate operators
(``dataset/format`` · ``dataset/dedup`` · ``dataset/profile`` · ``dataset/pii_scan``)
used to gather in four independent re-streams. The output fields carry the SAME
values those operators emit, so ``run-scan-all.yaml`` maps them onto the identical
``state.findings.*`` keys and the deterministic gate is byte-unchanged.

Tiering (ADR-0080 D2):
- **Full-corpus** (every row, cheap): dedup canonical hashes, the format-detection
  vote, and the deterministic SECRET regex — a rare leaked key can't hide from a
  sample (rejected design R4).
- **Reservoir sample** (``sample_size`` rows): format conformance, the profile
  stats, and the spaCy NER density — statistically valid for a go/no-go gate, and
  what keeps the check O(1) in dataset size.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from operator_dataset_checks.schemas.format import FormatFinding
from operator_dataset_checks.schemas.profile import LengthStats


class ScanAllInput(BaseModel):
    path: str = Field(
        ..., description="Path to a JSONL fine-tuning dataset (one example per line).",
    )
    # ── scan bound (statistical tiers) ────────────────────────────────────────
    sample_size: Optional[int] = Field(
        1000,
        description=(
            "Reservoir-sample this many rows for the STATISTICAL tiers (format "
            "conformance, profile stats, NER density) — a valid go/no-go estimate "
            "that keeps the check O(1) in dataset size (#737). Dedup hashes, the "
            "format-detection vote, and the SECRET regex always scan every row. "
            "null = run the statistical tiers over the full corpus too."
        ),
    )
    # ── format tier ───────────────────────────────────────────────────────────
    target_trainers: list[str] = Field(
        default_factory=lambda: ["trl", "unsloth", "axolotl"],
        description="Trainers to report format compatibility for (dataset/format).",
    )
    expected_format: Optional[str] = Field(
        None,
        description="Force a format instead of auto-detecting from the corpus vote.",
    )
    # ── dedup tier ────────────────────────────────────────────────────────────
    normalize: bool = Field(
        True,
        description="Collapse whitespace before hashing so rows differing only in "
        "spacing count as duplicates (dataset/dedup).",
    )
    # ── profile tier ──────────────────────────────────────────────────────────
    short_threshold: int = Field(
        10, description="Responses shorter than this many chars are flagged short (dataset/profile).",
    )
    refusal_markers: Optional[list[str]] = Field(
        None, description="Case-insensitive refusal substrings (default set used if None).",
    )
    strict: bool = Field(
        False,
        description="Malformed-line handling (#1008): False quarantines bad JSONL "
        "lines to a side file and counts them; True fails fast.",
    )
    # ── corpus engine (#1436) ─────────────────────────────────────────────────
    engine: str = Field(
        "python",
        description="Full-corpus pass engine: 'python' (one streaming read) | "
        "'duckdb' (row count, dedup distinct-key, every-row secret scan and the "
        "reservoir sample as SQL — the format vote then comes from the sample). "
        "Fails loudly without the duckdb package.",
    )
    # ── PII tier ──────────────────────────────────────────────────────────────
    backend: str = Field(
        "auto",
        description="PII backend: 'auto' | 'regex' | 'regexset' | 'presidio' | "
        "'cascade' (regex-in-presidio, ADR-0079) | 'duckdb' (the regex tier; "
        "sample density is scanned with the Python regex tier).",
    )
    text_fields: Optional[list[str]] = Field(
        None,
        description="Top-level field(s) holding training text to scan; null = auto "
        "free-text leaves (skip token/label lists, #1021 F4).",
    )
    preset: Literal["governed", "logs"] = Field(
        "governed", description="PII preset (governed = NER+regex+secrets; logs = regex+secrets).",
    )
    ner: Optional[bool] = Field(
        None, description="Detect names/locations with spaCy (null = preset value).",
    )
    model: Optional[Literal["blank", "sm", "lg"]] = Field(
        None, description="spaCy model powering NER (null = preset value).",
    )
    n_process: Optional[int] = Field(
        None, description="CPU workers for the NER pass; 0 = all cores (null = preset).",
    )
    analyze_chunk: Optional[int] = Field(
        None, description="Rows fed to the detector at once (null = preset value).",
    )
    secrets_full_scan: bool = Field(
        True,
        description="Scan every row for leaked secrets (ADR-0080 P1) and merge into "
        "the gate's blocking vocabulary even when NER density is sampled.",
    )
    max_findings: int = Field(
        1000, description="Cap the enumerated per-row PII/format `findings` lists (#891).",
    )
    # ── fused-only knob ───────────────────────────────────────────────────────
    ner_skip_below_conformant_pct: float = Field(
        40.0,
        description=(
            "Guard early-exit (ADR-0080 D2): when sampled format conformance is "
            "below this, the dataset already FAILs the format gate, so the CPU-heavy "
            "NER pass is skipped. Full-corpus secret scanning still runs (leaked keys "
            "still block), and only WARN-level sampled PII is skipped — never changes "
            "the verdict. 0 = never skip."
        ),
    )


class ScanAllConfig(BaseModel):
    """Deployment-tunable surface of ``dataset/scan_all`` (ADR-0068/0070) — the scan
    bound + PII backend/preset. **Operational** scope. A subset of ``ScanAllInput``;
    the safe defaults live on the input so an un-configured run is still bounded."""

    model_config = ConfigDict(extra="forbid")

    sample_size: Optional[int] = Field(
        1000, title="Rows to sample",
        description="Reservoir sample for the statistical tiers (null = full corpus). "
        "Dedup + secret detection always scan every row.",
    )
    preset: Literal["governed", "logs"] = Field(
        "governed", title="PII scan mode",
        description="governed = spaCy NER + regex + secrets; logs = regex + secrets only.",
    )
    backend: str = Field(
        "auto", title="PII backend",
        description="'auto' | 'cascade' | 'presidio' | 'regex' | 'regexset'.",
    )
    short_threshold: int = Field(
        10, title="Short-response threshold",
        description="Responses shorter than this many chars are flagged short.",
    )
    target_trainers: list[str] = Field(
        default_factory=lambda: ["trl", "unsloth", "axolotl"],
        title="Target trainers",
        description="Trainers to report format compatibility for.",
    )


class ScanAllOutput(BaseModel):
    # meta
    scanned: int = Field(..., description="Rows read over the FULL corpus (dedup/secret tier).")
    sampled: bool = Field(False, description="True when the statistical tiers ran over a sample.")
    sample_size: Optional[int] = Field(
        None, description="Rows in the statistical sample; None when the full corpus was scanned.",
    )
    ner_skipped: bool = Field(
        False,
        description="True when the guard early-exit skipped the NER pass (format already "
        "catastrophic); full-corpus secrets were still scanned.",
    )
    quarantined_rows: int = Field(
        0, description="Malformed source lines routed to quarantine instead of crashing (#1008).",
    )
    quarantine_path: Optional[str] = Field(
        None, description="Quarantine side-file path; None when nothing was quarantined.",
    )
    # ── format (maps to state.findings.format_*) ──────────────────────────────
    format_detected: str = Field(..., description="Winning format from the full-corpus vote.")
    format_conformant_pct: float = Field(..., description="Sampled conformance to the detected format.")
    format_findings: list[FormatFinding] = Field(
        default_factory=list, description="Per-row structural issues (capped at max_findings).",
    )
    format_trainer_compat: dict[str, bool] = Field(
        default_factory=dict, description="Per target trainer: is the detected format accepted?",
    )
    # ── dedup (maps to state.findings.dedup_*) ────────────────────────────────
    dedup_total: int = Field(..., description="Total rows read (full corpus).")
    dedup_exact: int = Field(..., description="Rows that are exact duplicates of an earlier row.")
    dedup_effective_size: int = Field(..., description="Distinct rows after dedup.")
    dedup_pct: float = Field(..., description="exact_duplicates / total * 100, rounded.")
    # ── profile (maps to state.findings.profile_*) ────────────────────────────
    profile_total: int = Field(..., description="Rows in the profile sample.")
    profile_lengths: LengthStats = Field(..., description="Response-length distribution (chars).")
    profile_refusal_pct: float = Field(..., description="Refusal-marker hit ratio, rounded.")
    profile_short: int = Field(..., description="Responses below short_threshold chars.")
    # ── pii (maps to state.findings.pii_*) ────────────────────────────────────
    pii_backend: str = Field(..., description="The PII backend actually used.")
    pii_rows: int = Field(..., description="Sampled rows with ≥1 PII/secret match.")
    pii_density: float = Field(..., description="pii_rows / sample * 100, rounded.")
    pii_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Per-kind counts; secret kinds are full-corpus and in the gate's "
        "lowercase blocking vocabulary (ADR-0080 P1).",
    )
    secret_rows: Optional[int] = Field(
        None, description="Rows carrying a leaked secret over the FULL corpus; None when off.",
    )
    pii_findings: list[dict] = Field(
        default_factory=list, description="Per-row PII kinds (sampled; capped at max_findings).",
    )
