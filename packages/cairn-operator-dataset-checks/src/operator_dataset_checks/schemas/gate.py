"""Schemas for dataset/gate — the deterministic readiness verdict.

The gate turns the six checks' findings into a pass / warn / fail verdict with
cited blocking + warning reasons. It is a pure function of its inputs: no I/O,
no LLM. This is where the *engine gates* (the §4 determinism rule) — the LLM
synthesis step only narrates the verdict this operator computes.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class GateThresholds(BaseModel):
    """Severity thresholds. Defaults match the design (§5.3). Deployment-tunable
    via the profile (ADR-0068, #742): the operator exposes these through
    ``GateConfig`` (its ``config_schema``), so ``[profile.config."dataset/gate"
    .thresholds]`` overrides them per environment. The ``title``/bounds drive
    ``cairn config scaffold`` and the console Guardrails form."""

    # Config surface forbids extras (ADR-0068) so a typo'd knob fails doctor
    # preflight AND runtime validation with the field list — never silently
    # ignored. GateThresholds is also GateInput.thresholds, so this hardens the
    # wired-input path too. scope=governance (ADR-0070): these are guardrails —
    # env-owned, never weakenable by a workflow/run.
    model_config = ConfigDict(extra="forbid", json_schema_extra={"scope": "governance"})

    format_min_conformant_pct: float = Field(
        99.0, ge=0, le=100,
        title="Minimum format conformance",
        description="Rows must parse into a trainer-accepted format. Below this %, the dataset is BLOCKED.",
    )
    decontam_max_overlap_pct: float = Field(
        5.0, ge=0, le=100,
        title="Max train/eval overlap",
        description="Above this % contamination between train and eval, the dataset is BLOCKED.",
    )
    refusal_warn_pct: float = Field(
        20.0, ge=0, le=100,
        title="Refusal-ratio warn threshold",
        description="Above this % of refusal-style responses, WARN (non-blocking).",
    )
    short_warn_frac: float = Field(
        0.05, ge=0, le=1,
        title="Too-short-response warn fraction",
        description="Above this fraction of too-short responses, WARN (non-blocking).",
    )


class GateConfig(BaseModel):
    """The deployment-tunable surface of ``dataset/gate`` (ADR-0068, #742).

    A SUBSET of ``GateInput`` — just the ``thresholds`` sub-model — so a profile's
    ``[profile.config."dataset/gate"]`` deep-merges cleanly under the gate's input
    at invocation, and ``run()`` still reads ``inp.thresholds`` unchanged. This is
    the operator's ``config_schema``: the published knob surface doctor validates,
    `cairn config scaffold` renders, and the console Guardrails form edits."""

    model_config = ConfigDict(extra="forbid")  # unknown operator key → fail (ADR-0068)

    thresholds: GateThresholds = Field(
        default_factory=GateThresholds,
        title="Dataset-readiness thresholds",
        description="Severity thresholds that decide the pass / warn / fail verdict.",
    )


class GateInput(BaseModel):
    """Findings the six deterministic checks wrote to ``state.findings``.

    Every field is optional: a check that did not run leaves its inputs unset,
    and the gate simply skips that dimension rather than failing.
    """

    # dataset/format
    format_conformant_pct: Optional[float] = None
    format_trainer_compat: dict[str, bool] = Field(default_factory=dict)
    # dataset/dedup
    dedup_pct: Optional[float] = None
    dedup_effective_size: Optional[int] = None
    # dataset/decontam
    decontam_checked: bool = False
    decontam_overlap_pct: Optional[float] = None
    # dataset/pii_scan
    pii_counts: dict[str, int] = Field(default_factory=dict)
    pii_rows: Optional[int] = None
    # dataset/profile
    profile_refusal_pct: Optional[float] = None
    profile_short: Optional[int] = None
    profile_total: Optional[int] = None
    # dataset/provenance
    prov_ok: Optional[bool] = None
    prov_missing: list[str] = Field(default_factory=list)

    # #579 — the validated dataset location (the run's dataset_path), passed
    # through so the gate can EMIT it as the ref a downstream consumer trains on
    # (ADR-0066: dataset pack produces a dataset_uri; a bundle injects it into
    # train). Pass-through only — the gate does no I/O.
    dataset_uri: Optional[str] = None

    thresholds: GateThresholds = Field(default_factory=GateThresholds)


class GateOutput(BaseModel):
    verdict: str = Field(..., description="pass | warn | fail.")
    dataset_uri: str = Field(
        default="",
        description=(
            "The validated dataset location (S3/HF/path) a downstream step trains "
            "on — the run's dataset_path, passed through (#579). Consumers should "
            "only train on it when verdict != 'fail'."
        ),
    )
    blocking: list[str] = Field(
        default_factory=list,
        description="Issues that must be fixed before training (each cites a check).",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-blocking concerns (each cites a check).",
    )
    passed: list[str] = Field(
        default_factory=list, description="Checks that came back clean."
    )
    checks_cited: list[str] = Field(
        default_factory=list,
        description="The deterministic checks whose results informed this verdict.",
    )
    audit_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Compact verdict record for the audit sink (blocking/warnings/thresholds).",
    )
    outcome: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "ADR-0075 run-level outcome metrics — the domain verdict + key check "
            "numbers as {name, value, pass, target}. Bound to $state.outcome so it "
            "surfaces on the standard output.outcome channel the pilot report / "
            "Playground read. Deterministic (mirrors the computed verdict); the LLM "
            "synthesis step never touches it."
        ),
    )
