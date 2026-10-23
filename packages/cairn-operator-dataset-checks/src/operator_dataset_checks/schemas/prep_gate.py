"""Schemas for dataset/prep_gate — the config-gated POST-SCRUB PII gate (#882).

prep re-scans its own scrubbed output (run-prep-verify → findings.post_scrub_pii_rows);
this gate turns that fact + a governance `mode` knob into a verdict:

- ``off``   — surface only (the report warning is enough); never blocks.
- ``block`` — fail-closed: a dirty scrub yields a FAIL verdict (the run completes
              blocked, the dataset/gate convention — not a crash).
- ``hitl``  — hold for human review: emits ``action=review`` so the prep template's
              guard routes to the HITL review node (#883).

Pure function of its inputs — no I/O, no LLM. Mirrors dataset/gate.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PrepGateMode = Literal["off", "block", "hitl"]


class PrepGateConfig(BaseModel):
    """The deployment-tunable surface of ``dataset/prep_gate`` (ADR-0070).

    A SUBSET of ``PrepGateInput`` (just ``mode``) so ``[profile.config."dataset/
    prep_gate"]`` deep-merges under the input and ``run()`` reads ``inp.mode``
    unchanged. ``scope=governance``: env-owned — a workflow/run can NOT switch its
    own PII gate off (that would be a governance bypass)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"scope": "governance"})

    mode: PrepGateMode = Field(
        "off",
        title="Post-scrub PII gate",
        description=(
            "off = surface only (report warns); block = fail-closed verdict on a "
            "dirty scrub; hitl = pause for human review of the residual PII."
        ),
    )


#: #889 — a "clean" verdict is only a GUARANTEE when the scan was full (not sampled)
#: AND ran on a real NER detector. regex/regexset miss names/phone/SSN/location, so
#: a regex "0 PII" is not certifiable. Presidio (the [pii] extra) is the certifying
#: backend; widen this set if regexset gains full-entity coverage.
_CERTIFYING_BACKENDS = {"presidio"}


class PrepGateInput(BaseModel):
    post_scrub_pii_rows: int = Field(
        0, ge=0, description="Residual PII rows in prep's SCRUBBED output (run-prep-verify)."
    )
    post_scrub_pii_counts: dict[str, int] = Field(
        default_factory=dict, description="Per-kind residual counts (for the cited reason)."
    )
    # #889/#890 — verdict provenance: was the re-scan a full scan on a real detector?
    post_scrub_pii_sampled: bool = Field(
        False, description="True if the re-scan ran over a reservoir SAMPLE (→ can't certify clean)."
    )
    post_scrub_pii_backend: str = Field(
        "", description="Detector backend the re-scan used (regex | regexset | presidio)."
    )
    # Governance knob, injected by the config layer (config_schema). Left unbound
    # in the operation YAML so a run can't set it.
    mode: PrepGateMode = Field("off", description="Enforcement mode (env-owned governance).")


class PrepGateOutput(BaseModel):
    action: Literal["pass", "block", "review"] = Field(
        ..., description="pass = proceed; block = fail-closed; review = route to HITL."
    )
    verdict: Literal["pass", "warn", "fail"] = Field(
        ...,
        description=(
            "fail = a dirty scrub was ENFORCED (block/hitl); warn = 0 PII found but "
            "the scan can't CERTIFY clean (sampled or non-NER backend, #889); else pass."
        ),
    )
    clean: bool = Field(..., description="post_scrub_pii_rows == 0 (found no PII).")
    verified: bool = Field(
        ...,
        description=(
            "#889 — clean is a GUARANTEE: 0 PII AND a full scan (not sampled) AND a "
            "certifying NER backend. A consumer/pipeline gates on THIS, not `clean`."
        ),
    )
    blocking: list[str] = Field(
        default_factory=list, description="Cited reason when the gate blocks / warns / holds for review."
    )
    #: ADR-0075 outcome-channel item so the run surfaces the post-scrub invariant
    #: alongside any other outcome (severity block only when enforced).
    outcome: list[dict] = Field(default_factory=list)
