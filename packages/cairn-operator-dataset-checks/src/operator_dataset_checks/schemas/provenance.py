"""Schemas for dataset/provenance — declared dataset-metadata check.

Sources metadata from HF ``DatasetInfo`` (license/source/citation that the Hub
card already carries) and/or a sidecar JSON (for local datasets + the
governance flags `datasets` doesn't have, e.g. synthetic_validated).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ProvenanceInput(BaseModel):
    path: Optional[str] = Field(
        None,
        description="Dataset path or HF Hub id — its DatasetInfo (license/homepage/citation) is read if present.",
    )
    metadata_path: Optional[str] = Field(
        None,
        description="Optional sidecar JSON of dataset metadata; overrides/fills DatasetInfo and carries governance flags (synthetic, synthetic_validated).",
    )
    required_fields: list[str] = Field(
        default_factory=lambda: ["license", "source"],
        description="Metadata fields that must be present and non-empty.",
    )


class ProvenanceConfig(BaseModel):
    """The deployment-tunable surface of ``dataset/provenance`` (ADR-0068/0070).

    A subset of ``ProvenanceInput`` — just ``required_fields``, tagged **governance**
    (ADR-0070): *which* metadata fields a dataset must declare is a data-governance
    policy (the org's data contract), not a per-run knob. Like a gate threshold it is
    **env-owned and never weakenable** by a workflow/run — a run cannot drop a required
    field to make a non-compliant dataset pass. ``required_fields`` is left UNBOUND in
    ``run-provenance`` (no ``$default`` clobber), so env config supplies it; the default
    ``["license", "source"]`` on ``ProvenanceInput`` is the floor."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"scope": "governance"})

    required_fields: list[str] = Field(
        default_factory=lambda: ["license", "source"],
        title="Required provenance fields",
        description=(
            "Metadata fields that must be present and non-empty for provenance to pass. "
            "Env-owned governance: a workflow/run cannot drop a required field."
        ),
    )


class ProvenanceOutput(BaseModel):
    metadata_sources: list[str] = Field(
        default_factory=list,
        description="Which sources contributed metadata: 'dataset_info' and/or 'sidecar'.",
    )
    license_present: bool = Field(..., description="A non-empty license is declared.")
    source_present: bool = Field(..., description="A non-empty source/homepage is declared.")
    synthetic: bool = Field(..., description="The data is declared synthetic.")
    synthetic_validated: bool = Field(
        ..., description="Synthetic data is declared validated.",
    )
    missing_fields: list[str] = Field(
        default_factory=list, description="Required fields that are absent or empty.",
    )
    provenance_ok: bool = Field(
        ...,
        description="No required field missing AND (not synthetic OR synthetic_validated).",
    )
