"""Schemas for ``eval/ifc-validate`` and the ``eval.ifc_validate`` flow stage (cairn-extensions#12).

Uncontracted (no ``contract://`` capability or stage contract yet, as ``eval/lm-eval``): the shapes
are this package's, versioned with it.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from operator_ifc_eval.validate import DEFAULT_SCHEMA


class IfcValidateInput(BaseModel):
    """Generated answers to check."""

    snippets: list[str] = Field(default_factory=list, description="The answers, one per row: STEP instances, "
                                                                  "fenced or not, or a whole IFC file.")
    schema_version: str = Field(DEFAULT_SCHEMA, description="The IFC schema the snippets target: IFC2X3, "
                                                            "IFC4 or IFC4X3_ADD2 (any case).")


class RowFailure(BaseModel):
    row: int = Field(..., description="The row's position in `snippets`, from 0.")
    stage: str = Field(..., description="parse: it did not parse; schema: it parsed with schema errors.")
    errors: list[str] = Field(default_factory=list)


class IfcValidateOutput(BaseModel):
    """The rates and the per-row failures. Never a verdict: a gate reads the rates."""

    checked: bool = Field(False, description="False when IfcOpenShell is not installed: no rate is reported.")
    note: str = ""
    schema_version: str = ""
    ifcopenshell_version: str = ""
    rows: int = 0
    parse_valid: int = 0
    parse_valid_rate: Optional[float] = Field(None, description="Snippets that parse / snippets.")
    schema_error_rate: Optional[float] = Field(None, description="Parsed snippets with a schema error / parsed "
                                                                 "snippets; None when none parsed.")
    valid_rate: Optional[float] = Field(None, description="Snippets that parse with no schema error / snippets.")
    failures: list[RowFailure] = Field(default_factory=list)


class IfcStageConfig(BaseModel):
    """The ``eval.ifc_validate`` stage's flat config (``$config.*`` substituted, absent keys omitted)."""

    model_config = ConfigDict(extra="ignore")

    samples_path: str = Field("", description="The generation samples file on this box (the generate leg's "
                                              "`samples_path`); used first when it exists.")
    samples_uri: str = Field("", description="The samples file's location otherwise: an s3:// URI, file:// or "
                                             "a path on the box.")
    samples_sha256: str = Field("", description="The samples file's content hash, as the generate leg "
                                                "reported it; the bytes are verified when it is set.")
    schema_version: str = Field(DEFAULT_SCHEMA, description="The IFC schema the answers target.")
    fields: str = Field("reference,base,candidate",
                        description="Comma-separated sample fields scored, each on its own. The reference "
                                    "answers are scored too: a reference that does not parse under the schema "
                                    "says the schema is the wrong one, not that the models are.")
    stub: bool = Field(False, description="The stub tier: the samples are placeholders; scored, never `checked`.")


__all__ = ["IfcStageConfig", "IfcValidateInput", "IfcValidateOutput", "RowFailure"]
