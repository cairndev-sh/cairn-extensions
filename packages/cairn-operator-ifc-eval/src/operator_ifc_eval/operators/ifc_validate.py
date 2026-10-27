"""eval/ifc-validate — do generated IFC snippets parse, and do they hold to the schema
(cairn-extensions#12)?

The rates and the per-row failures, from :mod:`operator_ifc_eval.validate`. It never gates on its
own: a verdict comes from a pipeline predicate (or a gate operator) that reads the rates. Without the
``[ifc]`` extra it reports ``checked: false`` and no rate, so a missing extra can never pass as a
zero or as a clean score.
"""
from __future__ import annotations

import asyncio

from cairn.core.operator import Operation, OperationContext, ReportSpec, register_operator

from operator_ifc_eval import validate
from operator_ifc_eval.schemas import IfcValidateInput, IfcValidateOutput, RowFailure


@register_operator(name="eval/ifc-validate", version="0.1.0", capabilities=["eval.ifc_validate"])
class IfcValidateOp(Operation[IfcValidateInput, IfcValidateOutput]):
    """Parse validity and schema conformance of generated IFC snippets."""

    input_schema = IfcValidateInput
    output_schema = IfcValidateOutput
    side_effect_free = True   # it reads its input and writes nothing
    risk = "read_only"
    reports = ReportSpec.NONE   # one quick pass: no stages or series to stream (ADR-0103)

    async def run(self, inp: IfcValidateInput, ctx: OperationContext) -> IfcValidateOutput:
        if not validate.available():
            return IfcValidateOutput(
                checked=False, schema_version=inp.schema_version, rows=len(inp.snippets),
                note="IfcOpenShell is not installed (pip install 'cairn-operator-ifc-eval[ifc]'): nothing checked")
        validate.resolve_schema(inp.schema_version)   # an unknown schema is a refusal, not a zero
        # IfcOpenShell is synchronous C++; a large batch should not hold the event loop.
        summary = await asyncio.to_thread(validate.summarize, list(inp.snippets), inp.schema_version)
        return IfcValidateOutput(
            checked=True, schema_version=validate.resolve_schema(inp.schema_version).name(),
            ifcopenshell_version=validate.version(), rows=summary.rows, parse_valid=summary.parse_valid,
            parse_valid_rate=summary.parse_valid_rate, schema_error_rate=summary.schema_error_rate,
            valid_rate=summary.valid_rate, failures=[RowFailure(**f) for f in summary.failures],
            note="" if summary.rows else "no snippets")


__all__ = ["IfcValidateOp"]
