"""ADR-0070 (#748) — dataset_checks operators that expose GOVERNANCE config knobs.

`dataset/anonymize.kinds` (which PII kinds must be redacted) and
`dataset/provenance.required_fields` (which metadata a dataset must declare) are
compliance policy, not per-run tuning — so both are tagged `scope=governance`
(env-owned, never weakenable by a workflow/run). These tests pin that tagging at
the operator surface so a future edit can't silently demote them to operational.
"""
from __future__ import annotations

from cairn.core.operator.config import GOVERNANCE, OPERATIONAL, partition_by_scope
from operator_dataset_checks.operators.anonymize import AnonymizeOp
from operator_dataset_checks.operators.provenance import ProvenanceOp
from operator_dataset_checks.schemas.anonymize import AnonymizeConfig
from operator_dataset_checks.schemas.provenance import ProvenanceConfig


def test_anonymize_kinds_is_governance():
    parts = partition_by_scope(AnonymizeConfig)
    gov = {f["path"] for f in parts[GOVERNANCE]}
    op = {f["path"] for f in parts[OPERATIONAL]}
    assert "kinds" in gov            # env-owned redaction policy
    assert op == set()               # nothing operational leaks in


def test_provenance_required_fields_is_governance():
    parts = partition_by_scope(ProvenanceConfig)
    gov = {f["path"] for f in parts[GOVERNANCE]}
    assert "required_fields" in gov  # env-owned data contract
    assert not parts[OPERATIONAL]


def test_config_schema_is_a_subset_of_input_schema():
    # ADR-0068 invariant: config_schema fields exist on input_schema (same names)
    # so the config layer deep-merges cleanly under the bound input.
    assert set(AnonymizeConfig.model_fields) <= set(AnonymizeOp.input_schema.model_fields)
    assert set(ProvenanceConfig.model_fields) <= set(ProvenanceOp.input_schema.model_fields)


def test_operators_publish_the_config_schema():
    assert AnonymizeOp.config_schema is AnonymizeConfig
    assert ProvenanceOp.config_schema is ProvenanceConfig


def test_config_forbids_unknown_knob():
    # extra="forbid" — a typo'd knob fails validation instead of being ignored.
    import pytest

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AnonymizeConfig(kindz=["EMAIL"])  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ProvenanceConfig(require_fields=["license"])  # type: ignore[call-arg]
