"""``eval/ifc-validate`` and the ``eval.ifc_validate`` stage (cairn-extensions#12).

The fixture (``fixtures/snippets.jsonl``) is ten answers a model might write, five that parse as
IFC4 and five that do not, each with the reason. The checks run against the real IfcOpenShell (the
``test`` extra installs it); the tests that need it skip, and say so, where it is absent. What is
proven:

* the operator and the stage are registered, and importing the package imports neither
  IfcOpenShell nor boto3;
* the fixture scores 0.5 parse validity, the same on every run, and each row for its own reason;
* a snippet that parses can still break the schema (attribute count, an undefined reference, a
  wrong type, a required attribute left out), and the schema is the one asked for;
* without the extra, the operator and the stage report ``checked: false`` and no rate;
* the stage scores each field of a samples file on its own, compares the candidate with the base
  model, writes and uploads a results file, and fails the leg on a missing or changed samples file;
* nothing gates: the outputs carry rates, never a verdict.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from operator_ifc_eval import box_io, flow_stages, validate

FIXTURE = Path(__file__).parent / "fixtures" / "snippets.jsonl"
ROWS = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
SNIPPETS = [r["snippet"] for r in ROWS]

needs_ifc = pytest.mark.skipif(not validate.available(), reason="IfcOpenShell (the [ifc] extra) is not installed")


@pytest.fixture(autouse=True)
def _no_bucket(monkeypatch):
    for name in ("CAIRN_CHECKPOINT_BUCKET", "CAIRN_CHECKPOINT_PREFIX", "CAIRN_RUN_ID", "CAIRN_PARENT_RUN",
                 "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(name, raising=False)


# ── registration ──────────────────────────────────────────────────────────────────────────────

def test_the_operator_and_the_stage_are_registered_and_import_nothing_heavy():
    from importlib.metadata import entry_points

    (op,) = [e for e in entry_points(group="cairn.operators") if e.name == "eval/ifc-validate"]
    (stage,) = [e for e in entry_points(group="cairn.flow_stages") if e.name == "eval.ifc_validate"]
    assert op.load().__name__ == "IfcValidateOp"
    fn = stage.load()
    assert fn is flow_stages.eval_ifc_validate and fn.risk_class == "external_write"
    assert not getattr(fn, "implements", ())   # uncontracted, as eval.lm_eval
    code = ("import sys, operator_ifc_eval.flow_stages, operator_ifc_eval.operators.ifc_validate; "
            "print(sorted(m for m in ('ifcopenshell', 'boto3') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_the_core_resolves_the_stage_and_the_operators_capability():
    from cairn.core.dataflow.defs import stage_risk_class
    from cairn.core.operator import default_registry, load_all_operators

    assert stage_risk_class("eval.ifc_validate") == "external_write"
    load_all_operators()
    meta = default_registry.get_metadata("eval/ifc-validate")
    assert "eval.ifc_validate" in meta.provides_capabilities and meta.risk == "read_only"


# ── the fixture ───────────────────────────────────────────────────────────────────────────────

def test_the_fixture_is_five_valid_and_five_broken():
    assert [r["expect"] for r in ROWS].count("valid") == 5 and [r["expect"] for r in ROWS].count("broken") == 5


@needs_ifc
def test_the_fixture_scores_half_parse_validity_deterministically():
    first, second = validate.summarize(SNIPPETS, "IFC4"), validate.summarize(SNIPPETS, "IFC4")
    assert first.parse_valid_rate == second.parse_valid_rate == 0.5
    assert first.schema_error_rate == 0.0 and first.valid_rate == 0.5
    assert first.failures == second.failures
    assert [f["row"] for f in first.failures] == [i for i, r in enumerate(ROWS) if r["expect"] == "broken"]
    assert {f["stage"] for f in first.failures} == {"parse"}


@needs_ifc
@pytest.mark.parametrize("row", ROWS, ids=[r["why"] for r in ROWS])
def test_each_fixture_row_for_its_own_reason(row):
    result = validate.check(row["snippet"], "IFC4")
    assert result.parse_valid is (row["expect"] == "valid"), result.error
    if row["expect"] == "valid":
        assert result.schema_errors == []
    else:
        assert result.error


def test_extract_takes_fenced_blocks_then_a_files_data_then_the_instances():
    fenced = "Sure:\n```ifc\n#1=IFCDIRECTION((0.,0.,1.));\n```\nand\n```\n#2=IFCDIRECTION((1.,0.,0.));\n```"
    assert validate.extract(fenced) == "#1=IFCDIRECTION((0.,0.,1.));\n\n#2=IFCDIRECTION((1.,0.,0.));"
    assert validate.extract(SNIPPETS[2]).startswith("#1=IFCWALL(") and "ENDSEC" not in validate.extract(SNIPPETS[2])
    assert validate.extract("The point is #7=IFCCARTESIANPOINT((0.,0.,0.)); done.") == "#7=IFCCARTESIANPOINT((0.,0.,0.));"
    assert validate.extract("no ifc here") == ""


def test_the_strict_reading_counts_attributes_and_references():
    (s,) = validate.statements("#4=IFCAXIS2PLACEMENT3D(#1,#2,#3);")
    assert (s.id, s.entity, s.arguments, s.references) == (4, "IFCAXIS2PLACEMENT3D", 3, [1, 2, 3])
    (s,) = validate.statements("#1=IFCLABEL('a, b (c) #9');")      # commas, parentheses and # in a string
    assert s.arguments == 1 and s.references == []
    for bad in ("#1=IFCX($);#1=IFCY($);", "#1=IFCX($)", "#1=IFCX('a);", "#1=(IFCX($));"):
        with pytest.raises(validate.SnippetSyntaxError):
            validate.statements(bad)


# ── the schema ────────────────────────────────────────────────────────────────────────────────

@needs_ifc
@pytest.mark.parametrize(("snippet", "expected"), [
    ("#1=IFCCARTESIANPOINT((0.,0.,0.),5);", "2 attributes, IFC4 declares 1"),
    ("#1=IFCAXIS2PLACEMENT3D(#9,$,$);", "references #9, which the snippet does not define"),
    ("#1=IFCDIRECTION('abc');", "DirectionRatios"),
    ("#1=IFCCARTESIANPOINT($);", "Attribute not optional"),
])
def test_a_snippet_that_parses_can_still_break_the_schema(snippet, expected):
    result = validate.check(snippet, "IFC4")
    assert result.parse_valid and not result.valid
    assert any(expected in e for e in result.schema_errors), result.schema_errors


@needs_ifc
def test_the_schema_is_the_one_asked_for():
    """IFC4 made an IfcProject's OwnerHistory optional; IFC2X3 requires it."""
    project = SNIPPETS[0]
    assert validate.check(project, "IFC4").valid
    ifc2x3 = validate.check(project, "ifc2x3")
    assert ifc2x3.parse_valid and any("IfcProject.OwnerHistory" in e for e in ifc2x3.schema_errors)
    with pytest.raises(ValueError, match="unknown IFC schema 'IFC5'"):
        validate.check(project, "IFC5")


@needs_ifc
def test_the_rates_are_over_what_they_say():
    s = validate.summarize([SNIPPETS[0], "#1=IFCCARTESIANPOINT($);", "prose"], "IFC4")
    assert (s.rows, s.parse_valid, s.schema_clean) == (3, 2, 1)
    assert s.parse_valid_rate == pytest.approx(2 / 3) and s.schema_error_rate == 0.5
    assert s.valid_rate == pytest.approx(1 / 3)
    assert [(f["row"], f["stage"]) for f in s.failures] == [(1, "schema"), (2, "parse")]
    assert validate.summarize([], "IFC4").parse_valid_rate is None


# ── the operator ──────────────────────────────────────────────────────────────────────────────

def _op(**kw):
    from cairn.core.operator.testing import run_operator_for_test
    from operator_ifc_eval.operators.ifc_validate import IfcValidateOp
    from operator_ifc_eval.schemas import IfcValidateInput

    return asyncio.run(run_operator_for_test(IfcValidateOp, IfcValidateInput(**kw)))


@needs_ifc
def test_the_operator_reports_the_rates_and_the_failures_and_no_verdict():
    out = _op(snippets=SNIPPETS, schema_version="ifc4")
    assert out.checked is True and out.schema_version == "IFC4" and out.rows == 10
    assert out.parse_valid_rate == 0.5 and out.schema_error_rate == 0.0 and out.valid_rate == 0.5
    assert [f.row for f in out.failures] == [5, 6, 7, 8, 9]
    assert not {"passed", "verdict", "gate"} & set(out.model_dump())


def test_without_the_extra_the_operator_reports_nothing_checked(monkeypatch):
    monkeypatch.setattr(validate, "available", lambda: False)
    out = _op(snippets=SNIPPETS)
    assert out.checked is False and "not installed" in out.note
    assert out.parse_valid_rate is None and out.schema_error_rate is None and out.failures == []


# ── the stage ─────────────────────────────────────────────────────────────────────────────────

def _samples(tmp_path: Path, rows: list[dict]) -> tuple[Path, str]:
    path = tmp_path / "generation_samples.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _generation(tmp_path: Path) -> tuple[Path, str]:
    """Five prompts: the references are the valid fixture rows, the base model answers with the
    broken ones, the candidate gets four of five right."""
    valid = [r["snippet"] for r in ROWS if r["expect"] == "valid"]
    broken = [r["snippet"] for r in ROWS if r["expect"] == "broken"]
    rows = [{"index": i, "prompt": f"q{i}", "reference": valid[i], "base": broken[i],
             "candidate": valid[i] if i < 4 else broken[i]} for i in range(5)]
    return _samples(tmp_path, rows)


@needs_ifc
def test_the_stage_scores_each_field_and_the_candidate_against_the_base(tmp_path):
    path, digest = _generation(tmp_path)
    out = flow_stages.eval_ifc_validate(tmp_path / "job", {"samples_path": str(path), "samples_sha256": digest,
                                                           "schema_version": "IFC4"}, [])
    assert out["checked"] is True and out["note"] == "" and out["schema_version"] == "IFC4" and out["rows"] == 5
    assert out["reference_parse_valid_rate"] == 1.0 and out["reference_valid_rate"] == 1.0
    assert out["base_parse_valid_rate"] == 0.0 and "base_schema_error_rate" not in out   # nothing parsed
    assert out["candidate_parse_valid_rate"] == 0.8 and out["candidate_schema_error_rate"] == 0.0
    assert out["delta_parse_valid_rate"] == 0.8 and out["delta_valid_rate"] == 0.8
    assert out["base_parse_failures"] == 5 and out["candidate_parse_failures"] == 1
    assert all(isinstance(v, (str, int, float, bool)) for v in out.values())
    results = tmp_path / "job" / flow_stages.RESULTS_FILE
    assert out["output"] == out["results_uri"] == str(results) and out["kind"] == "report"
    assert out["content_hash"] == hashlib.sha256(results.read_bytes()).hexdigest()
    report = json.loads(results.read_text())
    assert [f["row"] for f in report["fields"]["candidate"]["failures"]] == [4]
    assert report["deltas"]["parse_valid_rate"] == 0.8


@needs_ifc
def test_the_stage_scores_the_fields_it_is_given_and_skips_rows_without_one(tmp_path):
    path, _ = _samples(tmp_path, [{"base": SNIPPETS[0]}, {"other": "x"}, {"base": SNIPPETS[5]}])
    out = flow_stages.eval_ifc_validate(tmp_path / "job", {"samples_path": str(path), "fields": "base"}, [])
    assert out["base_rows"] == 2 and out["base_parse_valid_rate"] == 0.5 and "delta_parse_valid_rate" not in out
    report = json.loads((tmp_path / "job" / flow_stages.RESULTS_FILE).read_text())
    assert [f["row"] for f in report["fields"]["base"]["failures"]] == [2]   # the row's place in the file


def test_without_the_extra_the_stage_reports_nothing_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "available", lambda: False)
    path, digest = _generation(tmp_path)
    out = flow_stages.eval_ifc_validate(tmp_path / "job", {"samples_path": str(path), "samples_sha256": digest}, [])
    assert out["checked"] is False and "not installed" in out["note"] and out["rows"] == 5
    assert not [k for k in out if k.endswith("_rate")] and "output" not in out


@needs_ifc
def test_the_stub_tier_is_scored_and_never_checked(tmp_path):
    path, _ = _samples(tmp_path, [{"reference": SNIPPETS[0], "base": "stub output", "candidate": "stub output"}])
    out = flow_stages.eval_ifc_validate(tmp_path / "job", {"samples_path": str(path), "stub": True}, [])
    assert out["checked"] is False and "placeholders" in out["note"] and out["candidate_parse_valid_rate"] == 0.0


def test_a_missing_or_changed_samples_file_fails_the_leg(tmp_path):
    with pytest.raises(box_io.FetchError, match="no samples file"):
        flow_stages.eval_ifc_validate(tmp_path / "job", {}, [])
    with pytest.raises(box_io.FetchError, match="is not a file on this box"):
        flow_stages.eval_ifc_validate(tmp_path / "job", {"samples_uri": str(tmp_path / "gone.jsonl")}, [])
    path, digest = _generation(tmp_path)
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(box_io.FetchError, match="refusing to score a file that changed"):
        flow_stages.eval_ifc_validate(tmp_path / "job", {"samples_path": str(path), "samples_sha256": digest}, [])


@needs_ifc
def test_an_unknown_schema_fails_the_leg(tmp_path):
    path, _ = _generation(tmp_path)
    with pytest.raises(ValueError, match="unknown IFC schema"):
        flow_stages.eval_ifc_validate(tmp_path / "job", {"samples_path": str(path), "schema_version": "IFC9"}, [])


@pytest.fixture
def s3(monkeypatch):
    import boto3
    import moto

    for name, value in (("AWS_ACCESS_KEY_ID", "t"), ("AWS_SECRET_ACCESS_KEY", "t"),
                        ("AWS_DEFAULT_REGION", "us-east-1")):
        monkeypatch.setenv(name, value)
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="ckpts")
        yield client


@needs_ifc
def test_the_samples_are_read_from_s3_and_the_results_go_beside_the_checkpoints(tmp_path, monkeypatch, s3):
    path, digest = _generation(tmp_path)
    s3.upload_file(str(path), "ckpts", "cairn-checkpoints/run-gen/eval/generation_samples.jsonl")
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "ckpts")
    monkeypatch.setenv("CAIRN_PARENT_RUN", "run-ifc")
    out = flow_stages.eval_ifc_validate(tmp_path / "job", {
        "samples_path": str(tmp_path / "not-on-this-box.jsonl"), "samples_sha256": digest,
        "samples_uri": "s3://ckpts/cairn-checkpoints/run-gen/eval/generation_samples.jsonl"}, [])
    key = "cairn-checkpoints/run-ifc/eval/ifc_validate_results.json"
    assert out["output"] == out["results_uri"] == f"s3://ckpts/{key}" and out["checked"] is True
    body = s3.get_object(Bucket="ckpts", Key=key)["Body"].read()
    assert hashlib.sha256(body).hexdigest() == out["content_hash"]
