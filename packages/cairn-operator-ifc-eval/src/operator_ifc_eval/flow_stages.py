"""``eval.ifc_validate`` — a generation eval's samples, checked as IFC for each model
(cairn-extensions#12, #41).

A flow-stage kernel in the engine-neutral ``cairn.flow_stages`` entry-point group::

    eval_ifc_validate(workdir: Path, config: dict, inputs: list[str]) -> dict   # flat scalar stats

It reads a samples file: JSON lines, one per prompt, each carrying the answers to score as string
fields (``reference``, ``base``, ``candidate`` by default: what ``cairn-operator-eval-lm``'s
``eval.generate`` stage writes). Each field is scored on its own with
:func:`operator_ifc_eval.validate.summarize`, and the candidate's rates are compared with the base
model's (``delta_*`` = candidate − base). The reference answers are scored too: references that do
not parse under the schema mean the schema is the wrong one, not that the models are.

What it reports, as stats (the leg's findings, ``eval_ifc_validate_<key>``): per field
``<field>_parse_valid_rate``, ``<field>_schema_error_rate`` (when any row parsed),
``<field>_valid_rate`` and ``<field>_rows``; the deltas; and a results file with every row's
failures, uploaded beside the run's checkpoints when ``CAIRN_CHECKPOINT_BUCKET`` is set and
registered as the stage's ``report`` product.

What fails the leg: a samples file that is named but cannot be read, a file whose bytes are not the ones the generate leg reported,
an unknown schema. What does not: IfcOpenShell missing on the box, or no samples file named (the generate leg wrote
none because its own harness was missing or failed): ``checked: false`` with the reason and no
rate. It never gates: a pipeline predicate or a person reads the rates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

STAGE = "eval.ifc_validate"
RESULTS_FILE = "ifc_validate_results.json"
_RATES = ("parse_valid_rate", "schema_error_rate", "valid_rate")


def _rows(path: Path) -> list[dict]:
    from operator_ifc_eval.box_io import FetchError  # noqa: PLC0415

    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError as e:
                raise FetchError(f"{path.name} line {i} is not JSON: {e}") from e
            if isinstance(row, dict):
                rows.append(row)
    return rows


def eval_ifc_validate(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Score each field of the samples file as IFC; see the module docstring."""
    from operator_ifc_eval import box_io, validate  # noqa: PLC0415 — off the registry-enumeration path
    from operator_ifc_eval.schemas import IfcStageConfig  # noqa: PLC0415

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = IfcStageConfig.model_validate({k: v for k, v in (config or {}).items() if v is not None})
    fields = [f.strip() for f in cfg.fields.split(",") if f.strip()]
    if not cfg.samples_path and not cfg.samples_uri:
        # the generate leg wrote no samples (its harness was missing or failed: it says why); its
        # evaluation is "not evaluated", and so is this one, never a failed run
        return {"stub": cfg.stub, "schema_version": cfg.schema_version, "fields": ",".join(fields), "rows": 0,
                "checked": False, "note": "no samples file named: the generate leg wrote none (see its note)"}
    path = box_io.fetch_samples(local_path=cfg.samples_path, uri=cfg.samples_uri, dest_dir=workdir / "samples",
                                sha256=cfg.samples_sha256)
    rows = _rows(path)
    base: dict[str, Any] = {"stub": cfg.stub, "schema_version": cfg.schema_version, "fields": ",".join(fields),
                            "samples": str(path), "rows": len(rows)}
    if not validate.available():
        return {**base, "checked": False,
                "note": "IfcOpenShell is not installed on the box (install cairn-operator-ifc-eval[box]): "
                        "nothing checked"}
    schema = validate.resolve_schema(cfg.schema_version).name()
    base["schema_version"] = schema

    stats: dict[str, Any] = {}
    report_fields: dict[str, dict] = {}
    for name in fields:
        present = [(i, r[name]) for i, r in enumerate(rows) if isinstance(r.get(name), str)]
        s = validate.summarize([text for _, text in present], schema)
        failures = [{**f, "row": present[f["row"]][0]} for f in s.failures]
        rates = {"parse_valid_rate": s.parse_valid_rate, "schema_error_rate": s.schema_error_rate,
                 "valid_rate": s.valid_rate}
        report_fields[name] = {"rows": s.rows, "parse_valid": s.parse_valid, "schema_clean": s.schema_clean,
                               **rates, "failures": failures}
        stats[f"{name}_rows"] = s.rows
        stats[f"{name}_parse_failures"] = sum(f["stage"] == "parse" for f in failures)
        stats[f"{name}_schema_failures"] = sum(f["stage"] == "schema" for f in failures)
        for key, value in rates.items():
            if value is not None:
                stats[f"{name}_{key}"] = round(value, 6)
    if "base" in report_fields and "candidate" in report_fields:
        for key in _RATES:
            b, c = report_fields["base"][key], report_fields["candidate"][key]
            if b is not None and c is not None:
                stats[f"delta_{key}"] = round(c - b, 6)

    checked = bool(rows) and not cfg.stub
    note = ("stub tier: the samples are placeholders, so the rates say nothing about a model" if cfg.stub
            else "" if rows else "the samples file has no rows")
    report = {"schema": 1, "stage": STAGE, "checked": checked, "note": note, "schema_version": schema,
              "ifcopenshell_version": validate.version(), "samples": str(path), "fields": report_fields,
              "deltas": {k[len("delta_"):]: v for k, v in stats.items() if k.startswith("delta_")}}
    out = workdir / RESULTS_FILE
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    location, upload_error = str(out), ""
    target = box_io.results_target()
    if target is not None:
        try:
            location = box_io.upload(out, *target)
        except Exception as e:  # noqa: BLE001 — the rates are on the record; the file stays on the box
            upload_error = f"{type(e).__name__}: {e}"
    return {
        **base, "checked": checked, "note": note, "ifcopenshell_version": validate.version(), **stats,
        "results_uri": location, **({"results_upload_error": upload_error} if upload_error else {}),
        # the stage's product (the runner turns these into the registered descriptor)
        "output": location, "content_hash": digest, "kind": "report", "bytes": out.stat().st_size,
    }


# Reads a file on the box and writes its results file to the checkpoint bucket (ADR-0081 D2:
# fail-closed risk classes; the same class as the other on-box eval stages).
eval_ifc_validate.risk_class = "external_write"

__all__ = ["RESULTS_FILE", "STAGE", "eval_ifc_validate"]
