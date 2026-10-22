"""cairn-extensions#5 — a Hub source loaded at a pinned revision, and the bundled manifest.

No network: the Hub loader and the commit lookup are replaced, so each test controls which rows a
revision yields. Stage outputs and the manifest document are checked against the installed
contract wheel (cairn-contracts-dataset-stages), the same schemas the control plane validates."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from cairn.core.dataflow.conformance_check import bundle, problems
from cairn.core.pack.contracts import _installed

from operator_dataset_checks import _io
from operator_dataset_checks.dataflow_stages import dataset_manifest, ingest_normalize

_HUB_ID = "example-org/ifc-bim-alpaca"
#: What the fake Hub serves at each revision; v2 changed one answer.
_BY_REVISION = {
    "v1": [{"instruction": "What is an IfcWall?", "output": "A vertical building element."},
           {"instruction": "What is IFC?", "output": "Industry Foundation Classes.", "input": ""}],
    "v2": [{"instruction": "What is an IfcWall?", "output": "A vertical construction element."},
           {"instruction": "What is IFC?", "output": "Industry Foundation Classes.", "input": ""}],
}


@pytest.fixture
def fake_hub(monkeypatch):
    """Replace the Hub: ``load_dataset`` yields the rows of the revision asked for, the commit
    lookup returns a fixed sha per revision, and the card carries a licence and citation."""
    seen: list[dict] = []

    def load_dataset(source, *, split="train", streaming=False, revision=None, **_kw):
        seen.append({"source": source, "revision": revision, "streaming": streaming})
        return list(_BY_REVISION[revision])

    class _Info:
        license = "apache-2.0"
        citation = "@misc{ifc-bim-alpaca}"

    class _Builder:
        info = _Info()

    monkeypatch.setattr(_io, "load_dataset", load_dataset)
    monkeypatch.setattr(_io, "load_dataset_builder", lambda source, **kw: _Builder())
    monkeypatch.setattr(_io, "hub_commit", lambda source, revision=None: f"sha-of-{revision}")
    return seen


def _schema(uri: str) -> dict:
    _resolved, bodies = _installed()
    return bundle({u: body for u, (_d, body) in bodies.items()}, uri)


def _stage_output(stage: str, revision: str) -> dict:
    _resolved, bodies = _installed()
    return bodies[f"contract://dataset.stage.{stage}/{revision}"][1]["output"]


def _ingest(workdir: Path, revision: str | None, path: str = _HUB_ID) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    config = {"path": path} if revision is None else {"path": path, "revision": revision}
    return ingest_normalize(workdir, config, [])


def test_the_loader_passes_the_revision_to_the_hub(fake_hub):
    rows = list(_io.iter_rows(_HUB_ID, revision="v1"))
    assert rows == _BY_REVISION["v1"]
    assert fake_hub == [{"source": _HUB_ID, "revision": "v1", "streaming": True}]


def test_a_file_source_is_not_a_hub_id(tmp_path):
    f = tmp_path / "rows.jsonl"
    f.write_text('{"a": 1}\n', encoding="utf-8")
    assert not _io.is_hub_id(str(f))
    assert not _io.is_hub_id(f"file://{f}")
    assert not _io.is_hub_id("s3://bucket/rows.jsonl")
    assert _io.is_hub_id(_HUB_ID)


def test_one_revision_names_the_same_bytes_and_a_moved_one_does_not(fake_hub, tmp_path):
    first = _ingest(tmp_path / "a", "v1")
    again = _ingest(tmp_path / "b", "v1")
    moved = _ingest(tmp_path / "c", "v2")
    assert first["snapshot_hash"] == again["snapshot_hash"]
    assert first["snapshot_hash"] != moved["snapshot_hash"]
    assert (first["revision"], first["revision_sha"]) == ("v1", "sha-of-v1")
    # A Hub id has no file whose hash is a registry id: the runner keeps the URI parent (#1432).
    assert "source_hash" not in first
    assert first["rows"] == 2 and first["fetched_at"]


def test_the_ingest_output_conforms_to_its_stage_contract(fake_hub, tmp_path):
    out = _ingest(tmp_path, "v1")
    assert problems(_schema(_stage_output("ingest_normalize", "1.2.0")), out, "output") == []
    recorded = json.loads((tmp_path / "ingest.json").read_text(encoding="utf-8"))
    assert recorded == out


def test_the_ingest_configuration_takes_a_revision_and_nothing_else():
    _resolved, bodies = _installed()
    uri = bodies["contract://dataset.stage.ingest_normalize/1.2.0"][1]["configuration"]
    schema = _schema(uri)
    assert problems(schema, {"path": _HUB_ID, "revision": "v1"}, "configuration") == []
    assert problems(schema, {"path": _HUB_ID, "revison": "v1"}, "configuration")


def test_the_manifest_describes_the_dataset_and_carries_no_row(fake_hub, tmp_path):
    _ingest(tmp_path, "v1")
    out = dataset_manifest(tmp_path, {"store": "local"}, [])
    assert problems(_schema(_stage_output("dataset_manifest", "1.0.0")), out, "output") == []
    assert out["kind"] == "report" and "rows" not in out
    manifest_path = Path(out["output"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert problems(_schema("contract://dataset.manifest.document/1.0.0"), manifest, "manifest") == []
    assert manifest["source"] == _HUB_ID
    assert (manifest["revision"], manifest["revision_sha"]) == ("v1", "sha-of-v1")
    assert manifest["rows"] == 2 and manifest["fields"] == ["input", "instruction", "output"]
    assert (manifest["license"], manifest["citation"]) == ("apache-2.0", "@misc{ifc-bim-alpaca}")
    assert manifest["data_content_hash"] == out["data_content_hash"]
    text = manifest_path.read_text(encoding="utf-8")
    assert "IfcWall" not in text and "Industry Foundation" not in text


def test_a_file_source_has_no_licence_and_its_own_hash(tmp_path):
    src = tmp_path / "rows.jsonl"
    src.write_text('{"instruction": "q", "output": "a"}\n', encoding="utf-8")
    work = tmp_path / "work"
    out = _ingest(work, "v1", path=str(src))
    assert out["revision"] is None and out["revision_sha"] is None  # a file ignores `revision`
    manifest = json.loads(Path(dataset_manifest(work, {"store": "local"}, [])["output"]).read_text(encoding="utf-8"))
    assert (manifest["license"], manifest["citation"]) == (None, None)
    assert manifest["source_hash"] == out["source_hash"]
    assert problems(_schema("contract://dataset.manifest.document/1.0.0"), manifest, "manifest") == []


def test_the_manifest_needs_an_ingest_in_the_same_leg(tmp_path):
    (tmp_path / "current.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="ingest.normalize"):
        dataset_manifest(tmp_path, {"store": "local"}, [])


@pytest.mark.parametrize("card", ["empty", "unreadable"])
def test_a_hub_card_without_a_licence_records_null(fake_hub, monkeypatch, tmp_path, card):
    class _Empty:
        class info:  # noqa: N801 - mirrors DatasetInfo's attribute shape
            license = ""
            citation = ""

    def builder(source, **_kw):
        if card == "unreadable":
            raise OSError("card unavailable")
        return _Empty()

    monkeypatch.setattr(_io, "load_dataset_builder", builder)
    _ingest(tmp_path, "v1")
    manifest = json.loads(Path(dataset_manifest(tmp_path, {"store": "local"}, [])["output"]).read_text(encoding="utf-8"))
    assert (manifest["license"], manifest["citation"]) == (None, None)


def test_the_runner_records_a_report_after_the_export_and_keeps_the_uri_parent(fake_hub, tmp_path):
    """Lineage through the core runner's descriptor rules (in process, so the fake Hub holds):
    no `source_hash` for a Hub id, so stage 0 gets no invented registry id and the export is not
    its own parent; the manifest is a `report` whose parent is the export and has no rows."""
    from cairn.core.dataflow import DataflowSpec, StageRef  # noqa: PLC0415
    from cairn.core.dataflow.runner import run_dataflow  # noqa: PLC0415

    spec = DataflowSpec(stages=[
        StageRef(stage="ingest.normalize", config={"path": _HUB_ID, "revision": "v1"}),
        StageRef(stage="export.jsonl", config={"store": "local"}),
        StageRef(stage="dataset.manifest", config={"store": "local"}),
    ])
    ingest, export, report = run_dataflow(spec, tmp_path / "work").output_artifacts
    assert export.content_hash not in ingest.parents
    assert report.kind == "report" and report.rows is None
    assert report.parents == [export.content_hash]
    assert report.stats["data_content_hash"] == export.content_hash


def test_an_unpinned_hub_load_still_records_the_commit_it_got(fake_hub, monkeypatch, tmp_path):
    monkeypatch.setitem(_BY_REVISION, None, _BY_REVISION["v1"])
    out = _ingest(tmp_path, None)
    assert (out["revision"], out["revision_sha"]) == (None, "sha-of-None")
