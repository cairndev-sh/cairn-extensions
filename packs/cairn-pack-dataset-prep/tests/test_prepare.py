"""dataset-prep/prepare end to end, in process (cairn-extensions#16).

The pack's own flow is resolved through the core (`resolve_flow_ref`, with the pack loaded from
this checkout), run by the core runner on the 200-row IFC-shaped sample with the
`ifc-bim-alpaca` preset's rules, and registered through the leg's own registration path
(`register_leg_descriptors`, which stamps each version's intake verdict). Then the holdout door
runs on what was registered. Hermetic: a local file instead of the Hub, whitespace token counts,
no minhash extra (near dedup is an honest skip), sqlite."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

PACK = Path(__file__).resolve().parents[1]
SAMPLE = PACK / "data" / "sample-ifc.jsonl"


@pytest.fixture(autouse=True)
def _pack_from_checkout(monkeypatch):
    """Load this checkout's pack as a dev pack (CAIRN_PACK_PATH), as `cairn` does at start."""
    from cairn.core.pack import registry
    from cairn.core.pack.loader import load_all_packs

    monkeypatch.setenv("CAIRN_PACK_PATH", str(PACK))
    registry.clear()
    load_all_packs()
    yield
    registry.clear()


def _preset_config(**over) -> dict:
    doc = yaml.safe_load((PACK / "presets" / "ifc-bim-alpaca.yaml").read_text(encoding="utf-8"))
    (preset,) = [p for p in doc["presets"] if p["name"] == "ifc-bim-alpaca"]
    # The sample instead of the Hub; no tokenizer download; files in the workdir.
    return {**preset["inputs"], "path": str(SAMPLE), "revision": "", "base_model": "", "store": "local", **over}


def _prepare(workdir: Path, config: dict):
    from cairn.core.dataflow import Budget, DataflowSpec
    from cairn.core.dataflow.defs import resolve_flow_ref
    from cairn.core.dataflow.runner import run_dataflow

    stages = resolve_flow_ref("dataset-prep/prepare", config, kind="data")
    return run_dataflow(DataflowSpec(stages=stages, budget=Budget()), workdir)


async def _register(result, store) -> dict:
    from cairn.core.dispatch.leg_executor import register_leg_descriptors

    out_stats = dict(result.stats)  # the leg's stats are what the intake verdict reads
    await register_leg_descriptors(result.output_artifacts + result.named_outputs, store,
                                   run_id="run-prepare", out_stats=out_stats)
    return out_stats


@pytest.fixture
async def store(tmp_path):
    from cairn.core.storage.artifacts import SqliteArtifactStore

    s = SqliteArtifactStore(str(tmp_path / "registry.db"))
    yield s
    await s.close()


async def test_prepare_registers_three_versions_with_their_roles_and_verdicts(tmp_path, store):
    result = _prepare(tmp_path / "wd", _preset_config())
    stats = await _register(result, store)
    assert stats["dataset_versions"] == ["ifc-bim-alpaca-train@1", "ifc-bim-alpaca-val@1", "ifc-bim-alpaca-test@1"]
    for name, role in (("train", "training"), ("val", "validation"), ("test", "test")):
        meta = await store.get_dataset_version_meta(f"ifc-bim-alpaca-{name}")
        assert meta["role"] == role
    assert all(v.endswith(("=admitted", "=degraded")) for v in stats["intake_verdicts"]), stats["intake_verdicts"]
    # The audit: duplicates, prompt+response lengths, topics; the split kept every family on one side.
    assert (stats["ingest.normalize.rows"], stats["dedup.exact.dropped"]) == (200, 6)
    assert stats["profile.tokens.measure"] == "example" and stats["profile.tokens.prompt_tokens_p95"] > 0
    topics = dict(zip(stats["profile.fields.topic_values"], stats["profile.fields.topic_counts"]))
    assert topics["IFCWALLSTANDARDCASE"] == 24 and topics["other"] == 2
    assert stats["split.no_leakage"] is True
    assert sum(d.rows for d in result.named_outputs) == 194


async def test_the_manifest_and_the_record_carry_no_row_content(tmp_path, store):
    result = _prepare(tmp_path / "wd", _preset_config())
    stats = await _register(result, store)
    manifest = json.loads(Path(stats["dataset.manifest.output"]).read_text(encoding="utf-8"))
    assert manifest["rows"] == 194 and set(manifest["fields"]) == {"instruction", "input", "output", "topic", "family"}
    record = json.dumps(stats) + json.dumps(manifest)
    for text in ("shear wall", "IFCPROJECT(", "Industry Foundation Classes, an open"):
        assert text not in record


def test_the_same_source_and_seed_give_the_same_three_versions(tmp_path):
    first = _prepare(tmp_path / "a", _preset_config())
    again = _prepare(tmp_path / "b", _preset_config())
    other = _prepare(tmp_path / "c", _preset_config(seed=1))
    hashes = [[d.content_hash for d in r.named_outputs] for r in (first, again, other)]
    assert hashes[0] == hashes[1] and hashes[0] != hashes[2]


async def test_the_holdout_door_refuses_training_on_data_that_holds_the_test_split(tmp_path, store):
    from cairn.core.holdout import check_holdout

    result = _prepare(tmp_path / "wd", _preset_config())
    await _register(result, store)
    clean = await check_holdout(["dataset:ifc-bim-alpaca-train@1"], store=store, workspace_id="default")
    assert clean.checked and clean.ok and [h.name for h in clean.holdouts] == ["ifc-bim-alpaca-test"]
    # The whole cleaned corpus (the leg's export) registered as training data holds every test row.
    export = next(d for d in result.output_artifacts if d.stats.get("format") == "jsonl")
    await store.register_dataset_version("ifc-bim-alpaca-all", export.content_hash, actor="t")
    leaky = await check_holdout(["dataset:ifc-bim-alpaca-all@1"], store=store, workspace_id="default")
    assert leaky.checked and not leaky.ok and leaky.overlaps
