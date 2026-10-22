"""The catalog is generated, and the committed file is what the generator writes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("gen_catalog", ROOT / "scripts" / "gen_catalog.py")
gen_catalog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_catalog)


def test_the_committed_catalog_is_current():
    assert (ROOT / "catalog.json").read_text(encoding="utf-8") == gen_catalog.render(ROOT)


def test_an_empty_tree_writes_the_header_and_no_entries(tmp_path):
    assert json.loads(gen_catalog.render(tmp_path)) == {**gen_catalog.HEADER, "entries": []}


def test_a_domain_pack_is_listed_with_its_pipelines(tmp_path):
    pack = tmp_path / "packs" / "cairn-pack-foo"
    (pack / "pipelines").mkdir(parents=True)
    (pack / "pipelines" / "b.yaml").write_text("name: b\n")
    (pack / "pipelines" / "a.yaml").write_text("name: a\n")
    (pack / "pack.toml").write_text('[pack]\nname = "foo"\nversion = "0.1.0"\ndescription = "d"\n')
    (pack / "pyproject.toml").write_text('[project]\nname = "cairn-pack-foo"\nversion = "0.1.0"\n')
    [entry] = gen_catalog.build(tmp_path)["entries"]
    # the shape cairn.core.registry_catalog reads for a domain_pack: ref = [pack].name (the pipelines'
    # pack), install name = the distribution
    assert entry == {"ref": "foo", "kind": "domain_pack", "version": "0.1.0", "trust": "official",
                     "description": "d", "pipelines": ["a", "b"],
                     "install": {"index": "pypi", "name": "cairn-pack-foo"}}


def test_a_pack_with_agentic_flows_is_a_template_pack(tmp_path):
    pack = tmp_path / "packs" / "cairn-pack-bar"
    (pack / "flows").mkdir(parents=True)
    (pack / "flows" / "assistant.md").write_text("---\nname: assistant\n---\n")
    (pack / "pack.toml").write_text('[pack]\nname = "bar"\nversion = "0.1.0"\n')
    (pack / "pyproject.toml").write_text('[project]\nname = "cairn-pack-bar"\nversion = "0.1.0"\n')
    [entry] = gen_catalog.build(tmp_path)["entries"]
    assert (entry["kind"], entry["workflows"], "pipelines" in entry) == ("template_pack", ["assistant"], False)


def test_a_stale_catalog_fails_the_check(tmp_path, monkeypatch, capsys):
    (tmp_path / "catalog.json").write_text('{"schema": 1, "entries": ["stale"]}\n')
    monkeypatch.setattr(gen_catalog, "ROOT", tmp_path)
    assert gen_catalog.main(["--check"]) == 1
    assert "stale" in capsys.readouterr().err
    assert gen_catalog.main([]) == 0 and gen_catalog.main(["--check"]) == 0


def test_an_operator_pack_is_listed_from_its_manifest(tmp_path):
    pytest.importorskip("cairn.core.pack.manifest")
    pack = tmp_path / "packages" / "cairn-operator-foo"
    pack.mkdir(parents=True)
    (pack / "pyproject.toml").write_text('[project]\nname = "cairn-operator-foo"\nversion = "0.1.0"\n')
    (pack / "pack.toml").write_text(
        '[pack]\nname = "cairn-operator-foo"\nversion = "0.1.0"\ntype = "operator"\ndescription = "d"\n'
        'kind = "operator"\npublisher = "cairn"\ntrust_level = "official"\n\n[compat]\ncairn_engine = ">=0.1,<0.2"\n\n'
        '[operator."foo/bar"]\nversion = "0.1.0"\ninput_schema = "I"\noutput_schema = "O"\nrisk = "read_only"\n'
        'executors = ["in_process"]\nrequires_gpu = true\n\n[operator."foo/bar".capabilities]\nprovides = ["foo.bar"]\n')
    [entry] = gen_catalog.build(tmp_path)["entries"]
    assert (entry["kind"], entry["operators"], entry["provides"], entry["executors"], entry["gpu"]) == (
        "operator_pack", ["foo/bar"], ["foo.bar"], ["in_process"], True)
    assert entry["install"] == {"index": "pypi", "name": "cairn-operator-foo"}
