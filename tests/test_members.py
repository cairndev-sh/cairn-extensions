"""Every workspace member must build its own wheel; one that cannot is refused, never skipped."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("members", ROOT / "scripts" / "members.py")
members = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(members)


def _member(root: Path, rel: str, *, buildable: bool) -> None:
    path = root / rel / "pyproject.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    build = '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n' if buildable else ""
    path.write_text(f'{build}[project]\nname = "{Path(rel).name}"\nversion = "0.1.0"\n', encoding="utf-8")


def test_this_repository_has_no_unbuildable_member():
    assert members.unbuildable(ROOT) == []


def test_a_member_without_a_build_system_is_refused(tmp_path, monkeypatch, capsys):
    _member(tmp_path, "packages/cairn-operator-ok", buildable=True)
    _member(tmp_path, "packs/cairn-pack-folded", buildable=False)
    assert members.unbuildable(tmp_path) == ["packs/cairn-pack-folded"]
    monkeypatch.setattr(members, "ROOT", tmp_path)
    assert members.main([]) == 1
    assert "cairn-pack-folded" in capsys.readouterr().err


def test_buildable_members_are_listed(tmp_path, monkeypatch, capsys):
    _member(tmp_path, "packs/cairn-pack-b", buildable=True)
    _member(tmp_path, "packages/cairn-operator-a", buildable=True)
    monkeypatch.setattr(members, "ROOT", tmp_path)
    assert members.main([]) == 0
    assert capsys.readouterr().out.split() == ["packages/cairn-operator-a", "packs/cairn-pack-b"]


def test_operator_packs_are_selected_from_the_manifest_kind(tmp_path, monkeypatch, capsys):
    _member(tmp_path, "packages/cairn-operator-a", buildable=True)
    (tmp_path / "packages/cairn-operator-a/pack.toml").write_text("[pack]\nkind = 'operator'\n")   # single quotes
    _member(tmp_path, "packages/cairn-contracts-b", buildable=True)
    monkeypatch.setattr(members, "ROOT", tmp_path)
    assert members.main(["--operator-packs"]) == 0
    assert capsys.readouterr().out.split() == ["packages/cairn-operator-a"]
    assert members.main(["--bogus"]) == 2
