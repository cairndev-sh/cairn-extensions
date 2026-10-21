"""The venv check admits the public set and this repository's wheels, and nothing else of cairn's."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_venv", ROOT / "scripts" / "check_venv.py")
check_venv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_venv)


def _wheel(root: Path, rel: str, name: str, *, buildable: bool = True) -> None:
    path = root / rel / "pyproject.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    build = '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n' if buildable else ""
    path.write_text(f'{build}[project]\nname = "{name}"\nversion = "0.1.0"\n', encoding="utf-8")


def test_the_public_set_and_local_wheels_pass(tmp_path):
    _wheel(tmp_path, "packages/cairn-operator-foo", "cairn-operator-foo")
    installed = [("cairndev", "0.1.5"), ("cairn_client", "0.1.5"), ("cairn-operator-foo", "0.1.0"), ("requests", "2")]
    assert check_venv.private_installed(tmp_path, installed) == []


def test_a_cairn_wheel_not_built_here_is_refused(tmp_path):
    installed = [("cairndev", "0.1.5"), ("cairn-operator-secret", "0.1.0"), ("cairn-contracts-secret", "1.0.0")]
    assert check_venv.private_installed(tmp_path, installed) == [
        "cairn-contracts-secret 1.0.0", "cairn-operator-secret 0.1.0"]


def test_a_platform_distribution_is_refused(tmp_path):
    assert check_venv.private_installed(tmp_path, [("cairn-platform", "0.1.0")]) == ["cairn-platform 0.1.0"]


def test_a_member_without_a_build_system_is_not_a_local_wheel(tmp_path):
    _wheel(tmp_path, "packages/cairn-operator-foo", "cairn-operator-foo", buildable=False)
    assert check_venv.private_installed(tmp_path, [("cairn-operator-foo", "0.1.0")]) == ["cairn-operator-foo 0.1.0"]
