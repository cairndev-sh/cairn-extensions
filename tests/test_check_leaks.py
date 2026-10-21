"""The leak scan refuses what a public extension must not carry, and passes this tree.

Private-token detection is tested with a harmless stand-in word in the environment variable, so
this public file names nothing private either.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_leaks", ROOT / "scripts" / "check_leaks.py")
check_leaks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_leaks)
SRC = "packages/cairn-operator-foo/src/operator_foo/x.py"


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _categories(root: Path) -> list[str]:
    return sorted(h.split(":")[0] for h in check_leaks.scan(root))


def _stand_in(monkeypatch, words: str = "zebracorn\nquux-corp") -> None:
    monkeypatch.setenv(check_leaks.PRIVATE_TOKENS_ENV, words)


def test_this_repository_is_clean():
    assert check_leaks.scan(ROOT) == []


def test_a_module_importing_the_platform_is_refused(tmp_path):
    _write(tmp_path, SRC, "import cairn_platform.fidelity\n")
    assert "platform import" in _categories(tmp_path)
    assert check_leaks.main([str(tmp_path)]) == 1


def test_another_operator_pack_import_is_refused_and_its_own_is_not(tmp_path):
    _write(tmp_path, SRC, "from operator_foo import y\n")
    assert _categories(tmp_path) == []
    _write(tmp_path, SRC, "from operator_bar.z import w\n")
    assert _categories(tmp_path) == ["foreign operator import"]


def test_an_operator_import_outside_any_operator_pack_is_refused(tmp_path):
    _write(tmp_path, "packs/cairn-pack-foo/src/cairn_pack_foo/__init__.py", "import operator_foo\n")
    assert _categories(tmp_path) == ["foreign operator import"]


def test_package_source_may_import_only_the_allowed_core_modules(tmp_path):
    _write(tmp_path, SRC, "from cairn.core.operator import Operation\nfrom cairn.core.dataflow.fidelity import plan\n")
    assert _categories(tmp_path) == []
    _write(tmp_path, SRC, "from cairn.core.dispatch.start import launch\n")
    assert _categories(tmp_path) == ["core internals import"]
    # tests are not bound by the import allowlist
    _write(tmp_path, SRC, "")
    _write(tmp_path, "packages/cairn-operator-foo/tests/test_x.py", "from cairn.core.storage.memory import runs\n")
    assert _categories(tmp_path) == []


def test_domain_pack_and_platform_only_core_imports_are_refused(tmp_path):
    _write(tmp_path, "scripts/a.py", "from cairn.packs.training import x\nimport cairn.core.chat\n")
    assert _categories(tmp_path) == ["domain-pack import", "platform-only core surface"]


def test_a_private_token_is_refused_at_a_word_start_and_never_echoed(tmp_path, monkeypatch):
    _stand_in(monkeypatch)
    _write(tmp_path, "README.md", "see https://example.com/ZebraCorn-docs\nand zebracornish too\n"
                                  "but not azebracorn\nmail ops@QUUX-CORP.example\n")
    hits = check_leaks.scan(tmp_path)
    assert [h.split(":")[0] for h in hits] == ["private token"] * 3
    assert all("zebra" not in h.lower() and "quux" not in h.lower() for h in hits)


def test_without_tokens_the_check_is_skipped_locally_and_refused_when_required(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(check_leaks.PRIVATE_TOKENS_ENV, raising=False)
    _write(tmp_path, "README.md", "clean\n")
    assert check_leaks.main([str(tmp_path)]) == 0
    assert "skipped" in capsys.readouterr().out
    assert check_leaks.main(["--require-private-tokens", str(tmp_path)]) == 1


def test_private_tokens_have_no_exemption(tmp_path, monkeypatch):
    _stand_in(monkeypatch)
    _write(tmp_path, "scripts/check_leaks.py", "x = 'zebracorn'\n")
    assert _categories(tmp_path) == ["private token"]


def test_a_text_exemption_covers_only_its_own_categories(tmp_path):
    _write(tmp_path, "scripts/check_venv.py", 'NAMES = {"cairn-platform"}\nOLD = "obs-operator-x"\n')
    assert _categories(tmp_path) == ["legacy pack name"]


def test_public_text_patterns_are_refused_in_any_file(tmp_path):
    _write(tmp_path, "README.md", "name = obs-operator-foo\nrepo: cairn-platform\n")
    assert _categories(tmp_path) == ["legacy pack name", "platform name"]


def test_a_docstring_naming_a_module_is_not_an_import(tmp_path):
    _write(tmp_path, SRC, '"""Reads operator_bar output."""\n')
    assert _categories(tmp_path) == []


def test_the_git_file_list_is_scanned_including_a_force_added_ignored_file(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _write(tmp_path, ".gitignore", "build/\n")
    _write(tmp_path, "build/leak.py", "import cairn_platform\n")
    assert _categories(tmp_path) == []          # ignored and untracked: not part of a PR
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", "build/leak.py"], check=True)
    assert _categories(tmp_path) == ["platform import", "platform name"]


def test_a_token_that_could_never_match_is_refused(monkeypatch):
    import pytest

    monkeypatch.setenv(check_leaks.PRIVATE_TOKENS_ENV, "zebracorn\n.hidden")
    with pytest.raises(SystemExit, match="line"):
        check_leaks.private_tokens()

