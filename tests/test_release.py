"""The release workflow's checks: the tag names a distribution at its version, a changed distribution
bumps its version, and a file PyPI already holds is skipped, never uploaded again."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("release", ROOT / "scripts" / "release.py")
release = importlib.util.module_from_spec(_spec)
sys.modules["release"] = release  # a dataclass resolves its module by name
_spec.loader.exec_module(release)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                    "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args], check=True, capture_output=True)


def _member(root: Path, rel: str, version: str) -> None:
    path = root / rel / "pyproject.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
                    f'[project]\nname = "{Path(rel).name}"\nversion = "{version}"\n', encoding="utf-8")


def _commit(root: Path, msg: str = "c") -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _member(tmp_path, "packages/cairn-contracts-a", "1.0.0")
    _member(tmp_path, "packs/cairn-pack-b", "0.1.0")
    _commit(tmp_path)
    return tmp_path


def test_every_distribution_in_this_tree_has_a_comparable_version():
    for d in release.distributions(ROOT):
        release.parse_version(d.version)


def test_distributions_come_from_the_tree_not_a_list(repo):
    assert [d.name for d in release.distributions(repo)] == ["cairn-contracts-a", "cairn-pack-b"]
    _member(repo, "packs/cairn-pack-new", "0.1.0")
    assert "cairn-pack-new" in [d.name for d in release.distributions(repo)]


def test_with_no_tags_every_distribution_is_a_first_release(repo):
    lines, selected, errors = release.plan(repo, None, None)
    assert errors == []
    assert [d.name for d in selected] == ["cairn-contracts-a", "cairn-pack-b"]
    assert all("first release" in line for line in lines)


def test_a_change_without_a_version_bump_fails(repo):
    _git(repo, "tag", "cairn-contracts-a/v1.0.0")
    (repo / "packages/cairn-contracts-a/schema.json").write_text("{}")
    _commit(repo)
    _, _, errors = release.plan(repo, None, None)
    assert len(errors) == 1 and "changed since cairn-contracts-a/v1.0.0" in errors[0]


def test_a_change_with_a_version_bump_passes(repo):
    _git(repo, "tag", "cairn-contracts-a/v1.0.0")
    (repo / "packages/cairn-contracts-a/schema.json").write_text("{}")
    _member(repo, "packages/cairn-contracts-a", "1.1.0")
    _commit(repo)
    lines, _, errors = release.plan(repo, "cairn-contracts-a/v1.1.0", None)
    assert errors == []
    assert "cairn-contracts-a 1.1.0: bumped from cairn-contracts-a/v1.0.0" in lines


def test_another_distributions_unbumped_change_fails_the_release_too(repo):
    _git(repo, "tag", "cairn-pack-b/v0.1.0")
    (repo / "packs/cairn-pack-b/flow.yaml").write_text("x: 1\n")
    _commit(repo)
    _, selected, errors = release.plan(repo, "cairn-contracts-a/v1.0.0", None)
    assert [d.name for d in selected] == ["cairn-contracts-a"]
    assert any("cairn-pack-b" in e and "changed since" in e for e in errors)


def test_a_version_below_the_last_tag_fails(repo):
    _git(repo, "tag", "cairn-contracts-a/v2.0.0")
    _, _, errors = release.plan(repo, None, None)
    assert any("lower than its last release tag" in e for e in errors)


def test_a_tag_on_another_line_is_not_this_lines_last_release(repo):
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "packages/cairn-contracts-a/x").write_text("x")
    _member(repo, "packages/cairn-contracts-a", "9.0.0")
    _commit(repo)
    _git(repo, "tag", "cairn-contracts-a/v9.0.0")
    _git(repo, "checkout", "-q", "main")
    lines, _, errors = release.plan(repo, None, None)
    assert errors == [] and "cairn-contracts-a 1.0.0: first release (no cairn-contracts-a/v* tag)" in lines


@pytest.mark.parametrize("tag, needle", [
    ("cairn-contracts-a/v1.0.1", "says version 1.0.0, not 1.0.1"),
    ("cairn-contracts-zz/v1.0.0", "no distribution named cairn-contracts-zz"),
    ("v0.1.0", "is not <distribution>/vX.Y.Z"),
    ("cairn-contracts-a-v1.0.0", "is not <distribution>/vX.Y.Z"),
])
def test_a_tag_must_name_a_distribution_at_its_version(repo, tag, needle):
    _, _, errors = release.plan(repo, tag, None)
    assert any(needle in e for e in errors), errors


def test_a_tag_off_main_is_refused(repo):
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "packs/cairn-pack-b/x").write_text("x")
    _member(repo, "packs/cairn-pack-b", "0.2.0")
    _commit(repo)
    _, _, errors = release.plan(repo, "cairn-pack-b/v0.2.0", "main")
    assert any("not on main" in e for e in errors)
    _git(repo, "checkout", "-q", "main")
    _, _, errors = release.plan(repo, "cairn-pack-b/v0.1.0", "main")
    assert errors == []


def test_plan_writes_the_selection_for_the_workflow(repo, tmp_path, capsys):
    out = tmp_path / "gh_output"
    assert release.main(["plan", "--tag", "cairn-pack-b/v0.1.0", "--github-output", str(out)], root=repo) == 0
    assert out.read_text().splitlines() == ["distributions=cairn-pack-b", "distribution=cairn-pack-b",
                                            "version=0.1.0", "files=cairn_pack_b-0.1.0*"]
    assert release.main(["plan", "--tag", "cairn-pack-b/v0.9.0"], root=repo) == 1
    assert "::error::release plan:" in capsys.readouterr().out


# --- select: what PyPI already holds is skipped -----------------------------------------------


def _built(dist_dir: Path, stem: str) -> list[Path]:
    dist_dir.mkdir(exist_ok=True)
    files = [dist_dir / f"{stem}.tar.gz", dist_dir / f"{stem}-py3-none-any.whl"]
    for f in files:
        f.write_bytes(f.name.encode())
    return files


def _pypi(held: dict[str, dict[str, bytes]]):
    """A fake PyPI: {"<name>/<version>": {filename: content}}; anything else is a 404."""
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, bytes]:
        calls.append(url)
        key = url.removeprefix(release.PYPI + "/").removesuffix("/json")
        if key not in held:
            return 404, b""
        urls = [{"filename": n, "digests": {"sha256": hashlib.sha256(c).hexdigest()}} for n, c in held[key].items()]
        return 200, json.dumps({"urls": urls}).encode()

    fetch.calls = calls
    return fetch


def test_a_distribution_not_on_pypi_is_selected_whole(repo, tmp_path):
    files = _built(tmp_path / "dist", "cairn_pack_b-0.1.0")
    d = release.Dist("cairn-pack-b", "0.1.0", Path("packs/cairn-pack-b"))
    fetch = _pypi({})
    log, copied = release.select([d], tmp_path / "dist", tmp_path / "up", fetch)
    assert sorted(p.name for p in copied) == sorted(f.name for f in files)
    assert fetch.calls == ["https://pypi.org/pypi/cairn-pack-b/0.1.0/json"]
    assert log == ["upload: cairn-pack-b 0.1.0: cairn_pack_b-0.1.0.tar.gz, cairn_pack_b-0.1.0-py3-none-any.whl"]


def test_a_version_already_on_pypi_is_skipped_with_a_log_line(repo, tmp_path):
    files = _built(tmp_path / "dist", "cairn_pack_b-0.1.0")
    d = release.Dist("cairn-pack-b", "0.1.0", Path("packs/cairn-pack-b"))
    fetch = _pypi({"cairn-pack-b/0.1.0": {f.name: f.read_bytes() for f in files}})
    log, copied = release.select([d], tmp_path / "dist", tmp_path / "up", fetch)
    assert copied == []
    assert log[-1] == "skip: cairn-pack-b 0.1.0 is already on PyPI (every file); nothing to upload"


def test_a_partial_upload_sends_only_the_missing_file(repo, tmp_path):
    sdist, wheel = _built(tmp_path / "dist", "cairn_pack_b-0.1.0")
    d = release.Dist("cairn-pack-b", "0.1.0", Path("packs/cairn-pack-b"))
    fetch = _pypi({"cairn-pack-b/0.1.0": {sdist.name: b"another build"}})
    log, copied = release.select([d], tmp_path / "dist", tmp_path / "up", fetch)
    assert [p.name for p in copied] == [wheel.name]
    assert f"skip: {sdist.name} is already on PyPI (this build differs; PyPI keeps the file it has)" in log


def test_an_unknown_pypi_answer_fails_rather_than_uploading(repo, tmp_path):
    _built(tmp_path / "dist", "cairn_pack_b-0.1.0")
    d = release.Dist("cairn-pack-b", "0.1.0", Path("packs/cairn-pack-b"))
    with pytest.raises(RuntimeError, match="PyPI answered 503"):
        release.select([d], tmp_path / "dist", tmp_path / "up", lambda url: (503, b""))


def test_a_missing_build_fails(repo, tmp_path):
    (tmp_path / "dist").mkdir()
    d = release.Dist("cairn-pack-b", "0.1.0", Path("packs/cairn-pack-b"))
    with pytest.raises(FileNotFoundError, match="expected one sdist and at least one wheel"):
        release.select([d], tmp_path / "dist", tmp_path / "up", _pypi({}))


def test_select_reports_whether_anything_is_left_to_upload(repo, tmp_path):
    files = _built(tmp_path / "dist", "cairn_pack_b-0.1.0")
    out = tmp_path / "gh_output"
    fetch = _pypi({"cairn-pack-b/0.1.0": {f.name: f.read_bytes() for f in files}})
    args = ["select", "--dist-dir", str(tmp_path / "dist"), "--out", str(tmp_path / "up"),
            "--distribution", "cairn-pack-b", "--github-output", str(out)]
    assert release.main(args, root=repo, fetch=fetch) == 0
    assert out.read_text() == "upload=false\n"
    assert release.main([*args[:-4], "--distribution", "nope"], root=repo, fetch=fetch) == 1
