"""The release workflow's checks: the tag names a distribution at its version, a changed distribution
bumps its version, and a file PyPI already holds is skipped, never uploaded again."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import urllib.error
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
    _, _, errors = release.plan(repo, "cairn-pack-b/v0.2.0", "refs/heads/main")
    assert any("not on refs/heads/main" in e for e in errors)
    _git(repo, "checkout", "-q", "main")
    _, _, errors = release.plan(repo, "cairn-pack-b/v0.1.0", "refs/heads/main")
    assert errors == []


def test_a_tag_named_like_the_main_branch_cannot_stand_in_for_it(repo):
    # refs/remotes/origin/main is main; a tag literally named `origin/main` sits on an unmerged commit
    _git(repo, "update-ref", "refs/remotes/origin/main", "main")
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "packs/cairn-pack-b/x").write_text("x")
    _member(repo, "packs/cairn-pack-b", "0.2.0")
    _commit(repo)
    _git(repo, "tag", "origin/main")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    short = subprocess.run(["git", "-C", str(repo), "rev-parse", "origin/main"], capture_output=True, text=True).stdout
    assert short == head  # the hazard: the short name resolves to the tag, not the branch
    _, _, errors = release.plan(repo, "cairn-pack-b/v0.2.0", "origin/main")
    assert any("must be a fully qualified ref" in e for e in errors)
    _, _, errors = release.plan(repo, "cairn-pack-b/v0.2.0", "refs/remotes/origin/main")
    assert any("not on refs/remotes/origin/main" in e for e in errors)


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


def test_select_fails_when_pypi_cannot_be_reached(repo, tmp_path, monkeypatch, capsys):
    _built(tmp_path / "dist", "cairn_pack_b-0.1.0")

    def down(*a, **k):
        raise urllib.error.URLError("network is unreachable")

    monkeypatch.setattr(release.urllib.request, "urlopen", down)
    args = ["select", "--dist-dir", str(tmp_path / "dist"), "--out", str(tmp_path / "up"),
            "--distribution", "cairn-pack-b"]
    assert release.main(args, root=repo) == 1
    assert "network is unreachable" in capsys.readouterr().out
    assert not (tmp_path / "up").exists() or not any((tmp_path / "up").iterdir())


# --- mode: only a tag publishes ---------------------------------------------------------------


@pytest.mark.parametrize("event, ref_type, publish_input, publish, refused", [
    ("push", "tag", "", True, False),
    ("push", "branch", "", False, True),
    ("workflow_dispatch", "branch", "false", False, False),
    ("workflow_dispatch", "tag", "false", False, False),
    ("workflow_dispatch", "tag", "true", True, False),
    ("workflow_dispatch", "branch", "true", False, True),
])
def test_only_a_tag_ref_publishes(event, ref_type, publish_input, publish, refused):
    got, refusal = release.mode(event, ref_type, publish_input)
    assert got is publish and (refusal is not None) is refused


def test_publish_true_off_a_tag_fails_the_run(tmp_path, capsys):
    out = tmp_path / "gh_output"
    assert release.main(["mode", "--event", "workflow_dispatch", "--ref-type", "branch", "--publish-input", "true",
                         "--github-output", str(out)]) == 1
    assert "publish: true runs from a release tag" in capsys.readouterr().out
    assert not out.exists()
    assert release.main(["mode", "--event", "workflow_dispatch", "--ref-type", "branch", "--publish-input", "false",
                         "--github-output", str(out)]) == 0
    assert out.read_text() == "publish=false\n"


# --- check-environment: the pypi environment must be protected before anything publishes ------

_ENV = "https://api.github.com/repos/o/r/environments/pypi"
_REVIEWER = {"type": "required_reviewers", "reviewers": [{"type": "User", "reviewer": {"login": "owner"}}]}


def _github(env: dict | None, policies: list[dict] | None = None):
    def fetch(url: str) -> tuple[int, bytes]:
        if env is None:
            return 404, b""
        if url == _ENV:
            return 200, json.dumps(env).encode()
        if url == _ENV + "/deployment-branch-policies?per_page=100":
            return 200, json.dumps({"branch_policies": policies or []}).encode()
        return 404, b""
    return fetch


_PROTECTED = {"protection_rules": [_REVIEWER, {"type": "branch_policy"}],
              "deployment_branch_policy": {"protected_branches": False, "custom_branch_policies": True}}


def test_a_protected_environment_passes():
    assert release.check_environment("o/r", "pypi", _github(_PROTECTED, [{"name": "*/v*", "type": "tag"}])) == []


@pytest.mark.parametrize("env, policies, needle", [
    (None, None, "does not exist"),
    ({"protection_rules": [], "deployment_branch_policy": None}, None, "no required reviewer"),
    ({"protection_rules": [_REVIEWER], "deployment_branch_policy": None}, None, "any branch or tag"),
    (_PROTECTED, [], "no deployment tag rule"),
    (_PROTECTED, [{"name": "*/v*", "type": "tag"}, {"name": "main", "type": "branch"}], "from branch 'main'"),
])
def test_an_unprotected_environment_fails(env, policies, needle):
    errors = release.check_environment("o/r", "pypi", _github(env, policies))
    assert any(needle in e for e in errors), errors


def test_check_environment_fails_closed_and_the_dry_run_only_warns(capsys):
    args = ["check-environment", "--repo", "o/r"]
    assert release.main(args, github_fetch=_github(None)) == 1
    assert "::error::release environment:" in capsys.readouterr().out
    assert release.main([*args, "--warn-only"], github_fetch=_github(None)) == 0
    assert "::warning::release environment:" in capsys.readouterr().out
    assert release.main(args, github_fetch=lambda url: (403, b"")) == 1

    def down(url):
        raise urllib.error.URLError("network is unreachable")

    assert release.main(args, github_fetch=down) == 1
    assert release.main(args, github_fetch=_github(_PROTECTED, [{"name": "*/v*", "type": "tag"}])) == 0
