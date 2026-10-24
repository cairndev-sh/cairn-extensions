"""The release workflow's checks (``.github/workflows/release.yml``; ADR-0128 D3: each distribution
releases on its own tag, ``<distribution>/vX.Y.Z``, and there is no repository-wide version).

``python3 scripts/release.py plan [--tag TAG] [--main REF] [--github-output FILE]``
    Reads every distribution from the tree (``scripts/members.py``: each ``packages/*`` and
    ``packs/*`` with a ``pyproject.toml``; nothing is listed by hand) and, for each, its last release
    tag reachable from ``HEAD``. Fails when a distribution's directory changed since that tag but its
    version did not go up, or went down. With ``--tag`` it also fails unless the tag names a
    distribution in the tree at exactly the version its ``pyproject.toml`` carries, and, with
    ``--main``, unless the tagged commit is on that branch. Selects the tagged distribution, or every
    distribution without ``--tag`` (a dry run). Asks nothing of the network.

``python3 scripts/release.py select --dist-dir DIR --out DIR [--distribution NAME ...] [--github-output FILE]``
    For each named distribution (default: every one), finds its sdist and wheel in ``--dist-dir``
    and asks PyPI which of those files it already holds (``/pypi/<name>/<version>/json``). A file
    PyPI has is skipped with a log line, never uploaded again: an identical filename is refused by
    PyPI, and a rerun must not fail on it. The rest are copied to ``--out`` for the publish action.
    A distribution whose every file is on PyPI is skipped as a whole. Any answer other than 200 or
    404 fails: an unknown state is never read as "not on PyPI".

``python3 scripts/release.py mode --event E --ref-type T [--publish-input X] [--github-output FILE]``
    Whether this run publishes: a tag push does; ``workflow_dispatch`` does only with
    ``publish: true`` and only from a tag ref (``publish: true`` from a branch is refused);
    anything else is a dry run.

``python3 scripts/release.py check-environment --repo OWNER/REPO [--name pypi] [--warn-only] [--github-output FILE]``
    Fails unless the GitHub environment exists, has at least one required reviewer, and may
    deploy only from tags (custom deployment policies, every one of type ``tag``). GitHub creates a
    referenced environment on first use with no protection, and PyPI's trusted publisher matches
    the environment by name only, so without this check a tag pushed before the environment is
    protected would publish unapproved. Reads ``GITHUB_TOKEN`` when set (the repository is public,
    so the API also answers without one). ``--warn-only`` (the dry run) reports instead of failing.

Versions are ``X.Y.Z`` (integers); anything else is refused, so the order of two versions is never
a guess.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import members as _members  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PYPI = "https://pypi.org/pypi"
GITHUB_API = "https://api.github.com"
_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_TAG = re.compile(r"^(?P<name>[a-z0-9][a-z0-9-]*[a-z0-9])/v(?P<version>[0-9][0-9.]*)$")


@dataclass(frozen=True)
class Dist:
    name: str
    version: str
    path: Path  # relative to the repository root

    @property
    def file_stem(self) -> str:
        """The filename prefix of this distribution's sdist and wheel (PEP 625 / PEP 427)."""
        return f"{re.sub(r'[-_.]+', '_', self.name).lower()}-{self.version}"


def parse_version(text: str) -> tuple[int, int, int]:
    m = _VERSION.match(text)
    if not m:
        raise ValueError(f"version {text!r} is not X.Y.Z")
    return int(m[1]), int(m[2]), int(m[3])


def distributions(root: Path = ROOT) -> list[Dist]:
    out: list[Dist] = []
    for member in _members.members(root):
        project = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        out.append(Dist(project["name"], project["version"], member.relative_to(root)))
    return out


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def last_tag(root: Path, name: str) -> tuple[str, tuple[int, int, int]] | None:
    """The highest ``<name>/vX.Y.Z`` tag reachable from HEAD (a tag on another line is not this one's past)."""
    res = _git(root, "tag", "--merged", "HEAD", "--list", f"{name}/v*")
    if res.returncode:
        raise RuntimeError(f"git tag --merged HEAD failed: {res.stderr.strip()}")
    best: tuple[str, tuple[int, int, int]] | None = None
    for tag in res.stdout.split():
        try:
            v = parse_version(tag[len(name) + 2:])
        except ValueError:
            continue  # not a release tag of this distribution
        if best is None or v > best[1]:
            best = (tag, v)
    return best


def changed_since(root: Path, tag: str, path: Path) -> bool:
    res = _git(root, "diff", "--quiet", tag, "HEAD", "--", path.as_posix())
    if res.returncode not in (0, 1):
        raise RuntimeError(f"git diff {tag} HEAD -- {path} failed: {res.stderr.strip()}")
    return res.returncode == 1


def bump_report(root: Path, dists: list[Dist]) -> tuple[list[str], list[str]]:
    """(one line per distribution, the failures)."""
    lines: list[str] = []
    errors: list[str] = []
    for d in dists:
        try:
            v = parse_version(d.version)
        except ValueError as e:
            errors.append(f"{d.name}: {e}")
            continue
        prev = last_tag(root, d.name)
        if prev is None:
            lines.append(f"{d.name} {d.version}: first release (no {d.name}/v* tag)")
            continue
        tag, tv = prev
        changed = changed_since(root, tag, d.path)
        if v < tv:
            errors.append(f"{d.name}: version {d.version} is lower than its last release tag {tag}")
        elif v == tv and changed:
            errors.append(f"{d.name}: {d.path.as_posix()} changed since {tag} but its version is still "
                          f"{d.version}; bump it in {d.path.as_posix()}/pyproject.toml")
        elif v == tv:
            lines.append(f"{d.name} {d.version}: unchanged since {tag}")
        else:
            lines.append(f"{d.name} {d.version}: bumped from {tag}" + ("" if changed else " (no other change)"))
    return lines, errors


def plan(root: Path, tag: str | None, main_ref: str | None) -> tuple[list[str], list[Dist], list[str]]:
    dists = distributions(root)
    names = [d.name for d in dists]
    errors = [f"two members are both named {n}" for n in sorted({n for n in names if names.count(n) > 1})]
    lines, bump_errors = bump_report(root, dists)
    errors += bump_errors
    selected = dists
    if tag is not None:
        m = _TAG.match(tag)
        by_name = {d.name: d for d in dists}
        if not m:
            errors.append(f"tag {tag!r} is not <distribution>/vX.Y.Z")
            selected = []
        elif m["name"] not in by_name:
            errors.append(f"tag {tag}: no distribution named {m['name']} in the tree "
                          f"(there are: {', '.join(sorted(by_name))})")
            selected = []
        else:
            d = by_name[m["name"]]
            if m["version"] != d.version:
                errors.append(f"tag {tag}: {d.path.as_posix()}/pyproject.toml says version {d.version}, "
                              f"not {m['version']}")
            selected = [d]
        if main_ref is not None and not main_ref.startswith("refs/"):
            # `origin/main` would resolve to a tag of that name before the remote branch
            errors.append(f"--main {main_ref!r} must be a fully qualified ref (refs/remotes/origin/main)")
        elif main_ref is not None:
            res = _git(root, "merge-base", "--is-ancestor", "HEAD", main_ref)
            if res.returncode == 1:
                errors.append(f"tag {tag}: the tagged commit is not on {main_ref}; tag a commit of main")
            elif res.returncode:
                errors.append(f"tag {tag}: cannot check the commit against {main_ref}: {res.stderr.strip()}")
    return lines, selected, errors


def http_get(url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    """(status, body) for a GET; an HTTP error is a status, a network error (URLError) is raised."""
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "cairn-extensions-release", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def default_fetch(url: str) -> tuple[int, bytes]:
    return http_get(url)


def mode(event: str, ref_type: str, publish_input: str) -> tuple[bool, str | None]:
    """(publish?, the refusal). Only a tag publishes; a dispatch publishes only when asked, from a tag."""
    if event == "push":
        if ref_type != "tag":
            return False, f"a push of a {ref_type or 'non-tag'} ref never publishes; push a <distribution>/vX.Y.Z tag"
        return True, None
    if event == "workflow_dispatch" and publish_input == "true":
        if ref_type != "tag":
            return False, ("publish: true runs from a release tag (<distribution>/vX.Y.Z); this run is on a "
                           f"{ref_type or 'non-tag ref'}. Use publish: false for a dry run.")
        return True, None
    return False, None


def check_environment(repo: str, name: str, fetch: Callable[[str], tuple[int, bytes]],
                      api: str = GITHUB_API) -> list[str]:
    """What is missing from the environment's protection ([] when it is protected)."""
    url = f"{api}/repos/{repo}/environments/{name}"
    status, body = fetch(url)
    if status == 404:
        return [f"the {name!r} environment does not exist in {repo}; create it with a required reviewer and the "
                f"*/v* tag rule before any release (RELEASING.md, step 1)"]
    if status != 200:
        return [f"cannot read the {name!r} environment ({url} answered {status}); refusing to publish unchecked"]
    env = json.loads(body)
    rules = env.get("protection_rules") or []
    errors = []
    if not any(r.get("type") == "required_reviewers" and r.get("reviewers") for r in rules):
        errors.append(f"the {name!r} environment has no required reviewer; add the owner")
    policy = env.get("deployment_branch_policy") or {}
    if not policy.get("custom_branch_policies"):
        errors.append(f"the {name!r} environment may deploy from any branch or tag; restrict it to the */v* tag rule")
    else:
        status, body = fetch(f"{url}/deployment-branch-policies?per_page=100")
        if status != 200:
            errors.append(f"cannot read the {name!r} environment's deployment policies (answered {status})")
        else:
            policies = json.loads(body).get("branch_policies", [])
            if not policies:
                errors.append(f"the {name!r} environment has no deployment tag rule; add */v*")
            for p in policies:
                if p.get("type", "branch") != "tag":
                    errors.append(f"the {name!r} environment may deploy from branch {p.get('name')!r}; "
                                  f"allow only the */v* tag rule")
    return errors


def files_of(dist: Dist, dist_dir: Path) -> list[Path]:
    stem = dist.file_stem
    sdists = sorted(dist_dir.glob(f"{stem}.tar.gz"))
    wheels = sorted(dist_dir.glob(f"{stem}-*.whl"))
    if len(sdists) != 1 or not wheels:
        raise FileNotFoundError(f"{dist.name} {dist.version}: expected one sdist and at least one wheel named "
                                f"{stem}* in {dist_dir}, found {len(sdists)} sdist(s) and {len(wheels)} wheel(s)")
    return [*sdists, *wheels]


def on_pypi(dist: Dist, fetch: Callable[[str], tuple[int, bytes]]) -> dict[str, str]:
    """Filename -> sha256 of the files PyPI holds for this exact version ({} when it holds none)."""
    url = f"{PYPI}/{dist.name}/{dist.version}/json"
    status, body = fetch(url)
    if status == 404:
        return {}
    if status != 200:
        raise RuntimeError(f"{dist.name} {dist.version}: PyPI answered {status} for {url}; refusing to guess")
    return {f["filename"]: f.get("digests", {}).get("sha256", "") for f in json.loads(body).get("urls", [])}


def select(dists: list[Dist], dist_dir: Path, out: Path,
           fetch: Callable[[str], tuple[int, bytes]] = default_fetch) -> tuple[list[str], list[Path]]:
    """(log lines, the files copied to ``out``)."""
    out.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    copied: list[Path] = []
    for d in dists:
        files = files_of(d, dist_dir)
        held = on_pypi(d, fetch)
        todo = [f for f in files if f.name not in held]
        for f in files:
            if f.name in held:
                same = hashlib.sha256(f.read_bytes()).hexdigest() == held[f.name]
                log.append(f"skip: {f.name} is already on PyPI"
                           + ("" if same else " (this build differs; PyPI keeps the file it has)"))
        if not todo:
            log.append(f"skip: {d.name} {d.version} is already on PyPI (every file); nothing to upload")
            continue
        for f in todo:
            shutil.copy2(f, out / f.name)
            copied.append(out / f.name)
        log.append(f"upload: {d.name} {d.version}: {', '.join(f.name for f in todo)}")
    return log, copied


def _write_output(path: str | None, values: dict[str, str]) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for k, v in values.items():
            fh.write(f"{k}={v}\n")


def main(argv: list[str] | None = None, root: Path = ROOT,
         fetch: Callable[[str], tuple[int, bytes]] = default_fetch,
         github_fetch: Callable[[str], tuple[int, bytes]] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="release.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--tag")
    p.add_argument("--main", dest="main_ref")
    p.add_argument("--github-output")
    s = sub.add_parser("select")
    s.add_argument("--dist-dir", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--distribution", action="append", default=[])
    s.add_argument("--github-output")
    m = sub.add_parser("mode")
    m.add_argument("--event", required=True)
    m.add_argument("--ref-type", default="")
    m.add_argument("--publish-input", default="")
    m.add_argument("--github-output")
    e = sub.add_parser("check-environment")
    e.add_argument("--repo", required=True)
    e.add_argument("--name", default="pypi")
    e.add_argument("--warn-only", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "mode":
        publish, refusal = mode(args.event, args.ref_type, args.publish_input)
        if refusal:
            print(f"::error::release mode: {refusal}")
            return 1
        print(f"publish: {'true' if publish else 'false'}")
        _write_output(args.github_output, {"publish": "true" if publish else "false"})
        return 0

    if args.cmd == "check-environment":
        if github_fetch is None:
            token = os.environ.get("GITHUB_TOKEN", "")
            auth = {"Authorization": f"Bearer {token}"} if token else {}
            gh_fetch = lambda url: http_get(url, {"Accept": "application/vnd.github+json", **auth})  # noqa: E731
        else:
            gh_fetch = github_fetch
        try:
            errors = check_environment(args.repo, args.name, gh_fetch)
        except (OSError, ValueError) as exc:
            errors = [f"cannot read the {args.name!r} environment: {exc}"]
        level = "warning" if args.warn_only else "error"
        for err in errors:
            print(f"::{level}::release environment: {err}")
        if errors:
            return 0 if args.warn_only else 1
        print(f"environment {args.name}: required reviewer and tag-only deployment rules in place")
        return 0

    if args.cmd == "plan":
        try:
            lines, selected, errors = plan(root, args.tag, args.main_ref)
        except (RuntimeError, OSError, KeyError, tomllib.TOMLDecodeError) as e:
            print(f"release plan: {e}", file=sys.stderr)
            return 1
        for line in lines:
            print(line)
        for e in errors:
            print(f"::error::release plan: {e}")
        if errors:
            return 1
        names = " ".join(d.name for d in selected)
        print(f"selected: {names}")
        values = {"distributions": names}
        if args.tag is not None and len(selected) == 1:
            values |= {"distribution": selected[0].name, "version": selected[0].version,
                       "files": f"{selected[0].file_stem}*"}
        _write_output(args.github_output, values)
        return 0

    dists = distributions(root)
    by_name = {d.name: d for d in dists}
    unknown = [n for n in args.distribution if n not in by_name]
    if unknown:
        print(f"release select: no distribution named {', '.join(unknown)} in the tree", file=sys.stderr)
        return 1
    chosen = [by_name[n] for n in args.distribution] if args.distribution else dists
    try:
        log, copied = select(chosen, args.dist_dir, args.out, fetch)
    except (RuntimeError, OSError, ValueError) as e:
        print(f"::error::release select: {e}")
        return 1
    for line in log:
        print(line)
    _write_output(args.github_output, {"upload": "true" if copied else "false"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
