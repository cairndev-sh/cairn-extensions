"""Refuse a CI venv that holds anything private (ADR-0128 D3: the inverse of the core's
``check_public_set.py``).

A venv that proves an extension may hold ``cairndev`` and its public wheels, the wheels this
repository builds, and third-party distributions — nothing else of cairn's. Any other
``cairn-operator-*``, ``cairn-contracts-*`` or ``cairn-pack-*``, and any platform distribution,
means a test could have passed only because private code was importable. Exit 1 names each one.
"""
from __future__ import annotations

import re
import sys
import tomllib
from collections.abc import Iterable
from importlib.metadata import distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: cairn's public distributions outside this repository (cairndev-sh/cairn, ADR-0137).
PUBLIC_CORE = {"cairndev", "cairn-client", "cairn-operator-dataflow-metaflow",
               "cairn-operator-dataflow-skypilot", "cairn-pack-hello"}
#: Prefixes cairn's own distributions use; one of these not built here and not public is private.
CAIRN_PREFIXES = ("cairn-operator-", "cairn-contracts-", "cairn-pack-")
#: Distributions that are private whatever their prefix.
PRIVATE_NAMES = {"cairn-platform", "cairn-ask", "cairn-sdk"}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def local_wheels(root: Path = ROOT) -> set[str]:
    """The distributions this repository builds: every member with a build-system table."""
    names = set()
    for path in sorted([*root.glob("packages/*/pyproject.toml"), *root.glob("packs/*/pyproject.toml")]):
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
        if "build-system" in doc:
            names.add(_norm(doc["project"]["name"]))
    return names


def private_installed(root: Path = ROOT, installed: Iterable[tuple[str, str]] | None = None) -> list[str]:
    allowed = PUBLIC_CORE | local_wheels(root)
    if installed is None:
        installed = ((d.metadata["Name"] or "", d.version) for d in distributions())
    found = []
    for raw, version in installed:
        name = _norm(raw)
        if name in allowed:
            continue
        if name in PRIVATE_NAMES or name.startswith(CAIRN_PREFIXES):
            found.append(f"{name} {version}")
    return sorted(set(found))


def main(argv: list[str] | None = None) -> int:
    found = private_installed()
    if found:
        print("not a public venv — private or unknown cairn distributions installed:", file=sys.stderr)
        for f in found:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("public venv: no private cairn distribution installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
