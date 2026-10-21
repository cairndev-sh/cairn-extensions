"""The workspace members this repository builds — every ``packages/*`` and ``packs/*`` with a
``pyproject.toml``. Each one must be buildable (a ``[build-system]`` table): nothing here is folded
into another distribution, so a member without one would be skipped by every check while still
being listed in the catalog. That is refused instead.

``python scripts/members.py`` prints one member directory per line (CI's loops read it) and exits 1
naming any member that is not buildable. ``--operator-packs`` prints only the members whose
``pack.toml`` declares ``kind = "operator"`` (read with tomllib, not matched as text).
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def members(root: Path = ROOT) -> list[Path]:
    return sorted(p.parent for p in [*root.glob("packages/*/pyproject.toml"), *root.glob("packs/*/pyproject.toml")])


def unbuildable(root: Path = ROOT) -> list[str]:
    return [m.relative_to(root).as_posix() for m in members(root)
            if "build-system" not in tomllib.loads((m / "pyproject.toml").read_text(encoding="utf-8"))]


def operator_packs(root: Path = ROOT) -> list[Path]:
    out = []
    for m in members(root):
        manifest = m / "pack.toml"
        if manifest.is_file() and tomllib.loads(manifest.read_text(encoding="utf-8")).get("pack", {}).get("kind") == "operator":
            out.append(m)
    return out


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args not in ([], ["--operator-packs"]):
        print(f"members: unknown arguments {args}", file=sys.stderr)
        return 2
    bad = unbuildable(ROOT)
    if bad:
        print("members without a [build-system] table (every member here must build its own wheel):", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 1
    for m in (operator_packs(ROOT) if args else members(ROOT)):
        print(m.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
