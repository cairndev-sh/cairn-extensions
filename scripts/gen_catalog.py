"""Write ``catalog.json`` from the packs in this tree (the file says "do not edit by hand").

One entry per pack, in the shape ``cairn.core.registry_catalog`` reads and merges with live
discovery by ``ref``:

* an operator pack (``packages/*/pack.toml`` with ``kind = "operator"``): ``ref``, ``kind``
  ``operator_pack``, ``version``, ``trust``, ``description``, ``operators``, ``provides``,
  ``executors``, ``gpu``, ``install``;
* a domain pack (``packs/*/pack.toml``): the same rule the core's reader applies to a live pack —
  agentic ``flows/*.md`` make it a ``template_pack`` with its ``workflows``; otherwise it is a
  ``domain_pack`` with its ``pipelines`` (``pipelines/*.yaml`` stems, which the loader requires
  to equal each pipeline's ``name:``), installed when one is present as ``<ref>/<pipeline>``.
  ``domain_pack`` is read by the core release that adds it (cairndev-sh/cairn, next release); an
  older reader shows such an entry as not installed.

Contract wheels carry no ``pack.toml`` and are not listed. ``--check`` exits 1 when the committed
file differs from what this would write. Reading a manifest needs ``cairndev`` installed; an
empty tree needs nothing.
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = "catalog.json"
HEADER = {"schema": 1, "channel": "official", "generated_from": "cairn-extensions packs (do not edit by hand)"}


def _distribution(pack_dir: Path) -> str:
    return tomllib.loads((pack_dir / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"]


def _operator_entry(pack_dir: Path) -> dict | None:
    # lazy: only a tree that has packs needs the engine's manifest loader
    from cairn.core.pack.manifest import load_manifest  # noqa: PLC0415

    m = load_manifest(pack_dir / "pack.toml")
    if not m.operators:
        return None
    ops = m.operators.values()
    return {
        "ref": m.name, "kind": "operator_pack", "version": m.version,
        "trust": m.trust_level or "official", "description": m.description,
        "operators": sorted(m.operators),
        "provides": sorted({c for e in ops for c in e.provides_capabilities}),
        "executors": sorted({x for e in ops for x in e.executors}),
        "gpu": any(e.requires_gpu for e in ops),
        "install": {"index": "pypi", "name": _distribution(pack_dir)},
    }




def _domain_entry(pack_dir: Path) -> dict:
    pack = tomllib.loads((pack_dir / "pack.toml").read_text(encoding="utf-8"))["pack"]
    entry = {"ref": pack["name"], "version": pack.get("version", ""),
             "trust": pack.get("trust_level") or "official", "description": pack.get("description", "")}
    workflows = sorted({f.stem for d in ("flows", "templates") for f in (pack_dir / d).glob("*.md")})
    if workflows:
        entry.update(kind="template_pack", workflows=workflows)
    else:
        entry.update(kind="domain_pack", pipelines=sorted(p.stem for p in (pack_dir / "pipelines").glob("*.yaml")))
    entry["install"] = {"index": "pypi", "name": _distribution(pack_dir)}
    return entry


def build(root: Path = ROOT) -> dict:
    entries = []
    for manifest in sorted(root.glob("packages/*/pack.toml")):
        entry = _operator_entry(manifest.parent)
        if entry:
            entries.append(entry)
    for manifest in sorted(root.glob("packs/*/pack.toml")):
        entries.append(_domain_entry(manifest.parent))
    return {**HEADER, "entries": sorted(entries, key=lambda e: e["ref"])}


def render(root: Path = ROOT) -> str:
    return json.dumps(build(root), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = ROOT / CATALOG
    text = render(ROOT)
    if "--check" in args:
        if path.read_text(encoding="utf-8") != text:
            print(f"{CATALOG} is stale: run `python scripts/gen_catalog.py` and commit the result", file=sys.stderr)
            return 1
        print(f"{CATALOG} is current")
        return 0
    path.write_text(text, encoding="utf-8")
    print(f"wrote {CATALOG} ({len(build(ROOT)['entries'])} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
