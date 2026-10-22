"""Refuse a tree that carries what a public extension must not (ADR-0128 D1/D5).

Three scans over the repository's files (git's tracked and untracked-but-not-ignored files; a
directory that is not a git checkout is walked instead):

* **imports** (Python files, by their syntax tree, so a docstring that names a module is not a
  leak): no ``cairn_platform``, no ``cairn.packs``, no platform-only core surfaces; no
  ``operator_<x>`` import outside the operator pack ``packages/cairn-operator-<x>`` that owns it (an
  operator pack never imports another); and, in a package's ``src/``, ``cairn.*`` only from the
  modules ADR-0128 D5 allows an extension;
* **private tokens** (every file): a list this public repository never holds — not even as hashes —
  read from ``LEAK_SCAN_PRIVATE_TOKENS`` (one token per line; CI supplies it from a repository
  secret). Each matches case-insensitively at the start of a word, and a hit never echoes it.
  ``--require-private-tokens`` (CI on pushes and same-repository pull requests) fails when the list
  is empty, so the check cannot pass by being skipped; elsewhere it is skipped with a notice;
* **text** (every file): patterns that are not themselves private.

Exit 1 lists every hit as ``category: path:line: …``.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: (category, module prefixes) refused in any Python file.
IMPORT_LEAKS: list[tuple[str, tuple[str, ...]]] = [
    ("platform import", ("cairn_platform",)),
    ("domain-pack import", ("cairn.packs",)),
    ("platform-only core surface", ("cairn.core.console", "cairn.core.chat", "cairn.core.triggers.routers.ask", "cairn.ask")),
]
#: The ``cairn.*`` modules a package's own code may import: ADR-0128 D5 (operator, dataflow, and
#: llm / eval for those operator kinds) and ADR-0138 D4c (``cairn.core.backend_selection``, accepted
#: as recommended). A package's tests are not shipped and are not bound by this.
ALLOWED_CAIRN = ("cairn.core.operator", "cairn.core.dataflow", "cairn.core.llm", "cairn.core.eval",
                 "cairn.core.backend_selection")

#: The environment variable that carries the private tokens (one per line).
PRIVATE_TOKENS_ENV = "LEAK_SCAN_PRIVATE_TOKENS"

#: (category, regex) refused in any file's text; none of these is itself private.
TEXT_LEAKS: list[tuple[str, str]] = [
    ("old product name", r"\bobsagent\b|OBSAGENT_"),
    ("cloud instance id", r"\bi-0[0-9a-f]{16}\b"),
    ("platform name", r"cairn[-_]platform|\bcore-ext\b"),
    ("legacy pack name", r"\bobs-operator-"),
]
#: The only exemptions: a file that has to spell a text pattern out, for that category alone.
TEXT_EXEMPT: dict[str, set[str]] = {
    "scripts/check_leaks.py": {"old product name", "platform name", "legacy pack name"},
    "tests/test_check_leaks.py": {"platform name", "legacy pack name"},
    "scripts/check_venv.py": {"platform name"},
    "tests/test_check_venv.py": {"platform name"},
}
_OPERATOR = re.compile(r"^operator_([a-z0-9_]+)$")
_WORD_START = re.compile(r"(?<![a-z0-9])[a-z0-9]")
_WALK_SKIP = {".git", ".venv", ".venv-ci", "__pycache__", ".pytest_cache", ".ruff_cache"}


def _files(root: Path) -> Iterator[Path]:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, check=True).stdout.decode()
        listed = sorted({root / p for p in out.split("\0") if p})
    except (OSError, subprocess.CalledProcessError):
        listed = None
    if listed is not None and (root / ".git").exists():
        yield from (p for p in listed if p.is_file())
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and not (set(path.relative_to(root).parts) & _WALK_SKIP):
            yield path


def _owner(rel: Path) -> str | None:
    """The ``operator_<x>`` import root of the operator pack a path sits in, if any."""
    parts = rel.parts
    if len(parts) > 1 and parts[0] == "packages" and parts[1].startswith("cairn-operator-"):
        return "operator_" + parts[1].removeprefix("cairn-operator-").replace("-", "_")
    return None


def _is_package_source(rel: Path) -> bool:
    parts = rel.parts
    return len(parts) > 2 and parts[0] in ("packages", "packs") and parts[2] == "src"


def _imports(tree: ast.AST) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.lineno, node.module


def _under(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _import_hits(rel: Path, text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return [f"unparseable: {rel}:{e.lineno}: {e.msg}"]
    hits, owner, source = [], _owner(rel), _is_package_source(rel)
    for line, module in _imports(tree):
        for category, prefixes in IMPORT_LEAKS:
            if _under(module, prefixes):
                hits.append(f"{category}: {rel}:{line}: import {module}")
        top = module.split(".")[0]
        if _OPERATOR.match(top) and top != owner:
            hits.append(f"foreign operator import: {rel}:{line}: import {module}")
        if source and top == "cairn" and not _under(module, ALLOWED_CAIRN):
            hits.append(f"core internals import: {rel}:{line}: import {module}")
    return hits


def private_tokens() -> list[str]:
    """The private tokens, lowercased. A token must start with a letter or digit: matching begins
    only at a word start, so any other first character could never match — that is refused rather
    than silently checking nothing."""
    tokens = [t.strip().lower() for t in os.environ.get(PRIVATE_TOKENS_ENV, "").splitlines() if t.strip()]
    bad = [i for i, t in enumerate(tokens, 1) if not re.match(r"[a-z0-9]", t)]
    if bad:
        raise SystemExit(f"leak scan: {PRIVATE_TOKENS_ENV} line(s) {bad} must start with a letter or digit")
    return tokens


def _private_token_hits(rel: Path, text: str, tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    low, hits = text.lower(), []
    for m in _WORD_START.finditer(low):
        i = m.start()
        if any(low.startswith(t, i) for t in tokens):
            hits.append(f"private token: {rel}:{low.count(chr(10), 0, i) + 1}: (not shown)")
    return hits


def scan(root: Path = ROOT) -> list[str]:
    hits: list[str] = []
    tokens = private_tokens()
    for path in _files(root):
        rel = path.relative_to(root)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_bytes().decode("latin-1")  # binary: still scanned for private tokens
            hits.extend(_private_token_hits(rel, text, tokens))
            continue
        if path.suffix == ".py":
            hits.extend(_import_hits(rel, text))
        hits.extend(_private_token_hits(rel, text, tokens))
        exempt = TEXT_EXEMPT.get(rel.as_posix(), set())
        lines = text.splitlines()
        for category, pattern in TEXT_LEAKS:
            if category in exempt:
                continue
            for m in re.finditer(pattern, text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{category}: {rel}:{line}: {lines[line - 1].strip()[:120]}")
    return hits


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    require = "--require-private-tokens" in args
    args = [a for a in args if a != "--require-private-tokens"]
    root = Path(args[0]).resolve() if args else ROOT
    if not private_tokens():
        if require:
            print(f"leak scan: {PRIVATE_TOKENS_ENV} is empty; the private-token check cannot run", file=sys.stderr)
            return 1
        print(f"leak scan: private-token check skipped ({PRIVATE_TOKENS_ENV} not set)")
    hits = scan(root)
    if hits:
        print(f"leak scan: {len(hits)} hit(s) under {root}", file=sys.stderr)
        for hit in hits:
            print(f"  {hit}", file=sys.stderr)
        return 1
    print(f"leak scan: clean ({root})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
