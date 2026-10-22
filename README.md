# cairn extensions

First-party public extensions for [cairn](https://github.com/cairndev-sh/cairn): operator
packs, the domain contracts they implement, and domain packs.

Each extension ships as its own wheel (`cairn-operator-*`, `cairn-contracts-*`,
`cairn-pack-*`) and declares the cairn engine versions it supports, so it installs and
releases independently of the engine. The engine discovers installed extensions through
entry points; nothing here is imported by the engine itself.

The engine, the builtin operator packs and the core contracts live in
[cairndev-sh/cairn](https://github.com/cairndev-sh/cairn).

## Layout

| Path | What |
|---|---|
| `packages/cairn-operator-<name>/` | operator packs, one wheel each |
| `packages/cairn-contracts-<name>/` | the domain contracts those packs implement (data only) |
| `packs/cairn-pack-<name>/` | domain packs: pipelines, flows, profiles and presets, no code |
| `docs/` | a README per package where it outgrows the package (architecture docs are the core's) |
| `scripts/` | the checks CI runs and the catalog generator |
| `tests/` | the checks' own tests |

## Developing

Every wheel is proven against the **released** `cairndev` its own pins select, never a checkout
of the core:

```bash
scripts/test_wheels.sh               # a fresh venv: every wheel, its tests, the manifest and catalog checks
UV_PYTHON=3.12 scripts/test_wheels.sh
scripts/test_wheels.sh --core main   # the same against the core's main branch (what the nightly job runs)
python3 scripts/check_leaks.py       # nothing private in the tree (private tokens: set LEAK_SCAN_PRIVATE_TOKENS)
python3 scripts/gen_catalog.py       # regenerate catalog.json after a pack changes, and commit it
```

CI (`.github/workflows/ci.yml`) runs `policy` (the leak scan, lint, the checks' tests), `python`
on 3.12, 3.13 and 3.14, and `package` (builds every wheel). The private-token list the leak scan
checks lives in the repository secret `LEAK_SCAN_PRIVATE_TOKENS` (one token per line), never in the tree. Every commit carries a DCO
sign-off (`git commit -s`); the `dco` check refuses one without it.

A pack's `pack.toml` declares `[compat] cairn_engine` as a bounded range and its
`pyproject.toml` pins `cairndev` to the same range. A wheel that needs extra packages for its
tests declares them in a `test` extra, which CI installs.

## Status

The first extensions are landing: see the issues.

## License

Apache-2.0 (see [LICENSE](LICENSE)).
