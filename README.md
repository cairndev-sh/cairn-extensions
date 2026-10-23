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
| `scripts/` | the checks CI runs and the catalog generator |
| `tests/` | the checks' own tests |

Domain packs (`packs/cairn-pack-<name>/`: pipelines, flows, profiles and presets, no code) will
live beside them; none is on `main` yet.

## Packages

| Package | What | Contracts it implements |
|---|---|---|
| [`cairn-operator-dataset-checks`](packages/cairn-operator-dataset-checks/README.md) | dataset readiness checks, cleaning stages and splits | `cairn-contracts-dataset`, `cairn-contracts-dataset-stages` |
| [`cairn-operator-training`](packages/cairn-operator-training/README.md) | SFT with LoRA / QLoRA, DPO and distillation | `cairn-contracts-training`, `cairn-contracts-training-stages` |

## Install

Every package requires `cairndev` 0.2.x (`cairndev>=0.2.0,<0.3`). None is on PyPI yet; until
they are, install from a checkout, into the environment `cairndev` runs in, each operator pack
together with the contract wheels it pins:

```bash
git clone https://github.com/cairndev-sh/cairn-extensions && cd cairn-extensions
pip install ./packages/cairn-contracts-dataset ./packages/cairn-contracts-dataset-stages \
  ./packages/cairn-operator-dataset-checks
pip install ./packages/cairn-contracts-training ./packages/cairn-contracts-training-stages \
  ./packages/cairn-operator-training
cairn doctor && cairn pack check
```

Each package README lists its extras.

### With sign-in on (cairn 0.2.0 and later)

With sign-in on, cairn scopes connections and secrets to the account that owns the run's
workspace. Two consequences for these packages:

- **Engine targets.** A profile-bound `ml_target` (or `dataflow_target`) names a connection of
  the deployment. A team account's leg does not use the deployment's connections, so it needs a
  connection of that name in its own account, or the leg fails.
- **Credentials.** A team account's run receives the secrets an operator declares (`secrets:`)
  from its own connections, in `ctx.secrets` only; nothing is written to the process
  environment. These packages do not declare `secrets:` yet: credentials they read from the
  environment (`HF_TOKEN`, `WANDB_*`, `AWS_*`) are whatever the process or the box holds, for
  every account. See each package's `DESIGN.md`.

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

Two operator packs and their four contract wheels are on `main`, tested in CI against the
released `cairndev` 0.2.x. Their support tables (in each package README) say which paths have
been re-proven on a public release. Domain packs and PyPI releases are next: see the issues.

## License

Apache-2.0 (see [LICENSE](LICENSE)).
