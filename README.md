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
| `scripts/` | the checks CI runs and the catalog generator |
| `tests/` | the checks' own tests |

## Packages

| Package | What | Contracts it implements |
|---|---|---|
| [`cairn-operator-dataset-checks`](packages/cairn-operator-dataset-checks/README.md) | dataset readiness checks, cleaning stages and splits | `cairn-contracts-dataset`, `cairn-contracts-dataset-stages` |
| [`cairn-operator-eval-lm`](packages/cairn-operator-eval-lm/README.md) | `eval/lm-eval`: lm-evaluation-harness benchmarks against a served candidate, as a scorecard; `eval.lm_eval`: a base model and its adapter scored with identical settings on the GPU box, with held-out loss and a time probe; task presets as pack data | none yet |
| [`cairn-operator-ifc-eval`](packages/cairn-operator-ifc-eval/README.md) | `eval/ifc-validate`: do generated IFC snippets parse and hold to the targeted schema (IfcOpenShell, an extra); `eval.ifc_validate`: a generation eval's samples scored that way for the base model and the candidate on the GPU box; never gates | none yet |
| [`cairn-operator-training`](packages/cairn-operator-training/README.md) | SFT with LoRA / QLoRA, DPO and distillation | `cairn-contracts-training`, `cairn-contracts-training-stages` |

| Domain pack | What | Stages from |
|---|---|---|
| [`cairn-pack-dataset-prep`](packs/cairn-pack-dataset-prep/README.md) | `dataset-prep/prepare`: pin a source, audit it, split it by group and register `train`, `val` and `test` versions with their roles; a person reviews the run when dedup drops a large share of the source | `cairn-operator-dataset-checks` |
| [`cairn-pack-finetune`](packs/cairn-pack-finetune/README.md) | `finetune/sft`: LoRA / QLoRA on those versions, with a smoke run, the base model's validation loss as the baseline, benchmarks and held-out loss for the base model and the candidate, three approvals and a governed tag, starting its GPU box after approval and stopping it on every outcome; `finetune/eval-adapter`: the same evaluation for an adapter you already have (each with an `-own-box` copy for a box you manage, or one machine); the `finetune-ssh` profile for a GPU box over SSH | `cairn-operator-training`, `cairn-operator-eval-lm` |

## Install

Every operator and domain pack requires `cairndev` 0.2.x or 0.3.x (`cairndev>=0.2.0,<0.4`;
`>=0.2.1` for `cairn-operator-dataset-checks` and `cairn-pack-dataset-prep`, `>=0.3.1` for
`cairn-pack-finetune`, which starts and stops its GPU box, and for `cairn-operator-eval-lm` and
`cairn-operator-ifc-eval`); the contract wheels depend on
nothing. CI proves each wheel on the newest released `cairndev` its pins select. Once published, each installs by name:

```bash
pip install cairn-pack-dataset-prep cairn-pack-finetune   # the domain packs pull in their operator packs
```

The packages publish to PyPI soon (#10). Until then, install from a checkout, into the
environment `cairndev` runs in, each operator pack together with the contract wheels it pins, and
then the domain packs:

```bash
git clone https://github.com/cairndev-sh/cairn-extensions && cd cairn-extensions
pip install ./packages/cairn-contracts-dataset ./packages/cairn-contracts-dataset-stages \
  ./packages/cairn-operator-dataset-checks
pip install ./packages/cairn-contracts-training ./packages/cairn-contracts-training-stages \
  ./packages/cairn-operator-training
pip install ./packages/cairn-operator-eval-lm    # benchmarks; cairn-pack-finetune's evaluation stage
pip install './packages/cairn-operator-ifc-eval[ifc]'   # IFC snippet validation
pip install ./packs/cairn-pack-dataset-prep ./packs/cairn-pack-finetune
cairn doctor && cairn pack check
```

Each package README lists its extras. [`cairn-pack-finetune`](packs/cairn-pack-finetune/README.md)
walks through a governed fine-tune end to end: durability, the GPU box, preflight, a dry run on
the stub trainer, the three approvals, and retries and cancel.

The `install` entries in `catalog.json` name PyPI (`"index": "pypi"`). Each one becomes true
when its distribution is first published; until then, use the checkout install above.
[RELEASING.md](RELEASING.md) describes how a distribution is published.

### With sign-in on (cairn 0.2.0 and later)

With sign-in on, cairn scopes connections and secrets to the account that owns the run's
workspace. Two consequences for these packages:

- **Engine targets.** A connection bound in the deployment's profile (`ml_target`,
  `dataflow_target`) belongs to the deployment, and a team account's flow legs cannot use it.
  Connection names are unique across a deployment, so a team account creates a target connection
  under its own name (the console suggests `<account>-<provider>`) and names it in its pipelines
  or flows.
- **Credentials.** A team account's run receives the secrets an operator declares (`secrets:`)
  from its own connections, in `ctx.secrets`; it does not receive its account's credentials in the
  process environment. These packages do not declare `secrets:` yet: credentials they read from the
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

Releases (`.github/workflows/release.yml`) are one tag per distribution, `<distribution>/vX.Y.Z`,
published to PyPI with trusted publishing after the owner approves. See [RELEASING.md](RELEASING.md).

A pack's `pack.toml` declares `[compat] cairn_engine` as a bounded range and its
`pyproject.toml` pins `cairndev` to the same range. A wheel that needs extra packages for its
tests declares them in a `test` extra, which CI installs.

## Status

Four operator packs (`eval-lm` is experimental: proven against a faked harness and a faked in-process backend only; `ifc-eval` is proven on a fixture, not yet on a GPU box), the
four contract wheels and two domain packs (`dataset-prep`, `finetune`) are on `main`, tested in CI against the newest released `cairndev` in range. Their support
tables (in each README) say which paths have been re-proven on a public release. The first PyPI
release is next (#10); the GPU qualification run of `finetune/sft` is #9.

## License

Apache-2.0 (see [LICENSE](LICENSE)).
