# cairn-operator-eval-lm — design

- **Distribution:** `cairn-operator-eval-lm` (`packages/cairn-operator-eval-lm/` in cairndev-sh/cairn-extensions)
- **Capabilities provided:** `eval.lm_eval`
- **Status:** experimental — proven against a faked harness; never yet run
  against a real served model (needs a GPU box, see §7)
- **Decisions:** ADR-0102 D5 (evaluation after training), ADR-0103 (operator reporting),
  ADR-0068/0070 (config scoping), ADR-0128 D5 (what an extension may import) — all in
  cairndev-sh/cairn `docs/decisions/`

## 1. Purpose

Answer "does this candidate still score on the classic benchmarks" for the
artefact that will actually ship — the model already served behind an endpoint —
cheaply enough to ask on **every** candidate rather than once before a release.

It exists because an in-process benchmark producer (`eval/benchmark`, not public)
is shaped for a different moment: it loads base weights plus a LoRA adapter
*in-process on a GPU*, inside the training job. That is the right shape for "did
this training run work"; it is the wrong shape for "is the served model good",
because a merged or quantised endpoint can score differently from base+adapter
loaded a second way, and because it forces the eval onto the GPU that is busy
serving.

Deliberately NOT here: judged, rubric-scored or agentic evaluation (a separate
evaluation pack); deciding anything (a gate decides — `model/gate`, `eval/delta` or
a pipeline predicate — this produces); authoring benchmark tasks (lm-eval's own
YAML task system does that, and the presets here select from it).

## 2. Architecture

Wraps **EleutherAI lm-evaluation-harness** — chosen over lighteval and HELM in
the research behind ADR-0102:
widest local-backend support (it drives an OpenAI-compatible server directly),
YAML task authoring that a non-Python pack can ship as data, and **the only
decontamination story on any surveyed harness** (13-gram overlap, opt-in). HELM
entered maintenance mode 2026-06-01; lighteval pivoted to Inspect as a backend,
so wiring it would mean adding a layer over a library cairn wires
directly anyway.

The work runs **in-process on the control plane** and is HTTP: the candidate is
already served.

```
preset (pack data, TOML)
  -> tasks + limit + fewshot + decontamination switch
  -> harness.run_harness(model="local-completions", model_args=base_url…, tasks)
  -> results {results, higher_is_better, samples}
  -> ScorecardRow[]  (aggregate rows + per-case rows)
  -> budget assertion (RAISES)
  -> flat {task/metric: value} + directions  -> model/gate / eval/delta
```

State: none of its own. The only file it reads is its own preset data.

Concurrency is the harness's (`num_concurrent`); the served endpoint owns the
real batching. **Idempotency / retry:** the operator has no checkpoint — a
re-run re-runs the whole preset. That is affordable because a preset is bounded
by `limit`, and it is the honest position: partial benchmark results are not a
benchmark number. On any harness failure the operator returns a deterministic
stub with `checked=false` and no metrics, so a partial failure can never be read
as a low score.

## 3. Operators

| Operator | Capability | Risk | Mock | Input -> Output | Reports (stages · series · scalars) |
|---|---|---|---|---|---|
| `eval/lm-eval` | `eval.lm_eval` | external_write | yes | `LmEvalInput` -> `LmEvalOutput` | `resolve, evaluate, score` · — · `tasks_run, cases, cost_usd` |

`tasks_run` is the number of tasks the preset resolved to, `cases` the number of
per-document rows the harness logged (0 when it logged none), `cost_usd` this
run's summed per-case cost (0 when the endpoint reported no prices).

**What it refuses.** No `endpoint_url` → refusal, not a zero: this operator
measures a *served* candidate and a benchmark that silently measured nothing
would flow into a gate as absent scores. An unknown preset name → refusal naming
the presets that exist, rather than falling back to the default and reporting
numbers for tasks nobody asked for.

**Decontamination.** `decontaminate` defaults true. lm-eval's 13-gram check is
opt-in *per task* upstream (a task must override `should_decontaminate` and
`doc_to_decontamination_query`), so a preset can ask and get nothing back. The
output separates `decontamination_requested` from `decontaminated`, and the note
says plainly that the scores are NOT decontaminated when the twins are missing.
A decontaminated score keeps lm-eval's `_decontaminate` suffix in its metric
name all the way into the flat map, so it can never collide with the
contaminated number a gate is thresholding.

## 4. Configuration

| Setting | Where | Scope | Default | What it does |
|---|---|---|---|---|
| `preset` | `input_schema` | run | `smoke` | which pack-data task list to run |
| `tasks` | `input_schema` | run | `[]` | explicit lm-eval task ids, overriding the preset |
| `limit` / `num_fewshot` / `batch_size` | `input_schema` | run | preset's | per-run overrides of the preset |
| `endpoint_url` / `model` / `tokenizer` | `input_schema` | run | — | the served candidate |
| `api_key_env` | `input_schema` | run | `""` | env var read through `ctx.secrets`; never placed in `model_args` |
| `decontaminate` | `input_schema` | run | `true` | request the 13-gram check |
| `max_cost_usd_per_case` | `input_schema` | run | `0` | per-document cost ceiling; **raises** |
| `max_latency_ms_per_case` | `input_schema` | run | `0` | per-document latency ceiling; **raises** |
| `num_concurrent` | `config_schema` | operational | `4` | requests in flight |
| `max_total_cost_usd` | `config_schema` | **governance** | `0` | ENV-OWNED spend ceiling; a run-supplied value is **DROPPED** (ADR-0068/0070) |

**Per-run bounds belong in `input_schema`.** The two per-case ceilings started
life in `LmEvalConfig` at governance scope and the runtime discarded every value
the tests set (`ignored run-supplied governance knob(s)`) — the operator
advertised a limit it never enforced (a known failure of governance-scoped
per-run knobs). They now live only
on the input, and `test_a_per_case_ceiling_is_run_owned` holds that line. Only
the whole-evaluation ceiling is environment-owned, the same call a batch-inference
operator makes for `max_rows`.

Connections: none. Secrets: the endpoint's bearer key, by env-var NAME through
`ctx.secrets`, when the endpoint requires one. Absent, the operator still runs —
an unauthenticated in-cluster vLLM is the normal case; a 401 surfaces as a
harness failure and a `checked=false` stub.

## 5. Governance

`risk = external_write`. It does not write data, but it drives a served endpoint
and consumes its capacity — the same call that makes a batch-inference operator
`external_write` rather than `read_only`. A binding
should place an approval gate in front of it in any environment where endpoint
time is metered.

`requires_gpu = false` is deliberate and is a governance statement as much as a
scheduling one: this leg must never be scheduled onto the GPU box that is
serving the model it is measuring.

Auditable: `audit_metadata` carries the preset, the task list, the flat metrics,
`decontaminated` / `decontamination_requested`, the case count and the cost — so
the hash chain records not just the score but whether it was decontaminated.

A deployment can clamp `num_concurrent` and `max_total_cost_usd` through the
profile. It cannot weaken the decontamination reporting or the budget raise;
those are contract, not policy.

## 6. Failure modes

| Symptom | Cause | What the operator returns | Fail closed? |
|---|---|---|---|
| `checked=false`, note "harness unavailable ([harness] extra)" | `lm-eval` not installed | stub, no metrics | yes — no score to pass on |
| `checked=false`, note "no endpoint_url" | the serve leg did not run / was not wired | stub, no metrics | yes |
| `checked=false`, note "harness failed: …" | endpoint down, auth, task id typo | stub, no metrics | yes |
| `checked=true`, `decontaminated=false`, note naming the tasks | decontamination requested but no task implements it | real scores, flagged | reported, not hidden |
| `EvalBudgetExceeded` raised out of the leg | a document broke a per-case or total ceiling | exception | yes — a blown budget is not a result |
| a standard error appeared as a metric | `_DROP_SUFFIXES` checked against the raw key instead of the metric name — lm-eval spells it `exact_match_stderr,strict-match`, filter suffix LAST (caught while building the pack, held by `test_stderr_and_alias_keys_never_become_metrics`) | fixed: the check runs on the parsed metric name | — |

Known unknowns for the first live run, recorded rather than guessed: the exact
spelling and semantics of the decontamination argument to `simple_evaluate`
differ across lm-eval releases (the docs page survives from the 0.3 CLI flag), so
the seam passes `decontamination_ngrams_path` and it must be checked against the
installed release before the first real benchmark. Nothing in this pack has run
against a live harness.

## 7. Testing

`packages/cairn-operator-eval-lm/tests/`, run by `scripts/test_wheels.sh` (a fresh venv, the wheel
installed on the released `cairndev` its pins select).

Everything runs **without** the `[harness]` extra: `harness.run_harness` is the
single seam, replaced per test with a payload shaped like `simple_evaluate`'s.
`test_the_pack_imports_without_the_extra` fails if anything imports `lm_eval` at
module scope. Golden-ish fixtures: the fake results carry lm-eval's real key
spellings (`exact_match,strict-match`, `exact_match_stderr,strict-match`,
`acc_alias`, and the `_decontaminate` twins) so the parsing is tested against the
shapes the library actually emits.

Not covered here, and needing a GPU box serving the candidate: any real harness
call, the decontamination argument's real name, real per-case cost/latency
reporting, and whether `local-completions` needs a `tokenizer` for the
loglikelihood tasks in `knowledge`.

From a checkout:

```
scripts/test_wheels.sh                                   # every wheel, its tests, `cairn pack sync --check`
python -m pytest -q packages/cairn-operator-eval-lm/tests  # in a venv with this wheel and cairndev installed
```

`cairn pack sync --check packages/cairn-operator-eval-lm` reads the INSTALLED operator class, so
it runs after the wheel is installed (`test_wheels.sh` does both).

## 8. History

| Date | Change | PR |
|---|---|---|
| 2026-09-06 | Pack created: `eval/lm-eval`, presets as pack data, decontamination reporting, evaluation budget | an internal issue |
| 2026-10-26 | Moved public to `cairndev-sh/cairn-extensions` as its first release, 0.1.0. The operator ref, schemas, presets, risk class, capability and reports are unchanged; `[pack].name` is the distribution name (was a legacy name) and `[pack].kind` is `operator` (was `service_provider`, which is for substrate services; an operator pack's kind is what the repository's manifest check selects); `[compat]` and the `cairndev` pin are bounded to `>=0.3.1,<0.4`; private issue and document references are rewritten as plain text. No contract wheel: the operator implements no shared contract. `docs/sft-integration.md` proposes running the benchmarks on the `finetune/sft` GPU box. | |
