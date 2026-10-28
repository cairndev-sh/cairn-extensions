# cairn-operator-eval-lm — design

- **Distribution:** `cairn-operator-eval-lm` (`packages/cairn-operator-eval-lm/` in cairndev-sh/cairn-extensions)
- **Capabilities provided:** `eval.lm_eval` (the operator); the `eval.lm_eval` flow-stage kernel
  (`cairn.flow_stages`, §3a)
- **Status:** experimental — proven against a faked harness and a faked in-process backend; never
  yet run against a real served model or on a GPU box (§7)
- **Decisions:** ADR-0102 D5 (evaluation after training), ADR-0103 (operator reporting),
  ADR-0068/0070 (config scoping), ADR-0125 (in-process leg liveness), ADR-0143 D3 (a tenant's
  secrets never reach the process environment), ADR-0128 D5 (what an extension may import) — all in
  cairndev-sh/cairn `docs/decisions/`

## 1. Purpose

Answer "does this candidate still score on the classic benchmarks" for the
artefact that will actually ship — the model already served behind an endpoint —
cheaply enough to ask on **every** candidate rather than once before a release.

It exists because an in-process benchmark producer (`eval/benchmark`, one of cairn's
platform operators, not public) is shaped for a different moment: it loads base weights plus a LoRA adapter
*in-process on a GPU*, inside the training job. That is the right shape for "did
this training run work"; it is the wrong shape for "is the served model good",
because a merged or quantised endpoint can score differently from base+adapter
loaded a second way, and because it forces the eval onto the GPU that is busy
serving.

Deliberately NOT here: judged, rubric-scored or agentic evaluation (a separate
evaluation pack); deciding anything (a gate decides — a pipeline predicate, or a gate
operator such as the platform's `model/gate` or `eval/delta`, which are not public —
this produces); authoring benchmark tasks (lm-eval's own
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
already served. The harness call is synchronous and long, so `run` hands it to a
worker thread (`asyncio.to_thread`) and the control plane's event loop keeps
serving other runs (ADR-0125); lm-eval's HTTP backends also call `asyncio.run`
themselves (lm-eval 0.4.13 `models/api_models.py` 689, 831), which fails on a
thread already running a loop. **Cancel:** a thread cannot be killed. A cancel
stops the operator waiting; the harness call runs on to its end in the background,
still sending requests to the endpoint, and its result is discarded. That is why
the manifest says `supports_cancel = false`. `asyncio.to_thread` uses the loop's
default executor, so a long or cancelled harness call holds one of its worker slots
until it finishes; enough concurrent benchmarks can fill that pool and delay
everything else in the process that uses it.

**The installed release.** The `[harness]` extra is bounded to `lm-eval>=0.4.10,<0.4.14`,
the releases the seam was read against (`harness.py` cites 0.4.13's lines; 0.4.10 to
0.4.12 have the same code). `header` and `auth_token` first appear in 0.4.9.2, and
0.4.9 has neither. The seam does not rely on the bound alone:

- it refuses before building the model when `simple_evaluate` does not take `limit`
  or `num_fewshot`, because dropping `limit` would run every document of every task
  (smoke asks for 40) and dropping `num_fewshot` changes the scores;
- after building the model, it refuses unless the model's effective `header` is the
  one it passed. A release that ignored `header` would send `OPENAI_API_KEY` from the
  process environment, and `TemplateAPI.__init__` swallows unknown keyword arguments
  silently;
- any other argument the release does not take is dropped and named in the run's
  `note`. On 0.4.10-0.4.13 that includes the decontamination argument.

```
preset (pack data, TOML)
  -> tasks + limit + fewshot + decontamination switch
  -> harness.run_harness(model="local-completions", model_args=base_url…, tasks, api_key)
       (a worker thread; the seam builds the model with explicit headers)
  -> results {results, higher_is_better, samples}
  -> ScorecardRow[]  (aggregate rows + per-case rows)
  -> budget assertion (RAISES)
  -> flat {task/metric: value} + directions  -> a gate (a pipeline predicate, or a
                                                platform gate operator, not public)
```

State: none of its own. The only file of this pack's it reads is its own preset
data. **lm-eval itself reads and writes the Hugging Face cache:** on first use of a
task it downloads the task's dataset (and, for `local-completions` without a
remote tokenizer, the tokenizer named by `tokenizer`/`model`) from the Hugging Face
Hub into the HF cache (`HF_HOME`). The machine running it needs Hub access, or a
pre-filled cache, and a gated dataset or tokenizer needs a Hub token in that
process's environment; this pack declares none.

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
run's summed per-case cost (0 when the endpoint reported no prices). A case scored on
several metrics yields one row per metric; its cost rides on the first row only, so
`cost_usd` and the total ceiling count each case once.

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

## 3a. The on-box stage `eval.lm_eval` (cairn-extensions#42)

The in-process shape `docs/sft-integration.md` recommends (option b): a **flow-stage kernel** in
this package (`flow_stages.eval_lm_eval`, entry point `eval.lm_eval` in `cairn.flow_stages`), which
a domain pack's ml flow names (`cairn-pack-finetune`'s `sft-eval`) and runs as its own ml leg on
the GPU box. It is not an operator: the ml engine resolves kernels from the flow-stage group, and
an ml leg keeps every chokepoint (the launch door's estimate, the reservation and settlement of
the box's `hourly_usd`, retry and cancel, the record). No operator pack imports another: the
adapter arrives as a value wired from the `train` leg's findings, never by reading the training
pack's files, and the few helpers both need (the split fetch with pin verification, the
checkpoint-bucket location, ChatML) are small copies. Only ChatML is compared by a test
(`test_the_chatml_copy_is_the_trainers`); the split fetch follows the trainer's message shapes with
looser validation, and the results file goes under its own `eval/` key, not the trainer's layout.

```
config (flat, $config.* substituted) → EvalStageConfig
  → splits: fetch (path | file:// | s3://), verify the #sha256 pin, read rows
  → adapter: the train leg's dir on this box | s3://…/ (downloaded) | hf://repo | Hub id | path
  → Backend (inprocess.py): load the base once (4-bit nf4 for qlora; a pre-quantised checkpoint as is)
  → time probe: ~50 documents on the base (in proportion to the preset's limits) + 8 loss rows
       → projection = so far + Σ tasks (batches × s/batch + lm-eval overhead) × (1 + 1.25) + loss
       → over (time_budget_s × (1 − time_margin))?  raise TimeBudgetExceeded (the leg fails cleanly)
  → base pass: one simple_evaluate per task (HFLM over the loaded model), split loss
  → attach the adapter to the same weights (peft) → candidate pass: the same calls, split loss
  → per task/metric: base, candidate (± stderr), delta (± √(se_b² + se_c²)), documents
  → worst delta and regressions (delta below −its standard error, in the metric's direction)
  → lm_eval_results.json → s3://<CAIRN_CHECKPOINT_BUCKET>/<prefix>/<run>/eval/ (when set)
  → flat stats (the leg's findings, eval_lm_eval_<key>) + the `report` product (output, content_hash)
```

**Identical settings.** One process, one loaded base: the candidate is the base with the adapter
attached, so both passes share the weights, the quantisation, the tokenizer, the harness release,
the task list, the per-task limits (MMLU's per subject), the few-shot count, the batch size and the
four seeds lm-eval takes (python, numpy, torch, few-shot). The seam refuses a `simple_evaluate`
that does not take `limit`, `num_fewshot` or a seed.

**The time probe** exists because a hand-run of these benchmarks (cairn-extensions#9) was
under-estimated about 3×. lm-eval's own work (building tasks, downloading datasets, metrics) is
timed apart from the model's: the backend wraps the loaded `HFLM`'s request methods
(`loglikelihood`, `loglikelihood_rolling`, `generate_until`; 0.4.13 `evaluator.py` 600 calls them by
name) and counts requests and seconds. The model's time is scaled by batches (a fixed batch size
per preset, so the arithmetic holds; the probe's batches are part-full, so it over-estimates), the
overhead is counted once per pass, and the candidate is priced 1.25× the base (the LoRA matmuls).
A refusal names the projection per task and what to change.

**The split loss** is one method for both models, named on the record (`split_loss_method`):
full-sequence mean token NLL, each row rendered with the chat template the adapter was trained with
(its saved tokenizer's; else the trainer's rule), one row per forward pass (no padding, no
packing), truncated at `max_seq_len`. It is not the trainer's `eval_loss`, which batches and pads.

**The base model.** `base_model` is loaded as given. Under 4-bit loading Unsloth substitutes a
pre-quantised checkpoint (`unsloth/<model>-unsloth-bnb-4bit`), which the adapter records as its
`base_model_name_or_path`; the stage reports it (`adapter_base_model`) and says so in the note when
it differs, since a base quantised on load is not bit-identical. Both models are scored on the same
base either way, so the deltas compare like with like.

**What fails the leg** (a raise; a clean failure the engine does not retry): the projection over
budget; a split whose bytes do not match their pin, or a `dataset:` ref that reached the box
unresolved; an adapter that cannot be found, or a promoted tag (`<name>@<tag>`: the box has no
registry); an unknown preset; no `time_budget_s` outside the stub tier. **What is "not evaluated"**
(`checked=false`, the reason, no scores; the leg succeeds, so a training run is never lost to its
evaluation): the `[box]` libraries missing, or any error while loading or scoring (a CUDA
out-of-memory, a task that will not load). A process killed on the box fails the attempt; the leg's
`retry:` runs it again from the start.

**The stub tier** (`stub: true`): a backend that loads nothing and returns deterministic placeholder
numbers, so a pipeline's every leg, gate and record run without a GPU. Its output is `stub: true`
and never `checked`.

**Security.** The `hf` backend reads no endpoint key and no secret; nothing is written to
`os.environ`. The box's own credential chain reads the splits and writes the results file (as the
trainer's does). The results file holds settings, versions, scores and losses, never a row of data.

**Risk class:** `external_write` (paid GPU time, and it writes the results file to the checkpoint
bucket), the class of `train.qlora`. **Contract:** none yet (as `compress/distill`); a stage
contract in a contract wheel is a public contract and an owner decision.

## 4. Configuration

| Setting | Where | Scope | Default | What it does |
|---|---|---|---|---|
| `preset` | `input_schema` | run | `smoke` | which pack-data task list to run |
| `tasks` | `input_schema` | run | `[]` | explicit lm-eval task ids, overriding the preset |
| `limit` / `num_fewshot` / `batch_size` | `input_schema` | run | preset's | per-run overrides of the preset |
| `endpoint_url` / `model` / `tokenizer` | `input_schema` | run | — | the served candidate |
| `api_key_env` | `input_schema` | run | `""` | the NAME of a secret read through `ctx.secrets` (declared in the binding's `secrets:`), not an environment variable; the key goes to the harness call as a header, never into `os.environ` or `model_args` |
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

Connections: none. Secrets: the endpoint's bearer key, by NAME (`api_key_env`)
through `ctx.secrets`, when the endpoint requires one; the binding must declare
that name in its `secrets:` block. The key goes to the harness call only: the seam
builds lm-eval's model with an explicit `Authorization` header. It is never written
to `os.environ` (shared by every run in the process; ADR-0143 D3) and never put in
`model_args` (which lm-eval logs and copies into its results).

- `api_key_env` empty: the operator runs unauthenticated, the normal case for an
  in-cluster vLLM, and it still sends an explicit header with no key, so lm-eval
  never falls back to an `OPENAI_API_KEY` that happens to be in the process
  environment. A 401 surfaces as a harness failure and a `checked=false` stub.
- `api_key_env` named but not readable (not declared in the binding, not supplied
  by a connection, or empty): a **refusal** — a `checked=false` stub whose note names
  the key — and no request is sent. Calling the endpoint without the key the run
  asked for would only turn a configuration error into a 401.

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
| `checked=false`, note "harness failed: …" (the key, if the error echoed it, replaced by `[redacted]`) | endpoint down, auth, task id typo | stub, no metrics | yes |
| `checked=false`, note "refused: lm-eval X's simple_evaluate does not take limit" (or `num_fewshot`) | an lm-eval outside the bounded `[harness]` range | stub, no metrics, the model not built, no request sent | yes |
| `checked=false`, note "refused: lm-eval X built the model with a request header other than the one passed …" | a release that ignores `header` (it would send `OPENAI_API_KEY` from the environment) | stub, no metrics; only construction's remote-tokenizer probe (authenticated with `auth_token` alone) ran | yes |
| `checked=true`, note "not passed (lm-eval X does not take them): …" | the release does not take some other argument | real scores, the dropped arguments named | reported, not hidden |
| `checked=false`, note "api_key_env '…' names a key this run cannot read …" | the binding does not declare the key, or no connection supplies it | stub, no metrics, no request sent | yes |
| `checked=true`, `decontaminated=false`, note "… takes no decontamination argument, so it was not passed …" | decontamination requested, and the installed lm-eval (every release in the `[harness]` range) takes no decontamination argument | real scores, flagged | reported, not hidden |
| `checked=true`, `decontaminated=false`, note naming the tasks | decontamination requested and passed, but no task implements it | real scores, flagged | reported, not hidden |
| `EvalBudgetExceeded` raised out of the leg | a document broke a per-case or total ceiling | exception | yes — a blown budget is not a result |
| every default run returned "harness failed: … unexpected keyword argument 'decontamination_ngrams_path'" | the seam passed a decontamination argument that lm-eval 0.4.13's `simple_evaluate` does not take (`evaluator.py` 55-87) | fixed: the seam passes only the arguments the installed `simple_evaluate` accepts; the dropped request shows as the "not decontaminated" note | — |
| a tenant's key could reach another tenant's endpoint | the key was written to the process-wide `OPENAI_API_KEY` with `setdefault`, so the first run's key stuck for every later run | fixed: explicit per-call headers, the environment untouched (`test_two_runs_with_different_keys_do_not_leak_into_each_other`) | — |
| the control plane stalled for the whole benchmark, and a concurrent run failed inside `asyncio.run` | the synchronous harness ran on the event loop | fixed: a worker thread (`test_the_event_loop_stays_responsive_while_the_harness_runs`) | — |
| `cost_usd` and the total ceiling counted a case once per metric | each per-case metric row carried the case's full cost | fixed: the cost rides on the first row of each case | — |
| a standard error appeared as a metric | `_DROP_SUFFIXES` checked against the raw key instead of the metric name — lm-eval spells it `exact_match_stderr,strict-match`, filter suffix LAST (caught while building the pack, held by `test_stderr_and_alias_keys_never_become_metrics`) | fixed: the check runs on the parsed metric name | — |

Known unknowns for the first live run, recorded rather than guessed: lm-eval
0.4.10-0.4.13 (read, not run) take no decontamination argument in
`simple_evaluate`, so the request is dropped and the note says the scores are not
decontaminated; how decontamination is requested on the installed release (if at
all) must be checked before decontaminated numbers are claimed. The header and
model-construction path above is read from 0.4.13's source, not run. Nothing in
this pack has run against a live harness.

### The `eval.lm_eval` stage

`EvalStageConfig` (`schemas/eval_stage.py`), set by the flow's `$config.*` wiring:

| Setting | Default | What it does |
|---|---|---|
| `base_model` | — | Hugging Face id (or path) of the base |
| `adapter_uri` / `local_adapter_dir` | `""` | the adapter: the train leg's directory on this box first, else the URI (`s3://…/`, `hf://<repo>`, a Hub id, a path); a promoted tag is refused |
| `method` | `qlora` | `qlora` loads the base in 4 bit; `lora` in float16 |
| `preset` | `forgetting` | the task preset (`limits` per task, a fixed `batch_size`) |
| `seed` | `1234` | lm-eval's four seeds, the same for both models |
| `batch_size` | `0` (the preset's) | per-model batch size |
| `val_uri` / `test_uri` | `""` | the held-out splits, as the control plane resolved them (`<location>#sha256=<hash>`) |
| `max_seq_len` / `split_rows` | `2048` / `0` (all) | the split loss's truncation and rows |
| `time_budget_s` | required outside the stub tier | the leg's timeout, in seconds; the probe's ceiling |
| `time_margin` / `probe_items` | `0.1` / `50` | budget kept back; documents the probe scores |
| `stub` | `false` | the stub tier |

The findings a leg reports (`eval_lm_eval_<key>`): `checked`, `note`, `stub`, the settings (`preset`,
`tasks`, `seed`, `batch_size`, `num_fewshot`, `method`, `base_model`, `quantization`, `adapter`,
`adapter_source`, `adapter_base_model`, `chat_template_source`, `lm_eval_version`), per task and
metric `base_<task>_<metric>`, `candidate_…`, `delta_…` and each one's `_stderr`, `n_<task>` (documents
per model), `metrics_compared`, `regressions`, `regressed`, `worst_delta`, `worst_delta_metric`,
`split_loss_method`, `base_/candidate_/delta_{val,test}_loss`, `{val,test}_rows`, `{val,test}_tokens`,
`probe_documents`, `probe_s`, `probe_model_s_per_request`, `projected_s`, `elapsed_s`, `results_uri` (and `results_upload_error`
when the upload failed).

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

The stage (`tests/test_eval_stage.py`) runs on a faked `inprocess.Backend` and a fake clock: the
identical calls for both models on one loaded base, MMLU's own limit, the deltas and standard
errors in each metric's direction, the split loss on the same rendered rows, the probe's limits,
a projection that equals the length of a run whose cost is linear in batches, the refusal before
the full run, the results file (local, and uploaded to a fake S3 under the run's key), the adapter
and split resolution, the failure paths, the stub tier, and an untouched `os.environ`. Through
`cairn-pack-finetune`, the stage runs in the core's runner and on the core's launch door and bridge.
**Not proven:** `HfBackend` itself (model loading, `HFLM` over a pre-loaded and a PEFT-wrapped
model, 4-bit loading under transformers 5.5, the request-method timing, IFEval's generation cost on
a T4) and the projection's accuracy on real hardware. The first live run must check them, and
compare one base and one candidate score with a hand run of `lm_eval --model hf`.

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
| 2026-10-26 | Review fixes before the first release: the endpoint key goes to the harness call as an explicit header, never `os.environ` (ADR-0143 D3), and a named key the run cannot read is a refusal; the harness runs in a worker thread (ADR-0125); the seam drops arguments the installed `simple_evaluate` does not take (0.4.13 has no decontamination argument); a case's cost is counted once. Read against lm-eval 0.4.13's source; no contract change. | |
| 2026-10-26 | Second review: `[harness]` bounded to `lm-eval>=0.4.10,<0.5` (the releases read); the seam refuses when the installed `simple_evaluate` does not take `limit` or `num_fewshot`, and refuses when the built model's effective header is not the one passed; other dropped arguments, including the decontamination request, are named in the note; the key is redacted from failure notes. No contract change. | |
| 2026-10-26 | Third review: `[harness]` capped at `<0.4.14` (the releases whose source was read: 0.4.10–0.4.13). The tokenizer probe during construction authenticates only with the run's own key, or with none (`openai_completions.py:35`, `utils.py:873-874` in 0.4.13); a later release is admitted only after its source is read the same way. | |
| 2026-10-27 | 0.2.0 (cairn-extensions#42): the `eval.lm_eval` flow-stage kernel (§3a) — base model and base + adapter scored in process on the GPU box with identical settings, held-out split loss by one named method, a time probe that refuses a run projected past the leg's timeout, per-task scores ± standard error, deltas and a results file uploaded beside the run's checkpoints. The `[box]` and `test` extras; the `forgetting` and `forgetting-ifeval` presets and a per-task `limits` table; the result parsing moved to `scoring.py` (shared, unchanged); `eval/lm-eval` refuses a preset that sizes tasks apart (one harness call cannot honour per-task limits). The operator's ref, version, schemas and reports are unchanged. | |
| 2026-10-28 | 0.2.1 (cairn-extensions#46): `[harness]` is `lm-eval[api]`, the extra the served-candidate `local-completions` backend needs (aiohttp, tenacity, tiktoken); before, a real run failed at model construction and reported `checked=false`. Nothing else changes. | |
