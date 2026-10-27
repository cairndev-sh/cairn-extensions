# Proposal: lm-eval benchmarks inside `finetune/sft`, on the GPU box

**Status:** implemented as option (b) in cairn-extensions#42 (this package 0.2.0,
`cairn-pack-finetune` 0.2.0). The proposal below is kept as it was reviewed; where the
implementation differs:

- **The default preset** is a new `forgetting` (five log-likelihood tasks at 500 documents, MMLU at
  10 per subject), not `smoke`: 40 documents per task cannot resolve the deltas this check exists
  for. `forgetting-ifeval` adds IFEval for `finetune/eval-adapter`. Presets gain a per-task
  `limits` table, and the stage makes one `simple_evaluate` call per task.
- **`timeout: 3h`** on `sft`'s `eval` leg (the table's last row: 31 box-hours, $20.46, `max_hours:
  32`), with a **time probe** the proposal did not have: about 50 documents first, and a clean
  refusal when the projection does not fit (`time_budget_s`, which must equal the timeout).
- **The held-out loss** on `val_version` and `test_version` for both models, by one named method,
  is part of the stage.
- **The findings** are `eval_lm_eval_base_<task>_<metric>`, `…candidate_…`, `…delta_…` and their
  `_stderr`, `worst_delta`, `regressions`; `promote-model` shows the `acc` deltas of the six tasks
  with their standard errors, and the split losses. Shown, not gated.
- **The adapter** is the train leg's directory on the box (`train_qlora_local_adapter_dir`) when it
  is still there, else its `adapter_uri` (`s3://` downloaded, `hf://` from the Hub).
- **Box install:** `cairn-operator-eval-lm[box]` (`lm-eval[hf,ifeval]` and boto3) next to
  `cairn-operator-training[gpu]`. Resolved together once by hand (`uv pip compile`, Linux x86-64,
  Python 3.12, 2026-10-27): lm-eval 0.4.13 with the trainer's torch 2.11.0, transformers 5.5.0,
  peft 0.20.0, datasets 4.3.0, accelerate 1.14.0 and bitsandbytes 0.50.1. Not yet a CI check
  (item 1 below).
- **No stage contract** (item 4): uncontracted, like `compress/distill`.

## The ask

`finetune/sft` decides on validation loss alone (`improved`: the candidate's `eval_loss` below the
base model's `baseline_eval_loss`). A person approving `promote-model` cannot see whether the
adapter cost the model its general ability. The proposal: after `train` and before `release-box`,
run an lm-evaluation-harness preset **on the GPU box** for the base model and for the base model
plus the candidate adapter, and show the per-task delta on the `promote-model` card.

Constraints:

- **On the box, not on the control plane.** The control plane (a laptop, in the reference setup)
  cannot load a 4B model. The box has the GPU, the base model in its Hugging Face cache and the
  adapter on disk.
- **Before `release-box`.** The box stops right after the last GPU leg, so it does not bill while a
  person decides. The evaluation is the last GPU leg.
- **Another priced leg.** On a connection with `hourly_usd`, the launch door reserves
  `hourly_usd × timeout × attempts` for each ml leg, so the evaluation's timeout and retry count are
  part of the run's budget.
- **Boundaries.** An operator pack never imports another operator pack. A domain pack carries
  data only (pipelines, flows, profiles), never operators.

## What runs on the box

The `eval/lm-eval` operator in this pack does not fit as it is. It drives a **served** endpoint over
HTTP (`model="local-completions"`), it declares `requires_gpu = false`, and its design says it must
never run on the GPU box it measures. `finetune/sft` serves nothing. Its question, "did this
training run work", is the in-process shape: lm-eval's `hf` backend loads the base model and
applies the adapter with `peft=`.

```
lm_eval.simple_evaluate(model="hf", model_args="pretrained=<base_model>,…",                tasks, limit, num_fewshot)  # base
lm_eval.simple_evaluate(model="hf", model_args="pretrained=<base_model>,peft=<adapter_dir>,…", tasks, limit, num_fewshot)  # candidate
→ per (task, metric): base, candidate, delta = candidate − base, and the metric's direction
```

Both passes run in one process with the same harness release, preset, documents (`limit`) and seed,
so the delta compares like with like. The preset files, the result parsing (`_rows_from_results`:
the filter-suffix stripping, the dropped `_stderr`/`_alias` keys, the `_decontaminate` twins, the
metric directions) and the "no score unless it ran" rule already live in this pack. Every option
below has to run this code, or a copy of it, on the box.

## Options

| | (a) a kernel in `cairn-operator-training` | (b) a kernel in this pack | (c) a task leg through `remote/exec` |
|---|---|---|---|
| **Shape** | a new `eval.*` kernel in the training pack's `cairn.flow_stages` group, calling lm-eval directly; a new `sft-eval` flow beside `sft-train` (the flows live in `cairn-pack-finetune`, not in the training pack) | a new `eval.lm_eval` kernel in **this** pack's `cairn.flow_stages` group; `cairn-pack-finetune` adds an `sft-eval` flow that uses it and an `eval` leg | a pipeline `task:` leg runs `remote/exec` on the control plane, which runs a CLI from this pack on the box over SSH |
| **Boundaries** | legal (no cross-pack import), but a **second copy** of the harness seam, the presets and the result parsing, which drifts from this pack's. The training pack's own manifest scopes evaluation out ("evaluation … are other packs' operations") | legal: the kernel imports only its own package. The adapter reaches it as a config value wired from the `train` leg's finding, not by reading the training pack's `train_result.json` (no file-format coupling). `cairn-pack-finetune` gains a dependency on this pack (domain pack → operator pack, the normal direction) | legal, but it goes **around the ml-leg chokepoints**: the box time is not reserved or settled as an ml leg (the `hourly_usd × timeout × attempts` estimate applies to flow legs), stage-contract checks and live stage reports do not apply, and the scores come back through `remote/exec`'s stdout as a second, ad hoc result channel |
| **Box install** | `cairn-operator-training[gpu,eval]`: one extra beside the exact trainer pins (`images.py::TRAINER_PINS`), so one place resolves lm-eval against torch / transformers / datasets / peft | `cairn-operator-eval-lm[harness]` beside `cairn-operator-training[gpu]` in the box venv; lm-eval must co-resolve with the trainer's exact pins (torch 2.11.0, transformers 5.5.0, datasets 4.3.0, peft 0.20.0), so the `[harness]` floor becomes a tested pin | as (b), plus a console script. To reuse the HTTP operator unchanged it would also need a vLLM server on the box (base + adapter via `--enable-lora`), which brings its own torch pin and in practice a second venv |
| **Timeout / pricing** | an ml leg with its own `timeout` and `retry`, priced like `smoke` and `train` | same as (a) | not priced as an ml leg: GPU hours the budget does not see |
| **Retry** | `retry: {max_attempts: 2}`, as `smoke`/`train`: an engine-reported kill (the OOM killer) runs once more; a clean failure does not | same as (a) | `remote/exec`'s own retry semantics, separate from the ml-leg rule |
| **Delta to the gate card** | stage stats → leg findings (`eval_<…>`) → `promote-model` `show:` | the same: stats `delta_<task>_<metric>`, `worst_delta`, `regressions`, `checked`, `decontaminated` become findings `eval_lm_eval_<key>` (stage dots become underscores, as `train_qlora_eval_loss`) and are listed in `show:` | the scores are parsed from stdout into the operator's output, then `show:` |
| **Preset cost** | same work in every option (see the next section) | | |

### Same leg instead of a new one (a variant of b)

The kernel could also be a fifth stage of `sft-train`, after `package.adapter`. The adapter is
then in the same workdir and there is no extra model download or box round-trip. This proposal
does not recommend it:

- an evaluation killed for running out of memory fails the 12-hour `train` leg, and its retry
  resumes training before evaluating again;
- `train`'s timeout has to absorb the evaluation;
- `smoke` runs the same flow, so it needs a switch to skip the stage;
- the scores land on the `train` record, which mixes "the training ran" with "the benchmark ran".

## Recommendation: (b)

A kernel in this pack, `eval.lm_eval`, run as its own ml leg `eval` (flow `sft-eval` in
`cairn-pack-finetune`) after `train` and before `release-box`.

- **One copy of the capability.** The presets, the parsing and the decontamination reporting stay
  in this pack. The control-plane operator (`eval/lm-eval`, HTTP) and the box kernel
  (`eval.lm_eval`, in-process) share them, the same split as `train/local` and `train.qlora` in the
  training pack.
- **It stays inside every chokepoint.** It is an ml leg: launch-door estimate, reservation and
  settlement on the box's `hourly_usd`, engine retry and cancel, stage reports, the finalizer.
- **No import across packs.** The adapter path arrives as a wire from `train`'s findings.

Sketch, for review only (not committed anywhere):

```yaml
# cairn-pack-finetune/flows/sft-eval.yaml
name: sft-eval
kind: ml
stages:
  - stage: eval.lm_eval
    config:
      base_model: $config.base_model
      adapter_dir: $config.adapter_dir      # a path on the box (train's local_adapter_dir)
      method: $config.method                # qlora → load the base in 4 bit, as it was trained
      preset: $config.preset                # smoke by default
      max_seq_len: $config.max_seq_len
      seed: $config.seed
    resources: {gpu: 1}

# cairn-pack-finetune/pipelines/sft.yaml (the changed steps)
  - id: eval
    label: Benchmarks, base and candidate
    flow: sft-eval
    depends_on: [train]
    timeout: 1h                             # smoke; 3h for knowledge or reasoning
    with:
      retry: {max_attempts: 2}
      base_model: $input.base_model
      adapter_dir: $steps.train.train_qlora_local_adapter_dir
      method: $input.method
      preset: $input.eval_preset            # new input, default "smoke"
      max_seq_len: $input.max_seq_len
      seed: $input.seed
  - id: release-box
    depends_on: [train, eval]               # was [train]
  - gate: promote-model
    show:
      - …                                   # the existing losses
      - eval.eval_lm_eval_checked
      - eval.eval_lm_eval_worst_delta
      - eval.eval_lm_eval_regressions
      - eval.eval_lm_eval_delta_gsm8k_exact_match
      - eval.eval_lm_eval_delta_arc_easy_acc
      - eval.eval_lm_eval_delta_hellaswag_acc_norm
```

`show:` needs fixed keys, so the per-task lines follow the default preset. `worst_delta` (the
largest drop, measured in each metric's own direction) and `regressions` (how many metrics went
the wrong way by more than their standard error) carry the summary for any preset.

**Show, do not gate, at first.** The `smoke` preset scores 40 documents per task. At accuracy near
0.6 the standard error is about √(0.6·0.4/40) ≈ 0.077, so a drop of a few points is noise. The
kernel should report each delta's standard error (lm-eval's `_stderr` values, which the operator
drops from its metric map) and leave the decision to the person at `promote-model`. A
`no-regression` predicate gate comes later, on a preset whose `limit` makes it meaningful.

**A failed evaluation never blocks training.** As in the operator, a harness error returns
`checked=false` with a note and no scores; the card shows "not evaluated". Only a breached
evaluation budget raises. An OOM or a kill on the box fails the leg, the engine retries it once,
and a second failure fails the pipeline after `train` already produced the adapter. The finalizer
still stops the box.

### Timeout, pricing and presets

The launch estimate per ml leg is `hourly_usd × timeout × attempts`. With `sft.yaml`'s example rate
of $0.66/h:

| | box-hours reserved | estimate | `budget: {usd: 25}` holds up to | `acquire` `max_hours` |
|---|---|---|---|---|
| today: smoke 30m + train 12h, × 2 attempts | 25 | $16.50 | $1.00/h | 26 |
| + `eval`, `smoke` preset, 1h × 2 | 27 | $17.82 | $0.92/h | 28 |
| + `eval`, `knowledge` or `reasoning`, 3h × 2 | 31 | $20.46 | $0.80/h | 32 |

`acquire`'s lease (`max_hours`) has to cover the retried worst case plus its margin, or a sweep
could stop the box under a retried `eval`. The pipeline's budget comment ("a box dearer than $1/h
needs a larger budget") changes with it. What a person actually pays is the settled box time,
not the reservation.

Preset cost, both passes (base and candidate) on one box. These are estimates to be replaced by
the first run's measurements:

- **`smoke`** (gsm8k, arc_easy, hellaswag; 40 documents each; 0-shot): tens of loglikelihood
  batches plus 80 short generations per pass, two model loads, and the first-run dataset
  downloads. Expect well under the 1h timeout.
- **`reasoning`** (gsm8k 5-shot, arc_challenge, winogrande; 400 each): 800 long-prompt gsm8k
  generations dominate. Tens of minutes per pass; 3h timeout.
- **`knowledge`** (mmlu, truthfulqa_mc2, openbookqa; `limit` 200): `limit` applies **per task**,
  and `mmlu` is a group of 57 subtasks, so this is up to about 11,000 MMLU questions (four
  loglikelihood requests each) per pass, not 200. The preset's comment calls it "cheap enough to
  run on every cycle". That has to be measured before the preset is offered in `sft`, and it may
  need a per-subtask limit.

## Before implementing

1. **The dependency set.** Resolve `lm-eval` together with `cairn-operator-training[gpu]`'s exact
   pins (`uv pip compile`), pick the newest lm-eval release that accepts transformers 5.5, and pin
   it in `[harness]`. Add the resolution to CI so a trainer pin bump cannot break the box venv
   unnoticed.
2. **The `hf` backend's arguments on that release:** `peft=` for the adapter, and how 4-bit
   loading is requested under transformers 5 (`load_in_4bit` passed through `model_args`, or a
   quantization config). Evaluating a QLoRA adapter on a 16-bit base measures a different model
   from the one that was trained.
3. **The adapter on disk.** Observed in the `remote/exec` machinery: each job runs in
   `~/.cairn-jobs/<id>`, which nothing removes, and both legs run on the same `ml_target`.
   `train_qlora_local_adapter_dir` therefore still exists when `eval` starts. Check this on the box.
   With `artifact_target: s3` or `hf`, `adapter_uri` is the durable alternative (one download).
4. **A stage contract or none.** The training kernels implement `training.stage.*` contracts. An
   `eval.lm_eval` stage could ship uncontracted, as `compress/distill` does today, or with a new
   contract wheel (`cairn-contracts-eval-stages`). A new contract wheel is a public contract and an
   owner decision.
5. **Versions.** This pack 0.2.0 (the kernel and the `[harness]` pin), `cairn-pack-finetune` minor
   bump (the new flow, step, input and dependency), and the finetune README's walkthrough.
6. **Network on the box.** lm-eval downloads its task datasets from the Hugging Face Hub on first
   use. The box needs Hub access, as it already does for the base model. The harness release and
   the task versions it reports go on the record so a delta can be reproduced.

## Not proposed

- Serving the candidate on the box and pointing the existing HTTP operator at it. That is the
  right shape for measuring a deployed model, and this operator stays for it, but it needs vLLM in
  the box venv and a tunnel from the control plane to the box port, for a question that in-process
  evaluation answers.
- Caching the base model's scores across runs, keyed by base model, preset, harness release and
  `limit`. It would save one pass per run, but it is an optimisation for after the first
  measurements.
