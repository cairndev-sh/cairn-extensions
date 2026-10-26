# cairn-operator-training — design

- **Distribution:** `cairn-operator-training` (`packages/cairn-operator-training/` in cairndev-sh/cairn-extensions)
- **Capabilities provided:** `train.run` (stub · local), `compress.distill`; dataflow stages `train.qlora`, `train.dpo`, `train.distill`, `eval.checkpoints`, `package.adapter`, `dataset.dpo_pairs`
- **Status:** proven (QLoRA on an ssh GPU box, 2026-08-28; local and stub in CI-shaped pack tests)
- **Decisions:** ADR-0038 (remote job protocol), ADR-0043 (job channel), ADR-0064 (durable checkpoints), ADR-0095 (reproducibility), ADR-0099 §1 (job-channel door), ADR-0102 D4/D7 (live metrics, generic reporting), ADR-0103 (reporting contract), ADR-0081/ADR-0082 D7 (ml-flow stages on an engine); #136 #553 #1622 #1709 #1749 #2025

## 1. Purpose

Run ONE fine-tune (LoRA / QLoRA SFT, DPO, distillation) from a `TrainSpec` and hand back a
`TrainResult` — the adapter's location, the final metrics, a checkpoint URI when one is
durable — on whichever compute the environment selected. There is ONE trainer; WHERE it runs is
the engine's choice, not this pack's: the `sft-train` flow's `train.qlora` stage runs on the
profile's `ml_engine` / `ml_target` (an ssh GPU box today, the SkyPilot engine's clouds and
Kubernetes as that engine is proven), and `train/local` runs LoRA in-process on CPU/MPS.
The eval gate, the cost gate, promotion and lineage are other packs' operations; a pipeline
composes them. It does not choose hyperparameters (that is
the TrainSpec tiers, #2023), does not judge the model (`model/gate`), and does not serve it.

## 2. Architecture

There is ONE trainer: `train_entry.main` (Unsloth 4-bit QLoRA on CUDA) or
`operators/local._train_local` (plain PEFT LoRA on CPU/MPS, the hermetic path tests run). The heavy
environment — torch, unsloth, trl, peft, bitsandbytes at the pins in `images.py::TRAINER_PINS`
(#1622) — is the `[gpu]` extra on the GPU box, never the base wheel.

```
TrainSpec ──► train.qlora stage (on the engine's target) | train/local (in-process)
                 │  TrainerCallbacks:
                 │   • checkpoint_sync  → S3 mirror on on_save (ADR-0064)
                 │   • reporting        → progress/metrics/checkpoint/scalars (#2025)
                 │   • MLflow/W&B       → durable tracker (#136, #1709)
                 ▼
            adapter dir → publish (HF | S3 | local) ──► TrainResult / stage stats
```

**Where the work runs.** Not here. The flow stages are engine-neutral kernels (ADR-0082 D7):
the builtin engine runs them in-process or on an ssh box (`cairn-dataflow-run` over SSH, in a venv
with the `[gpu]` extra, or in a trainer image you provide), and any engine that carries stage contracts runs the same callables. This pack
ships no provider client; a managed GPU is an engine's concern.

**The live channel (#2025, ADR-0102 D4).** The trainer's `reporting.make_report_callback`
emits the declared vocabulary (§3) through whichever reporter the run has: the control plane's
typed `ctx.report` in-process, or `live.LiveReporter` on a box. The box side is stdlib-only
(`live.py`): every event goes to a local **spool** first, ships in batches over the **`http`**
transport (`POST /runs/{id}/events/ingest`, control by long-poll on `GET /runs/{id}/control`,
rank-0 only, ≤1 poll / 5 s, silent on failure) or over **NATS JetStream** (`CAIRN_JOB_CHANNEL=
nats://…`, the run's `cairn.jobs.<run>.events|control` subjects), and is acknowledged by the
control plane's idempotent ingest keyed on `(node, seq)`. A network cut is a replay on
reconnect; the run store — never the channel — is the log. Control down: `abort` and
`checkpoint-and-stop` (save at the current step, then stop). The engine hands the box its
identity and transport as env (`CAIRN_RUN_ID`, `CAIRN_NODE`, `CAIRN_ENGINE_URL` + a
run-scoped bearer when auth is on, or `CAIRN_JOB_CHANNEL` — sent with its `OBS_JOB_CHANNEL`
twin until 1.0, #2429).

**State.** Adapters land in `/tmp/<output_name>` on the trainer host, then wherever
`artifact_target` says. Checkpoints mirror to `s3://$CAIRN_CHECKPOINT_BUCKET/<prefix>/<run
key>/checkpoint-N/` on every save and are pulled back before a relaunch — but only under a
run-scoped key (#1749). Live events persist in the run store; MLflow/W&B stay the durable
experiment tracker.

**Idempotency / retries.** A resumed job resumes from its last checkpoint and the result says
so (`resumed_from_checkpoint`). A relaunch after a dead box is the control loop's decision
(#552), never the operator's. The reporter never raises into the trainer; a report failure
is one warning.

## 3. Operators

| Operator | Capability | Risk | Mock | Input -> Output | Reports (stages · series · scalars) |
|---|---|---|---|---|---|
| `train/stub` | `train.run` | read_only | yes | `TrainSpec` -> `TrainResult` | download, tokenise, train, eval, export · loss (nats), eval_loss (nats), learning_rate, grad_norm, epoch, tokens_per_s (tok/s), gpu_util_pct (%), gpu_mem_gb (GiB), gpu_mem_total_gb (GiB), gpu_temp_c (°C), gpu_power_w (W), cpu_pct (%), ram_gb (GB) · final_loss, best_eval_loss, tokens_seen, steps, train_runtime_s |
| `train/local` | `train.run` | read_only | yes | `TrainSpec` -> `TrainResult` | same declaration as `train/stub` (one `TRAIN_REPORTS`) |
| `compress/distill` | `compress.distill` | paid_compute | yes | `DistillSpec` -> `DistillResult` | (backfill #2035 — not yet declared) |

Both `train/*` operations and the `train.qlora` stage share ONE declaration, `reporting.TRAIN_REPORTS`,
because they run one trainer: **stages** `download → tokenise → train → eval → export`; **series** the
loss/optimiser curve at every `logging_steps` (`loss`, `eval_loss`, `learning_rate`,
`grad_norm`, `epoch`, `tokens_per_s`) plus the `system` sample every 15 s (`gpu_util_pct`,
`gpu_mem_gb`, `gpu_mem_total_gb`, `gpu_temp_c`, `gpu_power_w`, `cpu_pct`, `ram_gb`, from
`nvidia-smi`/`psutil` when present); **scalars** at the end (`final_loss`, `best_eval_loss`,
`tokens_seen`, `steps`, `train_runtime_s`). `checkpoint` fires on every trainer save with the
step and the best eval loss so far. The stub emits a five-point curve of the same vocabulary
so a `stub: true` pipeline exercises the whole live path. With `CAIRN_CHECKPOINT_BUCKET` set
(and `max_steps`, `save_steps` and a workdir), the stub also keeps checkpoints the way the
trainer does: it pulls the run's newest one, continues from its `global_step`, and mirrors a
`checkpoint-<step>/trainer_state.json` every `save_steps`, so a retried leg can be shown to
resume (cairn-extensions#28). It reports `resumed_from_checkpoint` and `checkpoint_uri` as
`train_entry` does. The files hold only the step.

**The fidelity knob** (`sample.fidelity`, a `cairn.flow_stages` kernel in `flow_stages.py`) is
the other half of #2029: it keeps the `n` rows with the smallest `H(seed ‖ row_hash)` — a
choice that depends on the corpus and the seed and on nothing else — writes them as the run's
dataset, and records `fidelity_seed`, `fidelity_n` and an **order-independent row-hash
digest** both in the stage's stats and in `TrainSpec.extra`, which ADR-0103 D3 writes verbatim
into the run's `parameters` event. Two subsampled runs with the same seed reproduce the same digest;
two whose digests differ trained on different data and the record says so. The kernel is not
called `sample.rows` because that name belongs to dataset_checks' dataflow sampler in the same
shared registry — the two share the ALGORITHM (`cairn.core.dataflow.fidelity`), which is the thing that
must not be duplicated. It is a complete no-op when no fidelity is set.

**What the trainers refuse.** A `TrainSpec` whose `report_to` names a tracker that does not
answer its preflight is trained anyway with `tracking_error` on the result (#1709) — never
fatal before step 1. `CAIRN_TRAIN_VALIDATE` runs a tiny model for a bounded wall-time and
marks the result `validation: true` so it is never mistaken for a real train.

### The `train.run` contract (#210, ADR-0111)

`train/stub` and `train/local` implement `contract://train.run/1.0.0` (`cairn-contracts-training`):
`TrainSpec` is the shared input, `TrainResult` the shared output, the configuration is the empty closed
object (every knob rides the spec). The same distribution carries `train.teardown/1.0.0`, which nothing in
this wheel implements; the contract wheel is the unit, so it ships whole and its bytes are unchanged. Distinct from the public core's `cairn-contracts-train` (`train.finetune`, the
builtin `train_local`'s smaller contract).

### Stage contracts on the `sft-train` kernels (#158, ADR-0118)

**`train.qlora` implements `training.stage.train_qlora/1.1.0`** (cairn-extensions#4). Its configuration adds the
governed splits and the trainer knobs, and drops `extra`:

| Key | Default | What it does |
|---|---|---|
| `train_uri` | `""` | the training data (a registered train version); `dataset_uri` mirrors it when absent |
| `eval_uri` | `""` | the evaluation data; when set, `eval_split` is not used |
| `test_uri` | `""` | recorded on the result as `test_uri`; the trainer never opens it |
| `per_device_batch_size` | `0` | rows per device per step; 0 keeps the trainer's default (8 on the GPU path, 1 on CPU/MPS) |
| `gradient_accumulation_steps` | `0` | 0 keeps the default (1) |
| `gradient_checkpointing` | `null` | null keeps the default (Unsloth's on the GPU path) |
| `lora_dropout` | `null` | null keeps the default (0 on the GPU path, 0.05 on CPU/MPS) |
| `baseline_eval` | `false` | 1.2.0 (cairn-extensions#7): report `baseline_eval_loss`, the base model's loss on the evaluation data — measured after training with the adapter disabled (PEFT `disable_adapter`), on the same rows and trainer as `eval_loss`. Evaluating before `train()` broke the fp16 path on a T4. `baseline_skipped` names why when there is none |

Both trainers read them through `splits.py`, and record `effective_batch_size`, `eval_source` (`eval_uri`,
`eval_split` or `none`) and `test_uri` on the result. A fidelity subsample replaces `train_uri` when the split is
governed. The 1.0.0 files are unchanged; the output stays `training.stage.train_qlora.output/1.0.0`. Not in 1.1.0:
`completion_only_loss` — the rows are rendered into one `text` column, which TRL's completion-only masking does not
apply to, and Unsloth's response-only masking needs a GPU run to verify. The docx recipe's values (alpha 32,
seed 3407) are the pack preset's, not these defaults.

`sample.fidelity`, `train.qlora`, `eval.checkpoints` and `package.adapter` implement
`contract://training.stage.<kernel>/1.0.0` (`cairn-contracts-training-stages`), declared as `implements` beside
`risk_class` in `flow_stages.py`: the configuration is CLOSED (for `train.qlora` it is `TrainSpec`'s fields plus
`stub`), the output is open with the keys the retrain pipeline reads from the leg's findings `required`
(`adapter_uri`, `train_loss`, `eval_loss`, `steps`, …). The launch door records them, the control plane checks
the resolved config before submission and the descriptor stats before registration, and the runner checks
both on the box — so an engine that does not declare `carries_stage_contracts` is refused for `sft-train`,
`train-only`, `verifiable-finetune` and `dpo-train` (ADR-0118 D7): builtin, SSH and Metaflow declare it; the
SkyPilot engine does not yet (cairndev-sh/cairn#135). `train.distill`, `dataset.dpo_pairs` and the other kernels
here stay legacy until their pipelines' slices. Proof: `tests/core/test_retrain_cycle_stage_contracts.py`
(every kernel under its hermetic stub, and the box refusals on the real builtin engine for an ML kernel).

## 4. Configuration

| Setting | Where | Scope | Default | What it does |
|---|---|---|---|---|
| `base_model`, `dataset_uri`, `method`, `epochs`, `max_steps`, `learning_rate`, `lora_r`, `lora_alpha`, `max_seq_len`, `seed`, `eval_split`, `eval_steps`, `save_steps`, `packing`, `gpu`, `artifact_target`, `hf_repo`, `report_to`, `tracking_project`, `init_adapter_uri` | `input_schema` (`TrainSpec`) | run | schema defaults | per-run bound; the launch dialog's tiers (#2023) show all of them |
| `compute_connection` | `input_schema` (`TrainSpec`, #2074) | run | `""` | part of the published `train.run` input; nothing in this wheel reads it. |
| `CAIRN_JOB_CHANNEL` | env (control plane) | deployment | in-memory | `nats://…` (declared service) or `http` (the control plane's API) for the live channel; a remote leg on an unreachable NATS is refused at the door (ADR-0099 §1) |
| `CAIRN_ENGINE_URL` | env (control plane) | deployment | unset | the URL a box posts events to on the `http` channel, handed on under the same name (`OBS_ENGINE_URL` is its alias until 1.0, #2429); unset + `http` = the box runs blind, with one warning |
| `CAIRN_RUN_ID`, `CAIRN_NODE`, `CAIRN_ENGINE_URL`, `CAIRN_ENGINE_TOKEN`, `CAIRN_LIVE_SPOOL` | env (box, set by the engine) | job | — | the worker's identity, transport and spool path (`/tmp/cairn-live-<run>-<node>.jsonl`) |
| `CAIRN_CHECKPOINT_BUCKET`, `CAIRN_CHECKPOINT_PREFIX` | env | deployment | unset (no mirror) | durable checkpoint target (ADR-0064) |
| `CAIRN_TRAIN_SPEC`, `CAIRN_<FIELD>` | env (box) | job | — | how `cairn-train-local` / `train_entry` receive the spec |
| `CAIRN_TRAIN_VALIDATE[_SECONDS]`, `CAIRN_TRAIN_STUB`, `CAIRN_TRAIN_IGNORE_RAM` | env | dev | unset | validation / stub / RAM-guard switches |
| `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `WANDB_*`, `HF_TOKEN`, `AWS_*` | env / connections | deployment | — | trackers, Hub push, S3 — folded into the job env from `tracking.*` connections (#136) |

**Per-run bounds live in `TrainSpec`.** Which GPU a run gets is the environment's (the profile's
`ml_target`), never a run input.

**Accounts (cairn 0.2.0, sign-in on).** cairn resolves a run's connections and declared secrets in
the account that owns the run's workspace. Two consequences for this wheel:

- A connection bound as the profile's `ml_target` belongs to the deployment, and a team account's
  flow legs cannot use it. The account creates its own `compute.ssh` connection under its own name
  (names are unique across a deployment) and names it as the target in its pipelines or flows.
- This wheel declares no `secrets:`. A team account's declared secrets reach `ctx.secrets`; its
  run does not receive its account's credentials in the process environment, so `HF_TOKEN`, `WANDB_*` and `AWS_*` read from
  the environment are whatever the control-plane process or the box holds: the deployment's own
  values, or none, for every account. Follow-up: declare them as `secrets:` and read them from
  `ctx.secrets`, so a team account's run uses that account's credentials.

## 5. Governance

`train/local` and `train/stub` are `read_only` (nothing leaves the host). The flow stages that write
an adapter (`train.qlora`, `train.dpo`, `train.distill`, `package.adapter`) declare `external_write`,
and the flow leg's gate is decided on the stages' aggregate risk (`enforce_flow_gate`). Spend is the
engine target's: a pipeline places `compute/estimate` + a cost gate ahead of the train leg and
`model/gate` behind it. Auditable: the launch's effective `parameters` event (ADR-0103 D3), every
live event in the run store, `checkpoint_uri` on the result, the licence acceptance for gated base
models (#2027). A control request (`abort`, `checkpoint-and-stop`) is itself a `control` run event
carrying who asked.

## 6. Failure modes

| Symptom | Cause | What the operator returns | Fail closed? |
|---|---|---|---|
| a resumed run "trained" 400 steps in 13 s with loss 0.0003 | checkpoints pulled under a shared, non-run-scoped key (#1749) | refuses the pull; trains fresh | yes |
| `tracking_error` on a succeeded result | MLflow/W&B unreachable at preflight (#1709) | trains, drops the sink, keeps the reason | no (degrade + say so) |
| eval served a mid-training checkpoint, adapter far below experiment (#1715/#1776) | promote read the wrong artifact | now `checkpoint_uri` + `adapter_uri` are distinct on the result | — |
| two runs weeks apart trained on different trl/unsloth | unpinned trainer deps (#1622) | `TRAINER_PINS` in `images.py`, mirrored in the `[local]` extra | yes |
| `<EOS_TOKEN>` placeholder rejected by the tokenizer | trl imported before unsloth patched it (2026-08-19) | unsloth is imported first in `train_entry` | yes |
| no live curve on a remote job; console blind between MLflow polls (#553) | nothing consumed the job channel | #2025: spool → channel → idempotent ingest; `stage`/`metrics` events in the run store | — |
| a remote job on `CAIRN_JOB_CHANNEL=nats://…` with no broker | events would go nowhere | refused at the launch door naming `nats` (ADR-0099 §1) | yes |
| box link cut mid-train | transient network | the spool replays on reconnect; the store ends exact on `(node, seq)` | — |
| `_live_env`/`CAIRN_RUN_ID` present but no transport | misconfigured submit | trains blind; events stay in the spool; one warning | no |
| a chat-format corpus (`messages`, e.g. smoltalk) "trained" with a meaningful-looking loss | the SFT mapping read only `input`/`output`, so every chat row became two empty turns (#1458) | `_to_messages` applies `dataset/format`'s conversational rules to a `messages` row (roles in any case, `human`/`gpt` normalised; no empty non-system turn; alternating; ending on the assistant), additionally requires string content, and keeps `role`/`content` only; alpaca (non-empty `instruction`), `input`/`output` and `prompt`/`completion` map by the values a row carries; only `messages` reaches trl. Any other row — and sharegpt or raw `text`, not trained here yet (#2653) — is refused with its row number, before the model loads (`distill_entry` keeps its own mapping: on-policy distillation takes a prompt with no answer) | yes |

## 7. Testing

Pack tests live in `tests/` beside `src/` (rule 5): `test_live_reporting.py` (#2025 — the
200-step network-cut acceptance against the real idempotent ingest, `checkpoint-and-stop` at
the last logged step, abort, the system-sample cadence, the stub's declared curve, a fake
`transformers` host so no torch is needed), `test_checkpoint_sync.py` / `test_checkpoint_scoping.py`
(ADR-0064, #1749, moto S3), `test_tracking_preflight.py` (#1709), `test_trainer_pins.py`
(#1622), `test_local_entry.py`, `test_gpu_unsloth_dispatch.py`, `test_validate_mode.py`,
`test_flow_stages.py`, `test_dpo_stages.py`, `test_distill.py`, `test_distill_stage.py`, `test_sample_fidelity.py`
(#2029 — the `sample.fidelity` stage's seed reproducibility, its record on the `TrainSpec`, and
the stub's declared curves). Needs a GPU and is not in CI:
`train_entry.main` itself (validated live on the C2 GPU run, 2026-08-19) and the NATS transport
against a broker. From the repository root, with the wheel installed: `pytest packages/cairn-operator-training/tests`;
the manifest sync/drift gates (`cairn pack sync --check`, `cairn pack check`) read the INSTALLED
pack — install it before trusting them.

## 8. History

| Date | Change | PR |
|---|---|---|
| 2026-06 | train.run providers (stub/modal/kaggle), remote_job protocol | #124, #127 |
| 2026-07 | RunPod/Vast HTTP providers on the shared base; tracking creds folded into the job env | #136, #1544 |
| 2026-08 | Durable checkpoints on S3, run-scoped pull; trainer pins; tracking preflight; two-stage init adapter | #1466, #1749, #1622, #1709, #1463 |
| 2026-09 | Live metrics over the job channel: `reporting.py` + `live.py`, `reports = TRAIN_REPORTS` on every trainer, `checkpoint-and-stop`, the NATS door; this design record | #2025 |
| 2026-09-07 | ADR-0104 D1/D6: one declared `GPU_IMAGE` pin published by CI (was `:latest` ×2 + a Modal pip line), `ManagedTrainLeg` so a pipeline `task:` leg reaches a managed provider, `TrainSpec.compute_connection`, `train/teardown`, and a managed-train pipeline | #2074 |
| 2026-09-06 | T10 proxy ranking: `train/proxy-trial` (ask/report + SuccessiveHalvingPruner), the `sample.fidelity` kernel + deterministic row-hash digest, `extra.proxy` on the record, `stub_curve` so trials differ | #2029 |
| 2026-09-11 | #2189 (T04): `package.adapter` emits the candidate's content identity (`output` + `content_hash` over the weight files + `kind: model`, `spec_digest`); `TrainResult` gains `local_adapter_dir` (additive); the stub tier leaves deterministic adapter bytes (spec knobs + the data's hash) and publishes nothing | #2189 |
| 2026-10-15 | #1458: the SFT trainer maps chat, alpaca, `input`/`output` and `prompt`/`completion` rows with the data plane's turn rules, refuses an empty turn or an unsupported shape with the row number, and does so before the model loads (it trained chat corpora on empty strings) | #1458 |
| 2026-10-16 | #2643: `train.distill` — `compress/distill` pipelines reach the GKD trainer through the `distill` ml flow (GPU target), not a `task:` leg that ran it in the control plane; the operator and the kernel share one `distill_body` | #2643 |
| 2026-10-21 | Moved public to `cairndev-sh/cairn-extensions` with its two contract wheels: `train/proxy-trial` and its Optuna ranking stay with the platform; `sample.fidelity` imports `cairn.core.dataflow.fidelity` (cairndev 0.2.0); the `[gpu]` extra is the exact trainer pin set; `[compat]` and the `cairndev` pin are bounded to `>=0.2.0,<0.3`. |
| 2026-10-22 | #4: `train.qlora` implements `training.stage.train_qlora/1.1.0` (governed train/eval/test splits; batch, accumulation, checkpointing and dropout knobs; `extra` out of the configuration); `splits.py` shared by both trainers; `cairn-contracts-training-stages` 1.1.0. |
| 2026-10-21 | One trainer, no provider clients: `train/modal`, `train/kaggle`, `train/runpod`, `train/vast`, `train/stage-dataset`, `train/teardown`, `managed_leg` and the published-image pin leave the public wheel; where a run goes is the engine's choice (ssh today). The `modal` and `kaggle` extras go with them. |
| 2026-10-26 | #28: the stub tier keeps run-scoped checkpoints in `CAIRN_CHECKPOINT_BUCKET` and resumes from them, so the `sft` retry is proven on the stub |
