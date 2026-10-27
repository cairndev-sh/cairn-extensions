# cairn-pack-finetune

Fine-tune a model for [cairn](https://github.com/cairndev-sh/cairn) with LoRA or 4-bit QLoRA, on
prepared data only: registered `train`, `val` and `test` versions, as
[`cairn-pack-dataset-prep`](../cairn-pack-dataset-prep) registers them (or your own JSONL splits
registered with those roles). A rerun or a sweep reads the same frozen versions.

A domain pack: YAML, a profile and presets, no code. Every stage comes from
[`cairn-operator-training`](../../packages/cairn-operator-training).

## Which pipeline

| Your GPU box | Pipeline |
|---|---|
| An EC2 instance that cairn starts after approval and stops when it is done (a `compute.ssh` connection with `instance_id`, `region` and `hourly_usd`) | `finetune/sft` |
| An SSH box you start and stop yourself (any provider), or one machine | `finetune/sft-own-box` |

`sft-own-box` is `sft` without its three box steps (`acquire`, `release-box` and the `release`
finalizer) and without the `compute_connection` input; its gates, legs, timeouts, retries, budget
and outputs are the same (a test pins the two files together). On `sft-own-box` cairn never
starts or stops the box, and a cancel does not stop a trainer already running on it: stop the
job or the box yourself. Both price their ml legs on a connection with `hourly_usd`, and both
show the estimate on `approve-run`. The rest of this README describes `sft`; for `sft-own-box`
skip the box steps and `compute_connection`, and launch `finetune/sft-own-box`.

## The `sft` pipeline

```
approve-run ⏸ → acquire → smoke → smoke-ok → train → release-box → improved → promote-model ⏸ → promote ⏸
                                                     finally (done, failed or cancelled): release
```

- **Admission, at launch.** Before any record exists, cairn's launch door checks the three
  versions:
  - each has the role its input declares (ADR-0146), so a test set given as training data, or a
    path instead of a registered version, refuses the launch;
  - the train and val versions carry an intake verdict, and a blocked or invalid one refuses;
  - training data that meets a registered test or golden version refuses (the holdout door);
  - the base model's licence.

  The decision is on the pipeline record and on the first gate's card.
- **`approve-run`:** a person reads it, and the GPU estimate, before GPU time is spent.
- **`acquire`:** `compute/acquire` starts the box `compute_connection` names (an EC2 instance),
  only after the approval. It stamps the `cairn:managed` and `cairn:release-after` lease tags
  first, then starts the instance and waits up to 600 s for its SSH port (if the port is still
  closed, `acquire` succeeds with `ssh_ready: false` and `smoke` fails on its own).
- **`smoke`:** the [`sft-train`](flows/sft-train.yaml) flow on a recorded subsample
  (`smoke_rows`, default 64) for `smoke_steps` (default 5). It proves the data, the tokenizer,
  the trainer and the packaging cheaply.
- **`train`:** the full run. At the end the trainer evaluates on `val_version` twice: the
  candidate (`eval_loss`), and the base model with the adapter switched off
  (`baseline_eval_loss`) — the same rows, the same trainer. `max_eval_loss` fails it fast.
  `package.adapter` registers the adapter as a model.

  Both ml legs get a second attempt when the first one's process is killed, and it resumes from
  that run's checkpoints (see [Checkpoints, retries and cancel](#8-checkpoints-retries-and-cancel)).
- **`release-box`:** `compute/release` stops the box as soon as `train`, the last GPU leg, is
  done, so it does not bill while a person decides on promotion.
- **`improved`:** the candidate's validation loss must be below the base model's.
- **`promote-model`:** a person decides, with the losses on the card.
- **`promote`:** `artifact/promote` points `tag` at the adapter, after its own approval.
- **`release`:** the same `compute/release`, declared `finalizer: true`: it runs once the pipeline
  is done, failed or cancelled, whatever the chain reached, and a box already stopped is a
  success. It is the backstop for a run that never reached `release-box`.

**Not yet:** the one-time comparison on `test_version`, and a generation-quality evaluation. Both
need the evaluation operators, which are not public yet. Until then the verdict is validation loss
against the base model's, and `test_version` is recorded, never trained on and never evaluated.

## Run it

Two machines: the **control plane** (the engine, the data leg, the gates and every record; no
torch) and a **GPU box** reached over SSH, which runs only the trainer. Every command below is
the `cairn` CLI; `<pl_id>` is a pipeline id (`pl_…`), `<run_id>` one leg's run.

**Which cairndev.** This pack (0.1.1) requires `cairndev>=0.3.1,<0.4`: 0.3.1 ships
`compute/acquire` and `compute/release`, prices a leg on a box with `hourly_usd`, and cancels the
job on the box before a cancelled pipeline's release stops it. CI proves it on the newest release
in that range.

### 1. Install

```bash
pip install 'cairndev[ssh]' cairn-pack-dataset-prep cairn-pack-finetune   # the control plane (brings boto3)
pip install 'cairn-operator-training[gpu]'      # the GPU box; add ,track-mlflow for MLflow
```

To run unreleased changes, install from a checkout of this repository: on the control plane
`pip install 'cairndev[ssh]' ./packages/cairn-contracts-* ./packages/cairn-operator-*
./packs/cairn-pack-dataset-prep ./packs/cairn-pack-finetune`, and on the box `pip install
./packages/cairn-contracts-training ./packages/cairn-contracts-training-stages
'./packages/cairn-operator-training[gpu]'`. The control plane starts and stops the box through
boto3, a dependency of `cairn-pack-finetune` (cairndev declares it only in an extra). The box must run the same `cairn-contracts-training*`
versions as the control plane, and needs `cairn-dataflow-run` on its non-interactive SSH `PATH`
(`ssh <box> 'which cairn-dataflow-run'`), or `CAIRN_DATAFLOW_REMOTE_CMD=<venv>/bin/cairn-dataflow-run`
on the control plane.

### 2. Durability

```bash
export CAIRN_STORAGE=sqlite:///$HOME/.cairn/train.db         # one host, one process
export CAIRN_STORAGE=postgres://user:pass@host:5432/cairn     # several processes: pip install 'cairndev[storage-postgres]'
pip install 'cairndev[runtime-dbos]' && export CAIRN_RUNTIME=dbos   # optional: resume a run at the next boot
```

Without `CAIRN_STORAGE` runs live in memory and are gone on restart. The schema migrates when the
server first opens the store; upgrade every process that shares one database together. **Not
Temporal:** `sft` has `task:` legs (`acquire`, the two releases, `promote`), which the Temporal
pipeline owner does not run, and a `finalizer:` step, which it does not implement (`sft-own-box`
has the `promote` task leg). The launch is refused under `CAIRN_RUNTIME=temporal` before any
record exists (`cairn remote launch` exits 1
with the reason; `cairn pipeline run` exits 3). Use `inprocess` (the default) or `dbos`.
`cairn dev` sets its own: SQLite at `--db` (default `~/.cairn/dev.db`) and `inprocess`.

### 3. The profile, the buckets and the box

```bash
cairn ls finetune                                  # lists profiles/finetune-ssh.toml with its installed path
export CAIRN_PROFILE=<that path>
export CAIRN_ARTIFACT_S3_BUCKET=<bucket>           # the registered dataset versions (CAIRN_ARTIFACT_S3_PREFIX optional)
export CAIRN_AUDIT_LOG=$HOME/.cairn/audit.jsonl     # the profile binds the hash-chained audit trail here (default: ./audit.jsonl)
export GPU_SSH_KEY="$(cat ~/.ssh/id_ed25519)"      # read by the engine; cairn never writes it to disk
cairn connect add ssh --name gpu --set host=<box> --set user=<user> \
  --set known_hosts=$HOME/.ssh/known_hosts --secret PRIVATE_KEY=GPU_SSH_KEY \
  --set instance_id=<i-…> --set region=<aws-region> --set hourly_usd=<rate>
cairn connect ls && cairn connect test gpu         # the box must be running for the test
```

`finetune-ssh` sends the ml legs to the `compute.ssh` connection named `gpu`, in the box's
virtualenv, and selects the s3 artifact store, because two machines need a store both can reach.
Its header lists every setting. Cancel and the checkpoint pull-back use the connection's
`PRIVATE_KEY` (`SSH_PRIVATE_KEY` is needed only for a job started under cairndev 0.2.1).

**The box is an EC2 instance that `sft` starts and stops.** `instance_id` and `region` name it
(give it an Elastic IP, so `host` stays the same across a stop); `hourly_usd` prices its legs.
The control plane needs AWS credentials that allow `ec2:DescribeInstances`, `ec2:CreateTags`,
`ec2:StartInstances` and `ec2:StopInstances` on that instance (the standard AWS chain: an
instance role, `AWS_PROFILE`, or `AWS_*` variables). `sft`'s `compute_connection` input names the
box to start and stop, and it **must be the connection `ml_target` selects** (`gpu` in
`finetune-ssh`, the input's default). cairndev does not check that the two match yet
([cairndev-sh/cairn#216](https://github.com/cairndev-sh/cairn/issues/216)); step 4's `explain`
prints both (`acquire`'s `with.connection` and each ml leg's `compute:`). With a
mismatch, `acquire` starts one box while the ml legs run on another, and a cancel does not stop
the job on the box the legs ran on. A box without an `instance_id` fails `acquire`, and the
finalizer's release fails the same way; use `sft-own-box` for such a box.

**On the box** the trainer reads `CAIRN_CHECKPOINT_BUCKET` (checkpoints, and the adapter with
`artifact_target: s3`), `CAIRN_CHECKPOINT_PREFIX`, `AWS_DEFAULT_REGION` and, for MinIO,
`AWS_ENDPOINT_URL`. Set them in the box's environment, or set them on the
control plane and name them in the profile's `remote_env` (the header shows the line); the engine
forwards each one that is set and refuses a name containing `KEY`, `SECRET`, `TOKEN` or
`PASSWORD`. Credentials stay on the box (an instance role, or its own `AWS_*`): read access to
the artifact bucket, write access to the checkpoint bucket. `MLFLOW_TRACKING_URI` set on the
control plane is forwarded by the engine itself.

### 4. Start the control plane and preflight

```bash
cairn dev --engine-only --port 8471 &                       # a laptop: sign-in dev / dev
cairn remote login -u dev -p dev -e http://localhost:8471
cairn doctor --strict                                       # configuration and reachability
cairn pipeline show finetune/sft                            # legs, gates, and which runtime runs each leg
cairn pipeline explain finetune/sft -i train_version=dataset:ifc-bim-alpaca-train@latest   # acquire's connection = the ml legs' compute
```

A server runs `cairn-server` with the same environment plus sign-in (`CAIRN_JWT_SECRET`,
`CAIRN_API_USER`, `CAIRN_API_PASSWORD_HASH` from `cairn remote hash-password`). `explain` is a dry
run: nothing starts and nothing is written.

### 5. Prepare the data, then a dry run without a GPU

```bash
cairn remote launch dataset-prep/prepare --preset ifc-bim-alpaca --watch   # once; see cairn-pack-dataset-prep
cairn remote advance <pl_id> source-ok                                      # a person confirms the source
cairn remote findings <pl_id>                                               # the three versions it registered
cairn remote launch finetune/sft --preset ifc-bim-qwen3-4b -i stub=true --watch
```

The stub trainer (`stub=true`) runs every leg, gate and record on the box without a GPU and
trains nothing. Its validation loss is not below its baseline, so `improved` stops it: a stub
adapter never reaches a person who could promote it. With `CAIRN_CHECKPOINT_BUCKET` set on the
box, the smoke leg writes a stub checkpoint there, which shows the box can write to the bucket.
It is a real run on the box, so after `approve-run` it starts the box (real EC2 calls, a few
minutes of box time) and stops it after `train`; the finalizer finds it stopped.

### 6. Train, and the three approvals

```bash
cairn remote presets finetune/sft
cairn remote launch finetune/sft --preset ifc-bim-qwen3-4b --watch
cairn remote advance <pl_id> approve-run                    # 1. before any GPU time, with the estimate on the card
cairn remote advance <pl_id> promote-model                  # 2. after `improved` (the box is already stopped), with the losses on the card
cairn remote advance <pl_id> promote                        # 3. artifact/promote's own approval
```

`approve-run`'s card carries `estimate_usd` and `estimate_basis` in its evidence: the worst case
the launch door checked, per ml leg (for example `train: $0.66/h × 12h on gpu × 2 (2 attempts)
= $15.84`). `acquire` is `paid_compute`: under a profile that requires approval for paid compute
it pauses for its own approval after `approve-run` (`cairn remote advance <pl_id> acquire`).

Override any input with `-i key=value` and keep the result with `--save-preset <name>`. A `list`
input takes JSON (`-i 'report_to=["mlflow"]'`) or a comma string
(`-i report_to=mlflow`). `cairn remote pipeline <pl_id>` shows which step is waiting. `--watch` returns when the pipeline pauses at a gate; `approvers:
ml-owners` names who decides, and with sign-in on the signed-in user is recorded.

The launch door refuses, before any record exists (`cairn remote launch` exits 1 with the reason): a test version given as
`train_version`, a train or val version without an intake verdict, training data that meets a
registered test or golden version, and a gated base model without `-i accept_license=true`.

### 7. Watch, logs, findings, replay

```bash
cairn remote watch                                  # a live view of the runs (Ctrl-C)
cairn remote pipeline <pl_id>                       # the durable record: plan, step states, admission, job handles
cairn remote status <run_id>                        # one leg's run
cairn remote logs <run_id> -f                       # its event log, followed live (--after-seq N resumes)
cairn remote findings <pl_id>                       # every step's findings: losses, steps, adapter URI, versions
cairn replay <run_id> --source db                   # the recorded events as a timeline; runs nothing again
```

### 8. Checkpoints, retries and cancel

- The trainer saves a checkpoint every `save_steps` (50), keeps `save_total_limit` (2), and
  mirrors each one to `s3://<CAIRN_CHECKPOINT_BUCKET>/<prefix>/<run-id>/`. It writes under the
  leg's job directory on the box (`train-output/<output_name>`), so two runs never share
  checkpoints. The published adapter is the adapter files, without the checkpoints.
- **`smoke` and `train` declare `retry: {max_attempts: 2}`** (an ml leg otherwise gets one
  attempt). When an attempt's process dies without writing its exit code (the OOM killer, a
  kill on the box), the engine reports it retriable and the leg runs once more under the
  same run; the trainer pulls that run's newest checkpoint from `CAIRN_CHECKPOINT_BUCKET` and
  continues from its step. A leg that fails on its own error (a clean non-zero exit) is not
  retried, and neither is a poll that cannot reach the box. Without a checkpoint bucket the retry starts again from step 0.
- **Budget:** `governance.budget: {usd: 25}` on the `sft` series, sized to the priced legs'
  worst case. Before any record exists, the launch door checks `hourly_usd × timeout × 2
  attempts` for each ml leg against what the series has left: (30 min for `smoke` + 12 h for
  `train`) × 2 = 25 box-hours, **$16.50 at $0.66/h**. A box dearer than $1/h is refused at launch
  with the figures; give it a larger budget, or a copy of the pipeline with shorter timeouts. Each
  attempt reserves one attempt's figure before it is submitted, so an attempt the budget cannot
  cover fails the leg instead of starting, and every attempt settles its own box time (`Run.cost.gpu_usd`),
  failed or not. Not priced: the minutes between `acquire` and `smoke` (the boot) and between
  `train` and `release-box`; a connection without `hourly_usd` is "not priced", never $0.
- **Cancel:** `cairn remote cancel-pipeline <pl_id>` stops the pipeline: no further leg starts.
  If `smoke` or `train` is running on the box, its job is cancelled first (the trainer is
  killed, and the leg settles `cancelled`, not retried), then the finalizer `release` stops the
  box. A job cancel that fails is logged and audited (`leg_job_cancel`, `sent: false`), and the
  box is stopped anyway. The record reads `cancelled` once the release has settled.
- **The box:** `sft` starts it after `approve-run` and stops it after `train`, and the finalizer
  stops it on every outcome that reaches it. Two gaps: a cancel that lands while `acquire` is
  still starting the box can leave it running (core cancels flow legs, not a task leg in flight),
  and a `release-box` that fails after `train` (a transient EC2 error) fails the pipeline before
  `improved`: the candidate is registered, but promotion is manual. In both cases the finalizer
  or the lease (below) stops the box. `acquire` stamps a lease of 26 hours (`cairn:release-after`) before
  the start: the retried worst case, 2 attempts × (30 min + 12 h) = 25 h, plus an hour, so a
  sweep never stops the box under a retried `train`. If the control plane dies, `cairn compute
  sweep` lists the box once the lease has passed and `--apply` stops it. **The trade-off:** until
  then an orphaned box keeps billing, up to 26 h (about $17 at $0.66/h), so run the sweep from a
  cron, or stop the box yourself. `cairn compute status gpu` shows whether the box is up and what
  it costs per hour.

### 9. Optional: an admission policy

[`docs/policies/require-dedup.yaml`](docs/policies/require-dedup.yaml) is an example, not active:
in a production environment, each dataset version's intake verdict must show that exact and near
deduplication ran. The holdout door needs no policy: it runs on every launch.

```bash
mkdir -p ~/.cairn/policies                       # or $CAIRN_POLICIES_DIR
cp <the pack>/docs/policies/require-dedup.yaml ~/.cairn/policies/
cairn validate ~/.cairn/policies/require-dedup.yaml
```

`<the pack>` is this directory in a checkout, or the installed `cairn_pack_finetune` directory
(the one `cairn ls finetune` shows the profile in).

What the door decided is on the pipeline record (`admission`, with `policy@<digest>`) and on the
first gate's card. A check a policy names must be one the door knows; it refuses every launch
under an unknown one, `holdout` included.

**MLflow (optional):** `MLFLOW_TRACKING_URI` on the control plane with
`-i 'report_to=["mlflow"]'` (or `-i report_to=mlflow`)
records the run; the box installs `cairn-operator-training[gpu,track-mlflow]` and must reach the
URI. **On one machine:** use `finetune/sft-own-box` (`sft` needs an EC2 instance to start and
stop) with the default profile and no ssh settings; dataset-prep still registers its versions
through the s3 store.

## Presets and profiles

| | |
|---|---|
| `presets/ifc-bim-qwen3-4b.yaml` | for `sft`: `Qwen/Qwen3-4B-Base`, QLoRA r 16, alpha 32, lr 2e-4, 2048 tokens, seed 3407, one epoch, batch 2 × 8, on `ifc-bim-alpaca-{train,val,test}@latest` |
| `presets/ifc-bim-qwen3-4b-own-box.yaml` | the same preset, `ifc-bim-qwen3-4b`, for `sft-own-box` (a preset file names one pipeline; a test keeps the two equal) |
| `profiles/finetune-ssh.toml` | ml legs on the `gpu` ssh connection (`ml_engine = "builtin"`, `ml_target = "gpu"`, venv mode), the s3 artifact store; for both pipelines (with `sft-own-box` the connection needs no `instance_id`) |

## Support

| Path | Status | Evidence | Proven on `cairndev` |
|---|---|---|---|
| `sft` flow and gates on the stub trainer | EXPERIMENTAL | `tests/test_sft.py`: the pack's flow on the core runner with resolved, pinned versions; the pipeline's own predicates through the core's evaluator | 0.3.1 (the released wheel: a local run and CI) |
| `smoke` / `train` retry after a kill | EXPERIMENTAL | `tests/test_sft.py`: each leg through the core's leg executor with the stub trainer and a fake S3; an attempt killed after `checkpoint-2` is retried once and the retry writes `checkpoint-4` and `-6` from the pulled checkpoint; a leg's own error is not retried; one reservation per attempt | 0.3.1 (a local run and CI) |
| example admission policy | EXAMPLE | `tests/test_example_policy.py`: it validates, names only known checks, and the core's admission decision admits or refuses on it | 0.3.1 (a local run and CI) |
| `finetune-ssh` profile selection | EXPERIMENTAL | `tests/test_finetune_ssh_profile.py`: the core resolves the ml leg to the SSH engine on `gpu` (venv mode), the data leg to the local engine, the s3 store; no connection is made | 0.3.1 (the released wheel: a local run and CI) |
| the box started and stopped by `sft`: `acquire` after `approve-run`, `release-box` after `train`, the finalizer `release`; a cancel cancels the job on the box first | EXPERIMENTAL | `tests/test_sft_box.py`: the plan; the pipeline through the core's launch door and in-process bridge on the stub trainer, with a fake EC2 client (the core's `CAIRN_EC2_CLIENT_FACTORY` seam): the box starts only after the approval and stops right after `train`; a cancel mid-`train` cancels the job before the box stops; a failed `train` still stops the box | 0.3.1 with a fake EC2 only (a local run and CI). A real start and stop on a public release is not proven yet: cairn-extensions#9 |
| the GPU estimate at launch and on `approve-run` | EXPERIMENTAL | `tests/test_sft_box.py`: the core's launch door prices `smoke` and `train` on a `gpu` connection at $0.66/h ($16.50, two attempts each), `approve-run`'s evidence carries it and no later gate's does, and a $1.50/h box is refused against the $25 budget before any record | 0.3.1 (a local run and CI) |
| `sft-own-box`: `sft` without the box steps | EXPERIMENTAL | `tests/test_sft_box.py`: the file is pinned to `sft.yaml` key for key except the box steps and `compute_connection`; its plan has the same legs and gates and no compute step; the same stub run on the core's launch door and bridge makes no EC2 call and shows the estimate on `approve-run` | 0.3.1 (a local run and CI) |
| `sft` on an ssh GPU box with QLoRA | not yet | the qualification run, cairn-extensions#9 | not yet |

Apache-2.0 — see [`LICENSE`](LICENSE).
