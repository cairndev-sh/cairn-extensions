# cairn-pack-finetune

Fine-tune a model for [cairn](https://github.com/cairndev-sh/cairn) with LoRA or 4-bit QLoRA, on
prepared data only: registered `train`, `val` and `test` versions, as
[`cairn-pack-dataset-prep`](../cairn-pack-dataset-prep) registers them (or your own JSONL splits
registered with those roles). A rerun or a sweep reads the same frozen versions.

A domain pack: YAML, a profile and a preset, no code. Every stage comes from
[`cairn-operator-training`](../../packages/cairn-operator-training).

## The `sft` pipeline

```
approve-run ⏸ → smoke → smoke-ok → train → improved → promote-model ⏸ → promote ⏸
```

- **Admission, at launch.** Before any record exists, cairn's launch door checks the three
  versions:
  - each has the role its input declares (ADR-0146), so a test set given as training data, or a
    path instead of a registered version, refuses the launch;
  - the train and val versions carry an intake verdict, and a blocked or invalid one refuses;
  - training data that meets a registered test or golden version refuses (the holdout door);
  - the base model's licence.

  The decision is on the pipeline record and on the first gate's card.
- **`approve-run`:** a person reads it before GPU time is spent.
- **`smoke`:** the [`sft-train`](flows/sft-train.yaml) flow on a recorded subsample
  (`smoke_rows`, default 64) for `smoke_steps` (default 5). It proves the data, the tokenizer,
  the trainer and the packaging cheaply.
- **`train`:** the full run. At the end the trainer evaluates on `val_version` twice: the
  candidate (`eval_loss`), and the base model with the adapter switched off
  (`baseline_eval_loss`) — the same rows, the same trainer. `max_eval_loss` fails it fast.
  `package.adapter` registers the adapter as a model.

  Both ml legs get a second attempt when the first one's process is killed, and it resumes from
  that run's checkpoints (see [Checkpoints, retries and cancel](#8-checkpoints-retries-and-cancel)).
- **`improved`:** the candidate's validation loss must be below the base model's.
- **`promote-model`:** a person decides, with the losses on the card.
- **`promote`:** `artifact/promote` points `tag` at the adapter, after its own approval.

**Not yet:** the one-time comparison on `test_version`, and a generation-quality evaluation. Both
need the evaluation operators, which are not public yet. Until then the verdict is validation loss
against the base model's, and `test_version` is recorded, never trained on and never evaluated.

## Run it

Two machines: the **control plane** (the engine, the data leg, the gates and every record; no
torch) and a **GPU box** reached over SSH, which runs only the trainer. Every command below is
the `cairn` CLI; `<pl_id>` is a pipeline id (`pl_…`), `<run_id>` one leg's run.

**Which cairndev.** The packs require `cairndev>=0.2.1,<0.3` today; accepting 0.3 is a
separate change. Where cairndev 0.3.0 changes a step, the step says so: `remote_env`, cancel with
the connection's key, `cairn ls finetune` listing the profile, comma-separated `list` inputs, and
the Temporal refusal at launch.

### 1. Install

```bash
pip install 'cairndev[ssh]' cairn-pack-dataset-prep cairn-pack-finetune   # the control plane
pip install 'cairn-operator-training[gpu]'      # the GPU box; add ,track-mlflow for MLflow
```

The packages publish to PyPI soon (cairn-extensions#10). Until then, install from a checkout of
this repository: on the control plane `pip install 'cairndev[ssh]' ./packages/cairn-contracts-*
./packages/cairn-operator-* ./packs/cairn-pack-dataset-prep ./packs/cairn-pack-finetune`, and on
the box `pip install ./packages/cairn-contracts-training ./packages/cairn-contracts-training-stages
'./packages/cairn-operator-training[gpu]'`. The box must run the same `cairn-contracts-training*`
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
Temporal:** `sft` has a `task:` leg (`promote`), which the Temporal pipeline owner does not run.
From cairndev 0.3.0 the launch is refused under `CAIRN_RUNTIME=temporal` (exit 3) before any
record exists; on 0.2.1 the leg fails mid-run. Use `inprocess` (the default) or `dbos`.
`cairn dev` sets its own: SQLite at `--db` (default `~/.cairn/dev.db`) and `inprocess`.

### 3. The profile, the buckets and the box

```bash
cairn ls finetune                                  # cairndev 0.3.0: lists profiles/finetune-ssh.toml with its installed path
export CAIRN_PROFILE=<that path>                   # 0.2.1: <site-packages>/cairn_pack_finetune/profiles/finetune-ssh.toml
export CAIRN_ARTIFACT_S3_BUCKET=<bucket>           # the registered dataset versions (CAIRN_ARTIFACT_S3_PREFIX optional)
export GPU_SSH_KEY="$(cat ~/.ssh/id_ed25519)"      # read by the engine; cairn never writes it to disk
export SSH_PRIVATE_KEY="$GPU_SSH_KEY"              # cairndev 0.2.1 only: cancel and the checkpoint pull-back read this
cairn connect add ssh --name gpu --set host=<box> --set user=<user> \
  --set known_hosts=$HOME/.ssh/known_hosts --set hourly_usd=<rate> --secret PRIVATE_KEY=GPU_SSH_KEY
cairn connect ls && cairn connect test gpu
```

`finetune-ssh` sends the ml legs to the `compute.ssh` connection named `gpu`, in the box's
virtualenv, and selects the s3 artifact store, because two machines need a store both can reach.
Its header lists every setting. From cairndev 0.3.0, cancel and the checkpoint pull-back use the
connection's `PRIVATE_KEY`; `SSH_PRIVATE_KEY` is then needed only for jobs started before the
upgrade. `hourly_usd` (with `instance_id` and `region`) is what `cairn compute status gpu` reads.

**On the box** the trainer reads `CAIRN_CHECKPOINT_BUCKET` (checkpoints, and the adapter with
`artifact_target: s3`), `CAIRN_CHECKPOINT_PREFIX`, `AWS_DEFAULT_REGION` and, for MinIO,
`AWS_ENDPOINT_URL`. Set them in the box's environment, or, from cairndev 0.3.0, set them on the
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
cairn pipeline explain finetune/sft -i train_version=dataset:ifc-bim-alpaca-train@latest
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

### 6. Train, and the three approvals

```bash
cairn remote presets finetune/sft
cairn remote launch finetune/sft --preset ifc-bim-qwen3-4b --watch
cairn remote advance <pl_id> approve-run                    # 1. before any GPU time
cairn remote advance <pl_id> promote-model                  # 2. after `improved`, with the losses on the card
cairn remote ps --status awaiting_approval                  # 3. artifact/promote's own approval
cairn remote approve <run_id> --approver you@example.com    #    or: cairn remote reject <run_id>
```

Override any input with `-i key=value` and keep the result with `--save-preset <name>`. A `list`
input takes JSON (`-i 'report_to=["mlflow"]'`), and from cairndev 0.3.0 a comma string as well
(`-i report_to=mlflow`). `--watch` returns when the pipeline pauses at a gate; `approvers:
ml-owners` names who decides, and with sign-in on the signed-in user is recorded.

The launch door refuses, before any record exists (exit 3): a test version given as
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
  attempt). When an attempt's process is killed (a box reboot, the OOM killer, a dropped SSH
  session: no exit code), the engine reports it retriable and the leg runs once more under the
  same run; the trainer pulls that run's newest checkpoint from `CAIRN_CHECKPOINT_BUCKET` and
  continues from its step. A leg that fails on its own error (a clean non-zero exit) is not
  retried. Without a checkpoint bucket the retry starts again from step 0.
- **Budget:** `governance.budget: {usd: 25}` on the `sft` series. Each attempt reserves its own
  estimate before it is submitted, so a retry is new spend the budget must still cover, and an
  attempt the budget cannot cover fails the leg instead of starting. On cairndev 0.3.0 and
  earlier GPU time is not priced (the estimate is $0) and a failed attempt releases its hold.
  From 0.3.1 a leg on a connection with `hourly_usd` is priced: before any record exists, the
  launch checks `hourly_usd × timeout × 2` for each leg (30 min for `smoke`, 12 h for `train`)
  against what the series has left, about $16.50 at $0.66/h, and every attempt settles its own box
  time, failed or not. A dearer box needs a larger budget, or a copy of the pipeline with a
  shorter `timeout`.
- **Cancel:** `cairn remote cancel-pipeline <pl_id>` (every in-flight leg) or `cairn remote
  cancel <run_id>` (one leg). Over SSH it kills the trainer on the box: cairndev 0.2.1 needs
  `SSH_PRIVATE_KEY` in the control plane's environment; from 0.3.0 it uses the connection's key,
  and a cancel that cannot reach the box is reported as failed.
- **The box:** `cairn compute status gpu` shows whether it is up and what it costs per hour, and
  `cairn compute sweep` lists boxes whose lease expired (`--apply` stops them). You start and stop
  the box yourself (cairn-extensions#26 tracks doing it from the pipeline): stop it after a run.

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

**MLflow (optional):** `MLFLOW_TRACKING_URI` on the control plane with `-i report_to=mlflow`
records the run; the box installs `cairn-operator-training[gpu,track-mlflow]` and must reach the
URI. **On one machine**, use the default profile: none of the SSH settings apply, and
dataset-prep still registers its versions through the s3 store.

## Presets and profiles

| | |
|---|---|
| `presets/ifc-bim-qwen3-4b.yaml` | `Qwen/Qwen3-4B-Base`, QLoRA r 16, alpha 32, lr 2e-4, 2048 tokens, seed 3407, one epoch, batch 2 × 8, on `ifc-bim-alpaca-{train,val,test}@latest` |
| `profiles/finetune-ssh.toml` | ml legs on the `gpu` ssh connection (`ml_engine = "builtin"`, `ml_target = "gpu"`, venv mode), the s3 artifact store |

## Support

| Path | Status | Evidence | Proven on `cairndev` |
|---|---|---|---|
| `sft` flow and gates on the stub trainer | EXPERIMENTAL | `tests/test_sft.py`: the pack's flow on the core runner with resolved, pinned versions; the pipeline's own predicates through the core's evaluator | 0.2.1 (the released wheel: a local run and CI) |
| `smoke` / `train` retry after a kill | EXPERIMENTAL | `tests/test_sft.py`: each leg through the core's leg executor with the stub trainer and a fake S3; an attempt killed after `checkpoint-2` is retried once and the retry writes `checkpoint-4` and `-6` from the pulled checkpoint; a leg's own error is not retried; one reservation per attempt | 0.2.1 (a local run and CI) |
| example admission policy | EXAMPLE | `tests/test_example_policy.py`: it validates, names only known checks, and the core's admission decision admits or refuses on it | 0.2.1 (a local run and CI) |
| `finetune-ssh` profile selection | EXPERIMENTAL | `tests/test_finetune_ssh_profile.py`: the core resolves the ml leg to the SSH engine on `gpu` (venv mode), the data leg to the local engine, the s3 store; no connection is made | 0.2.1 (the released wheel: a local run and CI) |
| `sft` on an ssh GPU box with QLoRA | not yet | the qualification run, cairn-extensions#9 | not yet |

Apache-2.0 — see [`LICENSE`](LICENSE).
