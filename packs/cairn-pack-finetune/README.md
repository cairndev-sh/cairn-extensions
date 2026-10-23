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
- **`improved`:** the candidate's validation loss must be below the base model's.
- **`promote-model`:** a person decides, with the losses on the card.
- **`promote`:** `artifact/promote` points `tag` at the adapter, after its own approval.

**Not yet:** the one-time comparison on `test_version`, and a generation-quality evaluation. Both
need the evaluation operators, which are not public yet. Until then the verdict is validation loss
against the base model's, and `test_version` is recorded, never trained on and never evaluated.

The stub trainer (`stub: true`, no GPU) runs the whole path but trains nothing. Its validation
loss is not below its baseline, so `improved` stops it.

## Run it

```bash
pip install cairn-pack-finetune                       # the control plane: no torch
pip install 'cairn-operator-training[gpu]'            # on the GPU box
export GPU_SSH_KEY="$(cat ~/.ssh/id_ed25519)"        # in the engine's environment
cairn connect add ssh --name gpu --set host=<box> --set user=<user> --set known_hosts=$HOME/.ssh/known_hosts --secret PRIVATE_KEY=GPU_SSH_KEY
export CAIRN_PROFILE=<site-packages>/cairn_pack_finetune/profiles/finetune-ssh.toml
cairn remote launch dataset-prep/prepare --preset ifc-bim-alpaca --watch   # once
cairn remote launch finetune/sft --preset ifc-bim-qwen3-4b --watch
cairn remote advance <pipeline-id> approve-run        # then, after the run: promote-model
```

Buckets and trackers are the environment's, never the pack's:
- **Over SSH, two machines need a store both can reach.** `finetune-ssh` selects the s3 artifact
  store (`CAIRN_ARTIFACT_S3_BUCKET`), and the IFC preset publishes the adapter with
  `artifact_target: s3` (`CAIRN_CHECKPOINT_BUCKET`, which also mirrors checkpoints).
- **MLflow:** `MLFLOW_TRACKING_URI` with `-i report_to='[mlflow]'` records the run.
- **On one machine** (the default profile, the local store), none of these is needed.

**Give each new run its own `output_name`.** It is the adapter's directory on the box, and a run
that finds checkpoints there resumes from them. The smoke run uses `smoke_output_name`.

## Presets and profiles

| | |
|---|---|
| `presets/ifc-bim-qwen3-4b.yaml` | `Qwen/Qwen3-4B-Base`, QLoRA r 16, alpha 32, lr 2e-4, 2048 tokens, seed 3407, one epoch, batch 2 × 8, on `ifc-bim-alpaca-{train,val,test}@latest` |
| `profiles/finetune-ssh.toml` | ml legs on the `gpu` ssh connection (`ml_engine = "builtin"`, `ml_target = "gpu"`), the s3 artifact store |

## Support

| Path | Status | Evidence | Proven on `cairndev` |
|---|---|---|---|
| `sft` flow and gates on the stub trainer | EXPERIMENTAL | `tests/test_sft.py`: the pack's flow on the core runner with resolved, pinned versions; the pipeline's own predicates through the core's evaluator | 0.2.1 pre-release (local) |
| `sft` on an ssh GPU box with QLoRA | not yet | the qualification run, cairn-extensions#9 | not yet |

Apache-2.0 — see [`LICENSE`](LICENSE).
