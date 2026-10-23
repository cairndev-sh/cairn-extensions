# cairn-operator-training

One SFT trainer for [cairn](https://github.com/cairndev-sh/cairn): supervised fine-tuning with
LoRA or 4-bit QLoRA (Unsloth + TRL + PEFT + Transformers), as governed pipeline legs. The
trainer does not pick the GPU: the profile's ml engine does (`ml_engine` / `ml_target`), so the
same stage runs on an ssh GPU box today and on whatever the SkyPilot engine reaches as it is
proven (clouds, Kubernetes, RunPod, Vast).

- **`train.run` operations:** `train/stub` (deterministic, no GPU) and `train/local`
  (CPU/MPS LoRA, for smoke runs).
- **ml-flow stages** (`cairn.flow_stages`): `sample.fidelity` → `train.qlora` →
  `eval.checkpoints` → `package.adapter` (the `sft-train` flow), and `train.dpo`,
  `dataset.dpo_pairs`, `train.distill`.
- **`compress/distill`:** knowledge distillation (TRL's GKD trainer).
- **Contracts:** implements `train.run` (`cairn-contracts-training`) and the four
  `training.stage.*` stage contracts (`cairn-contracts-training-stages`); `train.qlora` implements
  `training.stage.train_qlora/1.1.0`, which takes governed `train_uri` / `eval_uri` / `test_uri`
  splits and the batch, accumulation, checkpointing and dropout knobs.

## Install

```bash
pip install cairn-operator-training          # the control plane: no torch, no CUDA
pip install 'cairn-operator-training[gpu]'   # a CUDA box the ssh engine runs the trainer on
pip install 'cairn-operator-training[gpu,track-mlflow]'   # the same box, reporting to MLflow
```

| Extra | What it is for |
|---|---|
| `gpu` | The Unsloth QLoRA trainer, pinned exactly to `images.py::TRAINER_PINS` (torch, unsloth, trl, peft, transformers, datasets, bitsandbytes, accelerate), plus `boto3` for `s3://` data, checkpoint mirroring and `artifact_target: s3`. Install it on the GPU box. |
| `local` | Plain LoRA on CPU/MPS (no unsloth or bitsandbytes), ranges floored at the same pins. |
| `track-mlflow` | MLflow run, dataset and model logging (`report_to: [mlflow]`). Not part of `gpu`: without it, a run asked to report to MLflow trains untracked and records why in `tracking_error`. |

boto3 is imported only when a run reads or writes S3, so a run without S3 needs none. In a flow
stage the trainer writes its checkpoints and adapter under the job's workdir
(`<workdir>/train-output/<output_name>`); the published adapter (`artifact_target: s3` or `hf`)
is the adapter files only, never the `checkpoint-*` directories.

The base install pins `cairndev>=0.2.0,<0.3` and the two contract wheels exactly. Check an
install with `cairn doctor` and `cairn pack check`.

## Support

Carried from cairn's support matrix as it stood at `cairndev` 0.1.3, when this pack was
private. **No row has been re-proven on this public wheel yet**: the first proof is the
qualification run in cairndev-sh/cairn-extensions#9 (an ssh GPU box, `cairndev` 0.2.0), and
each row will name the `cairndev` release it was proven on as it is re-run.

| Path | Status | Evidence so far | Proven on `cairndev` |
|---|---|---|---|
| SFT with LoRA / QLoRA on an ssh GPU box (`ml_target` = a `compute.ssh` connection) | SUPPORTED | 2026-08-28 GPU campaign, before the public 0.1 releases | not yet (#9) |
| `stub: true` trainer tier | SUPPORTED | pack tests | 0.2.0 pre-release (local run; CI lands with #1) |
| DPO (`train.dpo`) | EXPERIMENTAL | pack tests; no GPU run recorded | not yet |
| Distillation (`compress/distill`, `train.distill`) | EXPERIMENTAL | pack tests; no GPU run recorded | not yet |

Where a run goes is the engine's: an ssh GPU box (`cairn connect add ssh`, profile
`ml_engine = "builtin"`, `ml_target = "<connection>"`) is the proven path; the box installs
`cairn-operator-training[gpu]`. Design notes: [`DESIGN.md`](DESIGN.md).

Apache-2.0 — see [`LICENSE`](LICENSE).
