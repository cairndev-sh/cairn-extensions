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

This wheel and the contract wheels it pins are not on PyPI yet. Until they are, install from a
checkout of this repository, into the environment `cairndev` 0.2.x runs in:

```bash
git clone https://github.com/cairndev-sh/cairn-extensions && cd cairn-extensions
# the control plane: no torch, no CUDA
pip install ./packages/cairn-contracts-training ./packages/cairn-contracts-training-stages \
  ./packages/cairn-operator-training
# a CUDA box the ssh engine runs the trainer on
pip install ./packages/cairn-contracts-training ./packages/cairn-contracts-training-stages \
  './packages/cairn-operator-training[gpu]'
```

| Extra | What it is for |
|---|---|
| `gpu` | The Unsloth QLoRA trainer, pinned exactly to `images.py::TRAINER_PINS` (torch, unsloth, trl, peft, transformers, datasets, bitsandbytes, accelerate). Install it on the GPU box. |
| `local` | Plain LoRA on CPU/MPS (no unsloth or bitsandbytes), ranges floored at the same pins. |
| `track-mlflow` | MLflow run, dataset and model logging (`report_to: [mlflow]`). |

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

**With sign-in on (cairn 0.2.0 and later).** A profile-bound `ml_target` names a connection of
the deployment (the `default` account). A team account's leg does not use the deployment's
connections, so it needs a `compute.ssh` connection of that name in its own account, or the leg
fails. Credentials this wheel reads from the environment (`HF_TOKEN`, `WANDB_*`, `AWS_*`) are not
declared as `secrets:`; see [`DESIGN.md`](DESIGN.md) §4.

Apache-2.0 — see [`LICENSE`](LICENSE).
