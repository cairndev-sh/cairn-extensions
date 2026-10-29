# The laptop demo: a governed fine-tune on one machine

One script runs a fine-tune end to end under cairn's governance, on a Mac (Apple silicon) or a
Linux box with **no GPU, no cloud account and no API key**:

```
data prep -> LoRA train -> held-out-loss eval -> automatic gate -> human approval -> promote
```

| Stage | What runs | Who decides |
|---|---|---|
| `dataset-prep/prepare` | pins the pack's 200-row sample, drops duplicates, splits it by group, registers `laptop-demo-train`, `-val` and `-test` with their roles and intake verdicts | a person confirms the source (`source-ok`); `data-ok` is answered by the data |
| `finetune/sft-local` `approve-run` | nothing yet: the person is shown the three versions, the base model and the method | a person |
| `smoke`, `train` | a LoRA run on a 16-row subsample for 2 steps, then the full run (30 steps) with the base model's validation loss as the baseline | the `smoke-ok` gate checks that the smoke run ended in a packaged adapter |
| `eval` | the base model and the candidate scored on the held-out `val` and `test` versions with the same method (`heldout-loss` preset: no benchmark) | |
| `improved` | | the data: candidate validation loss below the base model's, or the pipeline stops |
| `promote-model`, `promote` | the person is shown the losses; then tags the adapter `sft-adapter@candidate` | a person, twice |

Two policies a person owns, outside the pipelines, are loaded from [`policies/`](policies):
training needs approval, and so does moving a promote tag.

## Prerequisites

- macOS on Apple silicon, or Linux (CPU, or an NVIDIA GPU, which the LoRA run then uses). Python
  3.12 (uv fetches it if you have another) and [uv](https://docs.astral.sh/uv/). Not an Intel Mac:
  PyTorch publishes no macOS x86_64 wheels for the torch this demo pins, so `install.sh` stops there.
- About 5 GB of disk for the virtualenv (torch is most of it; on Linux the PyPI torch wheel is the
  CUDA build and runs on the CPU too, `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`
  selects the small one).
- Network to PyPI, and to the Hugging Face Hub for the default model (about 270 MB, once). With
  `--tiny` or a local `--base-model` directory, the run needs no Hub access.
- The packs are installed from this checkout: the fixes the demo needs (the CPU/MPS eval,
  `heldout-loss`, the `file://` adapter hash) are on `main` and not on PyPI yet.

## Steps

```bash
git clone https://github.com/cairndev-sh/cairn-extensions && cd cairn-extensions
demos/laptop/install.sh          # a virtualenv in demos/laptop/.venv: cairndev 0.3.1 + this checkout's packs
demos/laptop/run.sh              # interactive: you approve each human gate
```

`run.sh` starts a local S3 (`moto_server`) and the control plane (`cairn dev --engine-only`) on
`127.0.0.1`, logs in as `dev`, and runs the two pipelines. At each human gate it prints what the
person is shown and how to decide:

```
  A person is asked:
  gate approve-run (approvers: ml-owners)
    input.train_version  dataset:laptop-demo-train@1
    ...
  approve with:  . demos/laptop/env.sh && cairn remote advance pl_32ca75e20eea42e4 approve-run --approver dev
  Press Enter to approve here as dev, type n to cancel the pipeline, or run the command above in another terminal.
```

Press Enter, or run the printed command from another shell. Options:

| | |
|---|---|
| `--auto-approve` | answers every human gate as `dev`, for CI or to watch the path once; the gates still run and the approver is recorded |
| `--tiny` | builds a 0.1M-parameter random model offline (`tiny_model.py`) and trains that: about 2.5 minutes on 4 CPU cores, no Hub access. The numbers mean nothing; the path is real |
| `--base-model X` | a Hub id or a local model directory instead of the default `HuggingFaceTB/SmolLM2-135M` |
| `-i k=v` | any `finetune/sft-local` input, e.g. `-i max_steps=100 -i learning_rate=0.0001` |

`LAPTOP_DEMO_PORT`, `LAPTOP_DEMO_S3_PORT`, `LAPTOP_DEMO_WORK` and `LAPTOP_DEMO_TIMEOUT` change the
ports, the work directory (default: a new temp directory, kept so you can look) and the give-up time.
When a pipeline fails, the run prints each failed leg's error and the end of its run's event log,
then the last `LAPTOP_DEMO_LOG_LINES` (default 60) lines of the control plane's `server.log`.

[`env.sh`](env.sh) holds every variable the run sets; `source` it to drive the same server from
another shell.

On a macOS virtual machine (a hosted CI runner, for example), torch reports an MPS device that
cannot allocate memory, and the first ml leg fails with `MPS backend out of memory`. Run the demo
on a physical Mac, or on Linux, where it trains on the CPU.

## Expected output

From `run.sh --tiny --auto-approve` on a 4-core Linux CPU (a Mac prints the same with
`eval device mps` or `cpu`); trimmed:

```
==> 1/2  data prep: dataset-prep/prepare on the pack's 200-row sample
    source-ok:awaiting_approval prepare:pending data-review:pending format:pending data-ok:pending
  --auto-approve: approved source-ok as dev
    source-ok:done prepare:done data-review:skipped format:done data-ok:done
    registered: dataset:laptop-demo-train@1  dataset:laptop-demo-val@1  dataset:laptop-demo-test@1

==> 2/2  fine-tune: finetune/sft-local (LoRA, .../tiny-model, held-out loss)
    approve-run:done smoke:done smoke-ok:done train:done eval:done improved:done promote-model:awaiting_approval
    ...
    approve-run:done smoke:done smoke-ok:done train:done eval:done improved:done promote-model:done promote:done

==> result
  pipeline                              pl_078cff8ace0644c2  done
  optimizer steps                       30
  validation loss, base model           6.1178
  validation loss, candidate            5.9866
  held-out val loss  base -> candidate  6.2229 -> 6.1064
  held-out test loss base -> candidate  6.2217 -> 6.1027
  eval device                           cpu
  promoted                              sft-adapter@candidate -> 4672ac085c184482…

==> who decided what (the pipeline records, the durable trail of the gates)
  pl_949f4640523d4eca  source-ok      approved by dev    asked ...  decided ...
  pl_078cff8ace0644c2  approve-run    approved by dev    asked ...  decided ...
  pl_078cff8ace0644c2  promote-model  approved by dev    asked ...  decided ...
  pl_078cff8ace0644c2  promote        approved by dev    asked ...  decided ...
```

Exit status 0 means both pipelines reached `done`. If the `improved` gate fails (the candidate did
not beat the base model on validation loss) the run stops there, which is the gate working; try
more steps (`-i max_steps=100`).

## Known limits

- **The audit log is nearly empty today.** The profile binds the hash-chained audit sink, and
  `run.sh` ends by running `cairn lineage --json` and checking `chain_ok` when the log has rows. In
  cairndev 0.3.1 the sink receives only a refused gate decision, template-path approvals, licence
  acceptance and LLM/MCP calls, not a pipeline's gate approvals or promotion
  ([cairndev-sh/cairn#233](https://github.com/cairndev-sh/cairn/issues/233)). On this run the log
  stays empty and `run.sh` says so; the approvals are on the pipelines' records (the "who decided
  what" table).
- **No trace of the run.** A pipeline run is not one trace: flow legs (the training and eval) and
  gate decisions emit no span ([cairndev-sh/cairn#222](https://github.com/cairndev-sh/cairn/issues/222)).
- **Nothing is priced, and the budget is declared, not measured.** A laptop leg has no
  `hourly_usd`, so the launch door estimates $0. `governance.budget` in `sft-local` is what its
  author wrote ([cairndev-sh/cairn#250](https://github.com/cairndev-sh/cairn/issues/250),
  [#251](https://github.com/cairndev-sh/cairn/issues/251)).
- **Cancel does not stop a running trainer.** Stop the process yourself.
- **The training policy is redundant here.** `sft-local` already asks before it trains, so
  [`training-needs-approval.yaml`](policies/training-needs-approval.yaml) injects nothing; it is
  there to show the environment's rule, and a test shows it inserts the gate into a pipeline that
  dropped its own.
- **Only the tiny model is run in CI, on a Linux runner's CPU.** The default SmolLM2 model and
  Apple-silicon MPS are not exercised by CI: the hosted macOS runners cannot allocate on MPS.
- **Nothing here is a production setup.** It is a single process on one machine with a throwaway
  store; the adapter stays in the leg's temporary directory.

## What is in this folder

| | |
|---|---|
| `install.sh` | the virtualenv and the installs |
| `run.sh` | the end-to-end run described above |
| `env.sh` | the environment the run sets |
| `policies/` | the two policies |
| `pipeline_view.py` | what `run.sh` prints from the control plane (gate cards, the result) |
| `tiny_model.py` | the offline tiny model for `--tiny` |

The pipeline is [`finetune/sft-local`](../../packs/cairn-pack-finetune/pipelines/sft-local.yaml)
and the profile [`finetune-local`](../../packs/cairn-pack-finetune/profiles/finetune-local.toml);
the [finetune pack README](../../packs/cairn-pack-finetune/README.md#on-a-laptop-sft-local) lists
what differs from `sft-own-box`.
