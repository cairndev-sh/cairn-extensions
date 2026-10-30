# The laptop demo: a governed fine-tune on one machine

A fine-tune end to end under cairn's governance, on a Mac (Apple silicon) or a Linux box with
**no GPU, no cloud account and no API key**, built only from what cairn already provides: two
declared pipelines in their own packs, joined by an event.

```
you launch                                   started by the event, not by you
dataset-prep/prepare ── dataset.prepared ──▶ finetune/sft-local
source-ok ⏸ → prepare → data-review ⏸         approve-run ⏸ → smoke → smoke-ok → train → eval
  → format → data-ok                            → improved → promote-model ⏸ → promote
```

- `dataset-prep/prepare` (pack `cairn-pack-dataset-prep`) emits `dataset.prepared` when it ends
  `done`, after its data gate, with its `outputs:`: the train, val and test versions by role and the
  base model (ADR-0161).
- `finetune/sft-local` (pack `cairn-pack-finetune`) declares a `kind: event` trigger on that topic
  (ADR-0131 D4). It is paused until a person turns it on in the workspace (ADR-0160); `run.sh` does
  that once with `cairn remote enable-trigger`. The fine-tune's record names the data prep that
  started it (its `trigger.publisher`).
- Neither pack references the other (ADR-0084 D5): the trigger names a topic, not a flow.

| Stage | What runs | Operators (wheel) | Who decides |
|---|---|---|---|
| `prepare` | pins the 200-row sample, drops exact and near duplicates, profiles, splits by group, registers `laptop-demo-train`, `-val`, `-test` with roles and intake verdicts | 9 stages of `cairn-operator-dataset-checks` | a person confirms the source (`source-ok`); `data-ok` is answered by the data |
| `format` | checks the example structure against the trainers | `dataset/format` (`cairn-operator-dataset-checks`) | |
| `approve-run` | — | | a person |
| `smoke`, `train` | LoRA on 16 rows for 2 steps, then the full run (30 steps) with the base model's validation loss as baseline | `sample.fidelity`, `train.qlora`, `eval.checkpoints`, `package.adapter` (`cairn-operator-training[local]`) | `smoke-ok`: the data |
| `eval` | base and candidate on the held-out val and test versions (`heldout-loss`) | `eval.lm_eval` (`cairn-operator-eval-lm`) | |
| `improved` | | | the data: candidate val loss below the base model's |
| `promote-model`, `promote` | tags the adapter `sft-adapter@candidate` | `artifact/promote` (cairndev) | a person |

These are the operators `finetune/sft` runs for real training on a GPU box; the demo adds no
operator of its own. Two policies a person owns, outside the pipelines, are loaded from the
finetune pack's [`docs/policies/laptop/`](../../packs/cairn-pack-finetune/docs/policies/laptop):
training needs approval, and so does moving a promote tag.

## Prerequisites

- macOS on Apple silicon, or Linux (CPU, or an NVIDIA GPU). Python 3.12 and
  [uv](https://docs.astral.sh/uv/). Not an Intel Mac: PyTorch publishes no macOS x86_64 wheels for
  the torch this demo pins.
- About 5 GB of disk for the virtualenv (`TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`
  selects the small CPU build on Linux).
- Network to PyPI, and to the Hugging Face Hub for the default model (about 270 MB, once). With
  `--tiny` or a local `--base-model` directory, the run needs no Hub access.
- **cairndev 0.3.3** (pipeline `emits:`, the trigger switch). Until it is released, install the core
  from a checkout: `CAIRNDEV_SOURCE=/path/to/cairn demos/laptop/install.sh`.

## Steps

```bash
git clone https://github.com/cairndev-sh/cairn-extensions && cd cairn-extensions
demos/laptop/install.sh          # a virtualenv in demos/laptop/.venv: cairndev + this checkout's packs
demos/laptop/run.sh              # interactive: you approve each human gate
```

`run.sh` starts a local S3 (`moto_server`) and the control plane (`cairn dev --engine-only`) on
`127.0.0.1`, logs in as `dev`, turns the chain on, launches `dataset-prep/prepare`, and follows both
pipelines. Every step is a `cairn` command you can type yourself:

```bash
. demos/laptop/env.sh
cairn remote triggers                               # sft-local's event trigger, paused
cairn remote enable-trigger <its id>
cairn remote launch dataset-prep/prepare -i path=packs/cairn-pack-dataset-prep/data/sample-ifc.jsonl \
    -i register=laptop-demo -i val_ratio=0.1 -i test_ratio=0.1 --watch
cairn remote advance <pipeline> source-ok --approver dev
cairn remote pipelines                              # the sft-local pipeline the event started
cairn remote findings <pipeline>                    # what each leg measured
cairn audit verify                                  # the hash-chained audit log
```

| Option | |
|---|---|
| `--auto-approve` | answers every human gate as `dev`, for CI or to watch the path once; the gates still run and the approver is recorded |
| `--tiny` | builds a 0.1M-parameter random model offline (`tests/fixtures/tiny_model.py`) and trains that; no Hub access. The numbers mean nothing; the path is real |
| `--base-model X` | a Hub id or a local model directory; it is `prepare`'s `base_model` (the tokenizer the lengths are counted with), and the event hands it to `sft-local` |

`LAPTOP_DEMO_PORT`, `LAPTOP_DEMO_S3_PORT`, `LAPTOP_DEMO_WORK` and `LAPTOP_DEMO_TIMEOUT` change the
ports, the work directory (default: a new temp directory, kept) and the give-up time.
[`env.sh`](env.sh) holds every variable the run sets.

On a macOS virtual machine (a hosted CI runner), torch reports an MPS device that cannot allocate
memory; run the demo on a physical Mac, or on Linux.

## Known limits

- **Training knobs cannot ride the event.** `sft-local` receives only what `prepare` emits (the
  versions and the base model); every other input takes its default. The old `-i key=value`
  pass-through is gone. An event trigger with a preset would bring it back (a core change).
- **`--tiny` trains at the default learning rate.** It used `learning_rate=0.002` so random weights
  showed a gain in 30 steps; the event cannot carry that. If `improved` fails on the tiny model, that
  is the gate working, and the tiny model or `max_steps` needs adjusting.
- **Two reads still parse JSON** in `run.sh` (a running pipeline's status, and its `trigger`):
  `cairn remote pipeline <id>` prints the full record and has no `--watch` for a pipeline you did not
  launch in the same command.
- **A gate's evidence is not printed** by the CLI yet (cairndev-sh/cairn#258); `run.sh` points at
  `cairn remote findings`.
- **Nothing is priced:** a laptop leg has no `hourly_usd`, so the launch door estimates $0.
- **The store is a local S3 (moto).** The profile's `artifact_store = "local"` is untested here.
- **Nothing here is a production setup:** one process on one machine with a throwaway store.

## What is in this folder

| | |
|---|---|
| `install.sh` | the virtualenv and the installs |
| `run.sh` | the end-to-end run described above |
| `env.sh` | the environment the run sets |

The pipelines are [`dataset-prep/prepare`](../../packs/cairn-pack-dataset-prep/pipelines/prepare.yaml)
and [`finetune/sft-local`](../../packs/cairn-pack-finetune/pipelines/sft-local.yaml), the profile
[`finetune-local`](../../packs/cairn-pack-finetune/profiles/finetune-local.toml).
