# cairn-operator-eval-lm

Classic public benchmarks (EleutherAI **lm-evaluation-harness**) for
[cairn](https://github.com/cairndev-sh/cairn), with the task lists shipped as **pack data**, and the
models' own answers to held-out prompts, in three shapes:

- `eval/lm-eval` (an operator) — provides capability **`eval.lm_eval`**. Runs a task preset against
  a candidate that is **already served** behind an OpenAI-compatible endpoint (a vLLM server, for
  example), decontamination on, and returns a scorecard (metric · slice · case · grader) plus the
  flat `{task/metric: value}` map and the metric directions a gate reads.
- `eval.lm_eval` (a flow-stage kernel, `cairn.flow_stages` entry point) — runs **on the GPU box**
  as an ml leg: loads a base model once (4-bit for QLoRA, as it was trained), scores it with
  lm-eval's `hf` backend, attaches an adapter to the same weights and scores it again with the same
  tasks, documents, few-shot count, batch size and seeds; measures both models' loss on held-out
  validation and test splits by one named method; and reports per-task scores ± standard error,
  deltas, sample sizes and a results file. A time probe scores about 50 documents first and refuses
  the leg when the projected run does not fit the leg's timeout. `cairn-pack-finetune` runs it
  after training (`finetune/sft`'s `eval` leg) and on an adapter you already have
  (`finetune/eval-adapter`).
- `eval.generate` (a flow-stage kernel, same group) — runs **on the GPU box** on the same loading
  code: draws `sample_n` prompts from a held-out test split with a seed, has the base model and then
  base + adapter answer them (the chat template the adapter was trained with, greedy decoding),
  scores both against the references with the baseline scorers (`exact_match`, `contains`), and
  writes every prompt, reference and answer to a samples file (the stage's report artifact) that a
  validator stage can read. The first batch projects the run and refuses it past the leg's timeout.
  `cairn-pack-finetune` runs it as `finetune/sft`'s `generate` leg.

Task presets live in `src/operator_eval_lm/presets/*.toml` — `smoke` (the operator's default;
minutes), `reasoning`, `knowledge`, and for the stage `forgetting` (ARC-Easy, ARC-Challenge,
HellaSwag, WinoGrande and PIQA at 500 documents each, MMLU at 10 per subject; the stage's default)
and `forgetting-ifeval` (plus IFEval, all 541 prompts). A preset can size a task on its own
(`[limits]`): lm-eval applies a limit per subtask, so `mmlu = 10` is 570 questions. Adding a preset
is a data edit.

`lm-eval` is optional and imported lazily: the `[harness]` extra for the operator, the `[box]`
extra for the stage. Without it the operator no-ops with `checked=false` and a note, and the stage
reports `checked=false` with the reason; neither ever emits a score a gate could pass on.

## Install

This wheel is not on PyPI yet. Until it is, install it from a checkout of this repository, into the
environment `cairndev` 0.3.x runs in:

```bash
git clone https://github.com/cairndev-sh/cairn-extensions && cd cairn-extensions
pip install ./packages/cairn-operator-eval-lm             # registers eval/lm-eval and eval.lm_eval; no-ops without the harness
pip install './packages/cairn-operator-eval-lm[harness]'  # where the operator's benchmarks actually run
cairn doctor && cairn pack check
```

On a GPU box that runs the `eval.lm_eval` stage, install it next to the trainer (the trainer's
extra pins torch, transformers, peft, accelerate and bitsandbytes exactly; lm-eval's own
requirements are floors, so one environment serves both):

```bash
pip install 'cairn-operator-training[gpu]' 'cairn-operator-eval-lm[box]'
```

The two resolve together (checked by hand with `uv pip compile` for Linux x86-64 and Python 3.12:
lm-eval 0.4.13 with the trainer's torch 2.11.0, transformers 5.5.0, peft 0.20.0, datasets 4.3.0);
CI does not check that resolution yet.

| Extra | What it is for |
|---|---|
| `harness` | `lm-eval>=0.4.10,<0.4.14`, the harness itself, for the `eval/lm-eval` operator: the releases the pack's harness seam was read against (they take an explicit request header, so the endpoint key never comes from the process environment). Heavy (torch, datasets): install it only where benchmarks run. It downloads task datasets into the Hugging Face cache. |
| `box` | `lm-eval[hf,ifeval]` in the same range, and boto3: the `eval.lm_eval` and `eval.generate` stages on the GPU box (lm-eval's `hf` backend, IFEval's checkers, S3 for the splits, an `s3://` adapter and the results file). Install it beside `cairn-operator-training[gpu]`. The box downloads the base model and the task datasets from the Hugging Face Hub (or reads its cache). No endpoint and no key: the stage reads no secret and writes nothing to the process environment. |
| `test` | `moto[s3]`, for this package's tests. |

The base install pins `cairndev>=0.3.1,<0.4`.

## Support

Carried over from cairn's private tree, where this pack was **experimental**: it is proven against
a faked harness only and has **never run against a real served model**. `DESIGN.md` §6 lists what
the first live run must check.

| Path | Status |
|---|---|
| Registration, presets, scorecard mapping, budget, no-op paths (no extra) | tested in CI on the released `cairndev` |
| A real harness call against a served endpoint | not yet proven |
| `eval.lm_eval` stage logic: identical calls for both models, deltas and standard errors, the split loss, the time probe and its refusal, the results file and its upload, the failure paths | tested in CI on the released `cairndev`, against a faked backend (`tests/test_eval_stage.py`) and through `cairn-pack-finetune`'s pipelines on the core's launch door and bridge |
| The stage's real backend (torch, transformers 5.5, peft, lm-eval 0.4.13 `hf`) on a GPU | not yet proven: read against the sources, never run (`DESIGN.md` §7) |
| `eval.generate` stage logic: the seeded draw, the same prompts and decoding for both models on one loaded base, the scorers and deltas, the first-batch projection and its refusal, the samples file and its upload, the failure paths, the stub tier | tested in CI on the released `cairndev`, against a faked backend (`tests/test_generate_stage.py`) and through `cairn-pack-finetune`'s pipelines on the core's launch door and bridge |
| `eval.generate`'s real generation (`HfBackend.generate`: greedy `model.generate`, left padding, end-of-turn stops) on a GPU | not yet proven: never run |

[`docs/sft-integration.md`](docs/sft-integration.md) is the design the stage implements (option b).

See `DESIGN.md` for the architecture, the configuration scopes and the failure modes.
