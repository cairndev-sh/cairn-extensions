# cairn-operator-eval-lm

Classic public benchmarks (EleutherAI **lm-evaluation-harness**) for
[cairn](https://github.com/cairndev-sh/cairn), run against a candidate that is **already served**
behind an OpenAI-compatible endpoint (a vLLM server, for example), with the task lists shipped as
**pack data**.

- `eval/lm-eval` — provides capability **`eval.lm_eval`**. Runs a task preset against the served
  candidate, decontamination on, and returns a scorecard (metric · slice · case · grader) plus the
  flat `{task/metric: value}` map and the metric directions a gate reads.

Task presets live in `src/operator_eval_lm/presets/*.toml` — `smoke` (the default; minutes),
`reasoning`, `knowledge`. Adding one is a data edit.

`lm-eval` is the optional `[harness]` extra, imported lazily. Without it the operator no-ops with
`checked=false` and a note; it never emits a score a gate could pass on.

## Install

This wheel is not on PyPI yet. Until it is, install it from a checkout of this repository, into the
environment `cairndev` 0.3.x runs in:

```bash
git clone https://github.com/cairndev-sh/cairn-extensions && cd cairn-extensions
pip install ./packages/cairn-operator-eval-lm             # registers eval/lm-eval; no-ops without the harness
pip install './packages/cairn-operator-eval-lm[harness]'  # where benchmarks actually run
cairn doctor && cairn pack check
```

| Extra | What it is for |
|---|---|
| `harness` | `lm-eval>=0.4.4`, the harness itself. Heavy (torch, datasets): install it only where benchmarks run. |

The base install pins `cairndev>=0.3.1,<0.4`.

## Support

Carried over from cairn's private tree, where this pack was **experimental**: it is proven against
a faked harness only and has **never run against a real served model**. `DESIGN.md` §6 lists what
the first live run must check.

| Path | Status |
|---|---|
| Registration, presets, scorecard mapping, budget, no-op paths (no extra) | tested in CI on the released `cairndev` |
| A real harness call against a served endpoint | not yet proven |

[`docs/sft-integration.md`](docs/sft-integration.md) proposes how `finetune/sft` could run these
benchmarks on its GPU box, for the base model and the candidate adapter. It is a proposal, not
implemented.

See `DESIGN.md` for the architecture, the configuration scopes and the failure modes.
