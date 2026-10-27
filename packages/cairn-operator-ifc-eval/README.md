# cairn-operator-ifc-eval

IFC snippet validation for [cairn](https://github.com/cairndev-sh/cairn): when a model is asked
for IFC, do its answers parse, and do they hold to the schema they target? Two shapes, one check:

- `eval/ifc-validate` (an operator) — provides capability **`eval.ifc_validate`**. Takes a list of
  answers and a schema version, and returns `parse_valid_rate`, `schema_error_rate`, `valid_rate`
  and the per-row failures.
- `eval.ifc_validate` (a flow-stage kernel, `cairn.flow_stages` entry point) — runs **on the GPU
  box** as an ml leg. It reads a generation eval's samples file (JSON lines with `reference`,
  `base` and `candidate` answers), scores each field on its own, and reports the candidate's rates
  against the base model's (`delta_*`) and a results file with every row's failures.

Neither gates on its own: a pipeline predicate or a person reads the rates.

## What is checked

Each answer goes through three steps, the same for every row and every model:

1. **Extract.** The fenced code blocks when there are any; else the `DATA;` section of a whole
   file; else the text from the first `#<n>=` to the last `;`. Nothing found is a parse failure.
2. **Parse.** Every statement must be `#<id>=<ENTITY>(<attributes>);`, with balanced parentheses,
   closed strings and no id used twice; every entity must be one the schema declares; and
   IfcOpenShell must read every instance back. IfcOpenShell's reader on its own is lenient (it
   skips a statement it cannot read), so the strict reading is what makes an answer "parse".
3. **Schema**, for an answer that parses: each instance has the number of attributes its entity
   declares, every `#<id>` it references is defined in the answer, and `ifcopenshell.validate`
   finds no wrong attribute type, missing required attribute or bad aggregate. WHERE rules and
   global rules are not checked.

| Rate | Is |
|---|---|
| `parse_valid_rate` | answers that parse / answers |
| `schema_error_rate` | parsed answers with at least one schema error / parsed answers (none when nothing parsed) |
| `valid_rate` | answers that parse with no schema error / answers |

The schema is `schema_version`: `IFC2X3`, `IFC4` (the default) or `IFC4X3_ADD2`, as IfcOpenShell
names them. The stage scores the reference answers too: references that do not parse under the
schema mean the schema is the wrong one, not that the models are.

Without IfcOpenShell, the operator and the stage report `checked: false` with the reason and no
rate.

## Install

This wheel is not on PyPI yet. Until it is, install it from a checkout of this repository, into the
environment `cairndev` 0.3.x runs in:

```bash
git clone https://github.com/cairndev-sh/cairn-extensions && cd cairn-extensions
pip install './packages/cairn-operator-ifc-eval[ifc]'
cairn doctor && cairn pack check
```

On a GPU box that runs the `eval.ifc_validate` stage, install the `[box]` extra.

| Extra | What it is for |
|---|---|
| `ifc` | `ifcopenshell>=0.9,<0.10` (LGPL-3.0): the IFC reader and the schema validator. |
| `box` | IfcOpenShell and boto3: the stage on the GPU box (an `s3://` samples file, and the results file uploaded beside the run's checkpoints when `CAIRN_CHECKPOINT_BUCKET` is set). |
| `test` | IfcOpenShell and `moto[s3]`, for this package's tests: the fixture runs against the real IfcOpenShell in CI. |

The base install pins `cairndev>=0.3.1,<0.4`. No contract wheel: the operator and the stage
implement no shared contract yet (as `cairn-operator-eval-lm`).

## Support

| Path | Status |
|---|---|
| The checks on the ten-snippet fixture (five valid, five broken: 0.5 parse validity), the schema errors, the operator, the stage with its samples and results files, the no-extra path | tested in CI on the released `cairndev`, with IfcOpenShell 0.9 |
| The stage on a GPU box, on a real model's answers | not yet proven |

Not yet: a `challenge` dataset role (expert-written cases unlike the training templates) and a stage
that scores on it (cairn-extensions#12).
