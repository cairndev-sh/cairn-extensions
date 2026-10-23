# cairn-pack-dataset-prep

Prepare a dataset once for [cairn](https://github.com/cairndev-sh/cairn): pin the source, audit
it, split it by group and register frozen `train`, `val` and `test` versions with their roles. A
fine-tune then reads the same three versions for its baseline, its validation and its one-time
test comparison, and cairn's holdout door refuses a training run that would meet the test set.
The fine-tune pack (`cairn-pack-finetune`) accepts these versions only.

A domain pack: YAML and a sample, no code. Every stage comes from
[`cairn-operator-dataset-checks`](../../packages/cairn-operator-dataset-checks).

## The `prepare` pipeline

```
source-ok ⏸ → prepare (one data leg) → format → data-ok
```

- **`source-ok`:** a person confirms the source, its revision and the name the versions are
  registered under, before anything is written.
- **`prepare`:** the stages in [`flows/prepare.yaml`](flows/prepare.yaml):
  1. `ingest.normalize`: reads the source; a Hub dataset is loaded at `revision`.
  2. `dedup.exact`: drops exact duplicates.
  3. `fields.derive`: labels for a dataset that has none, for example a topic or a template
     family.
  4. `dedup.near`: drops near duplicates, on the same text and 5-word shingles the holdout door
     compares, so no near-duplicate pair can end up split between train and test. Needs the
     `[all]` extra; skipped, with a note, without it.
  5. `profile.tokens`: prompt+response lengths against `max_seq_len`.
  6. `profile.fields`: topic counts, and group sizes.
  7. `export.jsonl`: the cleaned corpus.
  8. `split`: group-aware, and registers `<register>-train`, `-val` and `-test` with roles
     training, validation and test.
  9. `dataset.manifest`: source, revision and commit, hashes, rows, fields, licence and
     citation, and the fetch time.

  Each registered version gets its intake verdict.
- **`format`:** checks every example's structure against TRL and Unsloth.
- **`data-ok`:** stops the pipeline when a version came back blocked or invalid, or an example
  is malformed.

The record carries stats, hashes, counts and version ids, never row content. A template family
is prompt text, so it is reported by group size only.

## Run it

```bash
pip install cairn-pack-dataset-prep            # the pipeline; near dedup and tokenizers: [all]
cairn remote launch dataset-prep/prepare --preset ifc-bim-alpaca --watch
cairn remote advance <pipeline-id> source-ok  # a person confirms the source
```

Any dataset: `-i path=<hub id | file | s3://…> -i register=<name>`, plus `-i revision=<commit>`
for a Hub source. Add `group_by`, `derive`, `topic_fields` and `group_fields` when the rows carry,
or can be given, a group. On a remote engine set `store=s3`, so the control plane can read the
versions it registers.

## Presets

| Preset | What it prepares |
|---|---|
| `ifc-bim-alpaca` | `Dietmar2020/ifc-bim-high-quality-alpaca` at a pinned commit, 90/5/5. `topic` is the first building element the answer defines and `family` is the request with numbers and names replaced; the split groups by `family`. Lengths use the `Qwen/Qwen3-4B-Base` tokenizer against 2048. |

The topic is matched with a regular expression, not parsed. An IFC parse belongs to the IFC
validator (cairn-extensions#12).

## Support

| Path | Status | Evidence | Proven on `cairndev` |
|---|---|---|---|
| `prepare` on the builtin data engine, local store | EXPERIMENTAL | `tests/test_prepare.py`: the pack's flow, the core runner and registration in process, on the 200-row sample | 0.2.1 pre-release (local) |
| `prepare` on a Hub source with `revision`, through `cairn remote launch` | not yet | the qualification run, cairn-extensions#9 | not yet |

Apache-2.0 — see [`LICENSE`](LICENSE).
