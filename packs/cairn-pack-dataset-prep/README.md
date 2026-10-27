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
source-ok ⏸ → prepare (one data leg) → data-review ⏸ (only when cleaning dropped a lot) → format → data-ok
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
- **`data-review`:** a person looks at the numbers when cleaning removed a large share of the
  source; see [The data review](#the-data-review). Otherwise it is skipped and nothing waits.
- **`format`:** checks every example's structure against TRL and Unsloth.
- **`data-ok`:** stops the pipeline when a version came back blocked or invalid, or an example
  is malformed.

The record carries stats, hashes, counts and version ids, never row content. A template family
is prompt text, so it is reported by group size only.

## Run it

```bash
pip install cairn-pack-dataset-prep            # the pipeline; near dedup and tokenizers: [all]
cairn remote presets dataset-prep/prepare      # the shipped presets
cairn remote launch dataset-prep/prepare --preset ifc-bim-alpaca --watch
cairn remote advance <pipeline-id> source-ok  # a person confirms the source
cairn remote advance <pipeline-id> data-review   # only if the run paused there: a person read the numbers
cairn remote findings <pipeline-id>           # rows in, duplicates dropped, lengths, the three versions
```

The package publishes to PyPI soon (cairn-extensions#10). Until then, install it from a checkout
of this repository, with the operator pack and contract wheels it needs:
`pip install ./packages/cairn-contracts-dataset ./packages/cairn-contracts-dataset-stages
'./packages/cairn-operator-dataset-checks[minhash,tokenizers]' ./packs/cairn-pack-dataset-prep`.
Runs need a durable store (`CAIRN_STORAGE`) to survive a restart, as every pipeline does; the
`cairn-pack-finetune` README's "Run it" covers the control plane end to end.

Any dataset: `-i path=<hub id | file | s3://…> -i register=<name>`, plus `-i revision=<commit>`
for a Hub source. Add `group_by`, `derive`, `topic_fields` and `group_fields` when the rows carry,
or can be given, a group.

**The versions live in S3.** Registering them needs the s3 artifact store: `-i store=s3`, or a
profile with `artifact_store = "s3"` (as `cairn-pack-finetune`'s `finetune-ssh` has), plus
`CAIRN_ARTIFACT_S3_BUCKET` (`CAIRN_ARTIFACT_S3_PREFIX` optional, `CAIRN_ARTIFACT_S3_ENDPOINT` for
MinIO or another S3-compatible store). With the local store the split refuses to register:
its files sit in the leg's temporary workdir, which a reboot deletes and another machine cannot
read.

## The data review

Dedup can remove far more than expected. In the qualification run (cairn-extensions#9), near dedup at
threshold 0.8 dropped 36,385 of 42,680 rows (85%), and training went ahead on what was left
without anyone looking at that number. `data-review` makes a drop that size a decision.

After `prepare`, the pipeline pauses at `data-review` (approvers `data-owners`) when exact or near
dedup dropped more than `review_drop_ratio` of the rows it read. The default is `0.30`. Each pass
reports its own share, `dropped_fraction`: exact dedup's is of the rows read, near dedup's is of
the rows exact dedup kept. The gate compares each share with the limit separately, so two passes
just under it can together drop more. When neither share is over the limit, the gate is skipped
and the run goes straight to `format`.

The card shows, from the `prepare` leg's findings:

| Field | What |
|---|---|
| `ingest_normalize_rows` | rows in |
| `dedup_exact_dropped`, `dedup_exact_dropped_fraction` | exact duplicates dropped, and their share |
| `dedup_near_dropped_near`, `dedup_near_dropped_fraction` | near duplicates dropped, and their share of what exact dedup kept |
| `dedup_near_threshold`, `dedup_near_method` | the near threshold applied (`minhash`), or `skipped` without the `[all]` extra |
| `export_jsonl_rows` | rows kept, the cleaned corpus the split reads |
| `profile_tokens_tokens_total`, `profile_tokens_tokens_p95` | tokens, and the p95 example length |
| `profile_tokens_over_length_rows` | rows over `max_seq_len` |
| `split_rows_train`, `split_rows_val`, `split_rows_test` | split sizes |

- **Approve:** `cairn remote advance <pipeline-id> data-review`. The run goes on to `format`
  and `data-ok`.
- **Decline:** cairndev 0.3.1 has no reject action for a pipeline gate, so cancel the pipeline
  with `cairn remote cancel-pipeline <pipeline-id>`. The run ends and the open review is recorded
  as rejected. The three versions are already registered when the gate pauses (the `prepare` leg
  registers them when it lands), and declining does not mark them: the fine-tune pack's launch
  door still accepts them. Do not train on a declined run's versions; prepare again under
  another `register` name instead. Marking the versions of a declined review is not done yet.
- **Compare thresholds before you choose.** Launch the same source with another near threshold
  and compare the two runs' findings, for example `-i near_dup_threshold=0.9` beside the default
  0.8. A higher threshold drops only closer pairs, so it drops fewer rows. Exact dedup alone is what a lean
  install runs: without the `[all]` extra near dedup is skipped (`dedup_near_method: skipped`).
  The holdout door compares train and test at 0.8. Above that, near dedup keeps pairs the door
  still counts as near duplicates, and if such a pair lands on both sides of the split the
  fine-tune launch is refused. A 0.9 run therefore shows what a looser dedup would keep; it is
  not automatically a set to train on.
- **Change or turn off the limit:** `-i review_drop_ratio=0.5` pauses only above half;
  `-i review_drop_ratio=1.0` never pauses (a share is at most 1.0). A value that is not a number
  is refused at launch.

## Presets

| Preset | What it prepares |
|---|---|
| `ifc-bim-alpaca` | `Dietmar2020/ifc-bim-high-quality-alpaca` at a pinned commit, 90/5/5. `topic` is the first building element the answer defines and `family` is the request with numbers and names replaced; the split groups by `family`. Lengths use the `Qwen/Qwen3-4B-Base` tokenizer against 2048. |

The topic is matched with a regular expression, not parsed. An IFC parse belongs to the IFC
validator (cairn-extensions#12).

## Support

| Path | Status | Evidence | Proven on `cairndev` |
|---|---|---|---|
| `prepare` on the builtin data engine, s3 store | EXPERIMENTAL | `tests/test_prepare.py`: the pack's flow, the core runner and registration in process, on the 200-row sample and a mocked S3 bucket | 0.2.1 (the released wheel from PyPI: a local run and CI's `scripts/test_wheels.sh`) |
| `data-review` pauses above `review_drop_ratio` and is skipped below it | EXPERIMENTAL | `tests/test_data_review.py`: the pipeline through the core's launch door and in-process bridge, the flow on the core runner in process, a mocked S3 bucket; the near-duplicate route needs the `[minhash]` extra and is skipped without it | 0.3.1 (a local run and CI's `scripts/test_wheels.sh`) |
| `prepare` on a Hub source with `revision`, through `cairn remote launch` | not yet | the qualification run, cairn-extensions#9 | not yet |

Apache-2.0 — see [`LICENSE`](LICENSE).
