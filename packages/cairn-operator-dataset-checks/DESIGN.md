# cairn-operator-dataset-checks — design

- **Distribution:** `cairn-operator-dataset-checks` (`packages/cairn-operator-dataset-checks/` in cairndev-sh/cairn-extensions)
- **Capabilities provided:** `dataset.format_check`, `dataset.dedup`, `dataset.dedup_apply`,
  `dataset.pii_scan`, `dataset.scan_all`, `dataset.decontam`,
  `dataset.decontam_apply`, `dataset.profile`, `dataset.provenance`, `dataset.gate`,
  `dataset.prep_gate`, `dataset.prep_review`, `dataset.sample`, `dataset.anonymize`,
  `dataset.curate`, `dataset.split`, `dataset.replay`, `dataset.capture_to_sft`,
  `dataset.capture_to_pairs`, `dataset.capture_to_rewards`, `dataset.publish_fingerprints`
- **Status:** proven before the move (the readiness gate, the `clean` dataflow and the prep
  split ran in a deployment; `dataset/curate` is exercised less often); not yet re-proven on
  the public wheel (see the README's Support table)
- **Decisions:** ADR-0076 (loop-aware core, the check→apply pairing), ADR-0078 (columnar
  data plane), ADR-0080 (the data plane and the control/data split), ADR-0081 (dataflows
  as pack data), ADR-0095 (continual learning: replay + the capture projections),
  ADR-0102 D1/D3 (dataset roles, frozen child-version splits, the holdout door) ·
  #378–#384, #711, #870, #882/#883, #991, #1029–#1031, #1050, #1466, #1625, #1629, #2020

## 1. Purpose

This pack answers one question about a corpus — **can this data be trained on, and what
would an eval number over it actually mean?** — and then, under a gate, fixes what it
found. It is the only place in cairn that knows what "the same example" means, and every
surface that depends on that agreement (dedup, decontamination, the holdout door, the
cross-batch fingerprint index) reads its definition from here.

It covers: format conformance against a named trainer, exact and near-duplicate
detection, contamination against an eval/benchmark set, PII detection and redaction,
token-length and refusal profiling, licence/provenance declarations, a deterministic
group-aware split, quality curation via an LLM judge, rehearsal sampling from earlier
cycles, and the three signal→rung projections off a capture export.

It deliberately does **not**: train, serve, judge model outputs (`dataset/curate` scores
*examples*, not answers — the eval pack judges answers), decide policy (`dataset/gate`
turns findings into a verdict against thresholds the environment owns; it does not choose
the thresholds), or write to the registry on its own initiative. Registration happens
control-side and only when a caller explicitly asked for it.

## 2. Architecture

Two execution shapes live in one wheel, because they share one row vocabulary:

```
in-process operators   scan/verdict/apply over a path        (most of the pack)
dataflow stages        the same logic as engine kernels      (dataflow_stages.py)
```

* **`_canon.canonical_hash`** is the row identity: sorted-key JSON, whitespace-collapsed,
  sha256. `dataset/dedup` and `dataset/decontam` import it so "duplicate" and "leaked"
  cannot drift apart. `cairn.core.holdout` re-states the same three lines (so the control
  plane can close its door with no operator pack installed) and a core test compares the
  two implementations directly.
* **`_ngram`** does exact shingle-set Jaccard — O(train·eval), right for one eval set.
  **`_minhash`** wraps **datasketch**'s `MinHash` + `MinHashLSH` for the near-linear
  version; the band/row split is datasketch's problem, not ours. `_pii` layers a
  regex/`google-re2` secrets pass under **Presidio** NER. `_duck` runs the exact-dedup
  anti-join in **DuckDB** on the job machine. `_io` owns streaming reads
  (local · `s3://` · `https://` · HF Hub, via **datasets** and **boto3**), atomic writes,
  sharded output and the optional **polars** frame fast path. `_split` owns the
  deterministic hash assignment and the optional **scikit-learn** grouped/stratified
  allocation.
* **Every heavy dependency is an extra** (`[pii]`, `[fastpii]`, `[fast]`, `[minhash]`,
  `[tokenizers]`, `[split]`) with a hermetic fallback, so a base install runs the whole
  pack and CI needs none of them. A path that genuinely cannot degrade (`method='minhash'`)
  raises a named `RuntimeError` naming the extra rather than silently doing less.

**Streaming, not loading.** The default path is one pass with bounded memory
(`_io.iter_rows`, `ShardedJsonlWriter`); the materialised paths (`load_rows`, the sklearn
split) are used only where the algorithm is global by nature and are documented as such.

**Idempotency and partial failure.** Every check is a pure function of the bytes: same
input, same seed, same output. The `*_apply` operators and `dataset/split` write through
`_io`'s atomic replace, so an interrupted write leaves the previous file intact rather
than a truncated one.

**The admission chain (#2021, ADR-0102 D1).** `dataflow_stages.py` is not just a mirror
of the operators any more: it is what a domain pack's intake flow runs, and that flow IS
intake. The chain is `ingest.normalize -> dedup.exact -> dedup.near -> decontam.apply ->
pii.redact -> expectations.check -> profile.tokens -> export.jsonl`, and the leg that
registers the export folds every stage's flat stats into ONE intake verdict
(`admitted | degraded | blocked | invalid_format`) with a structured report on the
version, which the launch door reads (`cairn/core/intake.py`). Three consequences the
kernels here have to honour:

* **A default-chain kernel degrades; an opted-in one refuses.** `dedup.near` is an honest
  skip (`checked: false` + a note, corpus untouched) without the `[minhash]` extra,
  because every import runs it and a lean install must still be able to import.
  `dedup.exact` with an explicit `backend=duckdb` keeps the opposite posture — an opt-in
  that silently downgraded would fake the evidence a gate reads.
* **Only scalars cross the boundary** (ADR-0081 D3 / ADR-0080 R5), so the token histogram
  rides as `bucket_<lo>_<hi>` keys, the failed expectations ride as one JSON string, and
  the eval sets compared against ride as a comma-joined `eval_sets`. `cairn.core.intake`
  reassembles them; nothing here returns a nested dict.
* **Nothing is ever silently truncated.** `profile.tokens`'s `over_length` is
  `count` (default) / `drop` / `refuse`; a fourth value is a `ValueError`, and `refuse`
  leaves the rows in place so the verdict — not the kernel — refuses the version.
* **Every kernel is under a stage contract (#2242 V1-C, ADR-0118).** The seven intake
  kernels here and `scan.all` declare (the eighth intake stage, `expectations.check`, lives in
  another pack and opts in the same way) `implements = ("contract://dataset.stage.<kernel>/1.0.0",)`
  beside `risk_class`; the definitions ship in `cairn-contracts-dataset-stages`. The
  configuration schema is CLOSED: an unknown key, or a value of the wrong type that the
  kernel used to coerce (`"0.9"` for `threshold`), is refused before the kernel runs, on the
  control plane and again in the runner. The output schema is open but its always-emitted keys
  are `required`, checked on the raw stats before the done marker. The kernels must load on the
  control plane (`cairn doctor` reports any that cannot, #2363), and an engine that does not
  declare `carries_stage_contracts` is refused for every flow that runs them. Since #2396
  (ADR-0102 D1 as written) `ingest.normalize` emits `quarantined_rows` and `export.jsonl` emits
  `rows` (counted in the hash pass); `intake.py` makes any malformed line, or an export with 0
  rows, `invalid_format`, which the launch door refuses. Those two kernels implement revision
  1.1.0 of their stage contract, which requires the keys.
* **A pinned source and its manifest (cairn-extensions#5).** `ingest.normalize` implements
  1.2.0 (configuration 1.1.0 adds `revision`, output 1.2.0 requires the new keys). `revision`
  reaches only the Hub-id branch of the loaders and is recorded only for a Hub id (a file or URL
  keeps its `#sha256=` pin). The stats add `revision_sha` (the commit, looked up after the load:
  best effort, null offline; pin a commit sha to rule out a branch moving in between),
  `fetched_at` and `snapshot_hash`, the hash of what was materialised, for every source.
  `source_hash` keeps its meaning — the bytes of a local file — because the runner names it as
  stage 0's registry-id parent (#1432); a Hub id keeps the URI parent instead of an id that
  names no artifact. The same record goes to `ingest.json`.
  `dataset.manifest` (`dataset.stage.dataset_manifest/1.0.0`) reads it and `current.jsonl` and writes
  `manifest.json` (`dataset.manifest.document/1.0.0`); licence and citation come from the Hub
  card and are null when there is none. Its stats say `kind: report`, so the runner registers a
  report, not a dataset; it belongs after `export.*` so the report's parent is the export and
  its `data_content_hash` is the export's registry id (nothing enforces the order: placed
  before the export, the export's parent becomes the report). It reports `dataset_rows`, not `rows`, so
  the report's descriptor does not claim rows of its own.
* **Lengths, topics and families for preparation (cairn-extensions#16).** `profile.tokens`
  1.1.0 takes `measure: example`: prompt (`_prompt_text`: the turns before the last assistant
  reply, Alpaca `instruction` + `input`, or `prompt`) plus response, which is what a training
  sequence holds, and `max_seq_len` then applies to it; `response` (default) is 1.0.0.
  `fields.derive` adds fields from a regular expression over another field — `extract` (the
  first match, `other` when none: a topic) or `normalize` (matches replaced, whitespace
  collapsed: a template family) — for a dataset with neither column; it runs after dedup so a
  derived field cannot change what counts as a duplicate, and it refuses to overwrite a field
  the rows carry. `profile.fields` (read-only) reports a label field's top values and counts as
  two flat lists and a grouping field by size only, because a template family is prompt text.
  Both pass through when a pipeline binds nothing.

**Control-plane writes.** Two operators touch a database, and both do it lazily and
control-side, because the data plane's job box has no credentials (ADR-0081 D3):
`dataset/publish_fingerprints` writes the cross-batch index, and `dataset/split` registers
its child versions when `register_as` is set. The distinction between them is deliberate:
the fingerprint index is an optimisation, so losing it degrades to `published=0`; a
requested registration is a promise a downstream run depends on, so failing to make it
raises.

**Exported reader (#2022).** `pack.toml`/`pyproject` register `_io.iter_rows` under the
`cairn.dataset_readers` entry-point group (`rows`). Core (`cairn/core/dataset_readers.py`)
resolves it by group for the Add-dataset preview — a bounded, read-only sample of an
upload, object-store URI, Hub id or Parquet source — and never imports this package
directly (ADR-0084 D1). Without the pack, core previews JSONL/s3 through its own reader.

## 3. Operators

| Operator | Capability | Risk | Mock | Input -> Output |
|---|---|---|---|---|
| `dataset/format` | `dataset.format_check` | read_only | no | `FormatInput` -> `FormatOutput` |
| `dataset/dedup` | `dataset.dedup` | read_only | no | `DedupInput` -> `DedupOutput` |
| `dataset/dedup_apply` | `dataset.dedup_apply` | external_write | no | `DedupApplyInput` -> `DedupApplyOutput` |
| `dataset/decontam` | `dataset.decontam` | read_only | no | `DecontamInput` -> `DecontamOutput` |
| `dataset/decontam_apply` | `dataset.decontam_apply` | external_write | no | `DecontamApplyInput` -> `DecontamApplyOutput` |
| `dataset/pii_scan` | `dataset.pii_scan` | read_only | no | `PiiInput` -> `PiiOutput` |
| `dataset/anonymize` | `dataset.anonymize` | external_write | no | `AnonymizeInput` -> `AnonymizeOutput` |
| `dataset/profile` | `dataset.profile` | read_only | no | `ProfileInput` -> `ProfileOutput` |
| `dataset/provenance` | `dataset.provenance` | read_only | no | `ProvenanceInput` -> `ProvenanceOutput` |
| `dataset/scan_all` | `dataset.scan_all` | read_only | no | `ScanAllInput` -> `ScanAllOutput` |
| `dataset/gate` | `dataset.gate` | read_only | no | `GateInput` -> `GateOutput` |
| `dataset/prep_gate` | `dataset.prep_gate` | read_only | no | `PrepGateInput` -> `PrepGateOutput` |
| `dataset/prep_review` | `dataset.prep_review` | read_only (`requires_approval`) | yes | `PrepReviewInput` -> `PrepReviewOutput` |
| `dataset/sample` | `dataset.sample` | read_only | no | `SampleInput` -> `SampleOutput` |
| `dataset/curate` | `dataset.curate` | paid_compute | no | `CurateInput` -> `CurateOutput` |
| `dataset/split` | `dataset.split` | external_write | no | `SplitInput` -> `SplitOutput` |
| `dataset/replay` | `dataset.replay` | external_write | no | `ReplayInput` -> `ReplayOutput` |
| `dataset/capture_to_sft` | `dataset.capture_to_sft` | external_write | no | `CaptureToSftInput` -> `CaptureToSftOutput` |
| `dataset/capture_to_pairs` | `dataset.capture_to_pairs` | external_write | no | `CaptureToPairsInput` -> `CaptureToPairsOutput` |
| `dataset/capture_to_rewards` | `dataset.capture_to_rewards` | external_write | no | `CaptureToRewardsInput` -> `CaptureToRewardsOutput` |
| `dataset/publish_fingerprints` | `dataset.publish_fingerprints` | read_only | no | `PublishFingerprintsInput` -> `PublishFingerprintsOutput` |

### The check → apply pairing (ADR-0076 D3, #991)

`dataset/dedup` and `dataset/decontam` are `read_only` and report; `dataset/dedup_apply`
and `dataset/decontam_apply` are `external_write` and remove. They are separate operators
so a governed environment can place its approval gate between the finding and the
deletion. An apply operator refuses to guess: it takes the row indices the check produced,
not a threshold of its own.

### `dataset/decontam`

Three methods: `exact` (canonical hash), `ngram` (shingle-set Jaccard) and `minhash`
(LSH); `both` means exact + ngram, which is why decontam guards its `minhash` value
itself instead of going through `_minhash.resolve`. **`checked: false` is the field to
read first** — it means no eval set was supplied and nothing was compared, so
`contaminated: 0` says nothing at all.

### `dataset/split` (#383, #2020)

Deterministic, group-aware and optionally stratified. A group's split is a pure function
of its key hash, so no group can straddle two splits — `no_leakage` is true by
construction on the streaming path and checked explicitly on the sklearn one.

With `register_as` set (ADR-0102 D1) each split also becomes a **frozen child version** —
`<name>-train` (role `training`), `<name>-val` (`validation`), `<name>-test` (`test`) —
carrying its own content hash, a `parent_version` pointer, the parent's declared schema
and column mapping, and the **declared policy** (ratios, `seed`, `group_by`,
`stratify_by`, backend) on the record. Items never move: the same source under the same
policy re-registers the same hash and mints no new version, while a different policy mints
new children and leaves the old ones byte-identical at their old version numbers. A split
never produces a `golden` child — golden is human-rated by definition and no split can
manufacture a human's ruling. `parent_ref` naming a different dataset than `register_as`
raises rather than registering an invented lineage.

### `dataset/prep_gate` → `dataset/prep_review` (#882, #883)

`prep_gate` turns prep's post-scrub PII re-scan plus the env-owned `mode` into a route.
When it routes to review, `dataset/prep_review` is `requires_approval`, so the durable
approval gate fires *before* it runs and the prep run pauses `awaiting_approval`.
Both implement shared contracts (#2242 V1-D): the gate still fires first, before the body and
before any contract check, so a rejected review leaves nothing to check; an approved one runs
under `dataset.prep_review` like any other step. `prep_gate`'s `action` and `verdict` are
enums in `dataset.prep_gate`'s output schema.

### `dataset/replay` and the capture projections (ADR-0095)

`dataset/replay` implements `contract://dataset.replay/1.0.0` (`cairn-contracts-dataset` 1.2.0, #158 — the
retrain-cycle slice): its `ReplayConfig` is the closed configuration a profile may govern (ratio, strategy,
seed). The capture projections stay legacy until their pipeline's slice.

`dataset/replay` mixes rows from earlier cycles' registered versions into the new train
slice at a governed ratio, recording which source each row-group came from — anti-forgetting
as a data operation. `dataset/capture_to_sft` / `_to_pairs` / `_to_rewards` are pure
operators over the **exported** capture JSONL, never over the store: the export is where
the consent gate and the holdout split already ran. `capture_to_rewards` deliberately does
not emit `(prompt, completion, score)` triples — GRPO scores its own rollouts, so a
captured score is context for a reward function, never a training reward.

## 4. Configuration

| Setting | Where | Scope | Default | What it does |
|---|---|---|---|---|
| `kinds`, `preset`, `ner` (`dataset/anonymize`) | `config_schema` | **governance** | `governed` preset | ENV-OWNED: which PII kinds MUST be redacted. A run-supplied value is DROPPED (ADR-0068/0070) — a run must not be able to narrow the redaction set |
| `required_fields`, `metadata_sources` (`dataset/provenance`) | `config_schema` | **governance** | licence + source | ENV-OWNED: the metadata a dataset must declare. A run-supplied value is DROPPED |
| `mode` (`dataset/prep_gate`) | `config_schema` | **governance** | see schema | ENV-OWNED: `block` / `hitl` / `warn` on residual PII. A run-supplied value is DROPPED |
| `thresholds` (`dataset/gate`) | `config_schema` | operational | see `GateThresholds` | the pass/warn/fail bars the verdict is measured against |
| `threshold`, `ngram`, `num_perm` (`dataset/dedup`) | `config_schema` | operational | 0.8 / 5 / 128 | near-duplicate sensitivity |
| `overlap_threshold`, `ngram`, `num_perm` (`dataset/decontam`) | `config_schema` | operational | 0.8 / 13 / 128 | contamination sensitivity |
| `target_trainers`, `sample_size` (`dataset/format`) | `config_schema` | operational | — | which trainer's format to conform to; scan bound |
| `preset`, `ner`, `model`, `n_process`, `analyze_chunk`, `sample_size`, `backend`, `secrets_full_scan` (`dataset/pii_scan`) | `config_schema` | operational | `auto` backend | detection bound + backend (the *scan*; redaction policy is `anonymize`'s governance table) |
| `short_threshold`, `sample_size` (`dataset/profile`) | `config_schema` | operational | — | refusal/short-response profiling |
| `sample_size`, `preset`, `backend`, `short_threshold`, `target_trainers` (`dataset/scan_all`) | `config_schema` | operational | — | the fused single-pass scan's bounds |
| `n` (`dataset/sample`) | `config_schema` | operational | — | rows fed to the judge |
| `train_ratio`, `val_ratio`, `test_ratio`, `seed`, `shard_size` (`dataset/split`) | `config_schema` | operational | 0.9 / 0.1 / 0 / 13 / 0 | the split policy an environment may standardise |
| `ratio`, `strategy`, `seed` (`dataset/replay`) | `config_schema` | operational | — | rehearsal mix |
| `register_as`, `parent_ref` (`dataset/split`) | `input_schema` | run | `""` | #2020 — per-run: register the splits as frozen child versions of this dataset |
| every `path`, `output_dir`, `eval_path`, row-index list | `input_schema` | run | — | per-run bounds; a caller sets them |
| `threshold`, `ngram`, `num_perm`, `normalize` (`dedup.near` stage) | flow `config` | run | 0.8 / 3 / 128 / true | #2021 — near-duplicate sensitivity in the admission chain |
| `eval_path`, `eval_paths` (`decontam.apply` stage) | flow `config` | run | unset | #2021 — the contamination probe. An explicit `eval_path` wins; `eval_paths` is the LIST the control plane fills with every `golden`/`test` version in the workspace (the box has no registry), merged into one probe file. Neither = honest `checked: false` |
| `base_model`, `tokenizer_name`, `tokenizer_path` (`profile.tokens` stage) | flow `config` | run | unset | #2021 — whose tokenizer counts the tokens. Unset, unloadable, or no `[tokenizers]` extra → whitespace WORD counts, said out loud in `token_backend_note` |
| `max_seq_len`, `over_length` (`profile.tokens` stage) | flow `config` | run | 0 / `count` | #2021 — the over-length rule. `count` keeps and counts · `drop` removes and counts · `refuse` keeps them and makes the intake verdict `blocked`. **`truncate` is not a value** (ADR-0102 D1) |
| `AWS_DEFAULT_REGION`, standard AWS credentials | env | deployment | — | `s3://` reads and writes, via boto3 |
| `CAIRN_STORAGE` | env | deployment | unset | read (lazily) by `dataset/publish_fingerprints` and by `dataset/split`'s `register_as` path only |

**Per-run bounds belong in `input_schema`.** A ceiling declared in a governance-scoped
`config_schema` field is silently discarded at run time, so the pack would advertise a
limit it does not enforce (#905). `register_as` is a per-run request and lives on the
input schema for exactly that reason; the redaction policy is env-owned and lives on the
config schema for the opposite one.

**Connections and secrets.** `dataset/curate` is the only operator that calls a model, and
it goes through the governed LLM gateway rather than holding a key. Everything else needs
no connection. Object storage uses the ambient AWS credential chain; with none, an
`s3://` path fails loudly at read rather than degrading to an empty dataset.
`dataset/publish_fingerprints` **degrades** with no store (`skipped: store_unavailable`);
`dataset/split`'s `register_as` **fails closed**.

## 5. Governance

* `read_only` — every check, every verdict, plus `dataset/publish_fingerprints`, which
  writes only cairn's own index (no user data, no external system, nothing to gate).
* `external_write` — everything that writes a dataset file: the two `*_apply` operators,
  `anonymize`, `split`, `replay` and the three capture projections. A governed environment
  is expected to place ONE approval gate in front of the remediation leg rather than one
  per operator; the `clean` dataflow runs the whole repair as one leg.
* `paid_compute` — `dataset/curate` alone: it scores every candidate example with an LLM,
  so cost scales with corpus size and it belongs under a budget.
* `requires_approval` — `dataset/prep_review` alone. The durable gate fires before it
  runs, so a dirty scrub cannot proceed unattended.

**What is auditable.** Every operator's effective inputs and config land on the run record
before `run` (the executor writes them), so a verdict can always be re-derived. Registered
versions carry their producer run id, their content hash, their lineage parents and — since
#2020 — their role, declared schema, column mapping and frozen split policy. The holdout
door reads those roles at launch and stamps the eval-set version hashes it considered onto
the run, which is what makes "which holdout was this model held out from?" answerable
later.

**What a deployment may override.** The operational config tables above. What it may
**not** override from a run: the redaction kinds, the required provenance fields and the
prep-gate mode — all three are governance-scoped, which means a run-supplied value is
dropped, not honoured.

## 6. Failure modes

| Symptom | Cause | What the operator returns | Fail closed? |
|---|---|---|---|
| `contaminated: 0` believed, no eval set was given | `eval_path` empty; the honest skip | `checked: false` + `contaminated: 0` | reported, not enforced — read `checked` first (#378) |
| `RuntimeError: method='minhash' requires the [minhash] extra` | a MinHash path without datasketch | raises, naming the extra | yes — never silently downgrades to exact (#382) |
| Cross-batch dedup silently within-batch only | the `clean` dataflow published no fingerprints; the only publisher lived in the deprecated fix trio | `dataset/publish_fingerprints` now publishes control-side; `published=0` + a `skipped` reason when it cannot | no — a lost index degrades, never fails a good remediation (#1050) |
| Registration requested, nothing registered | the data plane has no database, so a box-side write is impossible | `dataset/split` with `register_as` raises; the leg fails | yes — a promised version must never silently vanish (#1466, #2020) |
| Trained bytes were never gated | the readiness gate saw the RAW source, prep then scrubbed it | prep re-scans its own output; `dataset/prep_gate` verdicts on `post_scrub_pii_rows` | yes, when `mode=block` (#870, #882) |
| A `dataset:` ref reached the box unresolved | the launch-time resolver only rewrote top-level STRINGS, so `dataset/replay`'s list of `sources` passed through untouched | the resolver and its gate now share one predicate | yes (#1625) |
| Sharded near-dup missed a cross-shard pair | signatures clustered per shard | `cluster_signatures` clusters every signature once, globally | yes (#962) |
| `s3://` read returns nothing | absent AWS credentials | raises at read | yes — never an empty dataset that reads as "clean" |
| A golden set registered without human ratings | none — refused at the registry write | `DatasetRoleError` naming the unrated rows | yes (#2020) |
| Near-duplicates survived every import | `dedup_minhash` existed as an operator nobody had wired into the default chain, so only byte-identical rows were dropped | `dedup.near` is in the default admission chain; `dropped_near` is reported beside `dropped_exact` | no — an honest skip without `[minhash]`, never a failed import (#2021) |
| `checked: true` on a contamination pass that compared against nothing useful | the flow's single `eval_path` was whatever the author remembered to pass | `eval_paths` defaults control-side to every `golden`/`test` version in the workspace; `eval_sets` names what was ACTUALLY read, and a probe that would not read is left out of it | reported (#2021) |
| A token profile that is not the model's tokens | no tokenizer named, or the Hub unreachable | falls back to whitespace WORD counts with `token_backend: whitespace` and a `token_backend_note` saying why | no — an unreachable Hub must not fail an import (#2021) |
| Over-length examples silently shortened | OpenAI's documented behaviour, named as the anti-pattern by ADR-0102 D1 | `over_length: count/drop/refuse`; `truncate` raises `ValueError` | yes — there is no code path that truncates a row (#2021) |

## 7. Testing

Tests live in this pack's own `tests/` beside `src/`. From the repository root, with the
three wheels and the `test` extra installed:

```bash
CAIRN_STORAGE= OBS_STORAGE= python -m pytest packages/cairn-operator-dataset-checks/tests -q -m "not expensive"
```

Golden-fixture coverage: format conformance per trainer, exact + near-dup dedup (DuckDB
and streaming), n-gram and MinHash decontamination, the PII cascade (regex · regexset ·
two-tier · sharded), token profiling, provenance, the gate's verdict table, prep gate and
review, sampling bounds, the capture projections, replay, the split (hash and sklearn
paths) and, since #2020, the frozen child versions the split registers — against a temp
sqlite `CAIRN_STORAGE`, with real files on disk because registration hashes bytes.
`test_intake_stages.py` (#2021) covers the admission chain's own kernels: the 3-exact +
2-near-duplicate fixture that must report BOTH counts, the honest skip without
`[minhash]`, the token profile's total/p50/p95/histogram, all three over-length policies
plus the refusal of `truncate`, and the multi-probe decontamination default (union
probe · explicit `eval_path` wins · an unreadable probe is left out of `eval_sets`).

Every extra-dependent test is `importorskip`-gated, so a base install is green: the
sklearn split paths, the MinHash paths and the Presidio backends skip cleanly. What does
**not** run in CI: nothing needs a GPU; `dataset/curate` is exercised against a
stub gateway rather than a paid model. `test_dataflow_kernels.py` builds the real builtin
engine (through the core's engine registry) over a synthetic dirty sample whose duplicates
and PII rows it plants itself, and is the slowest file here by an order of magnitude.

Entry points and the `pack.toml` governance tables are read from **installed** metadata, so
the manifest sync and drift gates (`cairn pack sync --check`, `cairn pack check`) pass
vacuously against an uninstalled copy: install the wheel before trusting them.

## 8. History

| Date | Change | PR |
|---|---|---|
| 2026-06 | The pack: format, dedup, decontam, PII scan, profile, provenance, gate, sample | #378–#382, #384 |
| 2026-06 | `dataset/split` — deterministic, group-aware, stratified | #383 |
| 2026-07 | Governed prep: `anonymize`, the post-scrub invariant, the prep gate and review | #711, #870, #882, #883 |
| 2026-07 | Check → apply pairing: `dedup_apply`, `decontam_apply` (ADR-0076 D3) | #991 |
| 2026-07 | Columnar fast paths + sharded output; global cross-shard near-dup clustering | #962, #891, ADR-0078 |
| 2026-08 | The data plane: `scan_all` (fused pass), `fix_batch` (remote job), the dataflow stages | #1029–#1031, ADR-0080 |
| 2026-08 | `dataset/publish_fingerprints` — the control-side half of cross-batch dedup | #1050 |
| 2026-08 | `dataset/replay` and the three capture projections (ADR-0095) | #1625, #1629 |
| 2026-09 | Frozen child-version splits: `register_as` / `parent_ref`, roles and the declared policy on the record (ADR-0102 D1); this design record | #2020 |
| 2026-09 | The admission chain: `dedup.near` and `profile.tokens` kernels, the multi-probe `decontam.apply` default, and the intake verdict the launch door reads (ADR-0102 D1) | #2021 |
| 2026-09 | `_io.iter_rows` exported under the `cairn.dataset_readers` entry-point group for the Add-dataset preview; core resolves it by group, never by import | #2022 |
| 2026-10 | `decontam`, `provenance`, `sample` and `gate` implement the shared `dataset.*` contracts (`cairn-contracts-dataset`); the `dedup.exact`, `dedup.near`, `pii.redact`, `decontam.apply` and `scan.all` kernels call the operators' sync bodies (`run_dedup_apply`, `run_anonymize`, `run_decontam_apply`, `run_scan_all`) with no event loop or stand-in context (M7a). `pii.redact` still ignores the env-governed `AnonymizeConfig.kinds` (pre-existing) | #2242 |
| 2026-10 | The seven intake kernels here and `scan.all` implement `dataset.stage.<kernel>/1.0.0` stage contracts (`cairn-contracts-dataset-stages`): closed configurations, open outputs with required keys; `cairn doctor` reports a kernel the control plane cannot load (ADR-0118) | #2242, #2363 |
| 2026-10 | `ingest.normalize` emits `quarantined_rows`, `export.jsonl` emits `rows` (one pass with the hash): the intake verdict's malformed-line and empty-export checks fire (ADR-0102 D1); stage contracts 1.1.0 require them | #2396 |
| 2026-10 | `profile.tokens` 1.1.0 measures prompt + response on request; new `fields.derive` (topic labels, template families) and read-only `profile.fields` (label counts, group sizes) | cairn-extensions#16 |
| 2026-10 | `ingest.normalize` 1.2.0 loads a Hub id at `revision` and records its commit, fetch time and snapshot hash; new `dataset.manifest` stage writes the bundled manifest (a report) | cairn-extensions#5 |
| 2026-10 | `publish_fingerprints`, `anonymize`, `split`, `pii_scan`, `prep_gate` and `prep_review` implement the shared `dataset.*` contracts (`cairn-contracts-dataset` 1.1.0), so `clean`, `prep` and `decontaminate` reach no uncontracted operator or stage | #2242 |
| 2026-10 | `dataset/fix_batch`, its on-box `cairn-dataset-fix` entrypoint and `_fix_pipeline` retired with the deprecated fix trio; `clean` is the remediation path | #2434, #2425 |
| 2026-10-22 | Moved public to `cairndev-sh/cairn-extensions` with its two contract wheels (definitions byte for byte unchanged): the `sample.rows` hash method imports `cairn.core.dataflow.fidelity` (cairndev 0.2.0); `[compat]` and the `cairndev` pin are bounded to `>=0.2.0,<0.3`; `[pack].name` is the distribution name; the engine-chain tests run over a synthetic sample instead of a domain pack's example file. | |
