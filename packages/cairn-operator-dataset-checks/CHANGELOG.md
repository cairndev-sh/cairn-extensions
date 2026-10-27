# Changelog

## Unreleased

- **Both dedup passes report the share they dropped** (cairn-extensions#39). `dedup.exact`
  (python and duckdb backends) and `dedup.near` add `dropped_fraction`: `dropped` divided by the
  rows the stage read, rounded to 4 places, 0.0 for an empty input or a skipped near pass. A
  pipeline expression cannot divide, so a gate that pauses when a pass drops too much compares
  this stat with its limit. Both kernels implement the 1.1.0 revision of their stage contract,
  which requires the stat (1.0.0 stays published, unchanged). Requires
  `cairn-contracts-dataset-stages==1.5.0`.

- **Prompt+response length, derived fields and field counts** (cairn-extensions#16).
  `profile.tokens` implements 1.1.0: `measure: example` counts prompt + response and reports
  each side's p50/p95/max (`measure: response`, the default, is unchanged). New stage
  `fields.derive` adds a field from a regular expression over another (`extract` or
  `normalize`) and never overwrites one. New read-only stage `profile.fields` reports label
  values and counts as flat lists and grouping fields by size only. Requires
  `cairn-contracts-dataset-stages==1.4.0`.

- **`split` returns train / val / test as products** (cairn-extensions#6). With `export: true`
  each non-empty split is a product of the stage; with `register: <name>` it registers as
  `<name>-train` / `-val` / `-test` with roles training / validation / test. `register` needs
  the s3 store (`store: s3`, or the environment's `artifact_store = "s3"`): a local split is a
  file in the leg's temporary workdir, and a version registered from it would name bytes that a
  reboot deletes and another machine cannot read, so the stage refuses it before splitting.
  `split` now implements `dataset.stage.split/1.0.0`, whose configuration is closed: a flow that
  passes `split` a key the contract does not name is refused by the stage-contract check.

- **`export.jsonl` with `register` needs the s3 store too** (cairn-extensions#6; owner decision
  2026-10-24: registering a dataset version needs an S3-compatible store, and local development
  uses MinIO or similar). On the local store the export was registered as a file in the leg's
  temporary workdir; it is now refused, before anything is written, with the same message as
  `split` (one helper). Set `store: s3`, or `artifact_store = "s3"` with
  `CAIRN_ARTIFACT_S3_BUCKET` (`CAIRN_ARTIFACT_S3_ENDPOINT` for MinIO). An export without
  `register` is unchanged on either store.

- **A pinned Hub revision and the bundled manifest** (cairn-extensions#5).
  `ingest.normalize` takes `revision` (stage contract 1.2.0: configuration 1.1.0, output 1.2.0)
  and records `revision` (Hub ids only), `revision_sha`, `fetched_at` and `snapshot_hash`, also
  in `ingest.json`; `source_hash` stays file-only, so lineage is unchanged. New stage
  `dataset.manifest` (`dataset.stage.dataset_manifest/1.0.0`) writes `manifest.json` after the export: source, revision,
  hashes, rows, field names, licence and citation (null when absent), fetch time; no row
  content. Requires `cairn-contracts-dataset-stages==1.2.0`.

- **`ingest.normalize` opens a registered version's `file://` URI** (#2171, C1).
  The `#sha256=` verification hashed `Path("file:///…")`, which collapses to the
  non-existent `file:/…`; the source is now resolved to its local path before the
  hash check, the read and the lineage hash. `file:/abs` (the collapsed form) is
  accepted by `_normalize_source` as well.

- **The admission chain is intake** (#2021, ADR-0102 D1) — two new dataflow
  stages and one changed default, all inside the intake flow, which
  is what every import, ingestion and watch runs.
  - **`dedup.near`** (NEW stage) — the near-duplicate pass, `DedupApplyOp` at
    `method=minhash`, now in the DEFAULT chain after `dedup.exact`. It reports
    `dropped_near` separately from `dropped_exact` because "3 copy-paste
    duplicates" and "2 paraphrases of the same example" are two different
    messages. Without the `[minhash]` extra it is an honest skip
    (`checked: false` + a note, corpus untouched) rather than a raise: every
    import runs it, and a lean install must still be able to import a dataset.
  - **`profile.tokens`** (NEW stage) — the token-length profile, automatic, in
    one streaming pass: total, p50/p95/max, a power-of-two histogram as flat
    `bucket_*` scalars, and the count of rows over `max_seq_len`. Uses the base
    model's tokenizer when one is named and loadable, and otherwise falls back
    to whitespace WORD counts **and says so** in `token_backend_note`. Its
    `over_length` knob is `count` (default) / `drop` / `refuse` — `truncate` is
    a `ValueError`, because nothing in this pack may shorten a user's row.
  - **`decontam.apply` gains `eval_paths`** — a LIST of probes merged into one
    index, which the control plane fills with every `golden` and `test` version
    in the workspace (the box has no registry). An explicit `eval_path` still
    wins. `eval_sets` reports which probes were actually read, so a set that
    would not load never counts toward `checked`.

- `dataset/capture_to_sft`, `dataset/capture_to_pairs`,
  `dataset/capture_to_rewards` (0.1.0, NEW operators, #1629, ADR-0095 §1.2) —
  the three signal→rung projections over the production-capture export. All
  three are PURE operators over the exported `capture-signal@N` JSONL (D4: an
  operator never touches the capture store; the export is where the training
  consent gate and the deterministic holdout split already ran), and all three
  share one row vocabulary in `_capture.py` so they cannot disagree about what
  an event means. Feedback rows usually carry only a correlation id — the SDK
  does not ask an application to re-send a prompt it already traced — so each
  resolves prompt/completion through the correlated trace rather than dropping
  the richest half of the corpus.
  - **`capture_to_sft`** → gold SFT rows. An `edit` outranks the completion it
    corrected (`edited_text` becomes the assistant turn); an `accept` endorses
    the completion as it stands; an unreviewed `trace` is output, not a target,
    and never becomes gold. Output rows carry `messages` and NOTHING else — the
    model/adapter attribution is reported in `by_adapter`, not stamped onto the
    row, because an unknown key can break a trainer's chat formatter (the same
    rule `dataset/replay` follows for its `mix`).
  - **`capture_to_pairs`** → `{prompt, chosen, rejected, kind}`, the exact shape
    `operator_training.flow_stages.dataset_dpo_pairs` emits, so `train.dpo`
    reads capture-derived preferences with no adapter in between. An `edit`
    yields the cleanest pair (both halves from one row). A `reject`/`retry`
    pairs only with a LATER endorsed text on the same `trace_id`/`run_id`;
    with no successor the row is DROPPED and counted in `unpaired_dropped`,
    never paired against a guess — a pair whose `chosen` was invented teaches
    the model to prefer the invention.
  - **`capture_to_rewards`** → GRPO reward CONTEXT, deliberately NOT
    `(prompt, completion, score)` triples. GRPO is on-policy: it scores its own
    rollouts each step, so a score attached to a completion the policy has moved
    away from is off-policy noise dressed as supervision, and it gets worse
    every cycle. Emits one row per (prompt, score scale) — `4 out of 5` is never
    averaged with `0.8` — carrying `mean_score`, `score_count`, the
    human-endorsed `reference` a grader compares against, and the scalar keys
    from the production verifier's `payload`.
  - Every projection takes `only_adapter` so a cycle can train on signal about
    ONE adapter: without it, cycle 9's training set silently absorbs complaints
    about cycle 3's behaviour, which is what the capture row's `model` /
    `adapter_uri` fields exist to prevent. Output ordering is a deterministic
    function of the input (source order, no sort), findings carry counts and
    adapter URIs only (the D6 payload budget), and duplicates are dropped by
    canonical hash within each output. Risk: `external_write` (each writes the
    dataset its rung consumes).

- `dataset/replay` (0.1.0, NEW operator, #1625, ADR-0095 §2.3) — rehearsal
  sampling over governed dataset versions: the anti-forgetting DEFENCE for the
  continual-learning loop, built as a DATA operation (EWC/GEM-style
  regularization stays rejected, ADR-0095 §2.2). Mixes rows sampled from prior
  cycles' dataset versions (#1466) into the new train slice at a governed
  `ratio` — whose denominator is the OUTPUT row count, so 0.2 means replay is
  ~20% of what gets written. `strategy` is an OPEN enum (ADR-0095 Amendment 1):
  `uniform` (each source's share ∝ its eligible rows, i.e. every prior row
  equally likely) and `recency_weighted` (share ∝ position+1 over an
  oldest-first `sources` list) are implemented; `hard_example` and
  `forgetting_aware` are RESERVED and raise a clear error naming the per-row,
  per-version eval history `eval/forgetting` (#1626) must supply — never a
  silent fallback to uniform. Apportionment is largest-remainder with capacity
  spill-over, so a short source contributes everything it has and the shortfall
  moves to sources with room; when the pool runs dry the op reports the true
  `ratio_actual` rather than oversampling duplicates. All sampling goes through
  one `random.Random(seed)`, so same inputs + seed → byte-identical output.
  `dedup_against_new` (default on) drops replay rows that duplicate a new row
  using the pack's existing `_canon.canonical_hash`. Provenance is recorded per
  row-GROUP — an output `mix` of `{source, rows_taken, rows_available}` plus a
  fixed group ordering (new rows first, then each source's block in `sources`
  order) — never as extra keys on the training rows, which can break a trainer's
  formatter. Findings carry counts and locations only, never row text (the D6
  payload budget). Reads a launch-resolved `uri#sha256=<hash>` source by
  stripping the pin the way the fetching kernels do, and keeps the pinned string
  in `mix` so the receipt names the exact version mixed in. Risk:
  `external_write`. No new dependency.
- `dataset/curate` (0.1.0, #384) — NEW LLM-judge-driven quality selection
  operator. Scores each example through the governed LLM chokepoint
  (`cairn.core.llm.client.llm_complete`, ADR-0031 — redaction/budget/cost/audit
  apply ambiently), keeps the ones that clear the bar, and writes a curated
  JSONL plus a score-distribution report. Two industry-standard methods:
  **AlpaGasus** (arXiv:2307.08701 — single 0-5 quality score, paper-default
  threshold 4.5) and **DEITA** (arXiv:2312.15685 — complexity × quality, each
  0-5). Selection by `threshold` OR top `keep_fraction`; `max_samples` caps the
  paid calls; scoring runs with bounded async concurrency
  (`asyncio.Semaphore`). Scores are parsed robustly from LLM prose (first
  number, clamped to 0-5) with parse failures counted and skipped. Reuses the
  pack's `_extract`/`_io` to pull (instruction, response) across the common
  fine-tuning formats. Risk `paid_compute` (calls a paid model). No new
  dependency — runs on the base install.
- `dataset/split` (0.1.0, NEW operator, #383) — deterministic, group-aware
  (+ optionally stratified) train/val[/test] split. New `_split.py`: a
  pure-stdlib **hash-based fallback** (`hash_split`) assigns each row's split
  by hashing its group key — or its own canonical hash when no `group_by` is
  set — into `[0, 1)` and bucketing against the cumulative ratios; hashing
  (not shuffling) is the standard reproducible-split technique and makes
  group-locality fall out for free (every row sharing a group key hashes
  identically). An optional `[split]` extra (`scikit-learn`) backs a
  `sklearn_split` path using `GroupShuffleSplit` / `StratifiedShuffleSplit` /
  `StratifiedGroupKFold` (sklearn >= 1.0) depending on which of `group_by`/
  `stratify_by` are set — we do not hand-roll the combined group+stratify
  balanced-assignment problem, sklearn already solves it. `backend='auto'`
  (default) picks sklearn when installed, else the hash fallback; `'hash'`
  forces the dependency-free path; `'sklearn'` without the extra raises a
  clear `RuntimeError` (mirrors `[minhash]`/`[fastpii]`). Writes
  `train.jsonl`/`val.jsonl`[/`test.jsonl`] to a configurable `output_dir` and
  returns `SplitOutput`: per-split counts, ratios requested vs. achieved, a
  `no_leakage` boolean (asserts zero group spans splits — the leakage-check
  evidence the readiness report cites) and a per-stratum breakdown when
  `stratify_by` is set. `test_ratio` defaults to 0 (test.jsonl omitted) per
  SFT practice: train + a small held-out val for loss monitoring, not a
  classic 70/15/15 split — final quality is the finetune eval-gate's
  benchmarks' job, not a test split here. Risk: `external_write`.
- `_minhash` + `dataset/dedup` (0.2.0) + `dataset/decontam` (0.3.0), #382 —
  MinHash+LSH near-duplicate detection behind the new `[minhash]` extra
  (`datasketch`). New shared `_minhash.py` builds a seeded `MinHash` over each
  row's word-shingle set (reusing `_ngram.shingles`, fixed `seed`/`num_perm`
  so signatures and LSH buckets are byte-reproducible) and clusters/queries via
  `datasketch.MinHashLSH` — no hand-rolled banding. `dataset/dedup` gains a
  `method` ('exact' default / 'minhash' / 'both') + `threshold`/`num_perm`/
  `ngram` inputs and `near_duplicate_groups`/`near_duplicates` outputs (rows
  clustered transitively via union-find). `dataset/decontam` gains a fourth
  `method` value 'minhash' (additive to #381's 'exact'/'ngram'/'both') that
  flags train rows whose MinHash Jaccard vs any eval row >= `overlap_threshold`
  via one LSH query per row (near-linear vs the ngram path's O(train×eval)),
  plus a `num_perm` input and `minhash_contaminated_rows` output. Default
  behaviour of both operators (exact hash / #381's ngram) is unchanged and
  needs no new dependency — `resolve()` raises a clear `RuntimeError` when a
  MinHash path is requested without the `[minhash]` extra.
- `_pii` — third `regexset` PII backend behind the new `[fastpii]` extra
  (#379): the same recognizer patterns as the zero-dep `regex` backend,
  combined into one named-group alternation and matched in a SINGLE linear
  pass via Google's `re2` (`google-re2`) instead of looping per-pattern —
  fixes the O(patterns × text) bottleneck on multi-million-row corpora.
  Verified byte-for-byte count/anonymize parity with `regex` in tests.
  `resolve("regexset")` raises a clear `RuntimeError` (mirroring presidio)
  when the extra isn't installed; `auto` is unchanged (still presidio > regex,
  never implicitly picks regexset). `dataset/pii_scan` and `dataset/anonymize`
  both pick it up automatically via the existing `backend` input — no operator
  or schema shape changes beyond the docstring listing the new option.
- `dataset/profile` (0.2.0, #381) — optional accurate subword token counts via
  HF `tokenizers` (Rust-backed), behind the new `[tokenizers]` extra. New
  `ProfileInput.token_backend` ('whitespace' default / 'tokenizers' / 'auto'),
  `tokenizer_name` (HF Hub id) and `tokenizer_path` (local `tokenizer.json`,
  fully offline) inputs; new `ProfileOutput.response_length_tokens` /
  `token_backend_used` outputs (both `None` unless tokenizers actually ran).
  Default behaviour (whitespace char/word counts) is unchanged — additive.
- `dataset/decontam` (0.2.0, #381) — optional word n-gram (shingle) Jaccard
  overlap detection, catching near-duplicate/reworded train↔eval leaks that
  exact hashing misses. New `DecontamInput.method` ('exact' default / 'ngram'
  / 'both'), `ngram` (shingle window size) and `overlap_threshold` inputs; new
  `DecontamOutput.ngram_contaminated_rows` output. Pure stdlib (`set`
  intersection/union) — no new dependency. Default behaviour (exact hash only)
  is unchanged — additive.
- `_io.iter_rows` — memory-safe streaming loader (#380): local `.jsonl` streams
  line-by-line (stdlib, O(1) memory, nested `messages` preserved); other formats
  and Hub ids stream via `datasets` `IterableDataset`. The keystone that lets
  single-pass operators avoid materializing a multi-M-row corpus into one list.
- `_io.scan_frame` — polars `LazyFrame` columnar fast path behind the new
  `[fast]` extra (feeds #381 token counts / #382 MinHash); returns `None` in the
  base install so callers fall back to `iter_rows`.

## 0.1.0

- Scaffolded by `cairn init operator`.
