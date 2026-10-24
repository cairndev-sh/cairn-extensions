# cairn-operator-dataset-checks

Dataset readiness for [cairn](https://github.com/cairndev-sh/cairn): can this corpus be
trained on, and what would an eval number over it mean? Deterministic checks find out, and
under a gate the apply operations fix what they found. One wheel, two execution shapes that
share one row vocabulary:

- **`dataset/*` operations** (`cairn.operators`): `format`, `dedup`, `dedup_apply`, `decontam`,
  `decontam_apply`, `pii_scan`, `anonymize`, `scan_all`, `profile`, `provenance`, `gate`,
  `prep_gate`, `prep_review`, `sample`, `curate`, `split` (deterministic, group-aware, no group
  straddles two splits), `replay`, `capture_to_sft`, `capture_to_pairs`, `capture_to_rewards`,
  `publish_fingerprints`.
- **Dataflow stages** (`cairn.dataflow_stages`): `ingest.normalize`, `dedup.exact`,
  `dedup.near`, `decontam.apply`, `decontam.scan`, `pii.redact`, `profile.tokens`, `split`,
  `scan.all`, `sample.rows`, `curate.score`, `filter.rows`, `export.jsonl`, `export.parquet`,
  `dataset.manifest`, `fields.derive`, `profile.fields` — the admission and cleaning chain an
  engine runs. `ingest.normalize`
  takes `revision` to load a Hub dataset at a commit, tag or branch; `dataset.manifest`, after
  the export, writes `manifest.json` (source, revision and its commit, hashes, rows, field names,
  licence, citation, fetch time; never a row). `profile.tokens` with `measure: example` counts
  prompt + response; `fields.derive` adds a topic label or a template family from a regular
  expression; `profile.fields` reports label counts, and group sizes only.
- **Dataset reader** (`cairn.dataset_readers`): `rows`, the bounded preview reader.
- **Contracts:** implements the `dataset.*` capability contracts (`cairn-contracts-dataset`)
  and the `dataset.stage.*` stage contracts (`cairn-contracts-dataset-stages`).

## Install

This wheel and the contract wheels it pins are not on PyPI yet. Until they are, install from a
checkout of this repository, into the environment `cairndev` 0.2.x runs in:

```bash
git clone https://github.com/cairndev-sh/cairn-extensions && cd cairn-extensions
pip install ./packages/cairn-contracts-dataset ./packages/cairn-contracts-dataset-stages \
  ./packages/cairn-operator-dataset-checks
# with every optional backend
pip install ./packages/cairn-contracts-dataset ./packages/cairn-contracts-dataset-stages \
  './packages/cairn-operator-dataset-checks[pii,minhash,tokenizers,split]'
```

Every heavy dependency is an extra with an honest fallback, so a base install runs the whole
pack; a path that cannot degrade raises an error naming its extra.

| Extra | What it is for |
|---|---|
| `pii` | Microsoft Presidio for PII detection and anonymisation (needs a spaCy model: `python -m spacy download en_core_web_lg`). Without it, the regex tier. |
| `fastpii` | `google-re2` for the single-pass `regexset` PII backend. |
| `fast` | polars + DuckDB: the columnar fast paths and the DuckDB exact-dedup pass. |
| `minhash` | datasketch MinHash + LSH for near-duplicate dedup and decontamination. Without it `dedup.near` is an honest skip (`checked: false`, corpus untouched). |
| `tokenizers` | HF `tokenizers` for subword token counts; without it, whitespace word counts, said so in the stats. |
| `split` | scikit-learn for grouped/stratified splits; without it, the deterministic hash split. |

The base install pins `cairndev>=0.2.1,<0.3` and the two contract wheels exactly. Check an
install with `cairn doctor` and `cairn pack check`.

## Support

Carried from cairn's support matrix as it stood at `cairndev` 0.1.3, when this pack was
private. **No row has been re-proven on this public wheel yet**; each row will name the
`cairndev` release it was proven on as it is re-run.

| Path | Status | Evidence so far | Proven on `cairndev` |
|---|---|---|---|
| Readiness checks and verdict (`format`, `dedup`, `decontam`, `pii_scan`, `profile`, `provenance`, `scan_all`, `gate`) | SUPPORTED | deployment runs before the public 0.1 releases; pack tests | not yet |
| Admission and cleaning stages on the builtin engine (`ingest.normalize` → `dedup.*` → `decontam.apply` → `pii.redact` → `profile.tokens` → `export.jsonl`) | SUPPORTED | deployment runs before the public 0.1 releases; pack tests run the chain through the real builtin engine | not yet |
| `dataset/split` (hash path; sklearn with `[split]`) | SUPPORTED | deployment runs before the public 0.1 releases; pack tests | not yet |
| `dataset/curate` (LLM judge) | EXPERIMENTAL | pack tests against a stub gateway; exercised less often | not yet |
| `dataset/replay` and the capture projections | EXPERIMENTAL | pack tests | not yet |

Design notes: [`DESIGN.md`](DESIGN.md). Changes: [`CHANGELOG.md`](CHANGELOG.md).

Apache-2.0 — see [`LICENSE`](LICENSE).
