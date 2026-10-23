# cairn-contracts-dataset-stages

Data-only contract definitions for [cairn](https://github.com/cairndev-sh/cairn): the stage contracts of the dataset admission kernels (`cairn.stage/1`, ADR-0118).

Definitions: `dataset.stage.ingest_normalize/1.0.0`, `/1.1.0` and `/1.2.0`, `dataset.stage.dedup_exact/1.0.0`, `dataset.stage.dedup_near/1.0.0`, `dataset.stage.decontam_apply/1.0.0`, `dataset.stage.pii_redact/1.0.0`, `dataset.stage.expectations_check/1.0.0`, `dataset.stage.profile_tokens/1.0.0`, `dataset.stage.export_jsonl/1.0.0` and `/1.1.0`, `dataset.stage.scan_all/1.0.0`, `dataset.stage.dataset_manifest/1.0.0` and their configuration and output schemas, and `dataset.manifest.document/1.0.0` (the `manifest.json` the manifest stage writes).

No Python package and no entry points: cairn discovers the descriptor (`cairn_contracts/cairn-contracts-dataset-stages/contracts.toml`) from the installed distribution. It is its own distribution because a core older than the `stage` kind refuses a distribution that carries one. A published definition is never edited; a change ships as a new revision. Implemented by [`cairn-operator-dataset-checks`](../cairn-operator-dataset-checks), except `dataset.stage.expectations_check`, which the pack that provides the `expectations.check` kernel implements.

Apache-2.0 — see [`LICENSE`](LICENSE).
