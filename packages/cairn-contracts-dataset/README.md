# cairn-contracts-dataset

Data-only contract definitions for [cairn](https://github.com/cairndev-sh/cairn): the `dataset.*` capability contracts.

Definitions: `dataset.decontam/1.0.0`, `dataset.provenance/1.0.0`, `dataset.sample/1.0.0`, `dataset.gate/1.0.0`, `dataset.publish_fingerprints/1.0.0`, `dataset.anonymize/1.0.0`, `dataset.split/1.0.0`, `dataset.pii_scan/1.0.0`, `dataset.prep_gate/1.0.0`, `dataset.prep_review/1.0.0`, `dataset.replay/1.0.0` and their input, configuration and output schemas.

No Python package and no entry points: cairn discovers the descriptor (`cairn_contracts/cairn-contracts-dataset/contracts.toml`) from the installed distribution. A published definition is never edited; a change ships as a new revision. Implemented by [`cairn-operator-dataset-checks`](../cairn-operator-dataset-checks).

Apache-2.0 — see [`LICENSE`](LICENSE).
