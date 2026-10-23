# cairn-contracts-training

Data-only contract definitions for [cairn](https://github.com/cairndev-sh/cairn): the `train.run` and `train.teardown` capability contracts.

Definitions: `train.run/1.0.0`, `train.teardown/1.0.0` and their input, configuration and output schemas.

No Python package and no entry points: cairn discovers the descriptor (`cairn_contracts/cairn-contracts-training/contracts.toml`) from the installed distribution. A published definition is never edited; a change ships as a new revision. Implemented by [`cairn-operator-training`](../cairn-operator-training).

Apache-2.0 — see [`LICENSE`](LICENSE).
