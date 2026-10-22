# cairn-contracts-training-stages

Data-only contract definitions for [cairn](https://github.com/cairndev-sh/cairn): the stage contracts of the `sft-train` flow's kernels.

Definitions: `training.stage.sample_fidelity/1.0.0`, `training.stage.train_qlora/1.0.0`, `training.stage.eval_checkpoints/1.0.0`, `training.stage.package_adapter/1.0.0` and their schemas, plus
`training.stage.train_qlora/1.1.0` with its 1.1.0 configuration (the governed splits and the trainer knobs).

No Python package and no entry points: cairn discovers the descriptor (`cairn_contracts/cairn-contracts-training-stages/contracts.toml`) from the installed distribution. A published definition is never edited; a change ships as a new revision. Implemented by [`cairn-operator-training`](../cairn-operator-training).

Apache-2.0 — see [`LICENSE`](LICENSE).
