"""The `finetune-ssh` profile, loaded through the core's own config plane (cairn-extensions#7).

What it must select, and so what a control plane started with `CAIRN_PROFILE=<this file>`
does: the data leg on the control plane's builtin engine, the `sft-train` ml leg on the builtin
engine's SSH dispatch to the `gpu` connection in venv mode, and the s3 artifact store. Nothing
connects anywhere: the engine is built, never submitted to. The header must keep naming what a
deployment has to set, so the profile cannot drift from its instructions unnoticed."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
PROFILE = PACK / "profiles" / "finetune-ssh.toml"

#: Environment the profile's selections can be overridden by; cleared so the profile decides.
_OVERRIDES = ("CAIRN_DATAFLOW_ENGINE", "CAIRN_DATAFLOW_TARGET", "CAIRN_ML_ENGINE", "CAIRN_ML_TARGET",
              "CAIRN_ARTIFACT_STORE", "CAIRN_REMOTE_EXEC_MODE", "CAIRN_TRAIN_IMAGE")


@pytest.fixture(autouse=True)
def _profile(monkeypatch):
    for name in _OVERRIDES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv("OBS_" + name.removeprefix("CAIRN_"), raising=False)
    monkeypatch.setenv("CAIRN_PROFILE", str(PROFILE))


def test_the_profile_loads_and_names_only_backends_the_core_declares():
    from cairn.core.pack.profile import load_profile_from_env
    from cairn.core.profile_backends import BACKENDS

    backends = dict(load_profile_from_env().backends)
    assert set(backends) <= {f.key for f in BACKENDS}, set(backends) - {f.key for f in BACKENDS}
    assert backends == tomllib.loads(PROFILE.read_text(encoding="utf-8"))["profile"]["backends"]


def test_the_selection_the_core_resolves():
    from cairn.core.backend_selection import provenance, selected
    from cairn.core.dataflow.engine_registry import selected_engine

    assert selected_engine("data") == ("builtin", "")     # dataset-prep, on the control plane
    assert selected_engine("ml") == ("builtin", "gpu")    # sft-train, on the `gpu` connection
    assert selected("artifact_store") == "s3"
    assert selected("remote_exec_mode") == "venv"
    assert {provenance(k) for k in ("ml_engine", "ml_target", "artifact_store", "remote_exec_mode")} == {"profile"}


def test_the_ml_leg_gets_the_ssh_engine_and_the_data_leg_the_local_one():
    """Built through the core's engine registry, as the bridge builds them (by name: a pack's
    tests import no operator package)."""
    from cairn.core.dataflow.engine_registry import resolve_engine

    ml = resolve_engine(kind="ml")
    assert (type(ml).__name__, ml.name, ml._target) == ("BuiltinSshEngine", "builtin-ssh", "gpu")
    data = resolve_engine(kind="data")
    assert (type(data).__name__, data.name) == ("BuiltinLocalEngine", "builtin")


def test_the_ssh_engine_accepts_the_sft_train_flow(monkeypatch):
    """The SSH engine checks a spec against the control plane's stage registry before it
    dispatches: every stage `sft-train` names must be there (the box installs the same)."""
    from cairn.core.dataflow import Budget, DataflowSpec
    from cairn.core.dataflow.defs import resolve_flow_ref
    from cairn.core.dataflow.engine_registry import resolve_engine
    from cairn.core.pack import registry
    from cairn.core.pack.loader import load_all_packs

    monkeypatch.setenv("CAIRN_PACK_PATH", str(PACK))
    registry.clear()
    load_all_packs()
    try:
        stages = resolve_flow_ref("finetune/sft-train", {"train_uri": "s3://b/train.jsonl",
                                                         "base_model": "Qwen/Qwen3-4B-Base"}, kind="ml")
        assert resolve_engine(kind="ml").validate(DataflowSpec(stages=stages, budget=Budget())) == []
    finally:
        registry.clear()


@pytest.mark.parametrize("needle", [
    "cairndev[ssh]",
    "cairn-operator-training[gpu]",
    "track-mlflow",
    "cairn-contracts-training",
    "CAIRN_DATAFLOW_REMOTE_CMD",
    "CAIRN_CHECKPOINT_BUCKET",
    "CAIRN_CHECKPOINT_PREFIX",
    "AWS_DEFAULT_REGION",
    "CAIRN_ARTIFACT_S3_BUCKET",
    "MLFLOW_TRACKING_URI",
    "cairndev 0.2.1, cancel and the engine's stage-checkpoint relay",
    "SSH_PRIVATE_KEY",
    "AWS_ENDPOINT_URL",
    "read access",
    "--secret PRIVATE_KEY=",
    "Venv mode only",
])
def test_the_header_names_what_a_deployment_must_set(needle):
    header = PROFILE.read_text(encoding="utf-8").split("[profile]")[0]
    assert needle in header
