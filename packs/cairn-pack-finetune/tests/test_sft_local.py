"""finetune/sft-local: `sft-own-box` with laptop defaults, and the `finetune-local` profile
(the laptop demo, demos/laptop/ in this repository).

Hermetic, like `test_sft_box.py` (the `env` fixture in `conftest.py`, from which it reuses the
fake box engine and the bridge): the pipeline goes through the core's own launch door and
in-process bridge with the stub trainer. The real run, with a real LoRA trainer on a tiny model,
is `demos/laptop/run.sh`, which CI runs on macOS.
"""
from __future__ import annotations

import copy
import tomllib

import pytest
import yaml
from test_sft_box import PACK, FakeBoxEngine, _bridge, _Capture, _INPUTS, _plan, _sft, _status, _wait

LOCAL_PROFILE = PACK / "profiles" / "finetune-local.toml"

#: What `sft-local` sets differently from `sft-own-box`, and nothing else: the laptop defaults.
_INPUT_DEFAULTS = {
    "base_model": "HuggingFaceTB/SmolLM2-135M", "method": "lora", "max_steps": 30, "lora_r": 8,
    "lora_alpha": 16, "max_seq_len": 256, "per_device_batch_size": 4, "gradient_accumulation_steps": 1,
    "gradient_checkpointing": False, "eval_steps": 10, "save_steps": 10, "smoke_rows": 16,
    "smoke_steps": 2, "eval_preset": "heldout-loss",
}
#: Input descriptions that name the default they changed.
_INPUT_DESCRIPTIONS = ("base_model", "method")
_TIMEOUTS = {"smoke": "15m", "train": "90m", "eval": "30m"}   # own-box: 30m, 12h, 3h
_BUDGET = {"usd": 5}                                           # own-box: 25, sized to a priced box


def _pipeline(name: str) -> dict:
    return yaml.safe_load((PACK / "pipelines" / f"{name}.yaml").read_text(encoding="utf-8"))


def _own_box_with_laptop_defaults() -> dict:
    """`sft-own-box` as `sft-local` must read: the documented defaults applied, nothing else."""
    d = copy.deepcopy(_pipeline("sft-own-box"))
    for key, value in _INPUT_DEFAULTS.items():
        assert d["inputs"][key]["default"] != value, f"{key} is not a laptop default"
        d["inputs"][key]["default"] = value
    for key in _INPUT_DESCRIPTIONS:
        d["inputs"][key]["description"] = _pipeline("sft-local")["inputs"][key]["description"]
    for step in d["steps"]:
        if step.get("id") in _TIMEOUTS:
            step["timeout"] = _TIMEOUTS[step["id"]]
        if step.get("id") == "eval":
            step["with"]["time_budget_s"] = 1800
    d["governance"]["budget"] = _BUDGET
    return d


def test_sft_local_is_sft_own_box_with_the_laptop_defaults():
    """There is no include between pipeline files, so `sft-local.yaml` is a copy: this pins it to
    `sft-own-box.yaml` key for key (gates, legs, retries, series, approvers, outputs) except the
    defaults above, and the laptop demo's event trigger, which only `sft-local` declares."""
    local, own = _pipeline("sft-local"), _own_box_with_laptop_defaults()
    assert local["name"] == "sft-local" and own["name"] == "sft-own-box"
    for key in ("name", "description", "outcome"):
        local.pop(key), own.pop(key)
    assert local.pop("triggers")[0]["topic"] == "dataset.prepared" and "triggers" not in own
    assert local == own


def test_sft_local_plans_the_same_legs_and_gates_and_no_compute_step():
    local, own = _plan("sft-local"), _plan("sft-own-box")
    assert not any(s.ref.startswith("task:compute/") or s.finalizer for s in local.steps)
    assert [(s.id, s.ref, s.depends_on) for s in local.steps] == [(s.id, s.ref, s.depends_on) for s in own.steps]
    assert [s.id for s in local.steps] == ["approve-run", "smoke", "smoke-ok", "train", "eval", "improved",
                                          "promote-model", "promote"]


def test_the_eval_leg_runs_the_heldout_loss_preset_and_its_time_budget_is_its_timeout():
    ev = next(s for s in _plan("sft-local").steps if s.id == "eval")
    assert ev.timeout_s == 30 * 60 and ev.inputs["time_budget_s"] == ev.timeout_s
    assert ev.inputs["retry"] == {"max_attempts": 2}
    assert ev.inputs["preset"] == "$input.eval_preset"
    assert _pipeline("sft-local")["inputs"]["eval_preset"]["default"] == "heldout-loss"


def test_the_laptop_never_asks_for_cuda():
    """qlora is 4-bit on CUDA; the eval leg refuses it elsewhere (cairn-extensions#52). The
    default method is the plain LoRA the CPU/MPS trainer runs."""
    assert _pipeline("sft-local")["inputs"]["method"]["default"] == "lora"


def test_the_person_gates_are_the_same_as_own_boxs():
    steps = {s.id: s for s in _plan("sft-local").steps}
    assert steps["approve-run"].depends_on == []
    assert steps["promote-model"].depends_on == ["improved"] and steps["promote"].depends_on == ["promote-model"]
    assert steps["smoke"].depends_on == ["approve-run"] and steps["improved"].depends_on == ["train", "eval"]


# ── the finetune-local profile ────────────────────────────────────────────────

@pytest.fixture
def local_profile(monkeypatch):
    for name in ("CAIRN_DATAFLOW_ENGINE", "CAIRN_DATAFLOW_TARGET", "CAIRN_ML_ENGINE", "CAIRN_ML_TARGET",
                 "CAIRN_ARTIFACT_STORE", "CAIRN_REMOTE_EXEC_MODE", "CAIRN_TRAIN_IMAGE"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv("OBS_" + name.removeprefix("CAIRN_"), raising=False)
    monkeypatch.setenv("CAIRN_PROFILE", str(LOCAL_PROFILE))


def test_the_local_profile_loads_and_names_only_backends_the_core_declares(local_profile):
    from cairn.core.pack.profile import load_profile_from_env
    from cairn.core.profile_backends import BACKENDS

    profile = load_profile_from_env()
    backends = dict(profile.backends)
    assert set(backends) <= {f.key for f in BACKENDS}
    assert backends == tomllib.loads(LOCAL_PROFILE.read_text(encoding="utf-8"))["profile"]["backends"]
    assert "ml_target" not in backends and "remote_exec_mode" not in backends
    # what `secure` adds to the default profile, bound here because a pack cannot `extends` it
    assert profile.bindings["audit.sink"] == "audit/store" and profile.bindings["redact.text"] == "redact/text"


def test_the_local_profile_runs_every_leg_on_this_machine(local_profile):
    from cairn.core.backend_selection import selected
    from cairn.core.dataflow.engine_registry import resolve_engine, selected_engine

    assert selected_engine("data") == ("builtin", "")
    assert selected_engine("ml") == ("builtin", "")
    assert selected("artifact_store") == "s3"
    assert type(resolve_engine(kind="ml")).__name__ == "BuiltinLocalEngine"
    assert type(resolve_engine(kind="data")).__name__ == "BuiltinLocalEngine"


# ── the launch door and the in-process bridge ─────────────────────────────────

async def test_the_launch_door_admits_sft_local_and_prices_nothing(env, monkeypatch, local_profile):
    from cairn.core.dispatch.pipeline_launch import launch_pipeline_def

    bridge = _Capture()
    await launch_pipeline_def(_sft("sft-local"), inputs=_INPUTS, bridge=bridge)
    estimate = (bridge.kw["admission"] or {}).get("cost_estimate") or {}
    assert not estimate.get("estimate_usd"), estimate   # a laptop leg is unpriced (cairndev-sh/cairn#250, #251)
    assert env.timeline == [], "nothing starts at the launch door"


async def test_sft_local_runs_the_stub_path_and_a_stub_adapter_is_not_promoted(env, monkeypatch, local_profile):
    """The same stub run as `sft-own-box`'s: three legs, no EC2 call, and `improved` stops a stub
    adapter (it trains and measures nothing)."""
    engine = FakeBoxEngine(env.box, env.timeline, monkeypatch)
    b, audits = await _bridge(engine, monkeypatch)
    try:
        from cairn.core.dispatch.pipeline_launch import launch_pipeline_def

        rec = await launch_pipeline_def(_sft("sft-local"), inputs=_INPUTS, bridge=b)
        pid = rec.pipeline_id
        await _wait(b.store, pid, _status("approve-run", "awaiting_approval"))
        assert env.timeline == [], "nothing runs before the approval"
        await b.advance_step(pid, "approve-run", approver="ml-owner")
        rec = await _wait(b.store, pid, lambda r: r.status in ("done", "failed", "cancelled"))

        assert env.timeline == ["submit:smoke", "submit:train", "submit:eval"], "no EC2 call"
        assert rec.status == "failed" and rec.step_state["improved"].status != "done"
        assert set(rec.step_state) == {s.id for s in _plan("sft-local").steps}
        assert getattr(rec.step_state.get("promote"), "status", "pending") == "pending"
    finally:
        b.stop()
