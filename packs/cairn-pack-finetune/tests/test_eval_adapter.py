"""finetune/eval-adapter: evaluate an adapter you already have, as one priced ml leg on a GPU box
cairn starts after approval and stops on every outcome (cairn-extensions#42).

The plan (order, the leg's timeout, retry and time budget, the lease, the budget), the own-box copy
pinned to it, the presets, the tag refusal before any box time; then the pipeline on the core's
own launch door and in-process bridge with the `finetune-ssh` profile and a fake EC2 (the fixtures
of `conftest.py`): the estimate on `approve-run`, the box started after the approval and
stopped right after `eval`, the stub evaluation stopped at `evaluated`, a measured one passing it.
Hermetic: no AWS call, no SSH connection, no model.
"""
from __future__ import annotations

import types

import pytest
import yaml

from test_sft_box import PACK, FakeBoxEngine, _bridge, _Capture, _status, _wait

_EVAL = "flow:finetune/sft-eval"
_BOX_STEPS = {"acquire", "release-box", "release"}


def _pipeline(name: str = "eval-adapter") -> dict:
    return yaml.safe_load((PACK / "pipelines" / f"{name}.yaml").read_text(encoding="utf-8"))


def _def(name: str = "eval-adapter"):
    from cairn.core.engine.pipeline import parse_pipeline_def

    return parse_pipeline_def((PACK / "pipelines" / f"{name}.yaml").read_text(encoding="utf-8"), pack="finetune")


def _plan(name: str = "eval-adapter"):
    from cairn.core.engine.pipeline import plan_pipeline

    return plan_pipeline(_def(name))


# ── the plan ──────────────────────────────────────────────────────────────────

def test_the_box_runs_only_the_eval_leg_between_approval_and_release():
    plan = _plan()
    steps = {s.id: s for s in plan.steps}
    assert [s.id for s in plan.steps if not s.finalizer] == [
        "adapter-ref", "approve-run", "acquire", "eval", "release-box", "evaluated"]
    assert steps["approve-run"].depends_on == ["adapter-ref"]
    assert steps["acquire"].ref == "task:compute/acquire" and steps["acquire"].depends_on == ["approve-run"]
    assert steps["eval"].ref == _EVAL and steps["eval"].depends_on == ["acquire"]
    assert steps["release-box"].ref == "task:compute/release" and steps["release-box"].depends_on == ["eval"]
    assert steps["evaluated"].depends_on == ["eval", "release-box"]
    assert [(s.id, s.ref, s.inputs) for s in plan.steps if s.finalizer] == [
        ("release", "task:compute/release", {"connection": "$input.compute_connection"})]


def test_the_eval_leg_is_priced_retried_and_its_time_budget_is_its_timeout():
    for name in ("eval-adapter", "eval-adapter-own-box"):
        ev = next(s for s in _plan(name).steps if s.id == "eval")
        assert ev.timeout_s == 6 * 3600 and ev.inputs["retry"] == {"max_attempts": 2}, name
        assert ev.inputs["time_budget_s"] == ev.timeout_s, name
        assert ev.inputs["adapter_uri"] == "$input.adapter_uri" and "local_adapter_dir" not in ev.inputs


def test_the_lease_and_the_budget_cover_the_retried_worst_case():
    """2 attempts × 6h = 12 box-hours: the lease is that plus at most two hours, and the budget
    holds it at the example $0.66/h ($7.92)."""
    plan = _plan()
    worst_h = sum(s.timeout_s * s.inputs["retry"]["max_attempts"] for s in plan.steps if s.ref == _EVAL) / 3600
    max_hours = next(s for s in plan.steps if s.id == "acquire").inputs["max_hours"]
    assert worst_h == 12.0 and worst_h < max_hours <= worst_h + 2
    assert round(0.66 * worst_h, 2) == 7.92 <= _pipeline()["governance"]["budget"]["usd"]


def test_a_promoted_tag_is_refused_before_any_box_time():
    """cairndev resolves `dataset:` refs in a leg's inputs, not artifact tags, and the box has no
    registry: `adapter-ref` stops a tag before `approve-run` and `acquire`."""
    from cairn.core.engine.bundle import eval_gate_predicate

    pred = next(s["predicate"] for s in _pipeline()["steps"] if s.get("gate") == "adapter-ref")
    for uri in ("hf://cairndev/qwen3-4b-ifc-bim-qlora", "cairndev/qwen3-4b-ifc-bim-qlora",
                "s3://ckpts/cairn-checkpoints/run-1/adapter/", "/home/u/adapter"):
        assert eval_gate_predicate(pred, bundle_input={"adapter_uri": uri}, steps={}), uri
    assert not eval_gate_predicate(pred, bundle_input={"adapter_uri": "ifc-bim-qwen3-4b@candidate"}, steps={})


def _without_the_box(d: dict) -> dict:
    d = yaml.safe_load(yaml.safe_dump(d))
    d["inputs"].pop("compute_connection")
    steps = [s for s in d["steps"] if s.get("id") not in _BOX_STEPS]
    for s in steps:
        if "depends_on" in s:
            s["depends_on"] = [x for dep in s["depends_on"] for x in
                               (["approve-run"] if dep == "acquire" else [] if dep == "release-box" else [dep])]
        if "show" in s:
            s["show"] = [f for f in s["show"] if f != "input.compute_connection"]
    d["steps"] = steps
    return d


def test_eval_adapter_own_box_is_eval_adapter_without_the_box_steps():
    own, full = _pipeline("eval-adapter-own-box"), _without_the_box(_pipeline())
    assert own["name"] == "eval-adapter-own-box" and full["name"] == "eval-adapter"
    for key in ("name", "description", "outcome"):
        own.pop(key), full.pop(key)
    assert own == full
    assert not any(s.ref.startswith("task:compute/") or s.finalizer for s in _plan("eval-adapter-own-box").steps)


def test_the_ifc_bim_preset_is_the_same_for_both_pipelines_and_names_a_uri():
    full = yaml.safe_load((PACK / "presets" / "ifc-bim-qwen3-4b-eval-adapter.yaml").read_text(encoding="utf-8"))
    own = yaml.safe_load((PACK / "presets" / "ifc-bim-qwen3-4b-eval-adapter-own-box.yaml").read_text(encoding="utf-8"))
    assert (full.pop("pipeline"), own.pop("pipeline")) == ("finetune/eval-adapter", "finetune/eval-adapter-own-box")
    assert own == full
    (preset,) = full["presets"]
    assert preset["name"] == "ifc-bim-qwen3-4b" and set(preset["inputs"]) <= set(_pipeline()["inputs"])
    assert preset["inputs"]["adapter_uri"] == "hf://cairndev/qwen3-4b-ifc-bim-qlora"
    assert preset["inputs"]["eval_preset"] == "forgetting-ifeval"


def test_every_preset_the_pipelines_name_ships_with_the_eval_pack(tmp_path, monkeypatch):
    """The default `eval_preset` of each pipeline loads in the eval stage: the `sft-eval` flow on the
    core runner, stub tier (an unknown preset fails the flow before anything runs)."""
    from cairn.core.dataflow import Budget, DataflowSpec
    from cairn.core.dataflow.defs import resolve_flow_ref
    from cairn.core.dataflow.runner import run_dataflow
    from cairn.core.pack import registry
    from cairn.core.pack.loader import load_all_packs

    monkeypatch.setenv("CAIRN_PACK_PATH", str(PACK))
    registry.clear()
    load_all_packs()
    try:
        for name in ("eval-adapter", "eval-adapter-own-box", "sft", "sft-own-box"):
            preset = _pipeline(name)["inputs"]["eval_preset"]["default"]
            stages = resolve_flow_ref("finetune/sft-eval", {"base_model": "tiny/model", "stub": True, "preset": preset,
                                                            "adapter_uri": "stub://adapters/x"}, kind="ml")
            stats = run_dataflow(DataflowSpec(stages=stages, budget=Budget()), tmp_path / name).stats
            assert stats["eval.lm_eval.preset"] == preset and stats["eval.lm_eval.stub"] is True, name
    finally:
        registry.clear()


# ── the launch door and the in-process bridge ─────────────────────────────────

_INPUTS = {"adapter_uri": "hf://cairndev/qwen3-4b-ifc-bim-qlora", "val_version": "dataset:demo-val@1",
           "test_version": "dataset:demo-test@1", "base_model": "tiny/model", "stub": True}


async def test_the_launch_door_prices_the_eval_leg_and_approve_run_shows_it(env):
    from cairn.core.dispatch.leg_hooks import estimate_evidence
    from cairn.core.dispatch.pipeline_launch import launch_pipeline_def

    bridge = _Capture()
    await launch_pipeline_def(_def(), inputs=_INPUTS, bridge=bridge)
    est = bridge.kw["admission"]["cost_estimate"]
    assert est["estimate_usd"] == pytest.approx(7.92)
    (leg,) = est["legs"]
    assert (leg["step"], leg["runs"], leg["connection"]) == ("eval", 2, "gpu")
    rec = types.SimpleNamespace(admission=bridge.kw["admission"], plan=bridge.plan)
    shown = estimate_evidence(rec, "approve-run")
    assert shown["estimate_usd"] == pytest.approx(7.92)
    assert "eval: $0.66/h × 6h on gpu × 2 (2 attempts) = $7.92" in shown["estimate_basis"]
    assert env.timeline == [], "nothing starts at the launch door"


async def test_the_launch_door_refuses_a_version_in_the_wrong_role(env):
    from cairn.core.admission_door import AdmissionRefusedError
    from cairn.core.dispatch.pipeline_launch import launch_pipeline_def

    bridge = _Capture()
    with pytest.raises(AdmissionRefusedError) as refused:
        await launch_pipeline_def(_def(), inputs={**_INPUTS, "test_version": "dataset:demo-train@1"}, bridge=bridge)
    assert any("test_version" in r for r in refused.value.decision.refusing_reasons)
    assert not hasattr(bridge, "plan") and env.timeline == []


async def _launch(b, inputs: dict) -> str:
    from cairn.core.dispatch.pipeline_launch import launch_pipeline_def

    rec = await launch_pipeline_def(_def(), inputs=inputs, bridge=b)
    await _wait(b.store, rec.pipeline_id, _status("approve-run", "awaiting_approval"))
    return rec.pipeline_id


async def test_the_stub_evaluation_runs_on_the_box_and_stops_at_evaluated(env, monkeypatch):
    """`stub: true`: the box starts only after `approve-run`, the evaluation runs on it and it
    stops right after; the stub's scores are placeholders, so `evaluated` fails the run (never a
    result), and the finalizer finds the box stopped."""
    engine = FakeBoxEngine(env.box, env.timeline, monkeypatch)
    b, audits = await _bridge(engine, monkeypatch)
    try:
        pid = await _launch(b, _INPUTS)
        assert env.timeline == [], "no box before the approval"
        (card,) = [d["gate_context"] for s, e, d in audits if s == "approve-run" and e == "awaiting"]
        assert card["evidence"]["estimate_usd"] == pytest.approx(7.92)
        assert "input.adapter_uri" in card["fields"]
        await b.advance_step(pid, "approve-run", approver="ml-owner")
        rec = await _wait(b.store, pid, lambda r: r.status in ("done", "failed", "cancelled"))

        assert env.timeline == ["ec2:tags", "ec2:start", "submit:eval", "ec2:stop"]
        assert rec.step_state["eval"].status == "done" and rec.step_state["release-box"].status == "done"
        assert rec.step_state["evaluated"].verdict == "gate_blocked"
        assert rec.status == "failed" and rec.step_state["release"].status == "done"
        assert env.ec2.state == "stopped"
    finally:
        b.stop()


class MeasuredBoxEngine(FakeBoxEngine):
    """The box engine, with an `eval` job that measured: its stats are the stage's, `checked`."""

    async def submit(self, spec, ctx):
        from cairn.core.dataflow import DataflowResult
        from cairn.core.dataflow.engine import EngineJobHandle, EngineJobStatus

        self.timeline.append("submit:eval")
        self.submitted = {k: v for k, v in spec.stages[0].config.items()}
        stats = {"eval.lm_eval.checked": True, "eval.lm_eval.delta_mmlu_acc": -0.004,
                 "eval.lm_eval.candidate_test_loss": 0.218, "eval.lm_eval.base_test_loss": 1.955}
        self.jobs["job-eval"] = ("eval", EngineJobStatus(state="succeeded"),
                                 DataflowResult(output_artifacts=[], stats=stats, summary="ok"))
        return EngineJobHandle(job_id="job-eval", engine=self.name)


async def test_a_measured_evaluation_passes_evaluated(env, monkeypatch):
    """An `eval` job that measured (`checked`): `evaluated` passes and the run is done, with the
    scores as findings and the leg's config as the plan wired it."""
    engine = MeasuredBoxEngine(env.box, env.timeline, monkeypatch)
    b, _ = await _bridge(engine, monkeypatch)
    try:
        pid = await _launch(b, {**_INPUTS, "stub": False})
        await b.advance_step(pid, "approve-run", approver="ml-owner")
        rec = await _wait(b.store, pid, lambda r: r.status in ("done", "failed", "cancelled"))

        assert rec.status == "done", {k: (v.status, v.verdict) for k, v in rec.step_state.items()}
        assert env.timeline == ["ec2:tags", "ec2:start", "submit:eval", "ec2:stop"]
        assert engine.submitted["adapter_uri"] == _INPUTS["adapter_uri"]
        assert engine.submitted["time_budget_s"] == 21600 and engine.submitted["preset"] == "forgetting-ifeval"
        assert engine.submitted["val_uri"].startswith("file://") and "#sha256=" in engine.submitted["test_uri"]
        run = await b.dispatcher.store.get(rec.step_state["eval"].run_id)
        findings = run.result_payloads[0]["output"]["findings"]
        assert findings["eval_lm_eval_checked"] is True and findings["eval_lm_eval_candidate_test_loss"] == 0.218
    finally:
        b.stop()


async def test_a_failed_eval_leg_still_stops_the_box(env, monkeypatch):
    engine = FakeBoxEngine(env.box, env.timeline, monkeypatch, fail="eval")
    b, _ = await _bridge(engine, monkeypatch)
    try:
        pid = await _launch(b, _INPUTS)
        await b.advance_step(pid, "approve-run", approver="ml-owner")
        rec = await _wait(b.store, pid, lambda r: r.status in ("done", "failed", "cancelled"))
        assert rec.status == "failed" and rec.step_state["release"].status == "done"
        assert env.timeline == ["ec2:tags", "ec2:start", "submit:eval", "ec2:stop"]
    finally:
        b.stop()
