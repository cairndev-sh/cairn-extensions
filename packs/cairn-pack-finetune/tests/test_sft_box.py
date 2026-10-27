"""finetune/sft starts and stops its GPU box, and the approver sees the estimate
(cairn-extensions#26, #27, #42, #41; cairndev-sh/cairn ADR-0151 D4, D9, D11).

The plan: `compute/acquire` after `approve-run` and before `smoke`; `compute/release` right after
the last box leg (`generate`, or `ifc` when the launch names an IFC schema: #41); a `finalizer:`
`compute/release` as the backstop. Then the pipeline on
the core's own launch door and in-process bridge, with this pack's `sft` definition and the
`finetune-ssh` profile: the door prices the ml legs on the `gpu` connection and the estimate is in
`approve-run`'s evidence; the box is started after the approval and stopped before a person
decides on promotion; a cancel mid-train cancels the job on the box before the box stops; a
failed leg still stops the box.

Hermetic (the `env` fixture in `conftest.py`): EC2 is a fake behind `CAIRN_EC2_CLIENT_FACTORY`
(the core's documented dev seam), the `gpu` connection lives in a throwaway SQLite connection
store, its SSH port is a local socket, and the ml legs run on a fake engine. No AWS call and no SSH connection is made. `sft-own-box`, the same
pipeline without the box steps, is pinned to `sft` here and runs the same stub path with no EC2 call.
"""
from __future__ import annotations

import asyncio
import tomllib
import types
from pathlib import Path

import pytest
import yaml

PACK = Path(__file__).resolve().parents[1]
PROFILE = PACK / "profiles" / "finetune-ssh.toml"
_FLOW = "flow:finetune/sft-train"
_EVAL = "flow:finetune/sft-eval"
_GENERATE = "flow:finetune/sft-generate"
_IFC = "flow:finetune/ifc-validate"
#: The ml legs, each priced on the box: the two trainer legs, the evaluation (#42), the generation
#: eval and the IFC validator after it (#41; priced whether or not a launch routes it off).
_ML = (_FLOW, _EVAL, _GENERATE, _IFC)


#: The steps `sft` has and `sft-own-box` does not: the box cairn starts and stops (#26).
_BOX_STEPS = {"acquire", "release-box", "release"}


def _pipeline(name: str = "sft") -> dict:
    return yaml.safe_load((PACK / "pipelines" / f"{name}.yaml").read_text(encoding="utf-8"))


def _plan(name: str = "sft"):
    from cairn.core.engine.pipeline import parse_pipeline_def, plan_pipeline

    return plan_pipeline(parse_pipeline_def((PACK / "pipelines" / f"{name}.yaml").read_text(encoding="utf-8"),
                                            pack="finetune"))


# ── the plan ──────────────────────────────────────────────────────────────────

def test_the_box_is_started_after_approval_and_stopped_after_the_last_gpu_leg():
    plan = _plan()
    steps = {s.id: s for s in plan.steps}
    order = [s.id for s in plan.steps if not s.finalizer]
    assert order == ["approve-run", "acquire", "smoke", "smoke-ok", "train", "eval", "generate", "ifc",
                     "release-box", "improved", "promote-model", "promote"]
    assert steps["acquire"].ref == "task:compute/acquire" and steps["acquire"].depends_on == ["approve-run"]
    assert steps["smoke"].depends_on == ["acquire"]
    gpu_legs = [s.id for s in plan.steps if s.ref in _ML]
    assert gpu_legs == ["smoke", "train", "eval", "generate", "ifc"]
    assert steps["eval"].ref == _EVAL and steps["eval"].depends_on == ["train"]
    assert steps["generate"].ref == _GENERATE and steps["generate"].depends_on == ["eval"]
    assert steps["ifc"].ref == _IFC and steps["ifc"].depends_on == ["generate"] and steps["ifc"].when
    # the release follows the last box leg directly, before the person gate on the model
    assert steps["release-box"].ref == "task:compute/release"
    assert steps["release-box"].depends_on == ["train", "eval", "generate", "ifc"]
    assert {"eval", "generate", "ifc", "release-box"} <= set(steps["improved"].depends_on)
    assert order.index("release-box") < order.index("promote-model")
    for s in (steps["acquire"], steps["release-box"]):
        assert s.inputs["connection"] == "$input.compute_connection"


def test_a_finalizer_release_is_the_backstop_on_every_outcome():
    """Wired from `$input`, never from `$steps.acquire`: it must resolve when acquire never ran."""
    finals = [s for s in _plan().steps if s.finalizer]
    assert [(s.id, s.ref, s.inputs) for s in finals] == [
        ("release", "task:compute/release", {"connection": "$input.compute_connection"})]


def test_both_releases_are_the_operator_the_core_cancels_jobs_before():
    """ADR-0151 D9 finds the box a pipeline releases by the `compute.release` capability."""
    from cairn.core.operator import default_registry, load_all_operators

    load_all_operators()
    for name in ("compute/acquire", "compute/release"):
        assert default_registry.get(name) is not None, name
    assert "compute.release" in default_registry.get_metadata("compute/release").provides_capabilities
    assert default_registry.get_metadata("compute/acquire").risk == "paid_compute"


def test_the_lease_covers_the_retried_worst_case_plus_a_margin():
    """A sweep must never stop the box under a retried leg: the lease is every attempt of every
    ml leg, 2 × (30m + 12h + 3h + 1h + 15m) = 33.5h, plus a margin of at most two hours (an
    orphaned box bills until the lease expires)."""
    plan = _plan()
    worst_h = sum(s.timeout_s * s.inputs["retry"]["max_attempts"] for s in plan.steps if s.ref in _ML) / 3600
    max_hours = next(s for s in plan.steps if s.id == "acquire").inputs["max_hours"]
    assert worst_h == 33.5 and max_hours == 35
    assert worst_h < max_hours <= worst_h + 2, (worst_h, max_hours)


def test_compute_connection_defaults_to_the_profiles_ml_target():
    """ADR-0151 D11: the box acquire starts must be the one the ml legs run on. cairndev does not
    check it, so the shipped default and the shipped profile must agree."""
    ml_target = tomllib.loads(PROFILE.read_text(encoding="utf-8"))["profile"]["backends"]["ml_target"]
    assert _pipeline()["inputs"]["compute_connection"]["default"] == ml_target == "gpu"


def test_the_budget_covers_the_documented_worst_case():
    """#27, #42, #41: (30m + 12h + 3h + 1h + 15m) × 2 attempts at the example $0.66/h fits
    `governance.budget.usd`."""
    plan = _plan()
    attempts = {s.id: s.inputs["retry"]["max_attempts"] for s in plan.steps if s.ref in _ML}
    hours = sum(s.timeout_s / 3600 * attempts[s.id] for s in plan.steps if s.ref in _ML)
    assert hours == 33.5 and round(0.66 * hours, 2) == 22.11 <= _pipeline()["governance"]["budget"]["usd"]


def test_the_eval_leg_is_priced_retried_and_its_time_budget_is_its_timeout():
    """#42: the evaluation is an ml leg like the trainer's (a timeout, two attempts), and the stage's
    `time_budget_s` (the time probe's ceiling; a stage cannot read its leg's timeout) equals it."""
    for name in ("sft", "sft-own-box"):
        ev = next(s for s in _plan(name).steps if s.id == "eval")
        assert ev.timeout_s == 3 * 3600 and ev.inputs["retry"] == {"max_attempts": 2}, name
        assert ev.inputs["time_budget_s"] == ev.timeout_s, name


def test_the_generation_legs_are_priced_and_retried_and_generates_time_budget_is_its_timeout():
    """#41: `generate` is an ml leg like `eval` (a timeout, two attempts, `time_budget_s` equal to the
    timeout); `ifc` is a 15-minute ml leg with two attempts, priced even when routed off."""
    for name in ("sft", "sft-own-box"):
        steps = {s.id: s for s in _plan(name).steps}
        gen, ifc = steps["generate"], steps["ifc"]
        assert gen.timeout_s == 3600 and gen.inputs["retry"] == {"max_attempts": 2}, name
        assert gen.inputs["time_budget_s"] == gen.timeout_s, name
        assert ifc.timeout_s == 900 and ifc.inputs["retry"] == {"max_attempts": 2}, name


# ── sft-own-box: the same pipeline on a box cairn does not start or stop ──────

def _without_the_box(d: dict) -> dict:
    """`sft` as `sft-own-box` must read: the box steps and the `compute_connection` input
    removed, and the dependencies that ran through them joined up again."""
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


def test_sft_own_box_is_sft_without_the_box_steps():
    """There is no include between pipeline files, so `sft-own-box.yaml` is a copy: this pins it
    to `sft.yaml` key for key (gates, legs, timeouts, retries, budget, outputs) except the box."""
    own, sft = _pipeline("sft-own-box"), _without_the_box(_pipeline("sft"))
    assert own["name"] == "sft-own-box" and sft["name"] == "sft"
    for key in ("name", "description", "outcome"):
        own.pop(key), sft.pop(key)
    assert own == sft


def test_sft_own_box_plans_the_same_legs_and_gates_and_no_compute_step():
    own, sft = _plan("sft-own-box"), _plan("sft")
    assert not any(s.ref.startswith("task:compute/") or s.finalizer for s in own.steps)
    assert [(s.id, s.ref, s.timeout_s) for s in own.steps] == [
        (s.id, s.ref, s.timeout_s) for s in sft.steps if s.id not in _BOX_STEPS]
    deps = {s.id: s.depends_on for s in own.steps}
    assert deps["smoke"] == ["approve-run"] and deps["improved"] == ["train", "eval", "generate", "ifc"]


def test_the_presets_are_the_same_for_both_pipelines():
    """A preset file names one pipeline: the `sft-own-box` copy must carry the same presets."""
    sft = yaml.safe_load((PACK / "presets" / "ifc-bim-qwen3-4b.yaml").read_text(encoding="utf-8"))
    own = yaml.safe_load((PACK / "presets" / "ifc-bim-qwen3-4b-own-box.yaml").read_text(encoding="utf-8"))
    assert (sft.pop("pipeline"), own.pop("pipeline")) == ("finetune/sft", "finetune/sft-own-box")
    assert own == sft
    declared = set(_pipeline("sft-own-box")["inputs"])
    assert all(set(p["inputs"]) <= declared for p in own["presets"])


# ── the launch door and the in-process bridge ─────────────────────────────────

class FakeBoxEngine:
    """The ml engine on the box: a job per submission, which runs the leg's flow with the stub
    trainer on the core runner in its own job directory, as a new SSH job does. `hold` keeps the
    named leg's job running until it is cancelled; `fail` ends it with a clean non-zero exit."""

    name = "builtin-ssh"
    carries_stage_contracts = True   # as the builtin SSH engine: the stage outputs are re-checked

    def __init__(self, root: Path, timeline: list[str], monkeypatch, *, hold: str = "", fail: str = "") -> None:
        self.root, self.timeline, self.monkeypatch = root, timeline, monkeypatch
        self.hold, self.fail = hold, fail
        self.jobs: dict[str, tuple] = {}
        self.cancelled: list[str] = []

    def validate(self, spec) -> list[str]:
        return []

    async def submit(self, spec, ctx):
        from cairn.core.dataflow.engine import EngineJobHandle, EngineJobStatus
        from cairn.core.dataflow.runner import run_dataflow

        stages = {st.stage for st in spec.stages}
        if "eval.lm_eval" in stages:
            step = "eval"
        elif "eval.generate" in stages:
            step = "generate"
        elif "eval.ifc_validate" in stages:
            step = "ifc"
        else:
            step = "smoke" if any((st.config or {}).get("fidelity_rows") for st in spec.stages) else "train"
        job = f"job-{step}"
        self.timeline.append(f"submit:{step}")
        status, result = EngineJobStatus(state="running"), None
        if step == self.fail:
            status = EngineJobStatus(state="failed", error="remote command exit 1: the trainer failed")
        elif step != self.hold:
            self.monkeypatch.setenv("CAIRN_PARENT_RUN", spec.parent_run)
            result = await asyncio.to_thread(run_dataflow, spec, self.root / job)
            status = EngineJobStatus(state="succeeded")
        self.jobs[job] = (step, status, result)
        return EngineJobHandle(job_id=job, engine=self.name)

    async def poll(self, handle):
        from cairn.core.dataflow.engine import EngineJobStatus

        if handle.job_id in self.cancelled:
            return EngineJobStatus(state="cancelled", error="cancelled")
        return self.jobs[handle.job_id][1]

    async def result(self, handle):
        return self.jobs[handle.job_id][2]

    async def cancel(self, handle):
        self.timeline.append(f"cancel:{self.jobs[handle.job_id][0]}")
        self.cancelled.append(handle.job_id)


_INPUTS = {"train_version": "dataset:demo-train@1", "val_version": "dataset:demo-val@1",
           "test_version": "dataset:demo-test@1", "base_model": "tiny/model", "stub": True}


def _sft(name: str = "sft"):
    from cairn.core.engine.pipeline import parse_pipeline_def

    return parse_pipeline_def((PACK / "pipelines" / f"{name}.yaml").read_text(encoding="utf-8"), pack="finetune")


class _Capture:
    """A bridge that records what the launch door hands it, and starts nothing."""

    async def start_pipeline(self, plan, **kw):
        self.plan, self.kw = plan, kw
        return types.SimpleNamespace(pipeline_id="pl-captured", plan=plan, **kw)


async def test_the_launch_door_prices_the_gpu_legs_and_approve_run_shows_the_estimate(env):
    from cairn.core.dispatch.leg_hooks import estimate_evidence
    from cairn.core.dispatch.pipeline_launch import launch_pipeline_def

    bridge = _Capture()
    await launch_pipeline_def(_sft(), inputs=_INPUTS, bridge=bridge)
    admission = bridge.kw["admission"]
    est = admission["cost_estimate"]
    assert est["estimate_usd"] == pytest.approx(22.11)
    legs = {e["step"]: e for e in est["legs"]}
    assert set(legs) == {"smoke", "train", "eval", "generate", "ifc"}
    assert {legs[s]["runs"] for s in legs} == {2}
    assert legs["train"]["connection"] == "gpu" and legs["train"]["hourly_usd"] == 0.66
    assert env.timeline == [], "nothing starts at the launch door"

    rec = types.SimpleNamespace(admission=admission, plan=bridge.plan)
    shown = estimate_evidence(rec, "approve-run")
    assert shown["estimate_usd"] == pytest.approx(22.11)
    assert "smoke: $0.66/h × 0.5h on gpu × 2 (2 attempts)" in shown["estimate_basis"]
    assert "train: $0.66/h × 12h on gpu × 2 (2 attempts)" in shown["estimate_basis"]
    assert "eval: $0.66/h × 3h on gpu × 2 (2 attempts) = $3.96" in shown["estimate_basis"]
    assert "generate: $0.66/h × 1h on gpu × 2 (2 attempts) = $1.32" in shown["estimate_basis"]
    assert "ifc: $0.66/h × 0.25h on gpu × 2 (2 attempts) = $0.33" in shown["estimate_basis"]
    # only the gate before the first priced leg carries it
    assert estimate_evidence(rec, "smoke-ok") == {} and estimate_evidence(rec, "promote-model") == {}


async def test_a_box_dearer_than_the_budget_allows_is_refused_before_any_record(env):
    """$1.50/h × 33.5 box-hours = $50.25 > $25: refused at the door, nothing started."""
    from cairn.core.admission_door import AdmissionRefusedError
    from cairn.core.connections import SqliteConnectionStore, default_connection_store
    from cairn.core.dispatch.pipeline_launch import launch_pipeline_def

    store = default_connection_store()
    assert isinstance(store, SqliteConnectionStore)
    conn = await store.get_by_name("gpu")
    await store.update(conn.id, config={**conn.config, "hourly_usd": "1.5"})
    bridge = _Capture()
    with pytest.raises(AdmissionRefusedError) as refused:
        await launch_pipeline_def(_sft(), inputs=_INPUTS, bridge=bridge)
    assert "estimated at $50.25" in str(refused.value)
    assert "the budget has $25.00 of $25.00 left in series 'sft'" in str(refused.value)
    assert not hasattr(bridge, "plan") and env.timeline == []


async def _wait(store, pid, pred, timeout=20.0):
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        rec = await store.get(pid)
        if pred(rec):
            return rec
        if loop.time() > deadline:
            raise AssertionError(f"stuck: {rec.status!r} " + str({k: v.status for k, v in rec.step_state.items()}))
        await asyncio.sleep(0.05)


def _status(step: str, status: str):
    return lambda r: getattr(r.step_state.get(step), "status", "") == status


class _Runner:
    default_environment = "production"

    async def run(self, *a, **k):
        return []


async def _bridge(engine, monkeypatch):
    from cairn.core.dispatch.dispatcher import RunDispatcher
    from cairn.core.dispatch.pipeline_bridge import PipelineBridge
    from cairn.core.storage import build_storage
    from cairn.core.storage.pipeline_store import InMemoryPipelineStore

    b = PipelineBridge(RunDispatcher(_Runner(), backend=build_storage()), store=InMemoryPipelineStore())
    monkeypatch.setattr(b, "_leg_engine", lambda s: engine)
    audits: list[tuple] = []
    real_audit = b._audit

    async def _spy(rec, step_id, event, detail):
        audits.append((step_id, event, dict(detail or {})))
        return await real_audit(rec, step_id, event, detail)

    monkeypatch.setattr(b, "_audit", _spy)
    await b.start()
    return b, audits


async def _launch_and_approve(b):
    from cairn.core.dispatch.pipeline_launch import launch_pipeline_def

    rec = await launch_pipeline_def(_sft(), inputs=_INPUTS, bridge=b)
    pid = rec.pipeline_id
    await _wait(b.store, pid, _status("approve-run", "awaiting_approval"))
    await b.advance_step(pid, "approve-run", approver="ml-owner")
    return pid


async def test_the_stub_run_starts_the_box_after_approval_and_stops_it_after_train(env, monkeypatch):
    """`stub: true` end to end on the bridge: the box starts only after `approve-run`, whose card
    carries the estimate; it stops right after the last box leg (`generate`, or `ifc` when it runs), before `improved` (which stops a stub
    adapter) is decided; the finalizer release finds it stopped and is a success."""
    engine = FakeBoxEngine(env.box, env.timeline, monkeypatch)
    b, audits = await _bridge(engine, monkeypatch)
    try:
        from cairn.core.dispatch.pipeline_launch import launch_pipeline_def

        rec = await launch_pipeline_def(_sft(), inputs=_INPUTS, bridge=b)
        pid = rec.pipeline_id
        await _wait(b.store, pid, _status("approve-run", "awaiting_approval"))
        assert env.timeline == [], "no box before the approval"
        (card,) = [d["gate_context"] for s, e, d in audits if s == "approve-run" and e == "awaiting"]
        assert card["evidence"]["estimate_usd"] == pytest.approx(22.11)
        assert "train: $0.66/h × 12h on gpu × 2 (2 attempts)" in card["evidence"]["estimate_basis"]

        await b.advance_step(pid, "approve-run", approver="ml-owner")
        rec = await _wait(b.store, pid, lambda r: r.status in ("done", "failed", "cancelled"))

        assert env.timeline == ["ec2:tags", "ec2:start", "submit:smoke", "submit:train", "submit:eval",
                                "submit:generate", "ec2:stop"]
        assert rec.step_state["eval"].status == "done" and rec.step_state["generate"].status == "done"
        # no `ifc_schema`: the ifc leg is routed off, and the box still stops after `generate`
        assert (rec.step_state["ifc"].status, rec.step_state["ifc"].verdict) == ("skipped", "routed_off")
        assert env.ec2.tags["cairn:managed"] and env.ec2.tags["cairn:release-after"]
        assert rec.step_state["release-box"].status == "done"
        events = [(s, e) for s, e, _ in audits]
        assert events.index(("improved", "gate_blocked")) > max(
            i for i, (s, _) in enumerate(events) if s == "release-box")
        assert rec.status == "failed" and rec.step_state["release"].status == "done"
        assert env.ec2.state == "stopped"
    finally:
        b.stop()


async def test_with_an_ifc_schema_the_ifc_leg_runs_on_the_box_before_it_stops(env, monkeypatch):
    """#41, #12: a launch that names `ifc_schema` runs the `ifc` leg after `generate`, on the same
    box, reading the samples file the generate leg wrote there; the box stops after it. On the stub
    tier the answers are placeholders, so the leg is never `checked`."""
    engine = FakeBoxEngine(env.box, env.timeline, monkeypatch)
    b, audits = await _bridge(engine, monkeypatch)
    try:
        from cairn.core.dispatch.pipeline_launch import launch_pipeline_def

        rec = await launch_pipeline_def(_sft(), inputs={**_INPUTS, "ifc_schema": "IFC4"}, bridge=b)
        pid = rec.pipeline_id
        await _wait(b.store, pid, _status("approve-run", "awaiting_approval"))
        await b.advance_step(pid, "approve-run", approver="ml-owner")
        rec = await _wait(b.store, pid, lambda r: r.status in ("done", "failed", "cancelled"))

        assert env.timeline == ["ec2:tags", "ec2:start", "submit:smoke", "submit:train", "submit:eval",
                                "submit:generate", "submit:ifc", "ec2:stop"]
        assert rec.step_state["ifc"].status == "done"
        _, _, ifc = engine.jobs["job-ifc"]
        assert ifc.stats["eval.ifc_validate.checked"] is False and ifc.stats["eval.ifc_validate.rows"] == 6
        assert rec.step_state["release-box"].status == "done" and env.ec2.state == "stopped"
    finally:
        b.stop()


async def test_a_cancel_mid_train_cancels_the_job_on_the_box_before_the_box_stops(env, monkeypatch):
    engine = FakeBoxEngine(env.box, env.timeline, monkeypatch, hold="train")
    b, audits = await _bridge(engine, monkeypatch)
    try:
        pid = await _launch_and_approve(b)
        await _wait(b.store, pid, lambda r: (_status("train", "running")(r)
                                             and bool(getattr(r.step_state.get("train"), "handle", None))))
        await b.cancel_pipeline(pid)
        rec = await _wait(b.store, pid, lambda r: r.status in ("done", "failed", "cancelled"))

        assert rec.status == "cancelled" and rec.step_state["release"].status == "done"
        assert env.timeline == ["ec2:tags", "ec2:start", "submit:smoke", "submit:train",
                                "cancel:train", "ec2:stop"]
        assert env.ec2.state == "stopped"
        assert getattr(rec.step_state.get("release-box"), "status", "pending") != "done"
        (cancel,) = [d for s, e, d in audits if e == "leg_job_cancel"]
        assert cancel["target"] == "gpu" and cancel["sent"] is True and cancel["before"] == "release"
    finally:
        b.stop()


async def test_a_failed_train_leg_still_stops_the_box(env, monkeypatch):
    engine = FakeBoxEngine(env.box, env.timeline, monkeypatch, fail="train")
    b, _ = await _bridge(engine, monkeypatch)
    try:
        pid = await _launch_and_approve(b)
        rec = await _wait(b.store, pid, lambda r: r.status in ("done", "failed", "cancelled"))

        assert rec.status == "failed" and rec.step_state["release"].status == "done"
        assert env.timeline == ["ec2:tags", "ec2:start", "submit:smoke", "submit:train", "ec2:stop"]
        assert env.ec2.state == "stopped"
    finally:
        b.stop()


async def test_a_failed_eval_leg_still_stops_the_box(env, monkeypatch):
    """#42: the evaluation fails on the box (a time-probe refusal, or a crash): the pipeline fails
    before a person is asked, and the finalizer stops the box."""
    engine = FakeBoxEngine(env.box, env.timeline, monkeypatch, fail="eval")
    b, _ = await _bridge(engine, monkeypatch)
    try:
        pid = await _launch_and_approve(b)
        rec = await _wait(b.store, pid, lambda r: r.status in ("done", "failed", "cancelled"))

        assert rec.status == "failed" and rec.step_state["release"].status == "done"
        assert env.timeline == ["ec2:tags", "ec2:start", "submit:smoke", "submit:train", "submit:eval", "ec2:stop"]
        assert getattr(rec.step_state.get("promote-model"), "status", "pending") == "pending"
    finally:
        b.stop()


async def test_sft_own_box_runs_on_the_stub_with_no_ec2_call(env, monkeypatch):
    """The same stub run on `sft-own-box`: the legs run on the profile's `gpu` box (priced, so
    `approve-run` still shows the estimate), and cairn never calls EC2."""
    engine = FakeBoxEngine(env.box, env.timeline, monkeypatch)
    b, audits = await _bridge(engine, monkeypatch)
    try:
        from cairn.core.dispatch.pipeline_launch import launch_pipeline_def

        rec = await launch_pipeline_def(_sft("sft-own-box"), inputs=_INPUTS, bridge=b)
        pid = rec.pipeline_id
        await _wait(b.store, pid, _status("approve-run", "awaiting_approval"))
        (card,) = [d["gate_context"] for s, e, d in audits if s == "approve-run" and e == "awaiting"]
        assert card["evidence"]["estimate_usd"] == pytest.approx(22.11)
        await b.advance_step(pid, "approve-run", approver="ml-owner")
        rec = await _wait(b.store, pid, lambda r: r.status in ("done", "failed", "cancelled"))

        assert env.timeline == ["submit:smoke", "submit:train", "submit:eval", "submit:generate"], "no EC2 call"
        assert rec.status == "failed" and rec.step_state["improved"].status != "done"
        assert set(rec.step_state) == {s.id for s in _plan("sft-own-box").steps}
    finally:
        b.stop()
