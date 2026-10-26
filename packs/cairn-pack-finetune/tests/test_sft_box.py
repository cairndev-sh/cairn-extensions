"""finetune/sft starts and stops its GPU box, and the approver sees the estimate
(cairn-extensions#26, #27; cairndev-sh/cairn ADR-0151 D4, D9, D11).

The plan: `compute/acquire` after `approve-run` and before `smoke`; `compute/release` right after
`train`, the last GPU leg; a `finalizer:` `compute/release` as the backstop. Then the pipeline on
the core's own launch door and in-process bridge, with this pack's `sft` definition and the
`finetune-ssh` profile: the door prices the ml legs on the `gpu` connection and the estimate is in
`approve-run`'s evidence; the box is started after the approval and stopped before a person
decides on promotion; a cancel mid-train cancels the job on the box before the box stops; a
failed leg still stops the box.

Hermetic: EC2 is a fake behind `CAIRN_EC2_CLIENT_FACTORY` (the core's documented dev seam), the
`gpu` connection lives in a throwaway SQLite connection store, its SSH port is a local socket, and
the ml legs run on a fake engine. No AWS call and no SSH connection is made.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import sys
import tomllib
import types
from pathlib import Path

import pytest
import yaml

PACK = Path(__file__).resolve().parents[1]
PROFILE = PACK / "profiles" / "finetune-ssh.toml"
_FLOW = "flow:finetune/sft-train"


def _pipeline() -> dict:
    return yaml.safe_load((PACK / "pipelines" / "sft.yaml").read_text(encoding="utf-8"))


def _plan():
    from cairn.core.engine.pipeline import parse_pipeline_def, plan_pipeline

    return plan_pipeline(parse_pipeline_def((PACK / "pipelines" / "sft.yaml").read_text(encoding="utf-8"),
                                            pack="finetune"))


# ── the plan ──────────────────────────────────────────────────────────────────

def test_the_box_is_started_after_approval_and_stopped_after_the_last_gpu_leg():
    plan = _plan()
    steps = {s.id: s for s in plan.steps}
    order = [s.id for s in plan.steps if not s.finalizer]
    assert order == ["approve-run", "acquire", "smoke", "smoke-ok", "train", "release-box",
                     "improved", "promote-model", "promote"]
    assert steps["acquire"].ref == "task:compute/acquire" and steps["acquire"].depends_on == ["approve-run"]
    assert steps["smoke"].depends_on == ["acquire"]
    gpu_legs = [s.id for s in plan.steps if s.ref == _FLOW]
    assert gpu_legs == ["smoke", "train"]
    # the release follows the last GPU leg directly, before the person gate on the model
    assert steps["release-box"].ref == "task:compute/release" and steps["release-box"].depends_on == ["train"]
    assert "release-box" in steps["improved"].depends_on
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


def test_the_lease_covers_the_gpu_legs_timeouts_plus_a_margin():
    plan = _plan()
    legs_h = sum(s.timeout_s for s in plan.steps if s.ref == _FLOW) / 3600
    max_hours = next(s for s in plan.steps if s.id == "acquire").inputs["max_hours"]
    assert legs_h == 12.5
    assert legs_h < max_hours <= legs_h + 1, (legs_h, max_hours)


def test_compute_connection_defaults_to_the_profiles_ml_target():
    """ADR-0151 D11: the box acquire starts must be the one the ml legs run on. cairndev does not
    check it, so the shipped default and the shipped profile must agree."""
    ml_target = tomllib.loads(PROFILE.read_text(encoding="utf-8"))["profile"]["backends"]["ml_target"]
    assert _pipeline()["inputs"]["compute_connection"]["default"] == ml_target == "gpu"


def test_the_budget_covers_the_documented_worst_case():
    """#27: (30m + 12h) × 2 attempts at the example $0.66/h fits `governance.budget.usd`."""
    plan = _plan()
    attempts = {s.id: s.inputs["retry"]["max_attempts"] for s in plan.steps if s.ref == _FLOW}
    hours = sum(s.timeout_s / 3600 * attempts[s.id] for s in plan.steps if s.ref == _FLOW)
    assert hours == 25.0 and round(0.66 * hours, 2) == 16.5 <= _pipeline()["governance"]["budget"]["usd"]


# ── the launch door and the in-process bridge ─────────────────────────────────

class FakeEC2:
    """The EC2 client the lifecycle functions call; writes to the shared timeline."""

    def __init__(self, timeline: list[str]) -> None:
        self.state, self.timeline, self.tags = "stopped", timeline, {}

    def describe_instances(self, **kw):
        return {"Reservations": [{"Instances": [{"InstanceId": "i-0sft", "State": {"Name": self.state}}]}]}

    def start_instances(self, **kw):
        self.timeline.append("ec2:start")
        self.state = "running"
        return {"StartingInstances": [{"CurrentState": {"Name": "pending"}}]}

    def stop_instances(self, **kw):
        self.timeline.append("ec2:stop")
        self.state = "stopped"
        return {"StoppingInstances": [{"CurrentState": {"Name": "stopping"}}]}

    def create_tags(self, **kw):
        self.timeline.append("ec2:tags")
        self.tags.update({t["Key"]: t["Value"] for t in kw.get("Tags") or []})


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


@pytest.fixture
async def env(tmp_path, monkeypatch):
    """The `finetune-ssh` profile; the `gpu` connection (an EC2 instance at $0.66/h, its SSH
    port a local socket) in a throwaway store; registered versions with admitted verdicts; the
    fake EC2 behind the core's seam."""
    import cairn.core.storage.artifacts as artifacts
    from cairn.core.connections import Connection, SqliteConnectionStore
    from cairn.core.dataflow.contract import ArtifactDescriptor
    from cairn.core.pack import registry
    from cairn.core.pack.loader import load_all_packs
    from cairn.core.storage.artifacts import SqliteArtifactStore

    for name in ("CAIRN_ML_ENGINE", "CAIRN_ML_TARGET", "CAIRN_DATAFLOW_ENGINE", "CAIRN_DATAFLOW_TARGET",
                 "CAIRN_ARTIFACT_STORE", "CAIRN_REMOTE_EXEC_MODE", "CAIRN_RUNTIME", "CAIRN_STORAGE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CAIRN_PROFILE", str(PROFILE))
    monkeypatch.setenv("CAIRN_PACK_PATH", str(PACK))
    monkeypatch.setenv("CAIRN_CONNECTIONS_DB", str(tmp_path / "connections.db"))

    ssh = socket.socket()
    ssh.bind(("127.0.0.1", 0))
    ssh.listen(8)   # the kernel completes the handshake: acquire's readiness probe sees an open port
    store = SqliteConnectionStore(str(tmp_path / "connections.db"))
    await store.create(Connection(name="gpu", kind="compute.ssh", config={
        "host": "127.0.0.1", "port": str(ssh.getsockname()[1]), "user": "u",
        "instance_id": "i-0sft", "region": "eu-central-1", "hourly_usd": "0.66"}))
    close = getattr(store, "close", None)
    if callable(close):
        close()

    timeline: list[str] = []
    ec2 = FakeEC2(timeline)
    seam = types.ModuleType("_sft_box_ec2_seam")
    seam.client = lambda region="": ec2
    monkeypatch.setitem(sys.modules, seam.__name__, seam)
    monkeypatch.setenv("CAIRN_EC2_CLIENT_FACTORY", f"{seam.__name__}:client")

    versions = SqliteArtifactStore(str(tmp_path / "registry.db"))
    for name, role, n in (("train", "training", 100), ("val", "validation", 6), ("test", "test", 6)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps({"instruction": "Answer.", "input": f"{name} q{i}",
                                            "output": f"{name} a{i}"}) + "\n" for i in range(n)))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        await versions.register(ArtifactDescriptor(uri=f"file://{path}", kind="dataset", content_hash=digest),
                                producer_run_id="prepare")
        await versions.register_dataset_version(f"demo-{name}", digest, actor="prepare", role=role)
    for name in ("demo-train", "demo-val"):
        await versions.set_dataset_version_verdict(name, 1, verdict="admitted", reasons=[], report={})
    monkeypatch.setattr(artifacts, "build_artifact_store", lambda *a, **k: versions)

    import cairn.core.dispatch.leg_executor as leg_executor
    monkeypatch.setattr(leg_executor, "_POLL_INTERVAL_S", 0.05)

    registry.clear()
    load_all_packs()
    try:
        yield types.SimpleNamespace(ec2=ec2, timeline=timeline, box=tmp_path / "box")
    finally:
        registry.clear()
        await versions.close()
        ssh.close()


def _sft():
    from cairn.core.engine.pipeline import parse_pipeline_def

    return parse_pipeline_def((PACK / "pipelines" / "sft.yaml").read_text(encoding="utf-8"), pack="finetune")


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
    assert est["estimate_usd"] == pytest.approx(16.5)
    legs = {e["step"]: e for e in est["legs"]}
    assert set(legs) == {"smoke", "train"}
    assert legs["smoke"]["runs"] == legs["train"]["runs"] == 2
    assert legs["train"]["connection"] == "gpu" and legs["train"]["hourly_usd"] == 0.66
    assert env.timeline == [], "nothing starts at the launch door"

    rec = types.SimpleNamespace(admission=admission, plan=bridge.plan)
    shown = estimate_evidence(rec, "approve-run")
    assert shown["estimate_usd"] == pytest.approx(16.5)
    assert "smoke: $0.66/h × 0.5h on gpu × 2 (2 attempts)" in shown["estimate_basis"]
    assert "train: $0.66/h × 12h on gpu × 2 (2 attempts)" in shown["estimate_basis"]
    # only the gate before the first priced leg carries it
    assert estimate_evidence(rec, "smoke-ok") == {} and estimate_evidence(rec, "promote-model") == {}


async def test_a_box_dearer_than_the_budget_allows_is_refused_before_any_record(env):
    """$1.50/h × 25 box-hours = $37.50 > $25: refused at the door, nothing started."""
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
    assert "estimated at $37.50" in str(refused.value)
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
    carries the estimate; it stops right after `train`, before `improved` (which stops a stub
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
        assert card["evidence"]["estimate_usd"] == pytest.approx(16.5)
        assert "train: $0.66/h × 12h on gpu × 2 (2 attempts)" in card["evidence"]["estimate_basis"]

        await b.advance_step(pid, "approve-run", approver="ml-owner")
        rec = await _wait(b.store, pid, lambda r: r.status in ("done", "failed", "cancelled"))

        assert env.timeline == ["ec2:tags", "ec2:start", "submit:smoke", "submit:train", "ec2:stop"]
        assert env.ec2.tags["cairn:managed"] and env.ec2.tags["cairn:release-after"]
        assert rec.step_state["release-box"].status == "done"
        events = [(s, e) for s, e, _ in audits]
        assert events.index(("improved", "gate_blocked")) > max(
            i for i, (s, _) in enumerate(events) if s == "release-box")
        assert rec.status == "failed" and rec.step_state["release"].status == "done"
        assert env.ec2.state == "stopped"
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
