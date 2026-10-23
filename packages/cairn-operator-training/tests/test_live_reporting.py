"""#2025 (ADR-0102 D4) — the training pack's live channel, worker side.

The acceptance the ticket names, proven here at the layer that owns it: a
200-step run whose link to the control plane is cut for a window ends with
exactly 200 ``metrics`` events in the store, contiguous ``seq``, no duplicates;
``checkpoint-and-stop`` yields a checkpoint whose step equals the last logged
step. The trainer is a fake ``transformers.TrainerCallback`` host — no torch.
"""
from __future__ import annotations

import asyncio
import sys
import types

import pytest

from operator_training import live
from operator_training.live import HttpTransport, LiveReporter, NullReporter, Spool, reporter_from_env


# ─── a fake `transformers` so the callback can be built without torch ─────


@pytest.fixture
def fake_transformers(monkeypatch):
    mod = types.ModuleType("transformers")

    class TrainerCallback:  # the real one is a plain base class with no-op hooks
        pass

    mod.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return mod


class _State:
    def __init__(self, max_steps: int) -> None:
        self.global_step = 0
        self.max_steps = max_steps
        self.num_input_tokens_seen = 0


class _Control:
    should_save = False
    should_training_stop = False


class _Args:
    output_dir = "/tmp/out"


def _drive(cb, steps: int, *, log_every: int = 1, save_every: int = 0) -> tuple[_State, _Control]:
    """Run the callback the way a Trainer would: step_end → log → (save)."""
    state, control, args = _State(steps), _Control(), _Args()
    cb.on_train_begin(args, state, control)
    for step in range(1, steps + 1):
        state.global_step = step
        state.num_input_tokens_seen += 64
        cb.on_step_end(args, state, control)
        if step % log_every == 0:
            cb.on_log(args, state, control, logs={"loss": 1.0 / step, "learning_rate": 1e-4})
        if control.should_save:
            cb.on_save(args, state, control)
            control.should_save = False
        elif save_every and step % save_every == 0:
            cb.on_save(args, state, control)
        if control.should_training_stop:
            break
    cb.on_train_end(args, state, control)
    return state, control


# ─── the control plane's end, real: ingest_events over an in-memory store ──


class _ControlPlane:
    """The real idempotent ingest behind a fake HTTP that can be cut."""

    def __init__(self) -> None:
        from cairn.core.operator import report_ingest as ri
        from cairn.core.storage.memory.runs import InMemoryRunStore

        ri._HWM.clear()
        self.ri, self.store = ri, InMemoryRunStore()
        self.down = False
        self.posts = 0
        self.control: list[dict] = []

    def post(self, url: str, body: dict) -> dict:
        self.posts += 1
        if self.down:
            raise OSError("link down")
        run_id = url.rsplit("/runs/", 1)[1].split("/")[0]
        res = asyncio.run(self.ri.ingest_events(self.store, run_id, body["events"]))
        return res.as_dict()

    def get(self, url: str) -> dict:
        if self.down:
            raise OSError("link down")
        after = int(url.split("after_seq=")[1].split("&")[0])
        newer = [c for c in self.control if c["seq"] > after]
        return {"control": newer[-1] if newer else None}

    def events(self, run_id: str, typ: str) -> list:
        return [e for e in asyncio.run(self.store.get_events(run_id)) if e.type == typ]


def _reporter(cp: _ControlPlane, *, clock) -> LiveReporter:
    tr = HttpTransport("http://cp", "run-t6", "tok", post=cp.post, get=cp.get)
    return LiveReporter("run-t6", "train", tr, flush_every_s=1.0, control_every_s=5.0, clock=clock)


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


# ─── 1. the acceptance: a cut link, an exact store ─────────────────────────


def test_network_cut_mid_run_ends_with_exactly_200_contiguous_metrics(fake_transformers):
    from operator_training.reporting import make_report_callback

    cp, clock = _ControlPlane(), _Clock()
    rep = _reporter(cp, clock=clock)
    cb = make_report_callback(rep, system_every_s=1e9, clock=clock, sample_system=lambda: {})

    state, control, args = _State(200), _Control(), _Args()
    cb.on_train_begin(args, state, control)
    for step in range(1, 201):
        clock.t += 1.0                      # one second per step → a flush every step
        if step == 60:
            cp.down = True                  # the cut …
        if step == 90:
            cp.down = False                 # … 30 s later the link is back
        state.global_step = step
        cb.on_step_end(args, state, control)
        cb.on_log(args, state, control, logs={"loss": 1.0 / step})
    cb.on_train_end(args, state, control)
    rep.close()

    metrics = cp.events("run-t6", "metrics")
    assert len(metrics) == 200
    seqs = [e.data["seq"] for e in metrics]
    assert len(set(seqs)) == 200
    steps = [e.data["step"] for e in metrics]
    assert steps == list(range(1, 201))
    assert rep.spool.unshipped == 0
    # the replay after the cut re-sent what the control plane never saw and
    # nothing it had: every metrics event landed once (seq contiguous per node)
    assert seqs == sorted(seqs) and all(b - a >= 1 for a, b in zip(seqs, seqs[1:]))
    progress = cp.events("run-t6", "progress")
    assert progress[0].data["stage"] == "train" and progress[-1].data["percent"] == 100.0
    scal = cp.events("run-t6", "scalars")
    assert len(scal) == 1 and scal[0].data["values"]["steps"] == 200 and scal[0].data["values"]["final_loss"] == pytest.approx(1 / 200)


def test_spool_survives_a_worker_restart(tmp_path):
    path = str(tmp_path / "spool.jsonl")
    rep = LiveReporter("r", "train", None, spool=Spool(path))
    rep.series(1, loss=1.0)
    rep.series(2, loss=0.5)
    again = LiveReporter("r", "train", None, spool=Spool(path))
    assert again.seq == 2                       # seq continues, never restarts at 1
    assert again.spool.unshipped == 2           # nothing was acked → everything replays
    again.series(3, loss=0.25)
    assert [i["data"]["seq"] for i in again.spool.pending()] == [1, 2, 3]


def test_flush_is_rate_limited_but_forced_on_close():
    cp, clock = _ControlPlane(), _Clock()
    rep = _reporter(cp, clock=clock)
    rep.series(1, loss=1.0)         # first flush at t0
    rep.series(2, loss=0.5)         # within flush_every_s → spooled, not sent
    assert cp.posts == 1 and rep.spool.unshipped == 1
    rep.close()
    assert rep.spool.unshipped == 0 and len(cp.events("run-t6", "metrics")) == 2


# ─── 2. control down: checkpoint-and-stop / abort ──────────────────────────


def test_checkpoint_and_stop_saves_at_the_last_logged_step(fake_transformers):
    from operator_training.reporting import make_report_callback

    cp, clock = _ControlPlane(), _Clock()
    rep = _reporter(cp, clock=clock)
    cb = make_report_callback(rep, system_every_s=1e9, clock=clock, sample_system=lambda: {})
    state, control, args = _State(100), _Control(), _Args()
    cb.on_train_begin(args, state, control)
    for step in range(1, 101):
        clock.t += 6.0                                  # past the control poll interval each step
        state.global_step = step
        if step == 40:
            cp.control.append({"seq": 7, "action": live.CHECKPOINT_AND_STOP, "by": "console"})
        cb.on_step_end(args, state, control)
        cb.on_log(args, state, control, logs={"loss": 1.0 / step})
        if control.should_save:
            cb.on_save(args, state, control)
        if control.should_training_stop:
            break
    cb.on_train_end(args, state, control)
    rep.close()

    assert cb.stopped_by == live.CHECKPOINT_AND_STOP
    assert state.global_step == 40
    ck = cp.events("run-t6", "checkpoint")
    assert len(ck) == 1 and ck[0].data["step"] == 40 and ck[0].data["ref"].endswith("checkpoint-40")
    last_logged = cp.events("run-t6", "metrics")[-1].data["step"]
    assert ck[0].data["step"] == last_logged


def test_abort_stops_without_saving(fake_transformers):
    from operator_training.reporting import make_report_callback

    cp, clock = _ControlPlane(), _Clock()
    rep = _reporter(cp, clock=clock)
    cb = make_report_callback(rep, system_every_s=1e9, clock=clock, sample_system=lambda: {})
    cp.control.append({"seq": 1, "action": live.ABORT})
    clock.t += 10
    state, control = _drive(cb, 50)
    assert cb.stopped_by == live.ABORT and state.global_step == 1
    assert not cp.events("run-t6", "checkpoint")


def test_control_polls_at_most_once_per_interval():
    cp, clock = _ControlPlane(), _Clock()
    rep = _reporter(cp, clock=clock)
    calls = []
    rep.transport._get = lambda url: (calls.append(url) or {"control": None})
    for _ in range(10):
        rep.control()
    assert len(calls) == 1
    clock.t += 5.0
    rep.control()
    assert len(calls) == 2


# ─── 3. the rest of the surface ────────────────────────────────────────────


def test_system_series_is_sampled_on_its_own_cadence(fake_transformers):
    from operator_training.reporting import make_report_callback

    cp, clock = _ControlPlane(), _Clock()
    rep = _reporter(cp, clock=clock)
    samples = iter([{"gpu_util_pct": 91.0, "gpu_mem_gb": 20.5}] * 3)
    cb = make_report_callback(rep, system_every_s=15.0, clock=clock, sample_system=lambda: next(samples))
    state, control, args = _State(30), _Control(), _Args()
    for step in range(1, 31):
        clock.t += 1.0
        state.global_step = step
        cb.on_step_end(args, state, control)
    rep.close()
    sys_events = [e for e in cp.events("run-t6", "metrics") if "gpu_util_pct" in e.data["series"]]
    assert len(sys_events) == 2                       # t=1 and t=16 within 30 s
    assert sys_events[0].data["series"]["gpu_mem_gb"] == 20.5


def test_reporter_from_env_picks_the_transport():
    assert isinstance(reporter_from_env({}), NullReporter)
    http = reporter_from_env({"CAIRN_RUN_ID": "r1", "CAIRN_NODE": "train", "CAIRN_ENGINE_URL": "http://cp/",
                              "CAIRN_ENGINE_TOKEN": "t", "CAIRN_LIVE_SPOOL": "/dev/null"})
    assert isinstance(http.transport, HttpTransport) and http.transport.base == "http://cp" and http.transport.token == "t"
    nats = reporter_from_env({"CAIRN_RUN_ID": "r1", "OBS_JOB_CHANNEL": "nats://nats:4222", "CAIRN_LIVE_SPOOL": "/dev/null"})
    assert isinstance(nats.transport, live.NatsTransport) and nats.node == "train"
    blind = reporter_from_env({"CAIRN_RUN_ID": "r1", "CAIRN_LIVE_SPOOL": "/dev/null"})
    assert blind.transport is None and blind.series(1, loss=1.0) is None and blind.spool.unshipped == 1


def test_reporter_from_env_prefers_the_cairn_names_and_falls_back_to_obs():
    """#2429: the wire carries CAIRN_JOB_CHANNEL and its OBS_ twin until 1.0; live.py (stdlib,
    no resolver) reads the CAIRN_ name first. OBS_ENGINE_URL is CAIRN_ENGINE_URL's alias."""
    base = {"CAIRN_RUN_ID": "r1", "CAIRN_LIVE_SPOOL": "/dev/null"}
    new = reporter_from_env({**base, "CAIRN_JOB_CHANNEL": "nats://new:4222", "OBS_JOB_CHANNEL": "nats://old:4222"})
    assert isinstance(new.transport, live.NatsTransport) and new.transport.url == "nats://new:4222"
    old = reporter_from_env({**base, "CAIRN_JOB_CHANNEL": "", "OBS_JOB_CHANNEL": "nats://old:4222"})
    assert isinstance(old.transport, live.NatsTransport) and old.transport.url == "nats://old:4222"
    url = reporter_from_env({**base, "CAIRN_ENGINE_URL": "http://new/", "OBS_ENGINE_URL": "http://old/"})
    assert url.transport.base == "http://new"
    assert reporter_from_env({**base, "OBS_ENGINE_URL": "http://old/"}).transport.base == "http://old"


def test_a_transport_that_raises_never_reaches_the_trainer(fake_transformers):
    from operator_training.reporting import make_report_callback

    class _Boom(live.Transport):
        def send(self, batch):
            raise RuntimeError("boom")

        def control(self, after_seq):
            raise RuntimeError("boom")

    rep = LiveReporter("r", "train", _Boom(), flush_every_s=0.0)
    cb = make_report_callback(rep, system_every_s=1e9, sample_system=lambda: {})
    state, _ = _drive(cb, 5)
    # 5 metrics + 7 progress marks (begin, one per log, end) + 1 scalars: all still spooled, nothing lost
    assert state.global_step == 5 and rep.spool.unshipped == 13


# ─── 4. the in-process path: the stub speaks the declared vocabulary ──────


async def test_stub_train_emits_only_what_it_declares():
    from cairn.core.operator.testing import run_operator_for_test
    from operator_training.operators.stub import StubTrainOp
    from operator_training.reporting import TRAIN_REPORTS
    from operator_training.schemas.train import TrainSpec

    sink: list = []
    spec = TrainSpec(base_model="stub/model", dataset_uri="file:///tmp/x.jsonl", output_name="adapter", max_steps=3)
    out = await run_operator_for_test(StubTrainOp, spec, report_sink=sink)
    assert out.status == "succeeded"
    types = [t for t, _ in sink]
    assert types.count("metrics") == 3 and "checkpoint" in types and types[-1] == "scalars"
    declared = TRAIN_REPORTS.series_names
    for t, payload in sink:
        if t == "metrics":
            assert set(payload["series"]) <= set(declared)
        if t == "progress":
            assert payload["stage"] in TRAIN_REPORTS.stages


# ─── 5. the flow path: the train.qlora kernel reports as its stage ────────


def test_flow_kernel_reports_as_its_stage_from_the_engine_env(tmp_path, monkeypatch):
    """A `flow:` leg's train stage runs in the engine's subprocess/box/pod with
    CAIRN_PARENT_RUN + the transport in its env (the engines set them); the
    kernel builds a LiveReporter from that and the stub's curve reaches the
    control plane under node `train.qlora`."""
    from operator_training import flow_stages, live

    batches: list[dict] = []

    def fake_post(self, url, body):
        batches.append({"url": url, **body})
        return {"accepted": len(body["events"]), "dropped": 0, "refused": [], "last_seq": {}}

    monkeypatch.setattr(live.HttpTransport, "_http_post", fake_post)
    monkeypatch.setenv("CAIRN_PARENT_RUN", "run-flow-1")
    monkeypatch.setenv("OBS_ENGINE_URL", "http://cp:8000")
    monkeypatch.setenv("CAIRN_LIVE_SPOOL", str(tmp_path / "spool.jsonl"))
    monkeypatch.delenv("CAIRN_RUN_ID", raising=False)
    monkeypatch.delenv("OBS_JOB_CHANNEL", raising=False)
    monkeypatch.delenv("CAIRN_JOB_CHANNEL", raising=False)
    monkeypatch.delenv("CAIRN_ENGINE_URL", raising=False)
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"input":"q","output":"a"}\n')

    out = flow_stages.train_qlora(tmp_path, {"stub": True, "dataset_uri": str(ds), "output_name": "adp", "max_steps": 3}, [])
    assert out
    assert batches and all(b["url"] == "http://cp:8000/runs/run-flow-1/events/ingest" for b in batches)
    events = [e for b in batches for e in b["events"]]
    assert {e["data"]["node"] for e in events} == {"train.qlora"}
    assert [e["data"]["step"] for e in events if e["type"] == "metrics"] == [1, 2, 3]
    assert events[-1]["type"] == "scalars"


def test_flow_kernel_without_a_run_identity_reports_nowhere(tmp_path, monkeypatch):
    from operator_training import flow_stages

    for k in ("CAIRN_PARENT_RUN", "CAIRN_RUN_ID", "OBS_ENGINE_URL", "CAIRN_ENGINE_URL", "OBS_JOB_CHANNEL",
              "CAIRN_JOB_CHANNEL"):
        monkeypatch.delenv(k, raising=False)
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"input":"q","output":"a"}\n')
    assert flow_stages.train_qlora(tmp_path, {"stub": True, "dataset_uri": str(ds), "output_name": "adp"}, [])


# ─── 6. the divergence guard (#2030): NaN at step 50 stops at step 50, never restarts ──


def test_nan_at_step_50_stops_at_step_50_with_a_divergence_event(fake_transformers):
    from operator_training.reporting import make_report_callback

    cp, clock = _ControlPlane(), _Clock()
    rep = _reporter(cp, clock=clock)
    cb = make_report_callback(rep, system_every_s=1e9, clock=clock, sample_system=lambda: {})
    state, control, args = _State(200), _Control(), _Args()
    cb.on_train_begin(args, state, control)
    for step in range(1, 201):
        clock.t += 1.0
        state.global_step = step
        cb.on_step_end(args, state, control)
        cb.on_log(args, state, control, logs={"loss": float("nan") if step == 50 else 1.0 / step})
        if control.should_training_stop:
            break
    cb.on_train_end(args, state, control)
    rep.close()
    assert state.global_step == 50 and cb.stopped_by == "divergence"
    div = cp.events("run-t6", "divergence")
    assert len(div) == 1 and div[0].data["kind"] == "nan" and div[0].data["step"] == 50
    assert cp.events("run-t6", "metrics")[-1].data["step"] == 49   # the NaN is not a series point


def test_a_loss_spike_is_divergence_too_and_a_plateau_is_not():
    from operator_training.reporting import divergence_check

    steady = [1.0, 0.9, 0.95, 0.85, 0.9]
    assert divergence_check(0.8, steady) is None
    assert divergence_check(1.5, steady) is None            # under 4× the median
    assert divergence_check(4.0, steady) == "spike"
    assert divergence_check(float("inf"), steady) == "nan"
    assert divergence_check(50.0, [1.0, 0.9]) is None        # no window yet → no spike judgement
