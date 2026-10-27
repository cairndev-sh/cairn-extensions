"""dataset-prep/prepare pauses for a person when cleaning drops a large share of the data
(cairn-extensions#39).

The plan: a human gate `data-review` between `prepare` and `format`, routed in by `when:` when
either dedup pass's `dropped_fraction` is above the `review_drop_ratio` input (default 0.30).
Then the pipeline on the core's own launch door and in-process bridge: the `prepare` leg runs
this pack's flow on the core runner in process; a source where most rows are duplicates pauses at
`data-review` with the numbers on the card, and a clean source runs through without pausing.

Hermetic: the 200-row IFC-shaped sample (and copies of it), whitespace token counts, sqlite, and
a moto-mocked S3 bucket. The engine the bridge hands the data leg is a stand-in that runs the
leg's flow with the core runner in this process, so the mocked bucket is the one it writes to.
Without the [minhash] extra near dedup is an honest skip; the near-duplicate route needs it.
"""
from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path

import pytest
import yaml

PACK = Path(__file__).resolve().parents[1]
SAMPLE = PACK / "data" / "sample-ifc.jsonl"
BUCKET = "cairn-test-artifacts"
REVIEW = "data-review"

#: What `when:` reads: the prepare leg's findings and the limit.
_FINDINGS = ("dedup_exact_dropped_fraction", "dedup_near_dropped_fraction")


@pytest.fixture(autouse=True)
def s3(monkeypatch):
    """The s3 artifact store the exports land in, mocked."""
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    monkeypatch.setenv("CAIRN_ARTIFACT_S3_BUCKET", BUCKET)
    monkeypatch.setenv("CAIRN_ARTIFACT_S3_PREFIX", "datasets")
    for name in ("CAIRN_ARTIFACT_S3_ENDPOINT", "OBS_ARTIFACT_S3_ENDPOINT", "OBS_ARTIFACT_S3_BUCKET"):
        monkeypatch.delenv(name, raising=False)
    with moto.mock_aws():
        client = boto3.client("s3", region_name="eu-central-1")
        client.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-central-1"})
        yield client


@pytest.fixture(autouse=True)
def _pack_from_checkout(monkeypatch):
    """Load this checkout's pack as a dev pack (CAIRN_PACK_PATH), as `cairn` does at start."""
    from cairn.core.pack import registry
    from cairn.core.pack.loader import load_all_packs

    for name in ("CAIRN_DATAFLOW_ENGINE", "CAIRN_DATAFLOW_TARGET", "CAIRN_ARTIFACT_STORE",
                 "CAIRN_RUNTIME", "CAIRN_STORAGE", "CAIRN_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CAIRN_PACK_PATH", str(PACK))
    registry.clear()
    load_all_packs()
    yield
    registry.clear()


def _pipeline() -> dict:
    return yaml.safe_load((PACK / "pipelines" / "prepare.yaml").read_text(encoding="utf-8"))


def _def():
    from cairn.core.engine.pipeline import parse_pipeline_def

    return parse_pipeline_def((PACK / "pipelines" / "prepare.yaml").read_text(encoding="utf-8"),
                              pack="dataset-prep")


def _review_step() -> dict:
    (step,) = [s for s in _pipeline()["steps"] if s.get("gate") == REVIEW]
    return step


# ── the plan ──────────────────────────────────────────────────────────────────

def test_data_review_is_a_person_gate_between_prepare_and_format():
    from cairn.core.engine.pipeline import plan_pipeline

    plan = plan_pipeline(_def())
    steps = {s.id: s for s in plan.steps}
    assert [s.id for s in plan.steps] == ["source-ok", "prepare", REVIEW, "format", "data-ok"]
    review = steps[REVIEW]
    assert review.depends_on == ["prepare"] and steps["format"].depends_on == [REVIEW]
    assert review.when and review.approval
    assert _review_step()["approvers"] == "data-owners"
    spec = _pipeline()["inputs"]["review_drop_ratio"]
    assert (spec["type"], spec["default"]) == ("float", 0.30)


def test_the_when_expression_compiles_and_reads_both_dedup_shares():
    from cairn.core.engine.expr import compile_predicate

    when = _review_step()["when"]
    compile_predicate(when, source="data-review when")
    for key in _FINDINGS:
        assert f"steps.prepare.findings.{key}" in when
    assert "input.review_drop_ratio" in when


def _routed_in(exact, near, ratio=0.30) -> bool:
    from cairn.core.engine.bundle import eval_gate_predicate

    findings = {k: v for k, v in zip(_FINDINGS, (exact, near)) if v is not None}
    return eval_gate_predicate(_review_step()["when"], bundle_input={"review_drop_ratio": ratio},
                               steps={"prepare": {"findings": findings}})


@pytest.mark.parametrize(("exact", "near", "ratio", "paused"), [
    (0.03, 0.0, 0.30, False),      # the shipped sample: 6 of 200 exact duplicates
    (0.30, 0.30, 0.30, False),     # at the limit is within it
    (0.6767, 0.0, 0.30, True),     # most rows exact duplicates
    (0.0, 0.8525, 0.30, True),     # the qualification run: 85% near duplicates
    (0.10, 0.31, 0.30, True),
    (0.6767, 0.8525, 1.0, False),  # 1.0 never pauses
    (0.20, 0.0, 0.10, True),       # a stricter limit
    (None, None, 0.30, True),      # no stats (an older dataset-checks): fails closed, pauses
    (0.0, None, 0.30, True),
])
def test_the_gate_is_routed_in_only_above_the_limit(exact, near, ratio, paused):
    assert _routed_in(exact, near, ratio) is paused


def test_every_card_field_is_a_prepare_finding():
    """cairndev 0.3.1 resolves `show:` against step outputs only (cairndev-sh/cairn#236)."""
    show = _review_step()["show"]
    assert show and all(ref.startswith("prepare.") for ref in show)


# ── the launch door and the in-process bridge ─────────────────────────────────

class InProcessDataEngine:
    """The data engine the bridge hands `prepare`: the leg's flow on the core runner, here."""

    name = "builtin"
    carries_stage_contracts = True   # as the builtin engine: the stage outputs are checked

    def __init__(self, root: Path) -> None:
        self.root, self.jobs, self.submitted = root, {}, []

    def validate(self, spec) -> list[str]:
        return []

    async def submit(self, spec, ctx):
        from cairn.core.dataflow.engine import EngineJobHandle
        from cairn.core.dataflow.runner import run_dataflow

        job = f"job-{len(self.jobs)}"
        self.submitted.append(job)
        result = await asyncio.to_thread(run_dataflow, spec, self.root / job)
        self.jobs[job] = result
        return EngineJobHandle(job_id=job, engine=self.name)

    async def poll(self, handle):
        from cairn.core.dataflow.engine import EngineJobStatus

        return EngineJobStatus(state="succeeded")

    async def result(self, handle):
        return self.jobs[handle.job_id]

    async def cancel(self, handle):
        pass


class _Runner:
    default_environment = "production"

    async def run(self, *a, **k):
        return []


@pytest.fixture
async def bridge(tmp_path, monkeypatch):
    """The core's PipelineBridge with an in-memory record store, the registry in sqlite, and the
    data leg on :class:`InProcessDataEngine`; every audit event recorded."""
    import cairn.core.dispatch.leg_executor as leg_executor
    import cairn.core.storage.artifacts as artifacts
    from cairn.core.dispatch.dispatcher import RunDispatcher
    from cairn.core.dispatch.pipeline_bridge import PipelineBridge
    from cairn.core.storage import build_storage
    from cairn.core.storage.artifacts import SqliteArtifactStore
    from cairn.core.storage.pipeline_store import InMemoryPipelineStore

    versions = SqliteArtifactStore(str(tmp_path / "registry.db"))
    monkeypatch.setattr(artifacts, "build_artifact_store", lambda *a, **k: versions)
    monkeypatch.setattr(leg_executor, "_POLL_INTERVAL_S", 0.05)
    engine = InProcessDataEngine(tmp_path / "engine")
    b = PipelineBridge(RunDispatcher(_Runner(), backend=build_storage()), store=InMemoryPipelineStore())
    monkeypatch.setattr(b, "_leg_engine", lambda s: engine)
    audits: list[tuple] = []
    real_audit = b._audit

    async def _spy(rec, step_id, event, detail):
        audits.append((step_id, event, dict(detail or {})))
        return await real_audit(rec, step_id, event, detail)

    monkeypatch.setattr(b, "_audit", _spy)
    await b.start()
    try:
        yield types.SimpleNamespace(b=b, audits=audits, engine=engine, versions=versions)
    finally:
        b.stop()
        await versions.close()


def _inputs(path: Path, register: str, **over) -> dict:
    """The `ifc-bim-alpaca` preset on a local file: no Hub, no tokenizer download, s3 exports."""
    doc = yaml.safe_load((PACK / "presets" / "ifc-bim-alpaca.yaml").read_text(encoding="utf-8"))
    (preset,) = doc["presets"]
    return {**preset["inputs"], "path": str(path), "revision": "", "base_model": "", "store": "s3",
            "register": register, **over}


def _copies(tmp_path: Path, n: int) -> Path:
    """The sample with every row `n` times: most of the source is exact duplicates."""
    lines = [line for line in SAMPLE.read_text(encoding="utf-8").splitlines() if line.strip()]
    path = tmp_path / f"sample-x{n}.jsonl"
    path.write_text("".join(f"{line}\n" for line in lines for _ in range(n)), encoding="utf-8")
    return path


async def _wait(store, pid, pred, timeout=60.0):
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        rec = await store.get(pid)
        if pred(rec):
            return rec
        if loop.time() > deadline:
            raise AssertionError(f"stuck: {rec.status!r} {(rec.trigger or {}).get('failure_reason')} "
                                 + str({k: v.status for k, v in rec.step_state.items()}))
        await asyncio.sleep(0.05)


def _status(step: str, *statuses: str):
    return lambda r: getattr(r.step_state.get(step), "status", "") in statuses


def _terminal(r) -> bool:
    return r.status in ("done", "failed", "cancelled")


async def _launch_past_source_ok(env, inputs: dict) -> str:
    from cairn.core.dispatch.pipeline_launch import launch_pipeline_def

    rec = await launch_pipeline_def(_def(), inputs=inputs, bridge=env.b)
    pid = rec.pipeline_id
    await _wait(env.b.store, pid, _status("source-ok", "awaiting_approval"))
    await env.b.advance_step(pid, "source-ok", approver="data-owner")
    return pid


def _card(env) -> dict:
    (card,) = [d["gate_context"] for s, e, d in env.audits if s == REVIEW and e == "awaiting"]
    return card


async def test_a_source_that_is_mostly_duplicates_pauses_at_data_review(bridge, tmp_path):
    """Every sample row three times: exact dedup drops 406 of 600 rows (0.6767 > 0.30). The run
    stops at `data-review` before `format`, and the card carries the numbers."""
    pid = await _launch_past_source_ok(bridge, _inputs(_copies(tmp_path, 3), "ifc-x3"))
    rec = await _wait(bridge.b.store, pid, lambda r: _terminal(r) or _status(REVIEW, "awaiting_approval")(r))
    assert rec.step_state[REVIEW].status == "awaiting_approval", rec.status
    assert rec.step_state["format"].status == "pending"

    card = _card(bridge)
    assert card["approvers"] == "data-owners"
    f = card["fields"]
    assert f["prepare.ingest_normalize_rows"] == 600
    assert (f["prepare.dedup_exact_dropped"], f["prepare.dedup_exact_dropped_fraction"]) == (406, 0.6767)
    near = f["prepare.dedup_near_dropped_near"]
    assert f["prepare.export_jsonl_rows"] == 194 - near
    assert (f["prepare.split_rows_train"] + f["prepare.split_rows_val"] + f["prepare.split_rows_test"]
            == 194 - near)
    assert f["prepare.profile_tokens_tokens_total"] > 0 and f["prepare.profile_tokens_over_length_rows"] == 0
    # every field resolves; the near threshold only when the near pass ran
    missing = {k for k, v in f.items() if v is None}
    assert missing <= ({"prepare.dedup_near_threshold"} if f["prepare.dedup_near_method"] == "skipped" else set())

    await bridge.b.advance_step(pid, REVIEW, approver="data-owner")
    rec = await _wait(bridge.b.store, pid, _terminal)
    assert rec.status == "done", {k: (v.status, v.verdict) for k, v in rec.step_state.items()}
    assert rec.step_state["format"].status == "done" and rec.step_state[REVIEW].approver == "data-owner"


async def test_a_clean_source_runs_through_without_pausing(bridge):
    """The shipped sample: 6 of 200 exact duplicates (0.03). `data-review` is routed off."""
    pid = await _launch_past_source_ok(bridge, _inputs(SAMPLE, "ifc-clean"))
    rec = await _wait(bridge.b.store, pid, _terminal)
    assert rec.status == "done", {k: (v.status, v.verdict) for k, v in rec.step_state.items()}
    assert rec.step_state[REVIEW].status == "skipped"
    assert (REVIEW, "routed_off") in [(s, e) for s, e, _ in bridge.audits]
    assert not [d for s, e, d in bridge.audits if s == REVIEW and e == "awaiting"]


async def test_a_limit_of_one_never_pauses(bridge, tmp_path):
    """`-i review_drop_ratio=1.0` (a string, as the CLI sends it) turns the review off."""
    pid = await _launch_past_source_ok(bridge, _inputs(_copies(tmp_path, 3), "ifc-x3-off", review_drop_ratio="1.0"))
    rec = await _wait(bridge.b.store, pid, _terminal)
    assert rec.status == "done" and rec.step_state[REVIEW].status == "skipped"


async def test_a_limit_that_is_not_a_number_is_refused_at_the_launch_door(bridge):
    from cairn.core.dispatch.pipeline_launch import InputCoercionError, launch_pipeline_def

    with pytest.raises(InputCoercionError, match="review_drop_ratio"):
        await launch_pipeline_def(_def(), inputs=_inputs(SAMPLE, "ifc-x", review_drop_ratio="a third"),
                                  bridge=bridge.b)
    assert bridge.engine.submitted == []


async def test_declining_the_review_cancels_the_pipeline_before_format(bridge, tmp_path):
    """cairndev 0.3.1 has no reject door for a pipeline gate: a person declines by cancelling."""
    pid = await _launch_past_source_ok(bridge, _inputs(_copies(tmp_path, 3), "ifc-x3-no"))
    await _wait(bridge.b.store, pid, _status(REVIEW, "awaiting_approval"))
    await bridge.b.cancel_pipeline(pid)
    rec = await _wait(bridge.b.store, pid, _terminal)
    assert rec.status == "cancelled" and rec.step_state["format"].status != "done"


_WORDS = ("river stone bridge tower garden window market harbor engine signal winter summer copper "
          "silver forest valley meadow castle library lantern compass ladder anchor feather "
          "thunder candle mirror basket saddle orchard quarry glacier canyon prairie").split()


def _near_copies(tmp_path: Path) -> Path:
    """60 distinct 40-word answers, each three times with only its last word changed: no exact
    duplicates, and two of every three rows a near duplicate of the first."""
    import random

    path = tmp_path / "near.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(60):
            rng = random.Random(i)
            answer = " ".join(rng.choice(_WORDS) for _ in range(40))
            for last in ("north", "south", "east"):
                f.write(json.dumps({"instruction": "Describe the scene.", "input": f"Scene {i}.",
                                    "output": f"{answer} {last}"}) + "\n")
    return path


async def test_a_source_that_is_mostly_near_duplicates_pauses_at_data_review(bridge, tmp_path):
    """The qualification run's case: the near pass, not the exact one, drops most rows."""
    pytest.importorskip("datasketch")
    inputs = _inputs(_near_copies(tmp_path), "near", derive=[], group_by="", topic_fields=[], group_fields=[])
    pid = await _launch_past_source_ok(bridge, inputs)
    rec = await _wait(bridge.b.store, pid, lambda r: _terminal(r) or _status(REVIEW, "awaiting_approval")(r))
    assert rec.step_state[REVIEW].status == "awaiting_approval", rec.status
    f = _card(bridge)["fields"]
    assert (f["prepare.ingest_normalize_rows"], f["prepare.dedup_exact_dropped_fraction"]) == (180, 0.0)
    assert f["prepare.dedup_near_dropped_fraction"] > 0.30
    assert f["prepare.dedup_near_method"] == "minhash" and f["prepare.dedup_near_threshold"] == 0.8
    assert None not in f.values()
