"""finetune/sft in process (cairn-extensions#7): the `sft-train` flow and the pipeline's gates.

Three small versions are registered with their roles; their `dataset:` refs are resolved the way
the control plane resolves a leg's config (`<uri>#sha256=<hash>`, which the stages verify); the
pack's flow runs on the core runner with the stub trainer (no GPU); and the pipeline's own gate
predicates are evaluated by the core's evaluator over the leg findings the pipeline bridge
derives. The stub trains nothing, so its validation loss is not below its baseline and
`improved` stops it: a stub adapter can never reach the person who would promote it."""
from __future__ import annotations

import asyncio
import hashlib
import json
import types
from pathlib import Path

import pytest
import yaml

PACK = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _pack_from_checkout(monkeypatch):
    """Load this checkout's pack as a dev pack (CAIRN_PACK_PATH), as `cairn` does at start."""
    from cairn.core.pack import registry
    from cairn.core.pack.loader import load_all_packs

    monkeypatch.setenv("CAIRN_PACK_PATH", str(PACK))
    registry.clear()
    load_all_packs()
    yield
    registry.clear()


def _pipeline() -> dict:
    return yaml.safe_load((PACK / "pipelines" / "sft.yaml").read_text(encoding="utf-8"))


def _gate(name: str) -> str:
    return next(s["predicate"] for s in _pipeline()["steps"] if s.get("gate") == name)


def _with(step_id: str) -> dict:
    return dict(next(s["with"] for s in _pipeline()["steps"] if s.get("id") == step_id))


@pytest.fixture
async def versions(tmp_path):
    """dataset:demo-{train,val,test}@1, registered with their roles."""
    from cairn.core.dataflow.contract import ArtifactDescriptor
    from cairn.core.storage.artifacts import SqliteArtifactStore

    store = SqliteArtifactStore(str(tmp_path / "registry.db"))
    for name, role, n in (("train", "training", 100), ("val", "validation", 6), ("test", "test", 6)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps({"instruction": "Answer.", "input": f"{name} q{i}",
                                            "output": f"{name} a{i}"}) + "\n" for i in range(n)))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        await store.register(ArtifactDescriptor(uri=f"file://{path}", kind="dataset", content_hash=digest),
                             producer_run_id="prepare")
        await store.register_dataset_version(f"demo-{name}", digest, actor="prepare", role=role)
    yield store
    await store.close()


async def _leg_config(store, step_id: str, inputs: dict) -> dict:
    """The leg's `with:`, its `$input.*` wires bound and its `dataset:` refs resolved."""
    from cairn.core.storage.artifacts import resolve_dataset_refs_in_config

    defaults = {k: v.get("default") for k, v in _pipeline()["inputs"].items() if isinstance(v, dict)}
    bound = {**defaults, **inputs}
    cfg = {k: (bound[v[len("$input."):]] if isinstance(v, str) and v.startswith("$input.") else v)
           for k, v in _with(step_id).items()}
    return await resolve_dataset_refs_in_config(store, cfg)


def _run(workdir: Path, config: dict):
    from cairn.core.dataflow import Budget, DataflowSpec
    from cairn.core.dataflow.defs import resolve_flow_ref
    from cairn.core.dataflow.runner import run_dataflow

    stages = resolve_flow_ref("finetune/sft-train", config, kind="ml")
    return run_dataflow(DataflowSpec(stages=stages, budget=Budget()), workdir)


def _findings(result) -> dict:
    """A flow leg's step output, exactly as the pipeline bridge derives it."""
    from cairn.core.dispatch.pipeline_bridge import _flow_leg_output

    return _flow_leg_output(types.SimpleNamespace(produces=[]), types.SimpleNamespace(ok=True, stats=result.stats))


_INPUTS = {"train_version": "dataset:demo-train@1", "val_version": "dataset:demo-val@1",
           "test_version": "dataset:demo-test@1", "base_model": "tiny/model", "stub": True}


async def test_smoke_trains_a_recorded_subsample_and_passes_its_gate(tmp_path, versions):
    from cairn.core.engine.bundle import eval_gate_predicate

    cfg = await _leg_config(versions, "smoke", _INPUTS)
    assert cfg["train_uri"].endswith("#sha256=" + cfg["train_uri"].rsplit("=", 1)[1])
    # in a thread: the trainer stage runs its own event loop, as it does inside an engine's process
    result = await asyncio.to_thread(_run, tmp_path / "smoke", cfg)
    stats = result.stats
    assert stats["sample.fidelity.fidelity_sampled"] is True and stats["sample.fidelity.fidelity_n"] == 64
    assert stats["train.qlora.status"] == "succeeded" and "train.qlora.baseline_eval_loss" not in stats
    assert eval_gate_predicate(_gate("smoke-ok"), bundle_input=_INPUTS, steps={"smoke": _findings(result)})


async def test_the_full_run_reports_the_baseline_and_a_stub_cannot_pass_improved(tmp_path, versions):
    from cairn.core.engine.bundle import eval_gate_predicate

    result = await asyncio.to_thread(_run, tmp_path / "train", await _leg_config(versions, "train", _INPUTS))
    stats = result.stats
    assert "train.qlora.baseline_eval_loss" in stats and stats["eval.checkpoints.passed"] is True
    assert [d.kind for d in result.output_artifacts][-1] == "model"  # package.adapter: the candidate
    out = _findings(result)
    assert not eval_gate_predicate(_gate("improved"), bundle_input=_INPUTS, steps={"train": out})


async def test_every_trainer_figure_the_promote_model_card_shows_is_in_the_findings(tmp_path, versions):
    """The person gate shows `train.train_qlora_*` findings: each must be a figure `train.qlora`
    reports (`train_qlora_steps` was empty on the GPU path; the trainer's own tests pin that path)."""
    result = await asyncio.to_thread(_run, tmp_path / "train", await _leg_config(versions, "train", _INPUTS))
    findings = _findings(result)["findings"]
    gate = next(s for s in _pipeline()["steps"] if s.get("gate") == "promote-model")
    shown = [f.split(".", 1)[1] for f in gate["show"] if f.startswith("train.train_qlora_")]
    assert "train_qlora_steps" in shown
    assert {k: findings.get(k) for k in shown if findings.get(k) is None} == {}


# ── #42: the eval leg, base vs candidate ─────────────────────────────────────────────────────

def _run_eval(workdir: Path, config: dict):
    from cairn.core.dataflow import Budget, DataflowSpec
    from cairn.core.dataflow.defs import resolve_flow_ref
    from cairn.core.dataflow.runner import run_dataflow

    stages = resolve_flow_ref("finetune/sft-eval", config, kind="ml")
    return run_dataflow(DataflowSpec(stages=stages, budget=Budget()), workdir)


def test_the_eval_leg_follows_train_and_reads_its_adapter_and_the_held_out_versions():
    ev = _with("eval")
    assert ev["local_adapter_dir"] == "$steps.train.train_qlora_local_adapter_dir"
    assert ev["adapter_uri"] == "$steps.train.train_qlora_adapter_uri"
    assert (ev["val_uri"], ev["test_uri"]) == ("$input.val_version", "$input.test_version")
    assert ev["base_model"] == _with("train")["base_model"] and ev["method"] == _with("train")["method"]
    assert ev["preset"] == "$input.eval_preset" and _pipeline()["inputs"]["eval_preset"]["default"] == "forgetting"


async def test_the_eval_flow_on_the_stub_tier_reports_both_models_and_every_figure_the_card_shows(tmp_path, versions):
    """The pack's `sft-eval` flow on the core runner with the stub tier and the train leg's stub
    adapter: the pinned val and test versions are verified and read, placeholder scores come back
    for both models (never `checked`), and every `eval.*` figure `promote-model` shows is a finding."""
    from cairn.core.engine.bundle import eval_gate_predicate

    train = await asyncio.to_thread(_run, tmp_path / "train", await _leg_config(versions, "train", _INPUTS))
    tf = _findings(train)["findings"]
    cfg = await _leg_config(versions, "eval", _INPUTS)
    cfg.update(local_adapter_dir=tf["train_qlora_local_adapter_dir"], adapter_uri=tf["train_qlora_adapter_uri"])
    cfg.pop("retry")
    result = await asyncio.to_thread(_run_eval, tmp_path / "eval", cfg)
    findings = _findings(result)["findings"]

    assert findings["eval_lm_eval_stub"] is True and findings["eval_lm_eval_checked"] is False
    assert findings["eval_lm_eval_adapter_source"] == "local_adapter_dir"
    assert findings["eval_lm_eval_val_rows"] == 6 and findings["eval_lm_eval_test_rows"] == 6
    assert [d.kind for d in result.output_artifacts] == ["report"]          # the results file
    gate = next(s for s in _pipeline()["steps"] if s.get("gate") == "promote-model")
    shown = [f.split(".", 1)[1] for f in gate["show"] if f.startswith("eval.")]
    assert len(shown) == 24 and [k for k in shown if findings.get(k) is None] == []
    # the stub's adapter cannot pass the verdict either way: `improved` reads train only
    assert not eval_gate_predicate(_gate("improved"), bundle_input=_INPUTS, steps={"train": tf, "eval": findings})


def test_improved_passes_only_when_the_candidate_beats_the_base_model():
    from cairn.core.engine.bundle import eval_gate_predicate

    def steps(eval_loss, baseline):
        f = {"train_qlora_eval_loss": eval_loss}
        if baseline is not None:
            f["train_qlora_baseline_eval_loss"] = baseline
        return {"train": {"findings": f}}

    pred = _gate("improved")
    assert eval_gate_predicate(pred, bundle_input={}, steps=steps(0.8, 1.2))
    assert not eval_gate_predicate(pred, bundle_input={}, steps=steps(1.3, 1.2))
    assert not eval_gate_predicate(pred, bundle_input={}, steps=steps(0.8, None))  # no baseline: closed


def test_the_full_run_never_trains_on_a_subsample_and_always_measures_the_baseline():
    smoke, train = _with("smoke"), _with("train")
    assert smoke["fidelity_rows"] == "$input.smoke_rows" and train["fidelity_rows"] == 0
    assert train["baseline_eval"] is True and "baseline_eval" not in smoke
    for key in ("train_uri", "eval_uri", "test_uri", "base_model", "lora_r", "lora_alpha", "learning_rate"):
        assert smoke[key] == train[key]
    # its own directory on the box, so the full run never resumes from the smoke run
    assert smoke["output_name"] == "$input.smoke_output_name" and train["output_name"] == "$input.output_name"


async def test_the_launch_door_refuses_a_test_set_given_as_training_data(versions, monkeypatch):
    """ADR-0146 through the real launch door with this pack's `sft` definition: the refusal comes
    before any record (the bridge is never called)."""
    import cairn.core.storage.artifacts as artifacts
    from cairn.core.admission_door import AdmissionRefusedError
    from cairn.core.dispatch.pipeline_launch import launch_pipeline_def
    from cairn.core.engine.pipeline import parse_pipeline_def

    monkeypatch.setattr(artifacts, "build_artifact_store", lambda *a, **k: versions)
    for name in ("demo-train", "demo-val"):
        await versions.set_dataset_version_verdict(name, 1, verdict="admitted", reasons=[], report={})

    class _Bridge:
        async def start_pipeline(self, *a, **k):
            raise AssertionError("a refused launch must not reach the bridge")

    d = parse_pipeline_def((PACK / "pipelines" / "sft.yaml").read_text(encoding="utf-8"), pack="finetune")
    swapped = {**_INPUTS, "train_version": "dataset:demo-test@1", "test_version": "dataset:demo-train@1"}
    with pytest.raises(AdmissionRefusedError) as refused:
        await launch_pipeline_def(d, inputs=swapped, bridge=_Bridge())
    reasons = refused.value.decision.refusing_reasons
    assert any("train_version" in r and "is a test version" in r for r in reasons), reasons
    assert any("test_version" in r and "is a training version" in r for r in reasons), reasons


# ── #28: a signal-killed ml leg is retried once and resumes from its own checkpoints ──────────

class _Killed(BaseException):
    """The trainer's process dies (SIGKILL, the OOM killer, a box reboot): nothing after the
    current instruction runs, no `except Exception` sees it, and no exit code is written."""


class _BoxEngine:
    """An engine whose job can die the way a remote one does (cairndev-sh/cairn#552). Each submission runs the
    flow on the core runner in a fresh job directory, as a new SSH job does, with
    `CAIRN_PARENT_RUN` set to the leg's run the way the engines stamp it. A death with no exit code
    is reported `retriable`; a stage's own error is a clean non-zero exit and is not."""

    name = "box"

    def __init__(self, root: Path, monkeypatch):
        self.root, self.monkeypatch, self.jobs = root, monkeypatch, {}

    def validate(self, spec) -> list[str]:
        return []

    async def submit(self, spec, ctx):
        from cairn.core.dataflow.engine import EngineJobHandle, EngineJobStatus
        from cairn.core.dataflow.runner import run_dataflow

        job = f"job-{len(self.jobs) + 1}"
        self.monkeypatch.setenv("CAIRN_PARENT_RUN", spec.parent_run)
        result = None
        try:
            result = await asyncio.to_thread(run_dataflow, spec, self.root / job)
            status = EngineJobStatus(state="succeeded")
        except _Killed:
            status = EngineJobStatus(state="failed", retriable=True,
                                     error="remote process exited without an exit code")
        except Exception as e:  # noqa: BLE001 — a stage's own error: the job exits 1
            status = EngineJobStatus(state="failed", error=f"remote command exit 1: {e}")
        self.jobs[job] = (status, result)
        return EngineJobHandle(job_id=job, engine=self.name)

    async def poll(self, handle):
        return self.jobs[handle.job_id][0]

    async def result(self, handle):
        return self.jobs[handle.job_id][1]

    async def cancel(self, handle):
        pass


#: The trainer's default checkpoint prefix (CAIRN_CHECKPOINT_PREFIX unset).
_PREFIX = "cairn-checkpoints"


@pytest.fixture
def s3(monkeypatch):
    """A fake S3 (moto) holding the checkpoint bucket, set the way the box sets it."""
    import boto3
    import moto

    for name in ("CAIRN_RUN_ID", "CAIRN_OUTPUT_NAME", "CAIRN_CHECKPOINT_PREFIX", "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (("AWS_ACCESS_KEY_ID", "t"), ("AWS_SECRET_ACCESS_KEY", "t"),
                        ("AWS_DEFAULT_REGION", "us-east-1"), ("CAIRN_CHECKPOINT_BUCKET", "ckpts")):
        monkeypatch.setenv(name, value)
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="ckpts")
        yield client


@pytest.fixture
def ledger(monkeypatch):
    """The budget calls the leg executor makes, one entry per call."""
    import cairn.core.dispatch.leg_executor as leg_executor

    calls: list[tuple] = []

    async def reserve(series_id, *, est_usd, **kw):
        calls.append(("reserve", series_id, est_usd))
        return len(calls)

    async def release(reservation):
        if reservation is not None:
            calls.append(("release", reservation))

    async def settle(reservation, *, actual_usd):
        calls.append(("settle", reservation))

    for name, fn in (("reserve", reserve), ("release", release), ("settle", settle)):
        monkeypatch.setattr(leg_executor, name, fn)
    return calls


async def _execute(step_id: str, cfg: dict, engine, run_id: str):
    """The leg as the bridge runs it: its `retry:` popped and parsed by the bridge's own reader,
    then the core's leg executor, under the pipeline's capped series."""
    from cairn.core.dispatch.leg_executor import execute_leg
    from cairn.core.dispatch.pipeline_bridge import _leg_retry_policy

    cfg = dict(cfg)
    retries: list[tuple[int, str]] = []

    async def on_retry(attempt, cause):
        retries.append((attempt, cause))

    out = await execute_leg(
        "flow:finetune/sft-train", inputs=cfg, engine=engine, run_id=run_id,
        retry=_leg_retry_policy(cfg.pop("retry", None)), on_retry=on_retry,
        series_id="sft", estimated_usd=1.5, spent_usd=0.0, budget_usd=25.0)
    return out, retries


def test_the_smoke_and_train_legs_declare_two_attempts():
    """ml legs default to ONE attempt (paid compute is never re-spent silently); both declare two,
    and the plan the bridge executes carries the block to the leg."""
    from cairn.core.dispatch.pipeline_bridge import _leg_retry_policy
    from cairn.core.engine.pipeline import parse_pipeline_def, plan_pipeline

    plan = plan_pipeline(parse_pipeline_def((PACK / "pipelines" / "sft.yaml").read_text(encoding="utf-8"),
                                            pack="finetune"))
    legs = {s.id: s for s in plan.steps}
    for step_id in ("smoke", "train"):
        assert legs[step_id].inputs.get("retry") == {"max_attempts": 2}, step_id
        assert _leg_retry_policy(legs[step_id].inputs["retry"]).maximum_attempts == 2


class _DiesAfterCheckpoint2:
    """The box's S3 client, which the trainer uploads its checkpoints through: the first time
    `checkpoint-2` has reached the bucket, the process dies (the upload returned, nothing after
    it runs). Every upload is recorded by checkpoint name."""

    def __init__(self, client, uploads: list[str]):
        self._client, self._uploads = client, uploads

    def __getattr__(self, name):
        return getattr(self._client, name)

    def upload_file(self, filename, bucket, key, *args, **kwargs):
        self._client.upload_file(filename, bucket, key, *args, **kwargs)
        self._uploads.append(key.split("/")[-2])
        if self._uploads == ["checkpoint-2"]:
            raise _Killed


@pytest.mark.parametrize("step_id,inputs", [
    ("train", {"max_steps": 6, "save_steps": 2}),
    ("smoke", {"smoke_steps": 6, "save_steps": 2}),
])
async def test_a_leg_killed_mid_run_is_retried_once_and_continues_from_its_checkpoint(
        tmp_path, versions, s3, ledger, monkeypatch, step_id, inputs):
    import boto3

    run_id = f"run-{step_id}-1"
    uploads: list[str] = []
    real_client = boto3.client
    monkeypatch.setattr(boto3, "client", lambda service, *a, **k: (
        _DiesAfterCheckpoint2(real_client(service, *a, **k), uploads) if service == "s3"
        else real_client(service, *a, **k)))
    engine = _BoxEngine(tmp_path / "box", monkeypatch)
    cfg = await _leg_config(versions, step_id, {**_INPUTS, **inputs})
    out, retries = await _execute(step_id, cfg, engine, run_id)

    assert out.state == "succeeded", out.error
    assert list(engine.jobs) == ["job-1", "job-2"]
    assert retries == [(1, "remote process exited without an exit code")]
    # the step count continued: the retry wrote checkpoint-4 and -6, never checkpoint-2 again
    assert uploads == ["checkpoint-2", "checkpoint-4", "checkpoint-6"]
    metrics = json.loads((tmp_path / "box" / "job-2" / "train_result.json").read_text())["metrics"]
    resumed = Path(metrics["resumed_from_checkpoint"])
    # pulled from S3 into the new job's directory
    assert resumed.name == "checkpoint-2" and resumed.is_relative_to(tmp_path / "box" / "job-2")
    assert metrics["checkpoint_uri"] == f"s3://ckpts/{_PREFIX}/{run_id}/"
    keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket="ckpts")["Contents"])
    assert keys == [f"{_PREFIX}/{run_id}/checkpoint-{n}/trainer_state.json" for n in (2, 4, 6)]
    # each attempt reserved its own estimate: the killed one released it, the retry settled
    assert ledger == [("reserve", "sft", 1.5), ("release", 1), ("reserve", "sft", 1.5), ("settle", 3)]


async def test_a_leg_that_fails_on_its_own_error_is_not_retried(tmp_path, versions, s3, ledger, monkeypatch):
    """The train version's bytes no longer match the hash it was pinned with at launch: the
    smoke run's subsample refuses it. A clean failure, not a death: one attempt, no retry."""
    engine = _BoxEngine(tmp_path / "box", monkeypatch)
    cfg = await _leg_config(versions, "smoke", _INPUTS)
    Path(cfg["train_uri"].split("#", 1)[0].removeprefix("file://")).write_text("changed\n")
    out, retries = await _execute("smoke", cfg, engine, "run-smoke-2")

    assert out.state == "failed" and "remote command exit 1" in out.error
    assert list(engine.jobs) == ["job-1"] and retries == []
    assert ledger == [("reserve", "sft", 1.5), ("release", 1)]
