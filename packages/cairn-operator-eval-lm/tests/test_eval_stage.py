"""``eval.lm_eval`` — base vs base + adapter on the GPU box, as an ml-flow stage (cairn-extensions#42).

Every test runs WITHOUT torch, transformers, peft or lm-eval: the stage reaches them only through
``operator_eval_lm.inprocess.make_backend``, which these tests replace with :class:`FakeBackend`,
and the stage's clock (``flow_stages._now``) is a fake one the backend advances, so the time probe
is exercised on exact numbers. What is proven:

* the kernel is registered in the engine-neutral ``cairn.flow_stages`` group and imports nothing
  heavy;
* both models are scored with identical calls (task, limit, few-shot, batch size, seeds), the
  candidate on the same loaded base; MMLU's limit is its own (per subject);
* deltas, their standard errors, the regressions and the worst delta, in each metric's direction;
* the held-out loss: one named method, the same rendered rows for both models;
* the time probe: about 50 documents first, a projection that matches a run whose cost is linear in
  batches, and a refusal (raised, before the full run) when it does not fit the leg;
* the results file: written, hashed, uploaded beside the run's checkpoints (a fake S3), and the
  stage's ``report`` product;
* what fails the leg (a split whose bytes changed, a promoted tag, no budget) and what is
  "not evaluated" instead (no harness, a harness error); the stub tier is never ``checked``;
* nothing is written to the process environment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from operator_eval_lm import box_io, flow_stages, inprocess, presets, scoring
from operator_eval_lm.inprocess import Loaded, Timing


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def now(self) -> float:
        return self.t


class FakeBackend:
    """lm-eval's ``hf`` path, faked: every call is recorded; time passes per batch of requests
    (4 requests per document) plus a fixed per-call overhead; ``scores`` sets the values."""

    name = "fake"

    def __init__(self, clock: Clock, *, per_batch_s: float = 0.0, overhead_s: float = 0.0,
                 candidate_factor: float = 1.0, scores: dict | None = None, fail_on: str = "",
                 adapter_base: str = "", missing_adapter: bool = False) -> None:
        self.clock, self.per_batch_s, self.overhead_s = clock, per_batch_s, overhead_s
        self.candidate_factor, self.scores, self.fail_on = candidate_factor, scores or {}, fail_on
        self._adapter_base = adapter_base
        self.missing_adapter = missing_adapter
        self.calls: list[tuple] = []
        self.loss_calls: list[tuple] = []
        self.loads: list[tuple] = []
        self.attached: list[tuple] = []

    def versions(self) -> dict[str, str]:
        return {"lm_eval": "0.4.13"}

    def load(self, base_model: str, *, quantize_4bit: bool, seed: int) -> Loaded:
        self.loads.append((base_model, quantize_4bit, seed))
        return Loaded(model="base", tokenizer=None, quantization="4-bit nf4 on load (fake)")

    def chat_template(self, loaded: Loaded, adapter: str) -> tuple[str, str]:
        return "TEMPLATE", "adapter"

    def adapter_base(self, adapter: str) -> str:
        return self._adapter_base

    def check_adapter(self, adapter: str) -> None:
        if self.missing_adapter:
            raise OSError(f"{adapter} does not appear to have a file named adapter_config.json")

    def render(self, loaded: Loaded, messages: list[dict], template: str) -> str:
        return template + "|" + "|".join(f"{m['role']}:{m['content']}" for m in messages)

    def attach(self, loaded: Loaded, adapter: str) -> Loaded:
        self.attached.append((loaded.model, adapter))
        return Loaded(model="candidate", tokenizer=None, quantization=loaded.quantization, adapter=adapter)

    def evaluate(self, loaded: Loaded, *, task: str, limit: int, num_fewshot: int, batch_size: int,
                 seed: int) -> tuple[dict, Timing]:
        self.calls.append((loaded.model, task, limit, num_fewshot, batch_size, seed))
        if self.fail_on and self.fail_on == f"{loaded.model}:{task}":
            raise RuntimeError("CUDA out of memory")
        leaves = [f"mmlu_{i:02d}" for i in range(57)] if task == "mmlu" else [task]
        n = {leaf: {"original": 2000, "effective": min(limit, 2000) if limit else 2000} for leaf in leaves}
        docs = sum(c["effective"] for c in n.values())
        requests = 4 * docs
        factor = self.candidate_factor if loaded.model == "candidate" else 1.0
        model_s = math.ceil(requests / batch_size) * self.per_batch_s * factor
        self.clock.t += model_s + self.overhead_s
        value, stderr, higher = self.scores.get((loaded.model, task), (0.5, 0.02, True))
        return ({"results": {task: {"acc,none": value, "acc_stderr,none": stderr, "alias": task}},
                 "higher_is_better": {task: {"acc": higher}}, "n-samples": n,
                 "versions": {task: 1.0}}, Timing(requests=requests, model_s=model_s))

    def split_loss(self, loaded: Loaded, texts: list[str], *, max_seq_len: int) -> tuple[float, int]:
        self.loss_calls.append((loaded.model, tuple(texts), max_seq_len))
        tokens = sum(len(t) for t in texts)
        return (0.25 if loaded.model == "candidate" else 1.5) * tokens, tokens


@pytest.fixture
def clock(monkeypatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(flow_stages, "_now", c.now)
    return c


@pytest.fixture(autouse=True)
def _no_bucket(monkeypatch):
    for name in ("CAIRN_CHECKPOINT_BUCKET", "CAIRN_CHECKPOINT_PREFIX", "CAIRN_RUN_ID", "CAIRN_PARENT_RUN",
                 "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(name, raising=False)


def _use(monkeypatch, backend) -> None:
    monkeypatch.setattr(inprocess, "make_backend", lambda *, stub: backend)


def _split(tmp_path: Path, name: str, n: int = 4) -> str:
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps({"instruction": "Name the entity.", "input": f"{name} {i}",
                                        "output": f"IfcWall {i}"}) + "\n" for i in range(n))
                    + json.dumps({"text": "a row the trainer would refuse"}) + "\n")
    return f"file://{path}#sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"


def _config(tmp_path: Path, **kw) -> dict:
    adapter = tmp_path / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}")
    cfg = {"base_model": "Qwen/Qwen3-4B-Base", "local_adapter_dir": str(adapter),
           "adapter_uri": "s3://bucket/never-read/", "val_uri": _split(tmp_path, "val"),
           "test_uri": _split(tmp_path, "test"), "time_budget_s": 10800}
    cfg.update(kw)
    return cfg


def _run(tmp_path: Path, **kw) -> dict:
    return flow_stages.eval_lm_eval(tmp_path / "job", _config(tmp_path, **kw), [])


# ── registration ──────────────────────────────────────────────────────────────────────────────

def test_the_stage_is_registered_in_the_flow_stage_group_and_imports_nothing_heavy():
    from importlib.metadata import entry_points

    (ep,) = [e for e in entry_points(group="cairn.flow_stages") if e.name == "eval.lm_eval"]
    fn = ep.load()
    assert fn is flow_stages.eval_lm_eval and fn.risk_class == "external_write"
    assert not getattr(fn, "implements", ())   # uncontracted, as compress/distill (no stage contract yet)
    code = ("import sys, operator_eval_lm.flow_stages, operator_eval_lm.inprocess, operator_eval_lm.box_io; "
            "print(sorted(m for m in ('torch', 'transformers', 'peft', 'lm_eval', 'boto3') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_the_core_resolves_the_stage_and_its_risk():
    from cairn.core.dataflow.defs import stage_risk_class

    assert stage_risk_class("eval.lm_eval") == "external_write"


# ── both models, identical settings ───────────────────────────────────────────────────────────

def test_both_models_are_scored_with_identical_calls_on_the_same_loaded_base(tmp_path, clock, monkeypatch):
    fake = FakeBackend(clock)
    _use(monkeypatch, fake)
    out = _run(tmp_path)

    assert fake.loads == [("Qwen/Qwen3-4B-Base", True, 1234)]           # once, 4-bit (qlora)
    assert fake.attached == [("base", str(tmp_path / "adapter"))]       # the same loaded weights
    full = {"arc_easy": 500, "arc_challenge": 500, "hellaswag": 500, "winogrande": 500, "piqa": 500, "mmlu": 10}
    base = [c[1:] for c in fake.calls if c[0] == "base" and c[2] == full[c[1]]]
    cand = [c[1:] for c in fake.calls if c[0] == "candidate"]
    assert base == cand == [(t, limit, 0, 16, 1234) for t, limit in full.items()]
    # base: the probe first (smaller limits), then the full pass; the candidate gets the full pass only
    assert [c for c in fake.calls if c[0] == "base"][:6] == [
        ("base", t, 10 if t != "mmlu" else 1, 0, 16, 1234) for t in full]
    assert out["n_mmlu"] == 570 and out["n_arc_easy"] == 500          # MMLU: 10 per subject × 57
    assert out["checked"] is True and out["note"] == ""
    assert out["preset"] == "forgetting" and out["seed"] == 1234 and out["batch_size"] == 16
    assert out["quantization"] == "4-bit nf4 on load (fake)" and out["lm_eval_version"] == "0.4.13"


def test_lora_loads_the_base_unquantised_and_a_mismatched_base_is_said(tmp_path, clock, monkeypatch):
    fake = FakeBackend(clock, adapter_base="unsloth/qwen3-4b-base-unsloth-bnb-4bit")
    _use(monkeypatch, fake)
    out = _run(tmp_path, method="lora")
    assert fake.loads == [("Qwen/Qwen3-4B-Base", False, 1234)]
    assert out["adapter_base_model"] == "unsloth/qwen3-4b-base-unsloth-bnb-4bit"
    assert "trained on 'unsloth/qwen3-4b-base-unsloth-bnb-4bit'" in out["note"]


def test_deltas_their_stderr_regressions_and_the_worst_delta(tmp_path, clock, monkeypatch):
    fake = FakeBackend(clock, scores={
        ("base", "arc_easy"): (0.76, 0.02, True), ("candidate", "arc_easy"): (0.70, 0.02, True),   # −0.06, > se
        ("base", "hellaswag"): (0.65, 0.02, True), ("candidate", "hellaswag"): (0.64, 0.02, True),  # −0.01, < se
        ("base", "piqa"): (0.30, 0.02, False), ("candidate", "piqa"): (0.20, 0.02, False),          # lower is better
    })
    _use(monkeypatch, fake)
    out = _run(tmp_path)

    assert out["base_arc_easy_acc"] == 0.76 and out["candidate_arc_easy_acc"] == 0.70
    assert out["delta_arc_easy_acc"] == pytest.approx(-0.06)
    assert out["delta_arc_easy_acc_stderr"] == pytest.approx(math.sqrt(2) * 0.02, abs=1e-6)
    assert out["base_arc_easy_acc_stderr"] == 0.02 and out["candidate_arc_easy_acc_stderr"] == 0.02
    assert out["delta_piqa_acc"] == pytest.approx(-0.10)                # an improvement: lower is better
    assert out["regressions"] == 1 and out["regressed"] == "arc_easy_acc"
    assert out["worst_delta"] == pytest.approx(-0.06) and out["worst_delta_metric"] == "arc_easy_acc"
    assert out["metrics_compared"] == 6


def test_the_held_out_loss_is_one_named_method_on_the_same_rows_for_both(tmp_path, clock, monkeypatch):
    fake = FakeBackend(clock)
    _use(monkeypatch, fake)
    out = _run(tmp_path, max_seq_len=1024)

    full = [c for c in fake.loss_calls if len(c[1]) == 4]               # the probe scored a few rows
    assert [c[0] for c in full] == ["base", "base", "candidate", "candidate"]
    assert full[0][1] == full[2][1] and full[1][1] == full[3][1]        # val, test: the same texts
    assert full[0][1][0] == "TEMPLATE|user:Name the entity.\n\nval 0|assistant:IfcWall 0"
    assert {c[2] for c in fake.loss_calls} == {1024}
    assert out["split_loss_method"] == flow_stages.SPLIT_LOSS_METHOD
    assert out["split_loss_method"].startswith("full-sequence mean token NLL")
    assert out["base_val_loss"] == 1.5 and out["candidate_val_loss"] == 0.25
    assert out["delta_test_loss"] == pytest.approx(-1.25)
    assert out["val_rows"] == 4 and out["val_rows_skipped"] == 1 and out["val_tokens"] > 0
    assert out["chat_template_source"] == "adapter"


# ── the time probe ────────────────────────────────────────────────────────────────────────────

def test_the_probe_scores_about_fifty_documents_in_proportion_to_the_limits():
    plan = [(t, presets.load("forgetting").limit_for(t)) for t in presets.load("forgetting").tasks]
    limits = flow_stages.probe_limits(plan, 50)
    assert limits == {"arc_easy": 10, "arc_challenge": 10, "hellaswag": 10, "winogrande": 10, "piqa": 10,
                      "mmlu": 1}                                         # 50 + one per MMLU subject


def test_a_run_projected_past_the_leg_is_refused_before_the_full_run(tmp_path, clock, monkeypatch):
    """10 s per batch of 16 requests: about 2.1 h per pass, past a 3 h leg for both."""
    fake = FakeBackend(clock, per_batch_s=10.0)
    _use(monkeypatch, fake)
    with pytest.raises(flow_stages.TimeBudgetExceeded) as refused:
        _run(tmp_path)

    msg = str(refused.value)
    assert "refused before the full run" in msg and "> 2.70 h" in msg and "the leg's 3 h" in msg
    assert "mmlu" in msg and "570 documents" in msg
    assert {c[2] for c in fake.calls} == {10, 1}                        # the probe only
    assert fake.attached == [] and [c for c in fake.calls if c[0] == "candidate"] == []
    assert not (tmp_path / "job" / flow_stages.RESULTS_FILE).exists()


def test_a_run_that_fits_is_admitted_and_the_projection_matches_a_linear_run(tmp_path, clock, monkeypatch):
    """Cost linear in batches, the candidate 1.25× the base: the projection is the run's length."""
    fake = FakeBackend(clock, per_batch_s=1.0, candidate_factor=flow_stages.CANDIDATE_FACTOR)
    _use(monkeypatch, fake)
    out = _run(tmp_path)
    assert out["checked"] is True
    assert out["projected_s"] == pytest.approx(out["elapsed_s"], abs=1.0)
    assert out["probe_documents"] == 50 + 57
    assert out["probe_model_s_per_request"] > 0
    assert out["projected_s"] < 10800 * 0.9


def test_the_projection_scales_model_time_by_batches_and_counts_overhead_once():
    probe = flow_stages.TaskProbe(task="arc_easy", limit=500, probe_limit=10, wall_s=30.0, model_s=6.0,
                                  requests=40, documents=10, full_documents=500)
    # 40 requests = 3 batches of 16 (2 s each); 2000 requests = 125 batches; 24 s of lm-eval overhead
    assert probe.projected_s(16) == pytest.approx(125 * 2.0 + 24.0)
    assert probe.projected_s(16, 1.25) == pytest.approx(125 * 2.0 * 1.25 + 24.0)


def test_outside_the_stub_tier_a_budget_is_required(tmp_path, clock, monkeypatch):
    _use(monkeypatch, FakeBackend(clock))
    with pytest.raises(ValueError, match="time_budget_s is required"):
        _run(tmp_path, time_budget_s=0)


# ── the results file ──────────────────────────────────────────────────────────────────────────

def test_the_results_file_is_the_stages_report_product(tmp_path, clock, monkeypatch):
    _use(monkeypatch, FakeBackend(clock))
    out = _run(tmp_path)
    path = tmp_path / "job" / flow_stages.RESULTS_FILE
    assert out["output"] == out["results_uri"] == str(path)             # no bucket: on the box
    assert out["kind"] == "report" and out["bytes"] == path.stat().st_size
    assert out["content_hash"] == hashlib.sha256(path.read_bytes()).hexdigest()
    report = json.loads(path.read_text())
    assert report["checked"] is True and report["split_loss"]["method"] == flow_stages.SPLIT_LOSS_METHOD
    row = next(r for r in report["scores"] if r["key"] == "mmlu_acc")
    assert row["documents_base"] == row["documents_candidate"] == 570 and row["delta_stderr"] is not None
    assert report["settings"]["seed"] == 1234 and report["versions"] == {"lm_eval": "0.4.13"}
    assert [t["limit"] for t in report["tasks"]] == [500, 500, 500, 500, 500, 10]


@pytest.fixture
def s3(monkeypatch):
    import boto3
    import moto

    for name, value in (("AWS_ACCESS_KEY_ID", "t"), ("AWS_SECRET_ACCESS_KEY", "t"),
                        ("AWS_DEFAULT_REGION", "us-east-1")):
        monkeypatch.setenv(name, value)
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="ckpts")
        yield client


def test_the_results_file_is_uploaded_beside_the_runs_checkpoints(tmp_path, clock, monkeypatch, s3):
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "ckpts")
    monkeypatch.setenv("CAIRN_PARENT_RUN", "run-eval-1")                 # the engine stamps the leg's run
    _use(monkeypatch, FakeBackend(clock))
    out = _run(tmp_path)
    key = "cairn-checkpoints/run-eval-1/eval/lm_eval_results.json"
    assert out["output"] == out["results_uri"] == f"s3://ckpts/{key}"
    body = s3.get_object(Bucket="ckpts", Key=key)["Body"].read()
    assert hashlib.sha256(body).hexdigest() == out["content_hash"]
    assert "results_upload_error" not in out


def test_a_failed_upload_keeps_the_scores_and_says_so(tmp_path, clock, monkeypatch, s3):
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "no-such-bucket")
    monkeypatch.setenv("CAIRN_RUN_ID", "run-eval-2")
    _use(monkeypatch, FakeBackend(clock))
    out = _run(tmp_path)
    assert out["checked"] is True and out["results_uri"] == str(tmp_path / "job" / flow_stages.RESULTS_FILE)
    assert "NoSuchBucket" in out["results_upload_error"]


# ── the adapter and the splits ────────────────────────────────────────────────────────────────

def test_the_adapter_is_found_on_the_box_then_by_its_uri(tmp_path, s3):
    local = tmp_path / "train-output"
    local.mkdir()
    assert box_io.resolve_adapter(local_dir=str(local), uri="hf://x/y", dest=tmp_path / "d") == (str(local), "local_adapter_dir")
    assert box_io.resolve_adapter(local_dir=str(tmp_path / "gone"), uri="hf://cairndev/a", dest=tmp_path / "d") == ("cairndev/a", "hf")
    assert box_io.resolve_adapter(local_dir="", uri="cairndev/a", dest=tmp_path / "d") == ("cairndev/a", "hub")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        s3.put_object(Bucket="ckpts", Key=f"cairn-checkpoints/run-7/adapter/{name}", Body=b"x")
    got = box_io.resolve_adapter(local_dir="", uri="s3://ckpts/cairn-checkpoints/run-7/adapter/", dest=tmp_path / "s3a")
    assert got == (str(tmp_path / "s3a"), "s3") and (tmp_path / "s3a" / "adapter_model.safetensors").is_file()


def test_a_promoted_tag_or_a_missing_adapter_fails_the_leg(tmp_path):
    with pytest.raises(box_io.FetchError, match="promoted tag"):
        box_io.resolve_adapter(local_dir="", uri="ifc-bim-qwen3-4b@candidate", dest=tmp_path)
    with pytest.raises(box_io.FetchError, match="not a directory"):
        box_io.resolve_adapter(local_dir="", uri="/no/such/adapter", dest=tmp_path)
    with pytest.raises(box_io.FetchError, match="stub trainer"):
        box_io.resolve_adapter(local_dir="", uri="stub://adapters/x", dest=tmp_path)


def test_a_split_whose_bytes_changed_since_registration_fails_the_leg(tmp_path, clock, monkeypatch):
    fake = FakeBackend(clock)
    _use(monkeypatch, fake)
    cfg = _config(tmp_path)
    Path(cfg["test_uri"].split("#")[0].removeprefix("file://")).write_text("changed\n")
    with pytest.raises(box_io.FetchError, match="Refusing to score on unverified data"):
        flow_stages.eval_lm_eval(tmp_path / "job", cfg, [])
    assert fake.loads == []                                             # before any model loads


def test_an_unresolved_dataset_ref_fails_the_leg(tmp_path):
    with pytest.raises(box_io.FetchError, match="unresolved"):
        box_io.fetch_dataset("dataset:demo-test@1", tmp_path)


# ── not evaluated, never a score ──────────────────────────────────────────────────────────────

def _no_scores(out: dict) -> bool:
    return not any(k.startswith(("base_", "candidate_", "delta_")) and k != "base_model" for k in out)


def test_no_harness_on_the_box_is_not_evaluated_not_a_failure(tmp_path, clock, monkeypatch):
    def missing(*, stub):
        raise ImportError("No module named 'lm_eval'")

    monkeypatch.setattr(inprocess, "make_backend", missing)
    out = _run(tmp_path)
    assert out["checked"] is False and "harness unavailable" in out["note"] and "[box]" in out["note"]
    assert _no_scores(out) and "output" not in out


def test_a_harness_failure_mid_run_reports_no_scores(tmp_path, clock, monkeypatch):
    _use(monkeypatch, FakeBackend(clock, fail_on="candidate:mmlu"))
    out = _run(tmp_path)
    assert out["checked"] is False and out["note"] == "harness failed: RuntimeError: CUDA out of memory"
    assert _no_scores(out) and "output" not in out
    assert all(isinstance(v, (str, int, float, bool)) for v in out.values())


def test_the_stub_tier_produces_placeholder_scores_and_is_never_checked(tmp_path, monkeypatch):
    out = flow_stages.eval_lm_eval(tmp_path / "job", {**_config(tmp_path), "stub": True, "time_budget_s": 0,
                                                      "local_adapter_dir": "", "adapter_uri": "stub://adapters/x"}, [])
    assert out["stub"] is True and out["checked"] is False and "placeholders" in out["note"]
    assert out["harness_backend"] == "stub" and out["adapter_source"] == "stub"
    assert "delta_arc_easy_acc" in out and "candidate_test_loss" in out and out["kind"] == "report"


def test_nothing_is_written_to_the_process_environment(tmp_path, clock, monkeypatch, s3):
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "ckpts")
    monkeypatch.setenv("CAIRN_PARENT_RUN", "run-eval-3")
    _use(monkeypatch, FakeBackend(clock))
    before = dict(os.environ)
    _run(tmp_path)
    assert dict(os.environ) == before


def test_the_chatml_copy_is_the_trainers():
    """The stage carries a copy of the trainer's ChatML (an operator pack never imports another, and
    neither does this test: the trainer's module is read as data). The split loss renders rows
    the way training did only if the two are the same text."""
    import ast
    from importlib.util import find_spec

    spec = find_spec("operator_training")      # located, not executed
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("cairn-operator-training is not installed")
    source = Path(list(spec.submodule_search_locations)[0]) / "chat_template.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    (value,) = [ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and [t.id for t in n.targets if isinstance(t, ast.Name)] == ["CHATML"]]
    assert inprocess.CHATML == value


# ── presets and parsing ───────────────────────────────────────────────────────────────────────

def test_the_forgetting_presets_size_mmlu_per_subject_and_add_ifeval_apart():
    f, fi = presets.load("forgetting"), presets.load("forgetting-ifeval")
    assert f.tasks == ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa", "mmlu"]
    assert [f.limit_for(t) for t in f.tasks] == [500] * 5 + [10] and f.batch_size == 16
    assert fi.tasks == f.tasks + ["ifeval"] and fi.limit_for("ifeval") == 541 and fi.limit_for("mmlu") == 10
    assert presets.load("smoke").limits == {}                            # the existing presets are unchanged


def test_the_served_operator_refuses_a_preset_that_sizes_tasks_apart():
    import asyncio

    from cairn.core.operator.testing import run_operator_for_test
    from operator_eval_lm.operators.lm_eval import LmEvalOp
    from operator_eval_lm.schemas.lm_eval import LmEvalInput

    out = asyncio.run(run_operator_for_test(LmEvalOp, LmEvalInput(
        enabled=True, endpoint_url="http://candidate.example:8000/v1", preset="forgetting")))
    assert out.checked is False and "sizes its tasks apart" in out.note


def test_task_scores_read_a_groups_aggregate_its_stderr_and_direction():
    results = {
        "results": {"mmlu": {"acc,none": 0.77, "acc_stderr,none": 0.01, "alias": "mmlu"},
                    "mmlu_humanities": {"acc,none": 0.7}, "gsm8k": {
                        "exact_match,strict-match": 0.6, "exact_match_stderr,strict-match": "N/A",
                        "exact_match,flexible-extract": 0.62}},
        "higher_is_better": {"mmlu": {"acc": True}, "gsm8k": {"exact_match": True}},
        "n-samples": {"mmlu_a": {"original": 100, "effective": 10}, "mmlu_b": {"original": 5, "effective": 5}},
    }
    (mmlu,) = scoring.task_scores(results, "mmlu")
    assert (mmlu.key, mmlu.value, mmlu.stderr, mmlu.higher_is_better) == ("acc", 0.77, 0.01, True)
    gsm = scoring.task_scores(results, "gsm8k")
    assert [(s.key, s.stderr) for s in gsm] == [("exact_match_flexible-extract", None),
                                                 ("exact_match_strict-match", None)]
    assert scoring.documents(results, 10) == (15, 15) and scoring.documents(results, 0) == (15, 105)


def test_a_missing_adapter_fails_before_any_model_loads(tmp_path, clock, monkeypatch):
    """A typo'd Hub id or path is found by reading the adapter's config (a few KB) first: the leg
    fails in seconds, with no base load, no probe and no pass (#45 review N4)."""
    fake = FakeBackend(clock, missing_adapter=True)
    _use(monkeypatch, fake)
    with pytest.raises(box_io.FetchError) as failed:
        _run(tmp_path)
    assert "can't be read" in str(failed.value)
    assert fake.loads == [] and fake.calls == [] and fake.attached == []


def test_the_refusal_headline_survives_the_log_tail(tmp_path, clock, monkeypatch):
    """The remote job keeps only the last 500 characters of its log as the run's error, so the
    projected-vs-allowed headline is the END of the message (#45 review N5)."""
    fake = FakeBackend(clock, per_batch_s=10.0)
    _use(monkeypatch, fake)
    with pytest.raises(flow_stages.TimeBudgetExceeded) as refused:
        _run(tmp_path)
    tail = str(refused.value)[-500:]
    assert "refused before the full run" in tail and "> 2.70 h" in tail
