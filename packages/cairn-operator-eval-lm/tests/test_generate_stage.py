"""``eval.generate`` — the base model's and the candidate's answers to held-out prompts, on the GPU
box, as an ml-flow stage (cairn-extensions#41).

Every test runs WITHOUT torch, transformers, peft or lm-eval: the stage reaches them only through
``operator_eval_lm.inprocess.make_backend``, which these tests replace with :class:`GenFake`, and the
stage's clock (``flow_stages._now``) is a fake one the backend advances. What is proven:

* the kernel is registered in the engine-neutral ``cairn.flow_stages`` group and imports nothing
  heavy;
* the prompts are drawn from the test split with the seed, the same ones for both models, split the
  way the trainer renders a row (the last assistant turn is the reference);
* both models answer the same rendered prompts (with the generation prompt) with the same decoding,
  the candidate on the same loaded base;
* the baseline scorers (exact match, contains), each model's mean and the delta;
* the samples file is the stage's ``report`` product, uploaded beside the run's checkpoints;
* the first batch projects the run, and a run past the leg's timeout is refused before the rest;
* what fails the leg and what is "not evaluated" instead; the stub tier scores both models and is
  never ``checked``.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from operator_eval_lm import box_io, flow_stages, generate_stage, inprocess
from operator_eval_lm.inprocess import Loaded, Timing


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def now(self) -> float:
        return self.t


class GenFake:
    """The ``hf`` generation path, faked. The base model answers ``wrong: <prompt>``; the candidate
    answers the reference (read back from the prompt) for the prompts in ``right``. Each batch costs
    ``per_batch_s`` (the candidate's ``candidate_factor`` times that)."""

    name = "fake"

    def __init__(self, clock: Clock, *, per_batch_s: float = 0.0, candidate_factor: float = 1.0,
                 right: set[str] | None = None, wrapped: set[str] | None = None, fail: str = "") -> None:
        self.clock, self.per_batch_s, self.candidate_factor = clock, per_batch_s, candidate_factor
        self.right, self.wrapped, self.fail = right or set(), wrapped or set(), fail
        self.loads: list[tuple] = []
        self.attached: list[tuple] = []
        self.generated: list[tuple] = []

    def versions(self) -> dict[str, str]:
        return {"lm_eval": "0.4.13", "torch": "2.11.0", "device": "fake-gpu"}

    def load(self, base_model: str, *, quantize_4bit: bool, seed: int) -> Loaded:
        self.loads.append((base_model, quantize_4bit, seed))
        return Loaded(model="base", tokenizer=None, quantization="4-bit nf4 on load (fake)")

    def chat_template(self, loaded: Loaded, adapter: str) -> tuple[str, str]:
        return "TEMPLATE", "adapter"

    def adapter_base(self, adapter: str) -> str:
        return ""

    def check_adapter(self, adapter: str) -> None:
        return None

    def render(self, loaded: Loaded, messages: list[dict], template: str, *,
               add_generation_prompt: bool = False) -> str:
        text = template + "|" + "|".join(f"{m['role']}:{m['content']}" for m in messages)
        return text + ("|assistant:" if add_generation_prompt else "")

    def attach(self, loaded: Loaded, adapter: str) -> Loaded:
        self.attached.append((loaded.model, adapter))
        return Loaded(model="candidate", tokenizer=None, quantization=loaded.quantization, adapter=adapter)

    def generate(self, loaded: Loaded, prompts: list[str], *, max_new_tokens: int, max_seq_len: int,
                 seed: int) -> tuple[list[str], Timing]:
        self.generated.append((loaded.model, tuple(prompts), max_new_tokens, max_seq_len, seed))
        if self.fail == loaded.model:
            raise RuntimeError("CUDA out of memory")
        cost = self.per_batch_s * (self.candidate_factor if loaded.model == "candidate" else 1.0)
        self.clock.t += cost
        out = []
        for p in prompts:
            question = p.split("|user:")[-1].split("|")[0].split("\n\n")[-1]   # "test <i>"
            answer = f"IfcWall {question.split()[-1]}"           # the fixture's reference for this question
            if loaded.model == "candidate" and question in self.right:
                out.append(answer)
            elif loaded.model == "candidate" and question in self.wrapped:
                out.append(f"Sure:  {answer}\n")
            else:
                out.append(f"wrong: {question}")
        return out, Timing(requests=len(prompts), model_s=cost)


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


def _split(tmp_path: Path, n: int = 12, name: str = "test.jsonl") -> str:
    """``n`` alpaca rows (question ``test <i>``, answer ``IfcWall <i>``) and one the trainer would refuse."""
    path = tmp_path / name
    path.write_text("".join(json.dumps({"instruction": "Name the entity.", "input": f"test {i}",
                                        "output": f"IfcWall {i}"}) + "\n" for i in range(n))
                    + json.dumps({"text": "a row the trainer would refuse"}) + "\n")
    return f"file://{path}#sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"


def _config(tmp_path: Path, **kw) -> dict:
    adapter = tmp_path / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}")
    cfg = {"base_model": "Qwen/Qwen3-4B-Base", "local_adapter_dir": str(adapter), "test_uri": _split(tmp_path),
           "sample_n": 5, "batch_size": 2, "time_budget_s": 3600}
    cfg.update(kw)
    return cfg


def _run(tmp_path: Path, **kw) -> dict:
    return generate_stage.eval_generate(tmp_path / "job", _config(tmp_path, **kw), [])


# ── registration ──────────────────────────────────────────────────────────────────────────────

def test_the_stage_is_registered_in_the_flow_stage_group_and_imports_nothing_heavy():
    from importlib.metadata import entry_points

    from cairn.core.dataflow.defs import stage_risk_class

    (ep,) = [e for e in entry_points(group="cairn.flow_stages") if e.name == "eval.generate"]
    fn = ep.load()
    assert fn is generate_stage.eval_generate and fn.risk_class == "external_write"
    assert not getattr(fn, "implements", ())   # uncontracted, as eval.lm_eval
    assert stage_risk_class("eval.generate") == "external_write"
    code = ("import sys, operator_eval_lm.generate_stage; "
            "print(sorted(m for m in ('torch', 'transformers', 'peft', 'lm_eval', 'boto3') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


# ── the prompts ───────────────────────────────────────────────────────────────────────────────

def test_the_prompts_are_drawn_with_the_seed_and_split_as_the_trainer_renders_a_row():
    rows = [{"instruction": "Q", "input": str(i), "output": f"A{i}"} for i in range(20)] + [{"text": "refused"}]
    first, again, other = (generate_stage.draw(rows, 5, 7), generate_stage.draw(rows, 5, 7),
                           generate_stage.draw(rows, 5, 8))
    assert first == again and first != other
    assert [r for r, _, _ in first] == sorted(r for r, _, _ in first) and len(first) == 5
    row, turns, reference = first[0]
    assert turns == [{"role": "user", "content": f"Q\n\n{row}"}] and reference == f"A{row}"
    assert [r for r, _, _ in generate_stage.draw(rows, 50, 7)] == list(range(20))   # every usable row
    chat = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                         {"role": "assistant", "content": "R"}]}
    assert generate_stage.draw([chat], 1, 0) == [(0, chat["messages"][:2], "R")]


def test_the_baseline_scorers():
    assert generate_stage.score("IfcWall  1\n", "IfcWall 1") == {"exact_match": True, "contains": True}
    assert generate_stage.score("Sure: IfcWall 1.", "IfcWall 1") == {"exact_match": False, "contains": True}
    assert generate_stage.score("IfcSlab", "IfcWall 1") == {"exact_match": False, "contains": False}
    assert generate_stage.score("anything", "") == {"exact_match": False, "contains": False}


# ── both models, the same prompts and decoding ────────────────────────────────────────────────

def test_both_models_answer_the_same_prompts_with_the_same_decoding_on_the_same_base(tmp_path, clock, monkeypatch):
    fake = GenFake(clock)
    _use(monkeypatch, fake)
    out = _run(tmp_path)

    assert fake.loads == [("Qwen/Qwen3-4B-Base", True, 1234)]           # once, 4-bit (qlora)
    assert fake.attached == [("base", str(tmp_path / "adapter"))]       # the same loaded weights
    base = [g[1:] for g in fake.generated if g[0] == "base"]
    cand = [g[1:] for g in fake.generated if g[0] == "candidate"]
    assert base == cand and [len(g[0]) for g in base] == [2, 2, 1]      # batches of 2, five prompts
    assert all(p.endswith("|assistant:") and p.startswith("TEMPLATE|user:") for g in base for p in g[0])
    assert {g[1:] for g in base} == {(512, 2048, 1234)}
    assert out["samples"] == 5 and out["split_rows"] == 13 and out["sample_n"] == 5
    assert out["checked"] is True and out["note"] == "" and out["decoding"] == "greedy"
    assert out["quantization"] == "4-bit nf4 on load (fake)" and out["torch_version"] == "2.11.0"
    assert out["device"] == "fake-gpu" and out["harness_backend"] == "fake"


def test_lora_loads_the_base_unquantised(tmp_path, clock, monkeypatch):
    fake = GenFake(clock)
    _use(monkeypatch, fake)
    _run(tmp_path, method="lora")
    assert fake.loads == [("Qwen/Qwen3-4B-Base", False, 1234)]


def test_scores_for_both_models_and_the_delta(tmp_path, clock, monkeypatch):
    rows = generate_stage.draw(box_io.read_jsonl(box_io.fetch_dataset(_split(tmp_path), tmp_path / "s")), 5, 1234)
    questions = [t[0]["content"].split("\n\n")[-1] for _, t, _ in rows]
    _use(monkeypatch, GenFake(clock, right=set(questions[:2]), wrapped={questions[2]}))
    out = _run(tmp_path)
    assert (out["base_exact_match"], out["candidate_exact_match"], out["delta_exact_match"]) == (0.0, 0.4, 0.4)
    assert (out["base_contains"], out["candidate_contains"], out["delta_contains"]) == (0.0, 0.6, 0.6)
    assert out["delta_exact_match_stderr"] == pytest.approx(((0.4 * 0.6) / 5) ** 0.5, abs=1e-6)
    assert all(isinstance(v, (str, int, float, bool)) for v in out.values())


# ── the samples file ──────────────────────────────────────────────────────────────────────────

def test_the_samples_file_is_the_stages_report_product(tmp_path, clock, monkeypatch):
    _use(monkeypatch, GenFake(clock))
    out = _run(tmp_path)
    path = tmp_path / "job" / generate_stage.SAMPLES_FILE
    assert out["output"] == out["samples_uri"] == out["samples_path"] == str(path)   # no bucket: on the box
    assert out["kind"] == "report" and out["rows"] == 5 and out["bytes"] == path.stat().st_size
    assert out["content_hash"] == out["samples_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    assert [s["index"] for s in samples] == list(range(5))
    first = samples[0]
    assert set(first) == {"index", "row", "prompt", "reference", "base", "candidate", "scores"}
    assert first["reference"] == f"IfcWall {first['row']}" and first["base"].startswith("wrong: ")
    assert first["scores"] == {"exact_match": {"base": False, "candidate": False},
                               "contains": {"base": False, "candidate": False}}


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


def test_the_samples_are_uploaded_beside_the_runs_checkpoints(tmp_path, clock, monkeypatch, s3):
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "ckpts")
    monkeypatch.setenv("CAIRN_PARENT_RUN", "run-gen-1")
    _use(monkeypatch, GenFake(clock))
    out = _run(tmp_path)
    key = "cairn-checkpoints/run-gen-1/eval/generation_samples.jsonl"
    assert out["output"] == out["samples_uri"] == f"s3://ckpts/{key}"
    assert out["samples_path"] == str(tmp_path / "job" / generate_stage.SAMPLES_FILE)   # still read on the box
    body = s3.get_object(Bucket="ckpts", Key=key)["Body"].read()
    assert hashlib.sha256(body).hexdigest() == out["content_hash"] and "samples_upload_error" not in out


def test_a_failed_upload_keeps_the_scores_and_says_so(tmp_path, clock, monkeypatch, s3):
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "no-such-bucket")
    monkeypatch.setenv("CAIRN_RUN_ID", "run-gen-2")
    _use(monkeypatch, GenFake(clock))
    out = _run(tmp_path)
    assert out["checked"] is True and out["samples_uri"] == out["samples_path"]
    assert "NoSuchBucket" in out["samples_upload_error"]


def test_the_lm_eval_results_keep_their_key():
    assert box_io.RESULTS_KEY == "eval/lm_eval_results.json" and box_io.SAMPLES_KEY == "eval/generation_samples.jsonl"


# ── the time projection ───────────────────────────────────────────────────────────────────────

def test_a_run_projected_past_the_leg_is_refused_after_the_first_batch(tmp_path, clock, monkeypatch):
    fake = GenFake(clock, per_batch_s=600.0)        # 3 batches: 600 + 2×600 + 3×600×1.25 = 4050 s > 0.9 h
    _use(monkeypatch, fake)
    with pytest.raises(flow_stages.TimeBudgetExceeded) as refused:
        _run(tmp_path)
    assert [g[0] for g in fake.generated] == ["base"] and fake.attached == []
    tail = str(refused.value)[-500:]
    assert "eval.generate refused before the full run: projected 1.12 h > 0.90 h" in tail


def test_a_run_that_fits_is_admitted_and_the_projection_is_recorded(tmp_path, clock, monkeypatch):
    _use(monkeypatch, GenFake(clock, per_batch_s=60.0))
    out = _run(tmp_path)
    assert out["first_batch_s"] == 60.0 and out["projected_s"] == pytest.approx(60 + 120 + 180 * 1.25)
    assert out["base_model_s"] == 180.0 and out["candidate_model_s"] == 180.0


def test_outside_the_stub_tier_a_budget_is_required(tmp_path, clock, monkeypatch):
    _use(monkeypatch, GenFake(clock))
    with pytest.raises(ValueError, match="time_budget_s is required"):
        _run(tmp_path, time_budget_s=0)


# ── what fails the leg and what does not ──────────────────────────────────────────────────────

def test_no_test_split_or_no_usable_row_fails_the_leg(tmp_path, clock, monkeypatch):
    _use(monkeypatch, GenFake(clock))
    with pytest.raises(box_io.FetchError, match="no test_uri"):
        _run(tmp_path, test_uri="")
    empty = tmp_path / "empty.jsonl"
    empty.write_text(json.dumps({"text": "refused"}) + "\n")
    with pytest.raises(box_io.FetchError, match="no row with a prompt and an answer"):
        _run(tmp_path, test_uri=str(empty))


def test_a_split_whose_bytes_changed_since_registration_fails_the_leg(tmp_path, clock, monkeypatch):
    _use(monkeypatch, GenFake(clock))
    uri = _split(tmp_path, name="pinned.jsonl")
    Path(uri[len("file://"):].split("#")[0]).write_text("{}\n")
    with pytest.raises(box_io.FetchError, match="Refusing to score on unverified data"):
        _run(tmp_path, test_uri=uri)


def test_no_backend_on_the_box_is_not_evaluated_not_a_failure(tmp_path, clock, monkeypatch):
    def missing(*, stub):
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(inprocess, "make_backend", missing)
    out = _run(tmp_path)
    assert out["checked"] is False and "install cairn-operator-eval-lm[box]" in out["note"]
    assert "output" not in out and "base_exact_match" not in out
    # present and empty: a validator leg wires them, and a wire to nothing fails the pipeline
    assert (out["samples_path"], out["samples_uri"], out["samples_sha256"]) == ("", "", "")


def test_a_failure_while_generating_reports_no_scores(tmp_path, clock, monkeypatch):
    _use(monkeypatch, GenFake(clock, fail="candidate"))
    out = _run(tmp_path)
    assert out["checked"] is False and out["note"] == "generation failed: RuntimeError: CUDA out of memory"
    assert "output" not in out and "candidate_exact_match" not in out
    assert not (tmp_path / "job" / generate_stage.SAMPLES_FILE).exists()


def test_the_stub_tier_scores_both_models_and_is_never_checked(tmp_path):
    out = generate_stage.eval_generate(tmp_path / "job", {**_config(tmp_path), "stub": True, "time_budget_s": 0,
                                                          "local_adapter_dir": "", "adapter_uri": "stub://adapters/x"}, [])
    assert out["stub"] is True and out["checked"] is False and "placeholders" in out["note"]
    assert out["harness_backend"] == "stub" and out["adapter_source"] == "stub"
    for key in ("base_exact_match", "candidate_exact_match", "delta_exact_match", "base_contains",
                "candidate_contains", "delta_contains"):
        assert key in out
    samples = [json.loads(line) for line in (tmp_path / "job" / generate_stage.SAMPLES_FILE).read_text().splitlines()]
    assert len(samples) == 5 and samples[0]["base"].startswith("stub base model answer")
    assert samples[0]["candidate"].startswith("stub candidate answer")


def test_nothing_is_written_to_the_process_environment(tmp_path, clock, monkeypatch, s3):
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "ckpts")
    monkeypatch.setenv("CAIRN_PARENT_RUN", "run-gen-3")
    _use(monkeypatch, GenFake(clock))
    before = dict(os.environ)
    _run(tmp_path)
    assert dict(os.environ) == before


def test_the_stub_backend_renders_the_generation_prompt():
    stub = inprocess.StubBackend()
    loaded = stub.load("m", quantize_4bit=True, seed=1)
    msgs = [{"role": "user", "content": "q"}]
    assert stub.render(loaded, msgs, "") == "<|im_start|>user\nq<|im_end|>\n"
    assert stub.render(loaded, msgs, "", add_generation_prompt=True).endswith("<|im_start|>assistant\n")
