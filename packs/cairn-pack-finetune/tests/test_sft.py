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
