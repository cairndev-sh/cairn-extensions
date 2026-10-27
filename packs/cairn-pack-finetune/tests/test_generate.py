"""finetune/sft's generation eval (cairn-extensions#41) and the IFC validator after it (#12).

The `generate` leg runs the `sft-generate` flow (cairn-operator-eval-lm's `eval.generate`); the
`ifc` leg runs the `ifc-validate` flow (cairn-operator-ifc-eval's `eval.ifc_validate`) on the samples
file the generate leg wrote, only when the launch names `ifc_schema`. Both flows run here on the
core runner on the stub tier, chained the way the pipeline wires them. The plan and the pricing
are in `test_sft_box.py`."""
from __future__ import annotations

import asyncio
import hashlib
import json
import types
from pathlib import Path

import pytest
import yaml

PACK = Path(__file__).resolve().parents[1]
_INPUTS = {"train_version": "dataset:demo-train@1", "val_version": "dataset:demo-val@1",
           "test_version": "dataset:demo-test@1", "base_model": "tiny/model", "stub": True}


@pytest.fixture(autouse=True)
def _pack_from_checkout(monkeypatch):
    from cairn.core.pack import registry
    from cairn.core.pack.loader import load_all_packs

    monkeypatch.setenv("CAIRN_PACK_PATH", str(PACK))
    for name in ("CAIRN_CHECKPOINT_BUCKET", "CAIRN_RUN_ID", "CAIRN_PARENT_RUN"):
        monkeypatch.delenv(name, raising=False)
    registry.clear()
    load_all_packs()
    yield
    registry.clear()


def _pipeline(name: str = "sft") -> dict:
    return yaml.safe_load((PACK / "pipelines" / f"{name}.yaml").read_text(encoding="utf-8"))


def _step(step_id: str, name: str = "sft") -> dict:
    return next(s for s in _pipeline(name)["steps"] if s.get("id") == step_id)


def _shown(prefix: str) -> list[str]:
    gate = next(s for s in _pipeline()["steps"] if s.get("gate") == "promote-model")
    return [f.split(".", 1)[1] for f in gate["show"] if f.startswith(prefix + ".")]


def _flow(ref: str, workdir: Path, config: dict):
    from cairn.core.dataflow import Budget, DataflowSpec
    from cairn.core.dataflow.defs import resolve_flow_ref
    from cairn.core.dataflow.runner import run_dataflow

    stages = resolve_flow_ref(ref, config, kind="ml")
    return run_dataflow(DataflowSpec(stages=stages, budget=Budget()), workdir)


def _findings(result) -> dict:
    from cairn.core.dispatch.pipeline_bridge import _flow_leg_output

    return _flow_leg_output(types.SimpleNamespace(produces=[]),
                            types.SimpleNamespace(ok=True, stats=result.stats))["findings"]


def _bind(step_id: str, inputs: dict, steps: dict) -> dict:
    """A leg's `with:` with its `$input.*` and `$steps.<id>.<finding>` wires bound (no retry)."""
    defaults = {k: v.get("default") for k, v in _pipeline()["inputs"].items() if isinstance(v, dict)}
    bound = {**defaults, **inputs}
    out = {}
    for k, v in _step(step_id)["with"].items():
        if k == "retry":
            continue
        if isinstance(v, str) and v.startswith("$input."):
            v = bound[v[len("$input."):]]
        elif isinstance(v, str) and v.startswith("$steps."):
            src, _, field = v[len("$steps."):].partition(".")
            v = steps[src].get(field)
        out[k] = v
    return out


def _pinned(path: Path) -> str:
    return f"file://{path}#sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"


# ── the plan ──────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["sft", "sft-own-box"])
def test_the_generate_leg_follows_eval_and_reads_trains_adapter_and_the_test_version(name):
    gen = _step("generate", name)
    assert gen["flow"] == "sft-generate" and gen["depends_on"] == ["eval"]
    w = gen["with"]
    assert w["local_adapter_dir"] == "$steps.train.train_qlora_local_adapter_dir"
    assert w["adapter_uri"] == "$steps.train.train_qlora_adapter_uri"
    assert w["test_uri"] == "$input.test_version" and w["seed"] == "$input.eval_seed"
    assert (w["sample_n"], w["max_new_tokens"]) == ("$input.generate_sample_n", "$input.generate_max_new_tokens")
    assert w["base_model"] == _step("train", name)["with"]["base_model"] == "$input.base_model"
    inputs = _pipeline(name)["inputs"]
    assert inputs["generate_sample_n"]["default"] == 50 and inputs["generate_max_new_tokens"]["default"] == 512


@pytest.mark.parametrize("name", ["sft", "sft-own-box"])
def test_the_ifc_leg_reads_the_generate_legs_samples_and_runs_only_when_a_schema_is_named(name):
    from cairn.core.engine.bundle import eval_gate_predicate

    ifc = _step("ifc", name)
    assert ifc["flow"] == "ifc-validate" and ifc["depends_on"] == ["generate"]
    assert ifc["with"]["samples_path"] == "$steps.generate.eval_generate_samples_path"
    assert ifc["with"]["samples_uri"] == "$steps.generate.eval_generate_samples_uri"
    assert ifc["with"]["samples_sha256"] == "$steps.generate.eval_generate_samples_sha256"
    assert ifc["with"]["schema_version"] == "$input.ifc_schema"
    assert _pipeline(name)["inputs"]["ifc_schema"]["default"] == ""
    assert not eval_gate_predicate(ifc["when"], bundle_input={**_INPUTS, "ifc_schema": ""}, steps={})
    assert eval_gate_predicate(ifc["when"], bundle_input={**_INPUTS, "ifc_schema": "IFC4"}, steps={})


def test_the_ifc_bim_presets_name_the_ifc_schema():
    for name in ("ifc-bim-qwen3-4b.yaml", "ifc-bim-qwen3-4b-own-box.yaml"):
        (preset,) = yaml.safe_load((PACK / "presets" / name).read_text(encoding="utf-8"))["presets"]
        assert preset["inputs"]["ifc_schema"] == "IFC4"


# ── the generate flow on the stub tier ────────────────────────────────────────────────────────

@pytest.fixture
def split(tmp_path) -> Path:
    path = tmp_path / "test.jsonl"
    path.write_text("".join(json.dumps({"instruction": "Answer.", "input": f"test q{i}", "output": f"test a{i}"})
                            + "\n" for i in range(6)))
    return path


async def test_the_generate_flow_on_the_stub_tier_scores_both_models_and_every_figure_the_card_shows(tmp_path, split):
    cfg = _bind("generate", {**_INPUTS, "test_version": _pinned(split)},
                {"train": {"train_qlora_local_adapter_dir": "", "train_qlora_adapter_uri": "stub://adapters/x"}})
    result = await asyncio.to_thread(_flow, "finetune/sft-generate", tmp_path / "generate", cfg)
    findings = _findings(result)

    assert findings["eval_generate_stub"] is True and findings["eval_generate_checked"] is False
    assert findings["eval_generate_samples"] == 6 and findings["eval_generate_time_budget_s"] == 3600
    for who in ("base", "candidate", "delta"):
        assert f"eval_generate_{who}_exact_match" in findings and f"eval_generate_{who}_contains" in findings
    assert [d.kind for d in result.output_artifacts] == ["report"]          # the samples file
    shown = _shown("generate")
    assert len(shown) == 10 and [k for k in shown if findings.get(k) is None] == []


# ── the IFC leg on the generate leg's samples ─────────────────────────────────────────────────

_VALID = ["#1=IFCPROJECT('3MDa0qu6X5eBFeTDmhEULg',$,'Sample Project',$,$,$,$,$,$);",
          "#1=IFCCARTESIANPOINT((0.,0.,0.));\n#2=IFCDIRECTION((0.,0.,1.));",
          "#10=IFCSIUNIT(*,.LENGTHUNIT.,.MILLI.,.METRE.);\n#11=IFCUNITASSIGNMENT((#10));",
          "#1=IFCDIRECTION((1.,0.,0.));"]


async def test_the_ifc_leg_reads_what_the_generate_leg_wrote(tmp_path):
    """The two flows chained as the pipeline wires them, on the stub tier: the generate leg's
    samples file (written by the real `eval.generate` stage) is found on the box by the ifc leg,
    verified against the hash the generate leg reported, and scored field by field. The references
    are IFC, so they parse; the stub's placeholder answers do not; nothing is `checked`. (A domain
    pack's tests import no operator code: the rates on real answers are proven in
    cairn-operator-ifc-eval's and cairn-operator-eval-lm's own tests.)"""
    pytest.importorskip("ifcopenshell")
    split = tmp_path / "ifc-test.jsonl"
    split.write_text("".join(json.dumps({"instruction": "Write the IFC.", "input": f"prompt {i}",
                                         "output": answer}) + "\n" for i, answer in enumerate(_VALID)))
    inputs = {**_INPUTS, "test_version": _pinned(split), "ifc_schema": "IFC4"}
    gen_cfg = _bind("generate", inputs, {"train": {"train_qlora_local_adapter_dir": "",
                                                   "train_qlora_adapter_uri": "stub://adapters/x"}})
    generated = _findings(await asyncio.to_thread(_flow, "finetune/sft-generate", tmp_path / "generate", gen_cfg))
    assert generated["eval_generate_samples"] == 4

    ifc_cfg = _bind("ifc", inputs, {"generate": generated})
    assert ifc_cfg["samples_path"] == generated["eval_generate_samples_path"]
    assert ifc_cfg["samples_sha256"] == generated["eval_generate_samples_sha256"]
    result = await asyncio.to_thread(_flow, "finetune/ifc-validate", tmp_path / "ifc", ifc_cfg)
    f = _findings(result)
    assert f["eval_ifc_validate_checked"] is False and "placeholders" in f["eval_ifc_validate_note"]
    assert f["eval_ifc_validate_schema_version"] == "IFC4" and f["eval_ifc_validate_rows"] == 4
    assert f["eval_ifc_validate_reference_parse_valid_rate"] == 1.0
    assert f["eval_ifc_validate_reference_schema_error_rate"] == 0.0
    assert f["eval_ifc_validate_base_parse_valid_rate"] == f["eval_ifc_validate_candidate_parse_valid_rate"] == 0.0
    assert f["eval_ifc_validate_delta_parse_valid_rate"] == 0.0
    assert [d.kind for d in result.output_artifacts] == ["report"]
    # every figure the card shows is a finding, but the schema error rates of models whose answers
    # did not parse at all (a rate over nothing is not reported)
    missing = [k for k in _shown("ifc") if f.get(k) is None]
    assert missing == ["eval_ifc_validate_base_schema_error_rate", "eval_ifc_validate_candidate_schema_error_rate"]


async def test_a_changed_samples_file_fails_the_ifc_leg(tmp_path, split):
    cfg = _bind("generate", {**_INPUTS, "test_version": _pinned(split)},
                {"train": {"train_qlora_local_adapter_dir": "", "train_qlora_adapter_uri": "stub://adapters/x"}})
    generated = _findings(await asyncio.to_thread(_flow, "finetune/sft-generate", tmp_path / "generate", cfg))
    samples = Path(generated["eval_generate_samples_path"])
    samples.write_text(samples.read_text() + "\n")
    ifc_cfg = _bind("ifc", {**_INPUTS, "ifc_schema": "IFC4"}, {"generate": generated})
    with pytest.raises(RuntimeError, match="refusing to score a file that changed"):   # the stage's FetchError
        await asyncio.to_thread(_flow, "finetune/ifc-validate", tmp_path / "ifc", ifc_cfg)


async def test_a_generate_leg_that_measured_nothing_leaves_the_ifc_leg_not_evaluated(tmp_path):
    """A generate leg without its backend reports `checked: false` and EMPTY samples keys (never
    absent ones: the core fails the pipeline on a wire that resolves to nothing). The ifc leg's
    wires then resolve to empty strings through the core, and the leg says there was nothing to
    read instead of failing the run."""
    from cairn.core.dispatch.pipeline_bridge import _flow_leg_output
    from cairn.core.engine.bundle import resolve_step_inputs
    from cairn.core.engine.pipeline import parse_pipeline_def, plan_pipeline

    stats = {"eval.generate.checked": False, "eval.generate.note": "backend unavailable",
             "eval.generate.samples_path": "", "eval.generate.samples_uri": "", "eval.generate.samples_sha256": ""}
    generate = _flow_leg_output(types.SimpleNamespace(produces=[]), types.SimpleNamespace(ok=True, stats=stats))
    plan = plan_pipeline(parse_pipeline_def((PACK / "pipelines" / "sft.yaml").read_text(encoding="utf-8"),
                                            pack="finetune"))
    ifc = next(s for s in plan.steps if s.id == "ifc")
    cfg = resolve_step_inputs(ifc, bundle_input={**_INPUTS, "ifc_schema": "IFC4"}, steps={"generate": generate})
    cfg.pop("retry")
    assert (cfg["samples_path"], cfg["samples_uri"], cfg["samples_sha256"]) == ("", "", "")
    f = _findings(await asyncio.to_thread(_flow, "finetune/ifc-validate", tmp_path / "ifc", cfg))
    assert f["eval_ifc_validate_checked"] is False and "no samples file named" in f["eval_ifc_validate_note"]
