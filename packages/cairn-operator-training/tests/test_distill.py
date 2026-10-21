"""compress/distill — hermetic tier (#1726, revives #467).

No torch, no trl, no GPU: the stub tier is the whole point of this file, and
the real GKD path is exercised only for its *contract* (what it maps onto trl,
and what it refuses) since trl cannot be installed on a CPU dev box.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from operator_training.operators.distill import CompressDistillOp
from operator_training.schemas.distill import DistillResult, DistillSpec
from operator_training.schemas.train import TrainSpec


class _Ctx:
    pipeline_id = "test"
    node_name = "distill"
    compute = None


def _spec(**over) -> DistillSpec:
    fields = {"teacher_model": "Qwen/Qwen2.5-7B-Instruct",
              "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
              "dataset_uri": "d.jsonl", "stub": True}
    fields.update(over)
    return DistillSpec.model_validate(fields)


def _run(spec: DistillSpec) -> DistillResult:
    return asyncio.run(CompressDistillOp().run(spec, _Ctx()))


# ── the beta collision, which is the reason this spec exists ───────────────

def test_distill_does_NOT_inherit_the_DPO_beta():
    """`TrainSpec.beta` is DPO preference strength; `GKDConfig.beta`
    interpolates the generalized JSD (0 ≈ forward KL, 1 ≈ reverse KL). Same
    word, unrelated objective. Reusing the field would train successfully with
    quietly the wrong loss — no error, no crash."""
    assert "beta" in TrainSpec.model_fields, "the DPO beta still exists on TrainSpec"
    assert "beta" not in DistillSpec.model_fields, \
        "DistillSpec must not carry a bare `beta` — it would read as the DPO one"
    assert "jsd_beta" in DistillSpec.model_fields


def test_jsd_beta_maps_onto_trls_beta_explicitly():
    """The rename only works if the entry actually translates it."""
    from operator_training import distill_entry

    src = inspect.getsource(distill_entry.main)
    assert 'beta=spec.get("jsd_beta"' in src, "jsd_beta must be mapped to GKDConfig.beta"


def test_the_import_path_is_top_level_trl_not_experimental():
    """Verified against the pinned trl 0.24: `from trl import GKDConfig,
    GKDTrainer`. An earlier note claimed trl.experimental.gkd — wrong, and it
    would have failed at import time on the GPU box."""
    from operator_training import distill_entry

    src = inspect.getsource(distill_entry)
    assert "from trl import GKDConfig, GKDTrainer" in src
    # Check the IMPORT lines, not the prose — the module comment legitimately
    # mentions trl.experimental to say it is NOT that.
    imports = [ln.split("#", 1)[0].strip() for ln in src.splitlines()
               if ln.strip().startswith(("from trl", "import trl"))]
    assert imports, "the trl import must be present"
    assert not any("experimental" in ln for ln in imports), \
        f"GKD is top-level in trl 0.24; got {imports}"


# ── refusals that save a GPU ───────────────────────────────────────────────

def test_a_teacher_equal_to_the_student_is_refused():
    """trl would accept it — GKDConfig.teacher_model_name_or_path defaults to
    the model itself — and happily self-distil. Nobody asks for that on
    purpose, so refuse before provisioning a GPU."""
    with pytest.raises(ValueError, match="LARGER teacher"):
        _run(_spec(teacher_model="Qwen/Qwen2.5-0.5B-Instruct",
                   base_model="Qwen/Qwen2.5-0.5B-Instruct"))


# ── the stub tier: the governed path runs on a laptop ──────────────────────

def test_the_stub_tier_needs_no_torch_and_reports_the_pair():
    res = _run(_spec(max_steps=7))
    assert res.status == "succeeded" and res.backend == "stub"
    assert res.teacher_model.endswith("7B-Instruct")
    assert res.student_model.endswith("0.5B-Instruct")
    assert res.metrics["steps"] == 7 and res.metrics["stub"] is True


def test_the_env_var_selects_the_stub_like_train_qlora(monkeypatch):
    monkeypatch.setenv("CAIRN_TRAIN_STUB", "1")
    assert _run(_spec(stub=False)).backend == "stub"


def test_the_stub_carries_the_gkd_knobs_so_a_gate_can_read_them():
    res = _run(_spec(lmbda=0.0, jsd_beta=1.0))
    assert res.metrics["lmbda"] == 0.0 and res.metrics["jsd_beta"] == 1.0


def test_mock_touches_nothing():
    res = CompressDistillOp().mock(_spec(), _Ctx())
    assert res.backend == "mock" and res.adapter_uri == ""


# ── governance declarations (CLAUDE.md rule 7) ─────────────────────────────

def test_risk_is_declared_on_the_CLASS_not_only_the_manifest():
    """This one holds a GPU and spends. A read_only default would let the
    bridge's restart path silently re-buy a training run — the #1697 bug."""
    assert CompressDistillOp.risk == "paid_compute"
    assert CompressDistillOp.long_running is True
    assert CompressDistillOp.requires_gpu is True


def test_the_class_matches_the_manifest():
    """The drift the sync gate catches only once the pack is INSTALLED."""
    import tomllib
    from pathlib import Path

    tbl = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pack.toml").read_text(encoding="utf-8"))["operator"]
    entry = tbl["compress/distill"]
    assert entry["risk"] == CompressDistillOp.risk == "paid_compute"
    assert entry["requires_gpu"] == CompressDistillOp.requires_gpu is True
    assert entry["input_schema"] == "DistillSpec"


# ── the dataset format boundary ────────────────────────────────────────────

def test_cairn_rows_are_converted_to_the_messages_format_gkd_wants():
    """GKD's dataset contract is `messages`, not the {input, output} pairs
    cairn datasets carry. A row that already has messages passes through."""
    from operator_training.distill_entry import _to_messages

    out = _to_messages({"input": "hi", "output": "hello"})
    assert out["messages"] == [{"role": "user", "content": "hi"},
                               {"role": "assistant", "content": "hello"}]

    passthrough = [{"role": "system", "content": "s"}]
    assert _to_messages({"messages": passthrough})["messages"] is passthrough

    # prompt-only rows are valid for on-policy distillation: the student
    # generates the completion and the teacher grades it.
    assert _to_messages({"input": "hi"})["messages"] == [{"role": "user", "content": "hi"}]


def test_the_config_build_is_version_tolerant():
    """The dpo_entry pattern: filter to the parameters THIS trl accepts, so a
    pinned-version bump that renames a knob degrades instead of raising
    TypeError mid-run on a GPU."""
    from operator_training import distill_entry

    src = inspect.getsource(distill_entry.main)
    assert "inspect.signature(GKDConfig.__init__).parameters" in src
    assert "if k in params" in src


def test_dataset_uri_is_resolved_not_fed_raw_to_the_loader():
    """#1714 regression guard. The pipeline declares `dataset_uri: {type:
    Dataset}`, so the launch resolver hands this entry a pinned
    `s3://…#sha256=` URI. Feeding that straight to `load_dataset` is the exact
    FileNotFoundError train_entry and dpo_entry were fixed for — HF datasets
    cannot read s3, and the fragment lands in the object key. The load MUST go
    through dataset_fetch.resolve_dataset_uri (strip pin, fetch s3, verify),
    the same as the other two entries. Asserted on source because the loader's
    deps live on the GPU backend and are not importable in the hermetic tier."""
    from operator_training import distill_entry

    src = inspect.getsource(distill_entry.main)
    assert "dataset_fetch.resolve_dataset_uri" in src, (
        "distill_entry must resolve the dataset URI (strip+verify the #1466 pin, "
        "fetch s3://) before load_dataset — a raw governed ref breaks the run")
    # and it must be the ARGUMENT to the loader, not resolved-and-ignored
    assert 'load_dataset("json", data_files=dataset_fetch.resolve_dataset_uri' in src, (
        "the resolved path must be what load_dataset actually reads")
