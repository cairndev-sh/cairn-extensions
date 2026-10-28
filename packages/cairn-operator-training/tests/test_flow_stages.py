"""ML-flow kernels (ADR-0082 D7/P1b, #1074): the stub tier end-to-end through
the kernel contract — train.qlora writes train_result.json + scalars,
eval.checkpoints gates on metrics (fail-closed ceilings), package.adapter
emits the candidate manifest. Hermetic: no torch, no network."""
from __future__ import annotations

import json

import pytest

from operator_training.flow_stages import (
    eval_checkpoints,
    package_adapter,
    train_qlora,
)


def _train(workdir, config=None, inputs=None):
    cfg = {"stub": True, "output_name": "t-adapter", "dataset_uri": "hf://d", **(config or {})}
    return train_qlora(workdir, cfg, inputs or [])


def test_train_qlora_stub_writes_result_and_scalars(tmp_path):
    stats = _train(tmp_path)
    assert stats["status"] == "succeeded"
    assert stats["backend"] == "stub"
    assert stats["adapter_uri"]  # stub reflects artifact_target (hf:// default)
    # scalars only — nothing row/tensor shaped crosses the boundary
    assert all(not isinstance(v, (dict, list)) for v in stats.values())
    result = json.loads((tmp_path / "train_result.json").read_text())
    assert result["adapter_uri"] == stats["adapter_uri"]


def test_train_qlora_dataset_uri_defaults_from_inputs(tmp_path):
    stats = train_qlora(
        tmp_path, {"stub": True, "output_name": "x"}, ["file:///data/promoted.jsonl"],
    )
    assert stats["status"] == "succeeded"


def test_eval_checkpoints_passes_without_ceilings(tmp_path):
    _train(tmp_path)
    stats = eval_checkpoints(tmp_path, {}, [])
    assert stats["passed"] is True and stats["evaluated"] is True


def test_eval_checkpoints_zero_ceiling_means_unset(tmp_path):
    # operation bindings default the knob to 0 via $default — 0 must never
    # mean "reject everything"
    _train(tmp_path)
    assert eval_checkpoints(tmp_path, {"max_train_loss": 0}, [])["passed"] is True


def test_eval_checkpoints_rejects_over_ceiling_and_fails_stage(tmp_path):
    _train(tmp_path)
    result = json.loads((tmp_path / "train_result.json").read_text())
    result["metrics"]["train_loss"] = 9.9
    (tmp_path / "train_result.json").write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="rejected"):
        eval_checkpoints(tmp_path, {"max_train_loss": 1.5}, [])
    verdict = eval_checkpoints(tmp_path, {"max_train_loss": 1.5, "fail_on_reject": False}, [])
    assert verdict["passed"] is False and "train_loss" in verdict["reason"]


def test_eval_checkpoints_missing_metric_with_ceiling_fails_closed(tmp_path):
    _train(tmp_path)
    result = json.loads((tmp_path / "train_result.json").read_text())
    result["metrics"].pop("eval_loss", None)
    (tmp_path / "train_result.json").write_text(json.dumps(result))
    verdict = eval_checkpoints(
        tmp_path, {"max_eval_loss": 1.0, "fail_on_reject": False}, [],
    )
    assert verdict["passed"] is False and "fail-closed" in verdict["reason"]


def test_package_adapter_emits_candidate_manifest(tmp_path):
    _train(tmp_path)
    stats = package_adapter(tmp_path, {"base_model": "m"}, [])
    assert stats["packaged"] is True
    candidate = json.loads((tmp_path / "candidate.json").read_text())
    assert candidate["adapter_uri"] == stats["candidate_uri"]
    assert candidate["packaged_at"]


# ── #2189 (T04, C6) — the candidate's content identity ───────────────────────

def test_the_stub_tier_leaves_deterministic_adapter_bytes(tmp_path):
    stats = _train(tmp_path)
    d = tmp_path / "adapter"
    assert (d / "adapter_config.json").is_file() and (d / "adapter_model.safetensors").is_file()
    assert stats["local_adapter_dir"] == str(d)
    result = json.loads((tmp_path / "train_result.json").read_text())
    assert result["local_adapter_dir"] == str(d)          # the carrier package.adapter reads


def test_package_adapter_emits_the_model_descriptor_keys(tmp_path):
    _train(tmp_path)
    stats = package_adapter(tmp_path, {"base_model": "m"}, [])
    assert stats["candidate_hashed"] is True and stats["kind"] == "model"
    assert stats["output"] == stats["candidate_uri"]
    assert len(stats["content_hash"]) == 64 and stats["bytes"] > 0
    assert len(stats["spec_digest"]) == 64
    candidate = json.loads((tmp_path / "candidate.json").read_text())
    assert candidate["content_hash"] == stats["content_hash"]
    assert all(not isinstance(v, (dict, list)) for v in stats.values())   # scalars only cross the boundary


def test_the_stub_identity_follows_the_spec_and_the_data(tmp_path):
    """What the stub makes testable is the HASHING RULE over real files: the
    same knobs on the same bytes hash the same; a knob change or a different
    dataset changes the identity. (A real trainer never yields byte-identical
    adapters twice — the registry's identical-bytes rule is what T04 proves.)"""
    ds1, ds2 = tmp_path / "d1.jsonl", tmp_path / "d2.jsonl"
    ds1.write_text('{"prompt": "a", "completion": "b"}\n'); ds2.write_text('{"prompt": "a", "completion": "c"}\n')
    dirs = {n: tmp_path / n for n in ("a", "b", "knob", "data")}
    for d in dirs.values():
        d.mkdir()
    _train(dirs["a"], {"dataset_uri": str(ds1)}); _train(dirs["b"], {"dataset_uri": str(ds1)})
    _train(dirs["knob"], {"dataset_uri": str(ds1), "lora_r": 8}); _train(dirs["data"], {"dataset_uri": str(ds2)})
    h = {n: package_adapter(d, {}, [])["content_hash"] for n, d in dirs.items()}
    assert h["a"] == h["b"]
    assert h["knob"] != h["a"] and h["data"] != h["a"]


def test_logs_beside_the_weights_do_not_change_the_identity(tmp_path):
    from operator_training.flow_stages import hash_candidate_dir
    _train(tmp_path)
    d = tmp_path / "adapter"
    before = hash_candidate_dir(d)
    (d / "training.log").write_text("noise\n")
    (d / "checkpoint-50").mkdir()
    (d / "checkpoint-50" / "trainer_state.json").write_text("{}")
    assert hash_candidate_dir(d) == before


def test_a_remote_only_adapter_reports_it_could_not_be_hashed(tmp_path):
    _train(tmp_path)
    result = json.loads((tmp_path / "train_result.json").read_text())
    result["local_adapter_dir"] = ""                       # bytes live nowhere reachable
    result["adapter_uri"] = "hf://someone/adapter"
    (tmp_path / "train_result.json").write_text(json.dumps(result))
    stats = package_adapter(tmp_path, {}, [])
    assert stats["candidate_hashed"] is False and "content_hash" not in stats and "kind" not in stats


def test_stage_order_enforced(tmp_path):
    with pytest.raises(RuntimeError, match="train.qlora must run"):
        eval_checkpoints(tmp_path, {}, [])
    with pytest.raises(RuntimeError, match="train.qlora must run"):
        package_adapter(tmp_path, {}, [])


def test_risk_classes_declared():
    # fail-closed discipline (ADR-0081 D2): declarations must exist + be right
    assert train_qlora.risk_class == "external_write"
    assert eval_checkpoints.risk_class == "read_only"
    assert package_adapter.risk_class == "external_write"


def test_spec_from_config_carries_tracking_fields(tmp_path):
    """#1235 — the flow path honors the SAME tracking fields the direct
    train/* path does (#136): report_to / tracking_project / tracking_run_name
    flow from the stage config into the TrainSpec unchanged."""
    from operator_training.flow_stages import _spec_from_config

    spec = _spec_from_config(
        {
            "dataset_uri": "hf://d",
            "base_model": "m",
            "report_to": ["mlflow"],
            "tracking_project": "cairn-finetune",
            "tracking_run_name": "audit-run",
        },
        [],
    )
    assert spec.report_to == ["mlflow"]
    assert spec.tracking_project == "cairn-finetune"
    assert spec.tracking_run_name == "audit-run"


# ── #1322 pre-flight RAM feasibility ─────────────────────────────────────────

def test_ram_preflight_refuses_an_infeasible_train(monkeypatch):
    from operator_training import flow_stages
    monkeypatch.setattr(flow_stages, "_available_ram_bytes", lambda: 1.0 * 2**30)
    monkeypatch.setattr(flow_stages, "_host_bytes_per_param", lambda config: flow_stages._TRAIN_BYTES_PER_PARAM)
    with pytest.raises(RuntimeError, match="refused before starting"):
        flow_stages._refuse_if_ram_infeasible("Qwen/Qwen2.5-0.5B-Instruct", {})


def test_ram_preflight_override_and_skip_paths(monkeypatch):
    from operator_training import flow_stages
    monkeypatch.setattr(flow_stages, "_available_ram_bytes", lambda: 1.0 * 2**30)
    # explicit override proceeds
    flow_stages._refuse_if_ram_infeasible("Qwen2.5-0.5B", {"ignore_ram_check": True})
    # env override proceeds
    monkeypatch.setenv("CAIRN_TRAIN_IGNORE_RAM", "1")
    flow_stages._refuse_if_ram_infeasible("Qwen2.5-0.5B", {})
    monkeypatch.delenv("CAIRN_TRAIN_IGNORE_RAM")
    # a name with no parseable size skips rather than blocks
    flow_stages._refuse_if_ram_infeasible("my-custom-model", {})
    # unknowable free RAM skips rather than blocks
    monkeypatch.setattr(flow_stages, "_available_ram_bytes", lambda: None)
    flow_stages._refuse_if_ram_infeasible("Qwen2.5-72B", {})


def test_ram_preflight_uses_the_gpu_rate_on_a_cuda_box(monkeypatch):
    """Found on the first GPU run: the CPU rate refused Qwen3-4B 4-bit QLoRA on a 15 GiB T4 box
    for an estimated 26.1 GiB; the run peaked at 2.3 GiB. On CUDA the rate is the GPU path's."""
    from operator_training import flow_stages
    monkeypatch.setattr(flow_stages, "_available_ram_bytes", lambda: 10.0 * 2**30)
    monkeypatch.setattr(flow_stages, "_host_bytes_per_param", lambda config: flow_stages._GPU_TRAIN_BYTES_PER_PARAM)
    flow_stages._refuse_if_ram_infeasible("Qwen/Qwen3-4B-Base", {})
    monkeypatch.setattr(flow_stages, "_host_bytes_per_param", lambda config: flow_stages._TRAIN_BYTES_PER_PARAM)
    with pytest.raises(RuntimeError, match="CPU path"):
        flow_stages._refuse_if_ram_infeasible("Qwen/Qwen3-4B-Base", {})


def test_the_rate_follows_the_path_the_trainer_takes(monkeypatch):
    import sys
    from operator_training import flow_stages
    monkeypatch.setitem(sys.modules, "torch", None)  # no torch ⇒ the CPU path
    assert flow_stages._host_bytes_per_param({}) == flow_stages._TRAIN_BYTES_PER_PARAM
    assert flow_stages._host_bytes_per_param({"validate_pipeline": True}) == flow_stages._TRAIN_BYTES_PER_PARAM


def test_ram_preflight_passes_a_feasible_train(monkeypatch):
    from operator_training import flow_stages
    monkeypatch.setattr(flow_stages, "_available_ram_bytes", lambda: 32.0 * 2**30)
    flow_stages._refuse_if_ram_infeasible("Qwen/Qwen2.5-0.5B-Instruct", {})


def test_param_count_parses_common_names():
    from operator_training.flow_stages import _param_count_from_name
    assert _param_count_from_name("Qwen/Qwen2.5-0.5B-Instruct") == 0.5e9
    assert _param_count_from_name("meta-llama/Llama-3-8b") == 8e9
    assert _param_count_from_name("my-custom-model") is None
    assert _param_count_from_name("roberta-base") is None


# ── #1715 — the FLOW path honors artifact_target ────────────────────────────
# The direct path publishes in local_entry; this path had no equivalent, so a
# run declaring `artifact_target: s3` ended with adapter_uri = /tmp/<name> — a
# path on the box the pipeline's own `release` finalizer then stops. Nothing
# fetches the adapter dir back here, so a swallowed failure destroys the very
# artifact the run exists to produce; publish failure is FATAL on this path.

def _adapter_on_disk(tmp_path, name="t-adapter"):
    d = tmp_path / name
    d.mkdir()
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_text("w")
    return d


def test_flow_path_s3_run_ends_with_an_s3_adapter_uri(tmp_path, monkeypatch):
    """THE regression pin from #1715: today it ends with /tmp/."""
    from operator_training import flow_stages as fs

    adapter = _adapter_on_disk(tmp_path)
    monkeypatch.setattr(fs, "publish_adapter_dir",
                        lambda t, d, **kw: "s3://buck/p/m/adapter/" if t == "s3" else "")

    class _Op:
        async def run(self, spec, ctx):
            from operator_training.schemas.train import TrainResult
            return TrainResult(status="succeeded", backend="local",
                               adapter_uri=str(adapter), metrics={"train_loss": 0.1})

    monkeypatch.setattr("operator_training.operators.local.LocalTrainOp", _Op)
    wd = tmp_path / "wd"; wd.mkdir()
    # the trainer is a fake: give the RAM pre-check a big box so its call site still runs,
    # instead of measuring the host (the 2.1 GiB CI runner refused it)
    monkeypatch.setattr(fs, "_available_ram_bytes", lambda: 256 * 2**30)
    stats = fs.train_qlora(wd, {"output_name": "m", "dataset_uri": "hf://d",
                                "artifact_target": "s3"}, [])

    assert stats["adapter_uri"].startswith("s3://"), stats["adapter_uri"]
    assert stats["local_adapter_dir"] == str(adapter)
    # package.adapter reads train_result.json — it must inherit the durable URI
    pkg = fs.package_adapter(wd, {}, [])
    assert pkg["candidate_uri"].startswith("s3://")


def test_flow_path_publish_failure_FAILS_the_stage(tmp_path, monkeypatch):
    """Unlike the direct path, there is no fetched copy to fall back on."""
    from operator_training import flow_stages as fs
    from operator_training.publish import AdapterPublishError

    adapter = _adapter_on_disk(tmp_path)

    def _boom(*_a, **_kw):
        raise AdapterPublishError("CAIRN_CHECKPOINT_BUCKET is unset")

    monkeypatch.setattr(fs, "publish_adapter_dir", _boom)

    class _Op:
        async def run(self, spec, ctx):
            from operator_training.schemas.train import TrainResult
            return TrainResult(status="succeeded", backend="local",
                               adapter_uri=str(adapter), metrics={})

    monkeypatch.setattr("operator_training.operators.local.LocalTrainOp", _Op)
    wd = tmp_path / "wd"; wd.mkdir()
    # the trainer is a fake: give the RAM pre-check a big box so its call site still runs,
    # instead of measuring the host (the 2.1 GiB CI runner refused it)
    monkeypatch.setattr(fs, "_available_ram_bytes", lambda: 256 * 2**30)
    with pytest.raises(AdapterPublishError):
        fs.train_qlora(wd, {"output_name": "m", "dataset_uri": "hf://d",
                            "artifact_target": "s3"}, [])


# ── #51 — a file:// adapter (the CPU/MPS trainer's shape) is hashed like a bare path ──

def _package_with(tmp_path, adapter, uri, local_uri):
    wd = tmp_path / f"wd-{abs(hash((uri, local_uri)))}"
    wd.mkdir()
    (wd / "train_result.json").write_text(json.dumps({
        "status": "succeeded", "backend": "local",
        "adapter_uri": uri, "local_adapter_dir": local_uri, "metrics": {}}))
    return package_adapter(wd, {}, [])


@pytest.mark.parametrize("which", ["both", "local_dir_only", "adapter_uri_only"])
def test_package_adapter_hashes_a_file_uri_like_the_bare_path(tmp_path, which):
    adapter = _adapter_on_disk(tmp_path).resolve()
    bare = _package_with(tmp_path, adapter, str(adapter), str(adapter))
    furi = f"file://{adapter}"
    uri, local = {"both": (furi, furi), "local_dir_only": (str(adapter), furi),
                  "adapter_uri_only": (furi, "")}[which]
    stats = _package_with(tmp_path, adapter, uri, local)
    assert stats["candidate_hashed"] is True and stats["kind"] == "model"
    assert stats["content_hash"] == bare["content_hash"] and stats["bytes"] == bare["bytes"]
    assert len(stats["spec_digest"]) == 64 and stats["spec_digest"] == bare["spec_digest"]


def test_train_qlora_with_the_cpu_result_shape_leaves_a_hashable_local_dir(tmp_path, monkeypatch):
    from operator_training import flow_stages as fs

    adapter = _adapter_on_disk(tmp_path).resolve()

    class _Op:
        async def run(self, spec, ctx):
            from operator_training.schemas.train import TrainResult
            return TrainResult(status="succeeded", backend="local",
                               adapter_uri=f"file://{adapter}", metrics={"train_loss": 0.1})

    monkeypatch.setattr("operator_training.operators.local.LocalTrainOp", _Op)
    monkeypatch.setattr(fs, "_available_ram_bytes", lambda: 256 * 2**30)
    monkeypatch.setattr(fs, "publish_adapter_dir", lambda *a, **kw: "")   # nothing published
    wd = tmp_path / "wd"; wd.mkdir()
    stats = fs.train_qlora(wd, {"output_name": "m", "dataset_uri": "hf://d"}, [])
    assert stats["local_adapter_dir"] == str(adapter)      # a directory, not a file:// string
    pkg = fs.package_adapter(wd, {}, [])
    assert pkg["candidate_hashed"] is True and pkg["kind"] == "model"
    assert len(pkg["content_hash"]) == 64 and len(pkg["spec_digest"]) == 64
