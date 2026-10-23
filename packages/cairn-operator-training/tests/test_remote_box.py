"""cairn-extensions#7 — what the trainer needs to run on a GPU box reached over SSH, with the
control plane elsewhere (the `finetune-ssh` profile). Hermetic: no GPU, no torch, moto for S3.

- A run that touches no S3 works without boto3 (it is the `[gpu]` extra's, imported lazily).
- The trainer writes under the job's workdir, never a shared `/tmp/<output_name>`.
- The published adapter is the adapter files only, never `checkpoint-*/` (optimizer state).
- The GPU path reports `steps`, which the `promote-model` gate shows.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from operator_training import checkpoint_sync as CS
from operator_training.schemas.train import TrainResult, TrainSpec

_ADAPTER = ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("CAIRN_CHECKPOINT_BUCKET", "CAIRN_CHECKPOINT_PREFIX", "CAIRN_RUN_ID",
              "CAIRN_PARENT_RUN", "CAIRN_OUTPUT_NAME", "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(k, raising=False)


def _trained_dir(root: Path) -> Path:
    """What a trainer leaves in its output dir: the adapter at the top, a checkpoint below."""
    root.mkdir(parents=True, exist_ok=True)
    for name in _ADAPTER:
        (root / name).write_text(name)
    ck = root / "checkpoint-5"
    ck.mkdir()
    (ck / "optimizer.pt").write_text("optimizer state")
    (ck / "adapter_model.safetensors").write_text("step-5 weights")
    return root


# ── boto3 is optional: only the S3 paths import it ──────────────────────────

_NO_BOTO3 = r"""
import json, sys
sys.modules["boto3"] = None  # `import boto3` now raises ImportError, as on a box without it
from pathlib import Path
import operator_training.train_entry, operator_training.local_entry, operator_training.publish  # noqa: F401
from operator_training import checkpoint_sync, dataset_fetch, flow_stages
wd, data = Path(sys.argv[1]), sys.argv[2]
assert checkpoint_sync.s3_target_from_env() is None
assert checkpoint_sync.pull_checkpoints_if_any(str(wd / "out")) is None
assert dataset_fetch.resolve_dataset_uri("file://" + data) == data
stats = flow_stages.train_qlora(wd, {"stub": True, "output_name": "nob", "dataset_uri": data}, [])
assert stats["status"] == "succeeded", stats
try:
    checkpoint_sync.s3_client()
except RuntimeError as e:
    print(json.dumps({"ok": True, "s3_error": str(e)}))
else:
    raise SystemExit("s3_client built a client without boto3")
"""


def test_a_run_without_s3_imports_and_trains_without_boto3(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"instruction": "q", "output": "a"}) + "\n")
    wd = tmp_path / "wd"
    wd.mkdir()
    proc = subprocess.run([sys.executable, "-c", _NO_BOTO3, str(wd), str(data)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "cairn-operator-training[gpu]" in out["s3_error"]


# ── the trainer's output dir is the job's ───────────────────────────────────

def test_the_output_dir_is_under_the_workdir_and_tmp_only_without_one(tmp_path):
    assert CS.trainer_output_dir("m", tmp_path) == str(tmp_path / "train-output" / "m")
    assert CS.trainer_output_dir("m") == "/tmp/m"
    assert CS.trainer_output_dir("") == f"/tmp/{CS.GENERIC_RUN_KEY}"


def test_train_qlora_trains_under_its_own_workdir(tmp_path, monkeypatch):
    """Two jobs with the same output_name get two directories: neither resumes from the other."""
    from operator_training import flow_stages as fs
    from operator_training.operators import local

    seen = []

    async def fake_train_local(spec, *, progress_path=None, reporter=None, output_dir=None):
        seen.append(output_dir)
        _trained_dir(Path(output_dir))
        return TrainResult(status="succeeded", backend="unsloth-gpu", adapter_uri=output_dir,
                           metrics={"train_loss": 0.1, "steps": 7})

    monkeypatch.setattr(local, "_train_local", fake_train_local)
    monkeypatch.setattr(fs, "_available_ram_bytes", lambda: 256 * 2**30)
    stats = {}
    for job in ("job-a", "job-b"):
        wd = tmp_path / job
        wd.mkdir()
        stats[job] = fs.train_qlora(wd, {"output_name": "sft-candidate", "dataset_uri": "hf://d",
                                       "artifact_target": "local"}, [])
    assert seen == [str(tmp_path / j / "train-output" / "sft-candidate") for j in ("job-a", "job-b")]
    assert stats["job-a"]["local_adapter_dir"] == seen[0] and stats["job-a"]["steps"] == 7


def test_the_gpu_path_hands_the_workdir_dir_to_train_entry(tmp_path, monkeypatch):
    import operator_training.train_entry as te
    from operator_training.operators import local

    called = {}

    def fake_main(spec_dict, reporter=None, *, output_dir=None):
        called["output_dir"] = output_dir
        return {"adapter_uri": output_dir, "metrics": {"train_loss": 0.5, "steps": 3}}

    monkeypatch.setattr(local, "_use_gpu_unsloth", lambda spec: True)
    monkeypatch.setattr(te, "main", fake_main)
    ctx = types.SimpleNamespace(report=None, workdir=tmp_path)
    res = asyncio.run(local.LocalTrainOp().run(TrainSpec(dataset_uri="ds", output_name="out"), ctx))
    assert called["output_dir"] == res.adapter_uri == str(tmp_path / "train-output" / "out")


def test_a_retry_resumes_from_its_own_checkpoints_and_another_run_does_not(tmp_path, monkeypatch):
    """Resume is keyed by the run (CAIRN_PARENT_RUN on the SSH path), not by a shared directory:
    the retry's fresh workdir pulls the run's S3 checkpoints; another run with the same
    output_name finds none."""
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "t")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "t")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "ckpts")
    with moto.mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="ckpts")
        monkeypatch.setenv("CAIRN_PARENT_RUN", "run-a")
        first = _trained_dir(Path(CS.trainer_output_dir("sft", tmp_path / "attempt-1")))
        CS.upload_dir(str(first / "checkpoint-5"), "ckpts", f"{CS.DEFAULT_PREFIX}/run-a/checkpoint-5")

        retry = CS.trainer_output_dir("sft", tmp_path / "attempt-2")
        assert CS.pull_checkpoints_if_any(retry) == os.path.join(retry, "checkpoint-5")

        monkeypatch.setenv("CAIRN_PARENT_RUN", "run-b")
        assert CS.pull_checkpoints_if_any(CS.trainer_output_dir("sft", tmp_path / "other")) is None


# ── the published adapter is the adapter only ───────────────────────────────

def test_s3_publish_uploads_the_adapter_files_and_no_checkpoint(tmp_path, monkeypatch):
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    from operator_training.publish import publish_adapter_dir

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "t")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "t")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "ckpts")
    monkeypatch.setenv("CAIRN_PARENT_RUN", "run-a")
    src = _trained_dir(tmp_path / "out")
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="ckpts")
        uri = publish_adapter_dir("s3", str(src))
        keys = sorted(o["Key"] for o in client.list_objects_v2(Bucket="ckpts")["Contents"])
    assert uri == f"s3://ckpts/{CS.DEFAULT_PREFIX}/run-a/adapter/"
    assert keys == [f"{CS.DEFAULT_PREFIX}/run-a/adapter/{n}" for n in sorted(_ADAPTER)]


def test_hf_publish_allows_only_the_adapter_files(tmp_path, monkeypatch):
    import huggingface_hub
    from operator_training.publish import publish_adapter_dir

    calls = {}

    class _Api:
        def create_repo(self, **kw):
            pass

        def upload_folder(self, **kw):
            calls.update(kw)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    src = _trained_dir(tmp_path / "out")
    assert publish_adapter_dir("hf", str(src), hf_repo="acme/a") == "hf://acme/a"
    assert calls["allow_patterns"] == sorted(_ADAPTER)


def test_the_direct_entrypoint_s3_save_skips_the_checkpoints(tmp_path, monkeypatch):
    """train_entry run as its own entrypoint (the image) publishes with `_save_adapter`."""
    import operator_training.train_entry as te

    got = {}
    monkeypatch.setattr(CS, "s3_target_from_env", lambda *a, **k: ("ckpts", "p", "run-a"))
    monkeypatch.setattr(CS, "upload_files", lambda d, names, b, k: got.update(names=names, key=k))
    src = _trained_dir(tmp_path / "out")
    assert te._save_adapter({"artifact_target": "s3"}, None, None, str(src)) == "s3://ckpts/p/run-a/adapter/"
    assert got == {"names": sorted(_ADAPTER), "key": "p/run-a/adapter"}


# ── the GPU path reports its steps ──────────────────────────────────────────

def test_the_gpu_metrics_carry_the_optimizer_steps():
    from operator_training.train_entry import _base_metrics

    stats = types.SimpleNamespace(global_step=353, training_loss=0.6, metrics={})
    assert _base_metrics(stats)["steps"] == 353
