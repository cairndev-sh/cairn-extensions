"""Tests for checkpoint_sync (ADR-0064 §1): durable S3 checkpoint mirroring +
pull-before-resume. Hermetic — moto mocks S3, no real network, no torch stack
(the ``on_save`` callback path is exercised via a fake TrainerCallback-shaped
``args``/``state``/``control``, not a real trainer).
"""
from __future__ import annotations

import os

import boto3
import pytest
from moto import mock_aws

from operator_training import checkpoint_sync
from operator_training.schemas.train import TrainSpec

BUCKET = "cairn-test-bucket"


def _clear_checkpoint_env(monkeypatch):
    for k in ("CAIRN_CHECKPOINT_BUCKET", "CAIRN_CHECKPOINT_PREFIX", "CAIRN_RUN_ID", "CAIRN_PARENT_RUN", "CAIRN_OUTPUT_NAME"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def aws_creds(monkeypatch):
    # moto needs *some* creds present, even fake ones.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)


class _Args:
    def __init__(self, output_dir):
        self.output_dir = output_dir


class _State:
    pass


class _Control:
    pass


# ---------------------------------------------------------------------------
# s3_target_from_env
# ---------------------------------------------------------------------------


def test_s3_target_from_env_none_without_bucket(monkeypatch):
    _clear_checkpoint_env(monkeypatch)
    assert checkpoint_sync.s3_target_from_env() is None


def test_s3_target_from_env_resolves_bucket_prefix_run_key(monkeypatch):
    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", BUCKET)
    monkeypatch.setenv("CAIRN_RUN_ID", "run-123")
    bucket, prefix, run_key = checkpoint_sync.s3_target_from_env()
    assert bucket == BUCKET
    assert prefix == checkpoint_sync.DEFAULT_PREFIX
    assert run_key == "run-123"


def test_s3_target_from_env_parent_run_beats_output_name(monkeypatch):
    """The SSH/dataflow path stamps CAIRN_PARENT_RUN (not RUN_ID): it must win over
    output_name so two runs with the same output_name never share a key."""
    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", BUCKET)
    monkeypatch.setenv("CAIRN_PARENT_RUN", "run-flow-abc")
    monkeypatch.setenv("CAIRN_OUTPUT_NAME", "my-adapter")
    assert checkpoint_sync.s3_target_from_env()[2] == "run-flow-abc"


def test_s3_target_from_env_falls_back_to_output_name_then_adapter(monkeypatch):
    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", BUCKET)
    monkeypatch.setenv("CAIRN_OUTPUT_NAME", "my-adapter")
    _, _, run_key = checkpoint_sync.s3_target_from_env()
    assert run_key == "my-adapter"

    monkeypatch.delenv("CAIRN_OUTPUT_NAME")
    _, _, run_key = checkpoint_sync.s3_target_from_env()
    assert run_key == "adapter"


def test_s3_target_from_env_custom_prefix(monkeypatch):
    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", BUCKET)
    monkeypatch.setenv("CAIRN_CHECKPOINT_PREFIX", "/custom-prefix/")
    _, prefix, _ = checkpoint_sync.s3_target_from_env()
    assert prefix == "custom-prefix"


# ---------------------------------------------------------------------------
# upload_dir / download_prefix (moto)
# ---------------------------------------------------------------------------


@mock_aws
def test_upload_dir_then_download_prefix_roundtrip(tmp_path, aws_creds):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=BUCKET)

    src = tmp_path / "checkpoint-10"
    src.mkdir()
    (src / "adapter_model.safetensors").write_bytes(b"weights")
    (src / "optimizer.pt").write_bytes(b"opt-state")
    nested = src / "sub"
    nested.mkdir()
    (nested / "extra.json").write_text("{}", encoding="utf-8")

    checkpoint_sync.upload_dir(str(src), BUCKET, "cairn-checkpoints/run-1")

    listed = client.list_objects_v2(Bucket=BUCKET, Prefix="cairn-checkpoints/run-1/")
    keys = {o["Key"] for o in listed.get("Contents", [])}
    assert keys == {
        "cairn-checkpoints/run-1/adapter_model.safetensors",
        "cairn-checkpoints/run-1/optimizer.pt",
        "cairn-checkpoints/run-1/sub/extra.json",
    }

    dest = tmp_path / "downloaded"
    checkpoint_sync.download_prefix(BUCKET, "cairn-checkpoints/run-1", str(dest))
    assert (dest / "adapter_model.safetensors").read_bytes() == b"weights"
    assert (dest / "optimizer.pt").read_bytes() == b"opt-state"
    assert (dest / "sub" / "extra.json").read_text(encoding="utf-8") == "{}"


@mock_aws
def test_upload_dir_missing_local_dir_is_noop(aws_creds):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=BUCKET)
    # must not raise even though the dir doesn't exist.
    checkpoint_sync.upload_dir("/no/such/dir", BUCKET, "prefix")
    listed = client.list_objects_v2(Bucket=BUCKET, Prefix="prefix/")
    assert listed.get("KeyCount", 0) == 0


def test_upload_dir_never_raises_on_client_error(tmp_path, monkeypatch):
    # No AWS creds/region configured at all + no moto → boto3 client construction
    # or the call itself may error; upload_dir must swallow it.
    src = tmp_path / "checkpoint-1"
    src.mkdir()
    (src / "f.txt").write_text("x", encoding="utf-8")
    checkpoint_sync.upload_dir(str(src), "no-such-bucket-xyz", "prefix")  # must not raise


def test_download_prefix_never_raises_when_bucket_absent(tmp_path):
    checkpoint_sync.download_prefix("no-such-bucket-xyz", "prefix", str(tmp_path / "out"))  # must not raise


# ---------------------------------------------------------------------------
# latest_checkpoint_dir
# ---------------------------------------------------------------------------


def test_latest_checkpoint_dir_picks_highest_step(tmp_path):
    for step in (10, 30, 20):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "not-a-checkpoint").mkdir()
    best = checkpoint_sync.latest_checkpoint_dir(str(tmp_path))
    assert best == str(tmp_path / "checkpoint-30")


def test_latest_checkpoint_dir_none_when_empty(tmp_path):
    assert checkpoint_sync.latest_checkpoint_dir(str(tmp_path)) is None


def test_latest_checkpoint_dir_none_when_dir_missing():
    assert checkpoint_sync.latest_checkpoint_dir("/no/such/dir") is None


# ---------------------------------------------------------------------------
# make_checkpoint_callback — on_save uploads the newest checkpoint dir
# ---------------------------------------------------------------------------


@mock_aws
def test_on_save_uploads_newest_checkpoint_and_noop_without_target(tmp_path, monkeypatch, aws_creds):
    pytest.importorskip("transformers")
    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", BUCKET)
    monkeypatch.setenv("CAIRN_RUN_ID", "run-42")

    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=BUCKET)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "checkpoint-5").mkdir()
    (output_dir / "checkpoint-5" / "adapter_model.safetensors").write_bytes(b"v5")
    (output_dir / "checkpoint-10").mkdir()
    (output_dir / "checkpoint-10" / "adapter_model.safetensors").write_bytes(b"v10")

    progress_path = tmp_path / "metrics.json"
    cb = checkpoint_sync.make_checkpoint_callback(str(progress_path))
    cb.on_save(_Args(str(output_dir)), _State(), _Control())

    # The checkpoint-N/ directory is part of the key (what the pull side +
    # resume_from_checkpoint need); a flat upload was the 2026-08-19 live bug.
    listed = client.list_objects_v2(
        Bucket=BUCKET, Prefix="cairn-checkpoints/run-42/checkpoint-10/adapter_model.safetensors"
    )
    assert listed.get("KeyCount", 0) == 1
    body = client.get_object(
        Bucket=BUCKET, Key="cairn-checkpoints/run-42/checkpoint-10/adapter_model.safetensors"
    )["Body"].read()
    assert body == b"v10"  # the NEWEST checkpoint (highest step), not checkpoint-5
    assert client.list_objects_v2(Bucket=BUCKET, Prefix="cairn-checkpoints/run-42/checkpoint-5/").get("KeyCount", 0) == 0

    import json

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["checkpoint_uri"] == f"s3://{BUCKET}/cairn-checkpoints/run-42/"


def test_on_save_noop_without_bucket_configured(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    _clear_checkpoint_env(monkeypatch)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "checkpoint-1").mkdir()

    progress_path = tmp_path / "metrics.json"
    cb = checkpoint_sync.make_checkpoint_callback(str(progress_path))
    # must not raise, and must not write a checkpoint_uri anywhere.
    cb.on_save(_Args(str(output_dir)), _State(), _Control())
    assert not progress_path.exists()


# ---------------------------------------------------------------------------
# pull_checkpoints_if_any
# ---------------------------------------------------------------------------


@mock_aws
def test_pull_checkpoints_if_any_downloads_when_local_empty(tmp_path, monkeypatch, aws_creds):
    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", BUCKET)
    monkeypatch.setenv("CAIRN_RUN_ID", "run-7")

    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=BUCKET)
    client.put_object(
        Bucket=BUCKET,
        Key="cairn-checkpoints/run-7/checkpoint-20/adapter_model.safetensors",
        Body=b"resumed-weights",
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    checkpoint_sync.pull_checkpoints_if_any(str(out_dir))

    downloaded = out_dir / "checkpoint-20" / "adapter_model.safetensors"
    assert downloaded.read_bytes() == b"resumed-weights"


@mock_aws
def test_pull_checkpoints_if_any_skips_when_local_checkpoint_exists(tmp_path, monkeypatch, aws_creds):
    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", BUCKET)
    monkeypatch.setenv("CAIRN_RUN_ID", "run-8")

    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=BUCKET)
    client.put_object(
        Bucket=BUCKET,
        Key="cairn-checkpoints/run-8/checkpoint-20/adapter_model.safetensors",
        Body=b"should-not-be-pulled",
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "checkpoint-5").mkdir()
    (out_dir / "checkpoint-5" / "marker.txt").write_text("local", encoding="utf-8")

    checkpoint_sync.pull_checkpoints_if_any(str(out_dir))

    # local checkpoint already present → no S3 pull attempted.
    assert not (out_dir / "checkpoint-20").exists()


def test_pull_checkpoints_if_any_noop_without_bucket(tmp_path, monkeypatch):
    _clear_checkpoint_env(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    checkpoint_sync.pull_checkpoints_if_any(str(out_dir))  # must not raise
    assert list(out_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# schema defaults — packing / save_total_limit (ADR-0064 §3)
# ---------------------------------------------------------------------------


def test_train_spec_packing_and_save_total_limit_defaults():
    spec = TrainSpec(dataset_uri="d.jsonl")
    assert spec.packing is True
    assert spec.save_total_limit == 2


def test_train_spec_packing_and_save_total_limit_overridable():
    spec = TrainSpec(dataset_uri="d.jsonl", packing=False, save_total_limit=0)
    assert spec.packing is False
    assert spec.save_total_limit == 0


def test_train_spec_save_total_limit_rejects_negative():
    with pytest.raises(Exception):  # noqa: PT011 - pydantic ValidationError
        TrainSpec(dataset_uri="d.jsonl", save_total_limit=-1)


# ---------------------------------------------------------------------------
# checkpoint_uri_for
# ---------------------------------------------------------------------------


def test_checkpoint_uri_for_shape():
    uri = checkpoint_sync.checkpoint_uri_for("b", "p", "r")
    assert uri == "s3://b/p/r/"
