"""#732 — cairn-train-local publishes the adapter to HF/S3 FROM THE BOX when
artifact_target is hf|s3 (the SSH path was file://-only). Hermetic: HfApi +
checkpoint_sync are mocked, no network. The SFTP-fetched adapter is unaffected;
publish is additive and best-effort (a failure records publish_error, never raises).
"""
from __future__ import annotations

import os
import tempfile

import pytest

from operator_training.local_entry import _publish_adapter
from operator_training.schemas.train import TrainResult, TrainSpec


@pytest.fixture
def adapter_dir():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "adapter_model.safetensors"), "w").close()
    return d


def _result(adapter_dir):
    return TrainResult(status="succeeded", backend="local",
                       adapter_uri=f"file://{adapter_dir}", metrics={"train_loss": 0.5})


def test_local_target_does_not_publish(adapter_dir):
    spec = TrainSpec(dataset_uri="x", output_name="m", artifact_target="local")
    r = _result(adapter_dir)
    _publish_adapter(spec, r)
    assert "published_uri" not in r.metrics and "publish_error" not in r.metrics


def test_hf_target_uploads_folder(adapter_dir, monkeypatch):
    calls = {}

    class _FakeApi:
        def create_repo(self, **kw): calls["create_repo"] = kw
        def upload_folder(self, **kw): calls["upload_folder"] = kw

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    spec = TrainSpec(dataset_uri="x", output_name="m", artifact_target="hf", hf_repo="acme/adapter")
    r = _result(adapter_dir)
    _publish_adapter(spec, r)
    assert r.metrics["published_uri"] == "hf://acme/adapter"
    assert calls["create_repo"]["repo_id"] == "acme/adapter"
    assert calls["upload_folder"]["folder_path"] == adapter_dir


def _fake_client(n_objects):
    """A boto3 stand-in whose listing reports ``n_objects`` under any prefix.

    #1715 — the publish now VERIFIES the upload landed, because
    ``checkpoint_sync.upload_dir`` is best-effort by contract (it logs and
    returns on error, never raises), so a mocked-or-real call returning cleanly
    proves nothing. Every s3 test therefore has to say what S3 now holds.
    """
    class _Paginator:
        def paginate(self, **_kw):
            return [{"Contents": [{"Key": f"k{i}"} for i in range(n_objects)]}]

    class _Client:
        def get_paginator(self, _name):
            return _Paginator()

    return _Client()


def test_s3_target_uploads_dir(adapter_dir, monkeypatch):
    from operator_training import checkpoint_sync
    up = {}
    monkeypatch.setattr(checkpoint_sync, "s3_target_from_env", lambda: ("buck", "cairn-checkpoints", "m"))
    monkeypatch.setattr(checkpoint_sync, "upload_dir", lambda d, b, k: up.update(dir=d, bucket=b, key=k))
    monkeypatch.setattr(checkpoint_sync, "_client", lambda: _fake_client(1))  # the fixture's 1 file

    spec = TrainSpec(dataset_uri="x", output_name="m", artifact_target="s3")
    r = _result(adapter_dir)
    _publish_adapter(spec, r)
    assert r.metrics["published_uri"] == "s3://buck/cairn-checkpoints/m/adapter/"
    assert up["bucket"] == "buck" and up["key"] == "cairn-checkpoints/m/adapter"


def test_s3_without_bucket_records_error_not_raise(adapter_dir, monkeypatch):
    from operator_training import checkpoint_sync
    monkeypatch.setattr(checkpoint_sync, "s3_target_from_env", lambda: None)
    spec = TrainSpec(dataset_uri="x", output_name="m", artifact_target="s3")
    r = _result(adapter_dir)
    _publish_adapter(spec, r)  # must not raise
    assert "CAIRN_CHECKPOINT_BUCKET" in r.metrics["publish_error"]
    assert "published_uri" not in r.metrics


def test_publish_failure_is_best_effort(adapter_dir, monkeypatch):
    import huggingface_hub

    class _BoomApi:
        def create_repo(self, **kw): raise RuntimeError("401 unauthorized")
        def upload_folder(self, **kw): ...

    monkeypatch.setattr(huggingface_hub, "HfApi", _BoomApi)
    spec = TrainSpec(dataset_uri="x", output_name="m", artifact_target="hf")
    r = _result(adapter_dir)
    _publish_adapter(spec, r)  # must not raise — the fetched adapter stands
    assert "401 unauthorized" in r.metrics["publish_error"]
    assert "published_uri" not in r.metrics


# ── #1715 — the publish is VERIFIED, and the two paths fail differently ──────

def test_s3_publish_that_silently_drops_files_is_NOT_reported_as_published(adapter_dir, monkeypatch):
    """upload_dir swallows its own errors, so the old code recorded published_uri
    for an upload that never landed. The verification is what makes the URI mean
    something."""
    from operator_training import checkpoint_sync

    monkeypatch.setattr(checkpoint_sync, "s3_target_from_env", lambda: ("buck", "p", "m"))
    monkeypatch.setattr(checkpoint_sync, "upload_dir", lambda d, b, k: None)  # "succeeds", uploads nothing
    monkeypatch.setattr(checkpoint_sync, "_client", lambda: _fake_client(0))

    spec = TrainSpec(dataset_uri="x", output_name="m", artifact_target="s3")
    r = _result(adapter_dir)
    _publish_adapter(spec, r)
    assert "published_uri" not in r.metrics
    assert "0/1 objects" in r.metrics["publish_error"]


def test_an_already_remote_uri_is_left_alone(adapter_dir):
    """A trainer that already returned hf:// — re-publishing a URI as if it
    were a directory would be wrong."""
    from operator_training.publish import publish_adapter_dir

    assert publish_adapter_dir("hf", "hf://acme/already-there") == ""
    assert publish_adapter_dir("s3", "s3://buck/p/adapter/") == ""


def test_a_directory_with_no_model_file_is_refused(tmp_path):
    from operator_training.publish import AdapterPublishError, publish_adapter_dir

    (tmp_path / "notes.txt").write_text("nothing to see")
    with pytest.raises(AdapterPublishError, match="does not look like an adapter dir"):
        publish_adapter_dir("s3", str(tmp_path))


def test_a_missing_directory_is_refused(tmp_path):
    from operator_training.publish import AdapterPublishError, publish_adapter_dir

    with pytest.raises(AdapterPublishError, match="no local adapter dir"):
        publish_adapter_dir("s3", str(tmp_path / "gone"))


def test_local_target_is_a_no_op_not_a_failure(adapter_dir):
    from operator_training.publish import publish_adapter_dir

    assert publish_adapter_dir("local", adapter_dir) == ""
