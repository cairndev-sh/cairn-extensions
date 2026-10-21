"""Tests for dataset_fetch (#701): resolve an s3:// / file:// / HF dataset_uri to
a path the trainer's loader can read. Hermetic — moto mocks S3, no real network,
no torch stack (dataset_fetch imports only boto3 via checkpoint_sync)."""
from __future__ import annotations

import os

import boto3
import pytest
from moto import mock_aws

from operator_training import dataset_fetch

BUCKET = "cairn-ds-bucket"
ROWS = b'{"instruction": "hi", "output": "hello"}\n{"instruction": "x", "output": "y"}\n'


@pytest.fixture
def aws_creds(monkeypatch):
    # moto needs *some* creds present; ensure no MinIO endpoint override leaks in
    # so the client hits moto's mock backend rather than a real/absent endpoint.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)


@mock_aws
def test_resolve_s3_downloads_to_local(aws_creds):
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=BUCKET, Key="prepared/ds.jsonl", Body=ROWS
    )
    local = dataset_fetch.resolve_dataset_uri(f"s3://{BUCKET}/prepared/ds.jsonl")
    assert os.path.isfile(local)
    assert local.endswith(".jsonl")  # basename preserved → loader picks the json path
    with open(local, "rb") as fh:
        assert fh.read() == ROWS


@mock_aws
def test_resolve_s3_is_reused_not_refetched(aws_creds):
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=BUCKET, Key="prepared/ds.jsonl", Body=ROWS
    )
    uri = f"s3://{BUCKET}/prepared/ds.jsonl"
    first = dataset_fetch.resolve_dataset_uri(uri)
    second = dataset_fetch.resolve_dataset_uri(uri)
    assert first == second  # deterministic temp path keyed by the uri hash


def test_resolve_file_uri_normalizes_to_path():
    assert dataset_fetch.resolve_dataset_uri("file:///tmp/ds.jsonl") == "/tmp/ds.jsonl"
    # percent-escapes are decoded
    assert dataset_fetch.resolve_dataset_uri("file:///tmp/my%20ds.jsonl") == "/tmp/my ds.jsonl"


def test_passthrough_hf_id_local_and_empty():
    assert dataset_fetch.resolve_dataset_uri("Qwen/Qwen2.5-3B-Instruct") == "Qwen/Qwen2.5-3B-Instruct"
    assert dataset_fetch.resolve_dataset_uri("/data/local.jsonl") == "/data/local.jsonl"
    assert dataset_fetch.resolve_dataset_uri("") == ""


def test_malformed_s3_uri_raises():
    with pytest.raises(ValueError):
        dataset_fetch.resolve_dataset_uri("s3://only-bucket-no-key")


# ── #1714: the `#sha256=` pin a launch-resolved `dataset:` ref carries ───────
# `resolve_dataset_refs_in_config` rewrites `dataset:facts-w1-train@2` to
# `s3://…/train-ready.jsonl#sha256=<hash>`. HF `datasets`/fsspec treat the
# fragment as part of the OBJECT KEY, so the first governed-ref train launch
# died with `FileNotFoundError: Unable to find 's3://…#sha256=951b…'` — the
# trainer had bypassed this module entirely. Stripping is only half: the pin is
# a RETRIEVAL rule, so the bytes get verified too (as ingest.normalize has since
# #1466).

def test_split_pin_separates_location_from_hash():
    uri, h = dataset_fetch.split_pin("s3://b/k/train.jsonl#sha256=abc123")
    assert uri == "s3://b/k/train.jsonl" and h == "abc123"


@pytest.mark.parametrize("raw", [
    "s3://b/k/train.jsonl",                 # no pin at all
    "/local/train.jsonl",
    "org/dataset-name",                     # an HF hub id
    "",
])
def test_split_pin_passes_unpinned_through(raw):
    assert dataset_fetch.split_pin(raw) == (raw, "")


def test_split_pin_ignores_a_fragment_that_is_not_a_sha_pin():
    """Only `#sha256=` is the pin; any other fragment is still not part of the
    key, but there is nothing to verify against."""
    assert dataset_fetch.split_pin("s3://b/k.jsonl#page=2") == ("s3://b/k.jsonl", "")


@mock_aws
def test_resolve_strips_the_pin_and_downloads(aws_creds):
    """THE regression: the pinned form must LOAD. Before #1714 the fragment rode
    into the S3 key and the object was never found."""
    import hashlib

    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=BUCKET, Key="d/train.jsonl", Body=ROWS)
    digest = hashlib.sha256(ROWS).hexdigest()

    got = dataset_fetch.resolve_dataset_uri(f"s3://{BUCKET}/d/train.jsonl#sha256={digest}")
    assert "#" not in got, "the pin must never reach the loader"
    assert os.path.isfile(got)
    with open(got, "rb") as fh:
        assert fh.read() == ROWS


@mock_aws
def test_resolve_fails_closed_when_the_bytes_do_not_match_the_pin(aws_creds):
    """A NAME must guarantee BYTES. An object overwritten since registration is
    refused rather than silently trained on."""
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=BUCKET, Key="d/train2.jsonl", Body=ROWS)

    with pytest.raises(RuntimeError, match="verification FAILED"):
        dataset_fetch.resolve_dataset_uri(
            f"s3://{BUCKET}/d/train2.jsonl#sha256={'0' * 64}")


def test_resolve_verifies_a_pinned_file_uri(tmp_path):
    """The same rule on the file:// path — no S3 required."""
    import hashlib

    p = tmp_path / "train.jsonl"
    p.write_bytes(ROWS)
    digest = hashlib.sha256(ROWS).hexdigest()

    assert dataset_fetch.resolve_dataset_uri(f"file://{p}#sha256={digest}") == str(p)
    with pytest.raises(RuntimeError, match="verification FAILED"):
        dataset_fetch.resolve_dataset_uri(f"file://{p}#sha256={'1' * 64}")


def test_an_unpinned_uri_is_never_verified(tmp_path):
    """No pin → nothing to check; the historical behaviour is untouched."""
    p = tmp_path / "train.jsonl"
    p.write_bytes(ROWS)
    assert dataset_fetch.resolve_dataset_uri(f"file://{p}") == str(p)


def test_a_passed_through_hf_id_keeps_its_pin_free_form():
    """An HF hub id is not fetched here, so there is no local file to verify —
    it must pass through rather than raise."""
    assert dataset_fetch.resolve_dataset_uri("org/dataset-name") == "org/dataset-name"
