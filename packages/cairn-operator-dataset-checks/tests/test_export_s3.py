"""#1442 — export.jsonl to the S3 artifact store and ingest back from it: the
leg-to-leg handoff that survives a box. moto-mocked S3 (hermetic)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from operator_dataset_checks import dataflow_stages as S  # noqa: E402

ROWS = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(5)]


@pytest.fixture
def s3env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    monkeypatch.setenv("OBS_ARTIFACT_S3_BUCKET", "cairn-test-artifacts")
    monkeypatch.setenv("OBS_ARTIFACT_S3_PREFIX", "campaign")
    monkeypatch.delenv("OBS_ARTIFACT_S3_ENDPOINT", raising=False)
    with moto.mock_aws():
        boto3.client("s3", region_name="eu-central-1").create_bucket(
            Bucket="cairn-test-artifacts", CreateBucketConfiguration={"LocationConstraint": "eu-central-1"})
        yield


def _wd(tmp_path: Path, name: str) -> Path:
    src = tmp_path / f"{name}.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    wd = tmp_path / f"wd-{name}"
    wd.mkdir()
    S.ingest_normalize(wd, {"path": str(src)}, [])
    return wd


def test_export_s3_then_ingest_from_s3(tmp_path: Path, s3env):
    wd1 = _wd(tmp_path, "leg1")
    r = S.export_jsonl(wd1, {"name": "train-ready.jsonl", "store": "s3"}, [])
    assert r["store"] == "s3" and r["format"] == "jsonl"
    assert r["output"] == f"s3://cairn-test-artifacts/campaign/dataflow/{wd1.name}/train-ready.jsonl"
    assert len(r["content_hash"]) == 64 and r["bytes"] > 0
    # the object exists and is byte-identical to the working file
    body = boto3.client("s3").get_object(Bucket="cairn-test-artifacts",
                                         Key=f"campaign/dataflow/{wd1.name}/train-ready.jsonl")["Body"].read()
    assert body == S._current(wd1).read_bytes()
    # a SECOND leg ingests by URI — the handoff
    wd2 = tmp_path / "wd-leg2"; wd2.mkdir()
    r2 = S.ingest_normalize(wd2, {"path": r["output"]}, [])
    assert r2["rows"] == 5
    assert [json.loads(l) for l in S._current(wd2).read_text().splitlines()] == ROWS


def test_export_store_follows_the_environment_backend(tmp_path: Path, s3env, monkeypatch):
    """`store` unset → the environment's artifact_store (profile → env → local)."""
    wd = _wd(tmp_path, "env")
    monkeypatch.setenv("OBS_ARTIFACT_STORE", "s3")
    monkeypatch.delenv("OBS_PROFILE", raising=False)
    r = S.export_jsonl(wd, {"name": "x.jsonl"}, [])
    assert r["store"] == "s3" and r["output"].startswith("s3://cairn-test-artifacts/")
    monkeypatch.setenv("OBS_ARTIFACT_STORE", "local")
    wd = _wd(tmp_path, "loc")
    r = S.export_jsonl(wd, {"name": "x.jsonl"}, [])
    assert r["store"] == "local" and r["output"] == str(wd / "x.jsonl") and len(r["content_hash"]) == 64


def test_export_s3_fails_loud_without_bucket(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OBS_ARTIFACT_S3_BUCKET", raising=False)
    wd = _wd(tmp_path, "nb")
    with pytest.raises(RuntimeError, match="CAIRN_ARTIFACT_S3_BUCKET"):
        S.export_jsonl(wd, {"store": "s3"}, [])


# ── cairn-extensions#6: `register` needs the s3 store ───────────────────────

@pytest.mark.parametrize("store, env", [("local", "s3"), ("", "local"), ("", "")])
def test_register_on_the_local_store_is_refused_before_anything_is_written(tmp_path: Path, monkeypatch, store, env):
    """A version registered from a local export would name a file in the leg's temporary workdir.
    Refused whether the stage asks for `local` or the environment's store is local (or unset),
    and before the export file is written."""
    for name in ("OBS_PROFILE", "CAIRN_PROFILE", "OBS_ARTIFACT_STORE"):
        monkeypatch.delenv(name, raising=False)
    if env:
        monkeypatch.setenv("CAIRN_ARTIFACT_STORE", env)
    else:
        monkeypatch.delenv("CAIRN_ARTIFACT_STORE", raising=False)
    wd = _wd(tmp_path, "reg")
    before = sorted(p.name for p in wd.iterdir())
    config = {"name": "x.jsonl", "register": "facts", **({"store": store} if store else {})}
    with pytest.raises(ValueError, match=r"export.jsonl: `register` needs `store: s3`.*CAIRN_ARTIFACT_S3_BUCKET") as refused:
        S.export_jsonl(wd, config, [])
    assert str(wd) in str(refused.value)
    assert sorted(p.name for p in wd.iterdir()) == before  # nothing written


@pytest.mark.parametrize("config, env", [({"store": "s3"}, "local"), ({}, "s3")])
def test_register_on_s3_uploads_and_asks_for_the_version(tmp_path: Path, s3env, monkeypatch, config, env):
    for name in ("OBS_PROFILE", "CAIRN_PROFILE", "OBS_ARTIFACT_STORE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CAIRN_ARTIFACT_STORE", env)
    wd = _wd(tmp_path, "s3reg")
    r = S.export_jsonl(wd, {"name": "x.jsonl", "register": "facts", **config}, [])
    assert r["store"] == "s3" and r["register_as"] == "facts"
    key = f"campaign/dataflow/{wd.name}/x.jsonl"
    body = boto3.client("s3").get_object(Bucket="cairn-test-artifacts", Key=key)["Body"].read()
    assert r["output"] == f"s3://cairn-test-artifacts/{key}" and body == S._current(wd).read_bytes()
