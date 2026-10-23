"""cairn-extensions#6 — the split stage returns train / val / test as products and registers them
with their roles, so the holdout door sees the test version a trainer must not meet.

The split runs through the core runner in process (`run_dataflow`), the registration through
the leg executor's own function, and the holdout check is the door's `check_holdout`, all on a
hermetic sqlite store and a moto-mocked S3 bucket: a registered split must be in the object store,
because a local one is a file in the leg's temporary workdir. Stage outputs are checked against
the installed contract wheel."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from cairn.core.dataflow import Budget, DataflowSpec, StageRef
from cairn.core.dataflow.conformance_check import bundle, problems
from cairn.core.dataflow.runner import run_dataflow
from cairn.core.pack.contracts import _installed

from operator_dataset_checks.dataflow_stages import split

#: Ten documents of three rows each; `doc` is the group no split may straddle.
_ROWS = [{"doc": f"d{d}", "instruction": f"Question {d}.{k} about IfcWall {d * 3 + k}?",
          "output": f"Answer {d}.{k}."} for d in range(10) for k in range(3)]
_CONFIG = {"seed": 13, "train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2,
           "group_by": "doc", "backend": "hash", "export": True, "register": "facts",
           "store": "s3"}
_BUCKET = "cairn-test-artifacts"


@pytest.fixture
def s3(monkeypatch):
    """A moto-mocked bucket named by the s3/put contract's environment."""
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    monkeypatch.setenv("CAIRN_ARTIFACT_S3_BUCKET", _BUCKET)
    monkeypatch.setenv("CAIRN_ARTIFACT_S3_PREFIX", "datasets")
    for name in ("CAIRN_ARTIFACT_S3_ENDPOINT", "OBS_ARTIFACT_S3_ENDPOINT", "OBS_ARTIFACT_S3_BUCKET"):
        monkeypatch.delenv(name, raising=False)
    with moto.mock_aws():
        client = boto3.client("s3", region_name="eu-central-1")
        client.create_bucket(Bucket=_BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-central-1"})
        yield client


def _lines(client, uri: str) -> list[str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8").splitlines()


def _source(tmp_path: Path) -> str:
    src = tmp_path / "corpus.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in _ROWS), encoding="utf-8")
    return str(src)


def _run(tmp_path: Path, config: dict):
    spec = DataflowSpec(stages=[StageRef(stage="ingest.normalize", config={"path": _source(tmp_path)}),
                                StageRef(stage="split", config=config)], budget=Budget())
    result = run_dataflow(spec, tmp_path / "wd")
    return result.output_artifacts + result.named_outputs  # per-stage first, then named (ADR-0141)


def _schema(uri: str) -> dict:
    _resolved, bodies = _installed()
    return bundle({u: body for u, (_d, body) in bodies.items()}, uri)


def _contract(part: str) -> dict:
    _resolved, bodies = _installed()
    return _schema(bodies["contract://dataset.stage.split/1.0.0"][1][part])


def test_without_export_the_split_is_unchanged(tmp_path):
    ingest, only = _run(tmp_path, {k: v for k, v in _CONFIG.items() if k not in ("export", "register", "store")})
    assert "outputs" not in only.stats and only.stats["train_path"].endswith("splits/train.jsonl")
    assert problems(_contract("output"), {k: v for k, v in only.stats.items() if k != "kernel_version"}, "output") == []


def test_each_split_is_a_product_with_its_role_and_no_group_straddles(tmp_path, s3):
    _ingest, stage, *products = _run(tmp_path, _CONFIG)
    assert [p.stats["output_name"] for p in products] == ["train", "val", "test"]
    roles = {p.stats["output_name"]: (p.stats["register_as"], p.stats["register_role"]) for p in products}
    assert roles == {"train": ("facts-train", "training"), "val": ("facts-val", "validation"),
                     "test": ("facts-test", "test")}
    groups = []
    for p in products:
        assert p.parents == stage.parents and p.kind == "dataset"
        assert p.uri.startswith(f"s3://{_BUCKET}/datasets/dataflow/")
        lines = _lines(s3, p.uri)
        assert p.rows == len(lines) > 0
        groups.append({json.loads(line)["doc"] for line in lines})
    assert sum(p.rows for p in products) == len(_ROWS)
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i + 1:])
    assert stage.stats["no_leakage"] is True


async def test_registration_gives_each_split_its_role_and_the_door_finds_no_overlap(tmp_path, s3):
    from cairn.core.dispatch.leg_executor import register_leg_descriptors
    from cairn.core.holdout import check_holdout
    from cairn.core.storage.artifacts import SqliteArtifactStore

    descriptors = _run(tmp_path, _CONFIG)
    store = SqliteArtifactStore(str(tmp_path / "a.db"))
    out_stats: dict = {}
    await register_leg_descriptors(descriptors, store, run_id="run-split", out_stats=out_stats)
    assert out_stats["dataset_versions"] == ["facts-train@1", "facts-val@1", "facts-test@1"]
    assert (await store.get_dataset_version_meta("facts-test"))["role"] == "test"
    verdict = await check_holdout(["dataset:facts-train@1"], store=store, workspace_id="default")
    assert verdict.checked and verdict.ok and not verdict.overlaps
    assert [h.name for h in verdict.holdouts] == ["facts-test"]
    await store.close()


def test_a_local_export_without_register_stays_a_local_product(tmp_path):
    config = {k: v for k, v in _CONFIG.items() if k != "register"} | {"store": "local"}
    _ingest, _stage, *products = _run(tmp_path, config)
    assert [p.stats["output_name"] for p in products] == ["train", "val", "test"]
    assert all(Path(p.uri.removeprefix("file://")).is_file() and "register_as" not in p.stats for p in products)


@pytest.mark.parametrize("store, env", [("local", "s3"), ("", "local"), ("", "")])
def test_register_on_the_local_store_is_refused_before_the_split_runs(tmp_path, monkeypatch, store, env):
    """cairn-extensions#6 — a version registered from a local split would name a file in the leg's
    temporary workdir, gone after a reboot and unreadable from another machine. Refused whether
    the stage asks for `local` or the environment's artifact store is local (or unset)."""
    monkeypatch.delenv("CAIRN_PROFILE", raising=False)
    monkeypatch.delenv("OBS_PROFILE", raising=False)
    monkeypatch.delenv("OBS_ARTIFACT_STORE", raising=False)
    if env:
        monkeypatch.setenv("CAIRN_ARTIFACT_STORE", env)
    else:
        monkeypatch.delenv("CAIRN_ARTIFACT_STORE", raising=False)
    (tmp_path / "current.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _ROWS), encoding="utf-8")
    config = {**_CONFIG, "store": store} if store else {k: v for k, v in _CONFIG.items() if k != "store"}
    with pytest.raises(ValueError, match=r"`register` needs `store: s3`.*CAIRN_ARTIFACT_S3_BUCKET") as refused:
        split(tmp_path, config, [])
    assert str(tmp_path) in str(refused.value)
    assert not (tmp_path / "splits").exists()


def test_register_follows_the_environment_s3_store(tmp_path, s3, monkeypatch):
    monkeypatch.delenv("CAIRN_PROFILE", raising=False)
    monkeypatch.delenv("OBS_PROFILE", raising=False)
    monkeypatch.setenv("CAIRN_ARTIFACT_STORE", "s3")
    config = {k: v for k, v in _CONFIG.items() if k != "store"}
    *_rest, test_split = _run(tmp_path, config)
    assert test_split.stats["register_as"] == "facts-test" and test_split.uri.startswith(f"s3://{_BUCKET}/")


@pytest.mark.parametrize("config, message", [
    ({"register": "facts"}, "`register` needs `export: true`"),
    ({"export": True, "shard_size": 5}, "needs shard_size 0"),
])
def test_an_export_that_cannot_be_one_product_per_split_is_refused(tmp_path, config, message):
    (tmp_path / "current.jsonl").write_text(json.dumps(_ROWS[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        split(tmp_path, config, [])


def test_the_configuration_is_closed():
    schema = _contract("configuration")
    assert problems(schema, _CONFIG, "configuration") == []
    assert problems(schema, {**_CONFIG, "exports": True}, "configuration")
