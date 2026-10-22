"""dataset/split registers FROZEN CHILD VERSIONS (#2020, ADR-0102 D1).

A split used to be files on disk: reproducible, but not citable. These tests
pin the three properties that make a split a registry object instead —

* each split becomes its own version with its own content hash, the right ROLE
  (train → training, val → validation, test → test) and lineage to the parent;
* the DECLARED policy (ratios, seed, group_by, stratify_by, backend) rides on
  every child, so the split is reproducible from the record alone;
* **items never move** — re-splitting under a different policy mints NEW
  children and leaves the old ones byte-identical at their old versions, while
  re-splitting under the SAME policy is idempotent.

Hermetic: a temp sqlite ``OBS_STORAGE`` and the pure-stdlib ``hash`` backend.
"""
from __future__ import annotations

import json

import pytest

from cairn.core.operator import OperationContext
from cairn.core.storage.artifacts import build_artifact_store
from operator_dataset_checks.operators.split import SplitOp
from operator_dataset_checks.schemas.split import SplitInput


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="split")


def _source(tmp_path, n: int = 40) -> str:
    path = tmp_path / "ds.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": f"r{i}", "prompt": f"q{i}", "source": f"s{i % 3}"})
                  for i in range(n)),
        encoding="utf-8")
    return str(path)


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("OBS_STORAGE", f"sqlite:///{tmp_path / 'cairn.db'}")
    monkeypatch.setenv("OBS_WORKSPACE", "default")
    return tmp_path


def _inp(tmp_path, src: str, *, seed: int = 13, out: str = "v1", **kw) -> SplitInput:
    return SplitInput(
        path=src, output_dir=str(tmp_path / out), backend="hash", seed=seed,
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        stratify_by="source", register_as="facts", **kw)


async def test_each_split_becomes_a_child_version_with_its_own_hash(registry):
    src = _source(registry)
    out = await SplitOp().run(_inp(registry, src), _ctx())

    assert set(out.child_versions) == {"train", "val", "test"}
    assert all(ref.endswith("@1") for ref in out.child_versions.values())
    assert out.child_versions["train"] == "dataset:facts-train@1"
    # three distinct artifacts, not one name pointing three ways
    assert len(set(out.child_hashes.values())) == 3

    store = build_artifact_store()
    roles = {}
    for split, ref in out.child_versions.items():
        name = ref[len("dataset:"):].split("@")[0]
        roles[split] = (await store.get_dataset_version_meta(name))["role"]
    assert roles == {"train": "training", "val": "validation", "test": "test"}
    await store.close()


async def test_the_declared_policy_rides_on_every_child(registry):
    """"Declared and stratified, never a random tail" — the record has to say
    what the policy WAS, or the split cannot be reproduced from it."""
    src = _source(registry)
    await SplitOp().run(_inp(registry, src), _ctx())

    store = build_artifact_store()
    meta = await store.get_dataset_version_meta("facts-val")
    assert meta["split_policy"]["seed"] == 13
    assert meta["split_policy"]["stratify_by"] == "source"
    assert meta["split_policy"]["group_by"] is None
    assert meta["split_policy"]["backend"] == "hash"
    assert pytest.approx(meta["split_policy"]["ratios"]["val"], abs=1e-6) == 0.2
    await store.close()


async def test_a_child_carries_lineage_to_the_parent_version(registry):
    src = _source(registry)
    store = build_artifact_store()
    from cairn.core.dataflow.contract import ArtifactDescriptor

    await store.register(ArtifactDescriptor(content_hash="parent-hash", kind="dataset", uri=src))
    await store.register_dataset_version("facts", "parent-hash", schema="prompt-completion",
                                         column_mapping={"prompt": "prompt"})
    await store.close()

    out = await SplitOp().run(_inp(registry, src), _ctx())

    store = build_artifact_store()
    meta = await store.get_dataset_version_meta("facts-train")
    assert meta["parent_version"] == "dataset:facts@1"
    # the declared shape is inherited, not re-guessed per split
    assert meta["schema"] == "prompt-completion"
    assert meta["column_mapping"] == {"prompt": "prompt"}
    # the artifact DAG walks from child to parent
    lineage = await store.lineage(out.child_hashes["train"])
    assert [p["id"] for p in lineage["parents"]] == ["parent-hash"]
    await store.close()


async def test_a_resplit_under_a_new_policy_mints_new_children_and_moves_nothing(registry):
    """The acceptance property: re-running a split yields new child hashes, and
    the old children are byte-identical at their old version numbers."""
    src = _source(registry)
    first = await SplitOp().run(_inp(registry, src, seed=13, out="v1"), _ctx())
    before = {split: open(path, encoding="utf-8").read()
              for split, path in first.output_paths.items()}

    second = await SplitOp().run(_inp(registry, src, seed=99, out="v2"), _ctx())

    assert all(ref.endswith("@2") for ref in second.child_versions.values())
    for split in ("train", "val", "test"):
        assert second.child_hashes[split] != first.child_hashes[split], (
            f"a re-split under a new seed must mint a new {split} hash")
        # the OLD file is untouched …
        assert open(first.output_paths[split], encoding="utf-8").read() == before[split]

    # … and the old VERSION still resolves to those exact bytes.
    store = build_artifact_store()
    for split in ("train", "val", "test"):
        version, rec = await store.get_dataset_version(f"facts-{split}", 1)
        assert (version, rec.id) == (1, first.child_hashes[split])
        assert rec.uri == first.output_paths[split]
    await store.close()


async def test_the_same_policy_twice_is_idempotent(registry):
    """Same rows + same policy = same bytes = the same version. A split that
    bumped a version on every re-run would make "which eval set?" unanswerable."""
    src = _source(registry)
    first = await SplitOp().run(_inp(registry, src, out="a"), _ctx())
    second = await SplitOp().run(_inp(registry, src, out="b"), _ctx())

    assert second.child_hashes == first.child_hashes
    assert second.child_versions == first.child_versions  # still @1

    store = build_artifact_store()
    assert [v["version"] for v in await store.list_dataset_versions("facts-train")] == [1]
    await store.close()


async def test_without_register_as_nothing_touches_the_registry(registry):
    """The back-compatible shape: files only, exactly as before #2020."""
    src = _source(registry)
    out = await SplitOp().run(
        SplitInput(path=src, output_dir=str(registry / "plain"), backend="hash"), _ctx())

    assert out.child_versions == {} and out.child_hashes == {}
    assert out.split_policy["seed"] == 13  # the policy is still reported
    store = build_artifact_store()
    assert await store.list_datasets() == []
    await store.close()


async def test_a_mismatched_parent_ref_fails_loudly(registry):
    """A requested registration that cannot be right must fail the step, not
    register children under a lineage nobody asked for."""
    src = _source(registry)
    with pytest.raises(ValueError, match="but register_as is"):
        await SplitOp().run(
            _inp(registry, src, parent_ref="dataset:other@1"), _ctx())
