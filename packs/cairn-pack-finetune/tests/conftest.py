"""Shared by the box tests (`test_sft_box.py`, `test_eval_adapter.py`): the `finetune-ssh`
profile, a `gpu` connection in a throwaway store, registered versions with their roles and admitted
verdicts, and a fake EC2 behind the core's `CAIRN_EC2_CLIENT_FACTORY` seam. No AWS call and no SSH
connection is made."""
from __future__ import annotations

import hashlib
import json
import socket
import sys
import types
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
PROFILE = PACK / "profiles" / "finetune-ssh.toml"


class FakeEC2:
    """The EC2 client the lifecycle functions call; writes to the shared timeline."""

    def __init__(self, timeline: list[str]) -> None:
        self.state, self.timeline, self.tags = "stopped", timeline, {}

    def describe_instances(self, **kw):
        return {"Reservations": [{"Instances": [{"InstanceId": "i-0sft", "State": {"Name": self.state}}]}]}

    def start_instances(self, **kw):
        self.timeline.append("ec2:start")
        self.state = "running"
        return {"StartingInstances": [{"CurrentState": {"Name": "pending"}}]}

    def stop_instances(self, **kw):
        self.timeline.append("ec2:stop")
        self.state = "stopped"
        return {"StoppingInstances": [{"CurrentState": {"Name": "stopping"}}]}

    def create_tags(self, **kw):
        self.timeline.append("ec2:tags")
        self.tags.update({t["Key"]: t["Value"] for t in kw.get("Tags") or []})


@pytest.fixture
async def env(tmp_path, monkeypatch):
    """The `finetune-ssh` profile; the `gpu` connection (an EC2 instance at $0.66/h, its SSH
    port a local socket) in a throwaway store; registered versions with admitted verdicts; the
    fake EC2 behind the core's seam."""
    import cairn.core.storage.artifacts as artifacts
    from cairn.core.connections import Connection, SqliteConnectionStore
    from cairn.core.dataflow.contract import ArtifactDescriptor
    from cairn.core.pack import registry
    from cairn.core.pack.loader import load_all_packs
    from cairn.core.storage.artifacts import SqliteArtifactStore

    for name in ("CAIRN_ML_ENGINE", "CAIRN_ML_TARGET", "CAIRN_DATAFLOW_ENGINE", "CAIRN_DATAFLOW_TARGET",
                 "CAIRN_ARTIFACT_STORE", "CAIRN_REMOTE_EXEC_MODE", "CAIRN_RUNTIME", "CAIRN_STORAGE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CAIRN_PROFILE", str(PROFILE))
    monkeypatch.setenv("CAIRN_PACK_PATH", str(PACK))
    monkeypatch.setenv("CAIRN_CONNECTIONS_DB", str(tmp_path / "connections.db"))

    ssh = socket.socket()
    ssh.bind(("127.0.0.1", 0))
    ssh.listen(8)   # the kernel completes the handshake: acquire's readiness probe sees an open port
    store = SqliteConnectionStore(str(tmp_path / "connections.db"))
    await store.create(Connection(name="gpu", kind="compute.ssh", config={
        "host": "127.0.0.1", "port": str(ssh.getsockname()[1]), "user": "u",
        "instance_id": "i-0sft", "region": "eu-central-1", "hourly_usd": "0.66"}))
    close = getattr(store, "close", None)
    if callable(close):
        close()

    timeline: list[str] = []
    ec2 = FakeEC2(timeline)
    seam = types.ModuleType("_sft_box_ec2_seam")
    seam.client = lambda region="": ec2
    monkeypatch.setitem(sys.modules, seam.__name__, seam)
    monkeypatch.setenv("CAIRN_EC2_CLIENT_FACTORY", f"{seam.__name__}:client")

    versions = SqliteArtifactStore(str(tmp_path / "registry.db"))
    for name, role, n in (("train", "training", 100), ("val", "validation", 6), ("test", "test", 6)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps({"instruction": "Answer.", "input": f"{name} q{i}",
                                            "output": f"{name} a{i}"}) + "\n" for i in range(n)))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        await versions.register(ArtifactDescriptor(uri=f"file://{path}", kind="dataset", content_hash=digest),
                                producer_run_id="prepare")
        await versions.register_dataset_version(f"demo-{name}", digest, actor="prepare", role=role)
    for name in ("demo-train", "demo-val"):
        await versions.set_dataset_version_verdict(name, 1, verdict="admitted", reasons=[], report={})
    monkeypatch.setattr(artifacts, "build_artifact_store", lambda *a, **k: versions)

    import cairn.core.dispatch.leg_executor as leg_executor
    monkeypatch.setattr(leg_executor, "_POLL_INTERVAL_S", 0.05)

    registry.clear()
    load_all_packs()
    try:
        yield types.SimpleNamespace(ec2=ec2, timeline=timeline, box=tmp_path / "box")
    finally:
        registry.clear()
        await versions.close()
        ssh.close()
