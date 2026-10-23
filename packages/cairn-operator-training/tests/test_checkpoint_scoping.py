"""#1749 — checkpoint keys must be RUN-SCOPED, and a pull must fail closed.

The bug this pins, reproduced end-to-end below: a BRAND-NEW output dir was
seeded with a different run's step-400 checkpoint because the run key fell back
to a shared literal. The run then "trained" 400 steps in 13 seconds with
train_loss 0.0003, exited 0, wrote a valid adapter and plausible metrics, and
nothing warned. A continual-learning campaign is a chain of runs that all look
alike to a shared key, so this silently fabricates experiments.
"""
from __future__ import annotations

import os

import pytest

from operator_training import checkpoint_sync as CS


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("CAIRN_CHECKPOINT_BUCKET", "CAIRN_CHECKPOINT_PREFIX",
              "CAIRN_RUN_ID", "CAIRN_PARENT_RUN", "CAIRN_OUTPUT_NAME"):
        monkeypatch.delenv(k, raising=False)


def _ckpt(root, step):
    d = os.path.join(root, f"checkpoint-{step}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "trainer_state.json"), "w").write("{}")
    return d


# ── the key ─────────────────────────────────────────────────────────────────

def test_no_bucket_is_a_noop(monkeypatch):
    assert CS.s3_target_from_env() is None


def test_run_id_wins_and_is_run_scoped(monkeypatch):
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "b")
    monkeypatch.setenv("CAIRN_RUN_ID", "run-abc")
    _, _, key = CS.s3_target_from_env()
    assert key == "run-abc" and CS.is_run_scoped(key)


def test_output_name_is_an_identity_of_last_resort(monkeypatch):
    """THE #1749 gap: the env vars are set by the control plane and absent when
    an entry is driven directly, which is how a fresh run got the shared key."""
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "b")
    _, _, key = CS.s3_target_from_env(output_name="cl-cycle3")
    assert key == "cl-cycle3" and CS.is_run_scoped(key)


def test_the_bare_fallback_is_NOT_run_scoped(monkeypatch):
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "b")
    _, _, key = CS.s3_target_from_env()
    assert key == CS.GENERIC_RUN_KEY
    assert CS.is_run_scoped(key) is False


# ── the pull fails CLOSED ───────────────────────────────────────────────────

def test_a_generic_key_REFUSES_to_pull(tmp_path, monkeypatch, caplog):
    """The fix. Refusing costs a re-run; pulling costs the result."""
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "b")
    called = {}
    monkeypatch.setattr(CS, "download_prefix",
                        lambda *a, **k: called.setdefault("pulled", True))
    out = tmp_path / "adapter"          # basename == the generic key
    out.mkdir()
    with caplog.at_level("WARNING"):
        assert CS.pull_checkpoints_if_any(str(out)) is None
    assert "pulled" not in called, "must not pull under a shared key"
    assert "#1749" in caplog.text


def test_a_run_scoped_key_DOES_pull_and_says_so(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "b")
    monkeypatch.setenv("CAIRN_RUN_ID", "run-xyz")
    out = tmp_path / "cl-cycle9"
    out.mkdir()

    def fake_download(bucket, prefix, dest):
        assert prefix.endswith("/run-xyz"), prefix
        _ckpt(dest, 400)
    monkeypatch.setattr(CS, "download_prefix", fake_download)
    with caplog.at_level("WARNING"):
        got = CS.pull_checkpoints_if_any(str(out))
    assert got and got.endswith("checkpoint-400")
    assert "RESUMING" in caplog.text, "a resumed run must announce itself"


def test_existing_local_checkpoints_are_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "b")
    monkeypatch.setenv("CAIRN_RUN_ID", "run-xyz")
    out = tmp_path / "cl-cycle9"
    out.mkdir()
    _ckpt(str(out), 50)
    monkeypatch.setattr(CS, "download_prefix",
                        lambda *a, **k: pytest.fail("must not pull over local checkpoints"))
    assert CS.pull_checkpoints_if_any(str(out)).endswith("checkpoint-50")


# ── the regression, reproduced ──────────────────────────────────────────────

def test_two_different_runs_do_not_share_a_key(monkeypatch):
    """The contamination itself. Before the fix both fresh runs resolved to
    'adapter', so the second could resume from the first's step-400 state."""
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "b")
    _, _, a = CS.s3_target_from_env(output_name="cl-cycle2")
    _, _, b = CS.s3_target_from_env(output_name="cl-cycle3-clean")
    assert a != b, "distinct runs must not share a checkpoint namespace"
    assert CS.is_run_scoped(a) and CS.is_run_scoped(b)


def test_a_fresh_run_cannot_inherit_a_strangers_checkpoint(tmp_path, monkeypatch):
    """End-to-end: run A pushes under its key; run B, brand new, must not get
    A's checkpoints — neither by sharing the key nor by pulling the fallback."""
    monkeypatch.setenv("CAIRN_CHECKPOINT_BUCKET", "b")
    store = {}                                    # fake s3: prefix -> step

    def fake_download(bucket, prefix, dest):
        if prefix in store:
            _ckpt(dest, store[prefix])
    monkeypatch.setattr(CS, "download_prefix", fake_download)

    _, pfx, key_a = CS.s3_target_from_env(output_name="run-A")
    store[f"{pfx}/{key_a}"] = 400                 # run A got to step 400

    out_b = tmp_path / "run-B"                    # a DIFFERENT, fresh run
    out_b.mkdir()
    assert CS.pull_checkpoints_if_any(str(out_b)) is None, \
        "run B must start fresh, not resume run A"
    assert not list(out_b.glob("checkpoint-*"))
