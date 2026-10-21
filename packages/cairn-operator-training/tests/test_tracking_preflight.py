"""#1709 — a tracking sink must never cost the training run.

Live on the GPU box (2026-08-26): the HF trainer's MLflowCallback resolves the
experiment at ``on_train_begin`` — an unguarded network call. A dead TLS vhost
turned it into 7 urllib3 retries and an ``MlflowException`` that killed the run
*before step 1*, with the model already loaded and the box already billing. The
epilogue's tracking was all guarded ("tracking must never fail the train"); the
callback's setup call was the one unguarded spot.

The fix is a cheap preflight in ``_tracking_setup``: one tracker call before any
heavy work. If it fails, mlflow is DROPPED from ``report_to`` (so the callback
is never constructed), the reason rides the finished run's metrics as
``tracking_error`` — the same soft shape as ``publish_error`` — and the spec +
env are pruned so the epilogue's readers short-circuit instead of re-running
the retry storm per call.
"""
from __future__ import annotations

import sys
import types

import pytest

from operator_training.train_entry import _preflight_mlflow, _tracking_setup


class _FakeClient:
    def __init__(self, exc: Exception | None):
        self._exc = exc

    def search_experiments(self, max_results: int = 1):
        if self._exc is not None:
            raise self._exc
        return []


def _fake_mlflow(monkeypatch, exc: Exception | None) -> types.ModuleType:
    mod = types.ModuleType("mlflow")
    mod.MlflowClient = lambda: _FakeClient(exc)
    monkeypatch.setitem(sys.modules, "mlflow", mod)
    return mod


# ── the probe itself ────────────────────────────────────────────────────────

def test_reachable_tracker_probes_clean(monkeypatch):
    _fake_mlflow(monkeypatch, None)
    assert _preflight_mlflow() == ""


def test_unreachable_tracker_returns_the_reason_not_an_exception(monkeypatch):
    """ANY failure — the live one was an MlflowException wrapping an SSLError —
    becomes a string, never a raise. Raising here is the bug."""
    _fake_mlflow(monkeypatch, ConnectionError("tlsv1 alert internal error"))
    err = _preflight_mlflow()
    assert "ConnectionError" in err and "tlsv1" in err


def test_probe_pins_retries_for_itself_and_restores_them(monkeypatch):
    """The live failure burned minutes in urllib3 retries. The probe must fail
    in seconds — retries/timeout pinned tight — WITHOUT changing the policy the
    run's own tracking calls see afterwards."""
    seen: dict[str, str | None] = {}

    class _Spy:
        def search_experiments(self, max_results=1):
            import os
            seen["retries"] = os.environ.get("MLFLOW_HTTP_REQUEST_MAX_RETRIES")
            seen["timeout"] = os.environ.get("MLFLOW_HTTP_REQUEST_TIMEOUT")
            return []

    mod = types.ModuleType("mlflow")
    mod.MlflowClient = lambda: _Spy()
    monkeypatch.setitem(sys.modules, "mlflow", mod)
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "7")   # the deployment's policy
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)

    assert _preflight_mlflow() == ""
    assert seen == {"retries": "1", "timeout": "10"}             # pinned during the probe
    import os
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "7"  # restored after
    assert "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ       # unset restored as unset


def test_mlflow_not_installed_is_a_reason_not_a_crash(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)  # import → ImportError
    assert "not installed" in _preflight_mlflow()


# ── the setup that consumes it ──────────────────────────────────────────────

def test_dead_tracker_is_dropped_and_the_reason_kept(monkeypatch):
    """THE regression. Before: report_to=['mlflow'] reached the HF trainer and
    its callback raised at on_train_begin. After: mlflow never reaches the
    trainer, and the reason survives to become metrics['tracking_error']."""
    _fake_mlflow(monkeypatch, ConnectionError("boom"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dead.example")
    spec = {"report_to": ["mlflow"], "tracking_project": "cl"}

    report_to, err = _tracking_setup(spec)

    assert report_to == []                       # the callback is never built
    assert "ConnectionError" in err
    # the epilogue reads the SPEC and the ENV, not the return value — both are
    # pruned so _mlflow_training_run/_mlflow_log_and_register short-circuit
    # instead of re-running the retry storm per call.
    assert spec["report_to"] == []
    import os
    assert "MLFLOW_TRACKING_URI" not in os.environ


def test_reachable_tracker_is_kept(monkeypatch):
    _fake_mlflow(monkeypatch, None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://live.example")
    spec = {"report_to": ["mlflow"], "tracking_project": "cl"}
    report_to, err = _tracking_setup(spec)
    assert report_to == ["mlflow"] and err == ""
    assert spec["report_to"] == ["mlflow"]


def test_no_mlflow_requested_means_no_probe(monkeypatch):
    """A run that never asked for mlflow must not pay a network call — nor
    import mlflow at all."""
    def _boom():
        raise AssertionError("probe ran for a run that never asked for mlflow")

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    import operator_training.train_entry as te
    monkeypatch.setattr(te, "_preflight_mlflow", _boom)
    report_to, err = _tracking_setup({"report_to": ["wandb"], "tracking_project": "x"})
    assert report_to == ["wandb"] and err == ""


def test_wandb_survives_a_dead_mlflow(monkeypatch):
    """Only the unreachable sink is dropped — not tracking as a whole."""
    _fake_mlflow(monkeypatch, ConnectionError("boom"))
    spec = {"report_to": ["mlflow", "wandb"], "tracking_project": "cl"}
    report_to, err = _tracking_setup(spec)
    assert report_to == ["wandb"] and err
    assert spec["report_to"] == ["wandb"]


def test_env_default_to_mlflow_still_preflights(monkeypatch):
    """The pre-#136 back-compat path — report_to unset, MLFLOW_TRACKING_URI
    present — defaults INTO mlflow, so it must go through the same probe."""
    _fake_mlflow(monkeypatch, ConnectionError("boom"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dead.example")
    report_to, err = _tracking_setup({"tracking_project": "cl"})
    assert report_to == [] and "ConnectionError" in err
