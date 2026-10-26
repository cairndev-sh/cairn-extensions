"""eval/lm-eval — classic benchmarks against a SERVED candidate (ADR-0102 D5).

Every test here runs WITHOUT ``lm-eval`` installed: the harness is an optional
``[harness]`` extra and the operator reaches it through one seam
(``operator_eval_lm.harness.run_harness``) that these tests replace. If the pack
ever imports ``lm_eval`` at module scope, ``test_the_pack_imports_without_the_extra``
goes red.

What is proven:

* task presets are **pack data**, not code — a TOML file under the pack, with a
  small default that finishes in minutes;
* decontamination is ON in the default preset and the ``_decontaminate`` suffix
  reaches the scorecard (the ticket's acceptance);
* decontamination requested but NOT produced is REPORTED, never hidden;
* per-case cost/latency budgets fail loudly;
* the flat metric map + directions land in the shape ``model/gate`` /
  ``eval/delta`` already consume.
"""
from __future__ import annotations

import asyncio
import sys

import pytest

from cairn.core.eval.scorecard import DECONTAM_SUFFIX, EvalBudgetExceeded, is_decontaminated
from cairn.core.operator.testing import run_operator_for_test

from operator_eval_lm import harness, presets
from operator_eval_lm.operators.lm_eval import LmEvalOp
from operator_eval_lm.schemas.lm_eval import LmEvalInput


def _served(**kw) -> LmEvalInput:
    base = dict(enabled=True, endpoint_url="http://candidate.example:8000/v1", model="cairn-candidate")
    base.update(kw)
    return LmEvalInput(**base)


def _fake_results(*, decontaminated: bool = True) -> dict:
    """A ``simple_evaluate``-shaped payload, including the decontaminated twin
    lm-eval emits when 13-gram decontamination is on."""
    gsm8k = {"exact_match,strict-match": 0.62, "exact_match_stderr,strict-match": 0.01}
    arc = {"acc,none": 0.71, "acc_alias": "acc"}
    if decontaminated:
        gsm8k[f"exact_match{DECONTAM_SUFFIX}"] = 0.55
        arc[f"acc{DECONTAM_SUFFIX}"] = 0.66
    return {
        "results": {"gsm8k": gsm8k, "arc_easy": arc},
        "higher_is_better": {"gsm8k": {"exact_match": True}, "arc_easy": {"acc": True}},
    }


def _run(inp: LmEvalInput):
    return asyncio.run(run_operator_for_test(LmEvalOp, inp))


# ── the optional extra ──────────────────────────────────────────────────


def test_the_pack_imports_without_the_extra():
    """lm-eval is an optional [harness] extra; nothing may import it eagerly."""
    assert "lm_eval" not in sys.modules, (
        "operator_eval_lm imported lm_eval at module scope — it is an optional extra"
    )


# ── presets are pack DATA ───────────────────────────────────────────────


def test_presets_are_shipped_as_data_files_not_python():
    names = presets.available()
    assert presets.DEFAULT_PRESET in names
    path = presets.preset_path(presets.DEFAULT_PRESET)
    assert path.suffix == ".toml" and path.is_file()


def test_the_default_preset_is_small_and_decontaminated():
    p = presets.load(presets.DEFAULT_PRESET)
    assert p.tasks, "the default preset must name at least one task"
    assert p.decontaminate is True, "ADR-0102 D5: decontamination on"
    assert 0 < p.limit <= 100, "the default preset must finish in minutes"


def test_an_unknown_preset_fails_loudly_listing_what_exists():
    with pytest.raises(ValueError) as exc:
        presets.load("does-not-exist")
    assert presets.DEFAULT_PRESET in str(exc.value)


def test_explicit_tasks_override_the_preset(monkeypatch):
    seen: dict = {}

    def _fake(**kw):
        seen.update(kw)
        return _fake_results()

    monkeypatch.setattr(harness, "run_harness", _fake)
    _run(_served(tasks=["hellaswag"]))
    assert seen["tasks"] == ["hellaswag"]


# ── the scorecard ───────────────────────────────────────────────────────


def test_the_default_preset_produces_a_scorecard_with_the_decontamination_suffix(monkeypatch):
    """The ticket's acceptance, against the fake harness."""
    monkeypatch.setattr(harness, "run_harness", lambda **kw: _fake_results())
    out = _run(_served())
    assert out.checked and out.decontaminated
    decon = [r for r in out.scorecard if is_decontaminated(r.metric)]
    assert decon, f"no decontaminated row in {[r.key for r in out.scorecard]}"
    assert all(r.decontaminated for r in decon)
    assert f"gsm8k/exact_match{DECONTAM_SUFFIX}" in out.metrics


def test_scorecard_rows_carry_slice_and_grader(monkeypatch):
    monkeypatch.setattr(harness, "run_harness", lambda **kw: _fake_results())
    out = _run(_served())
    row = next(r for r in out.scorecard if r.key == "gsm8k/exact_match")
    assert row.slice == "gsm8k" and row.grader == "lm-eval"
    assert row.direction == "higher_better"


def test_stderr_and_alias_keys_never_become_metrics(monkeypatch):
    monkeypatch.setattr(harness, "run_harness", lambda **kw: _fake_results())
    out = _run(_served())
    assert not [k for k in out.metrics if "stderr" in k or k.endswith("_alias")]


def test_decontamination_requested_but_absent_is_reported_not_hidden(monkeypatch):
    monkeypatch.setattr(harness, "run_harness", lambda **kw: _fake_results(decontaminated=False))
    out = _run(_served())
    assert out.checked is True
    assert out.decontaminated is False
    assert "decontamination" in out.note.lower()


# ── the served target ───────────────────────────────────────────────────


def test_the_endpoint_is_driven_as_an_openai_compatible_server(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(harness, "run_harness", lambda **kw: seen.update(kw) or _fake_results())
    _run(_served())
    assert seen["model"] == "local-completions"
    assert "base_url=http://candidate.example:8000/v1" in seen["model_args"]
    assert "model=cairn-candidate" in seen["model_args"]


def test_no_endpoint_is_a_refusal_not_a_zero(monkeypatch):
    monkeypatch.setattr(harness, "run_harness", lambda **kw: _fake_results())
    out = _run(LmEvalInput(enabled=True, endpoint_url=""))
    assert out.checked is False and not out.metrics
    assert "endpoint" in out.note.lower()


def test_a_missing_extra_no_ops_with_a_note_and_no_scores(monkeypatch):
    def _boom(**kw):
        raise ImportError("No module named 'lm_eval'")

    monkeypatch.setattr(harness, "run_harness", _boom)
    out = _run(_served())
    assert out.checked is False and out.metrics == {}
    assert "harness" in out.note.lower()


# ── the budget ──────────────────────────────────────────────────────────


def test_a_per_case_cost_ceiling_fails_loudly(monkeypatch):
    def _with_samples(**kw):
        r = _fake_results()
        r["samples"] = {"gsm8k": [
            {"doc_id": 0, "exact_match": 1.0, "cost_usd": 0.001, "latency_ms": 100.0},
            {"doc_id": 1, "exact_match": 0.0, "cost_usd": 0.900, "latency_ms": 100.0},
        ]}
        return r

    monkeypatch.setattr(harness, "run_harness", _with_samples)
    with pytest.raises(EvalBudgetExceeded) as exc:
        _run(_served(max_cost_usd_per_case=0.01))
    assert "gsm8k:1" in str(exc.value)


def test_a_per_case_ceiling_is_run_owned(monkeypatch):
    """ADR-0068/0070, caught while building this operator: the per-case
    ceilings were first declared in ``LmEvalConfig`` at governance scope, and the
    runtime DROPPED every value the tests set (``ignored run-supplied governance
    knob(s)``) — the operator advertised a bound it never enforced. A per-run
    bound belongs in ``input_schema``; only the whole-evaluation ceiling is
    env-owned."""
    from operator_eval_lm.schemas.lm_eval import LmEvalConfig

    assert "max_cost_usd_per_case" not in LmEvalConfig.model_fields
    assert "max_latency_ms_per_case" not in LmEvalConfig.model_fields
    scope = LmEvalConfig.model_fields["max_total_cost_usd"].json_schema_extra
    assert scope == {"scope": "governance"}


def test_no_budget_means_no_ceiling(monkeypatch):
    monkeypatch.setattr(harness, "run_harness", lambda **kw: _fake_results())
    assert _run(_served()).checked is True


# ── mock / disabled paths stay hermetic ─────────────────────────────────


def test_mock_produces_no_scores_a_gate_could_pass_on():
    out = LmEvalOp().mock(_served(), None)
    assert out.checked is False and out.metrics == {}


def test_disabled_is_the_default():
    assert LmEvalInput().enabled is False
