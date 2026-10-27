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
* the flat metric map + directions land in the shape a gate reads (a pipeline
  predicate, or the platform's ``model/gate`` / ``eval/delta``, which are not public);
* the endpoint key reaches the harness call only, never the process environment;
* the harness runs off the event loop.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

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


# ── the endpoint key never touches the process environment (ADR-0143 D3) ──


def _run_with(inp: LmEvalInput, **kw):
    return asyncio.run(run_operator_for_test(LmEvalOp, inp.model_dump(), **kw))


def _recording(seen: list):
    """A fake harness that records the key it was handed and what the process
    environment held while it ran."""
    def _fake(**kw):
        seen.append({"api_key": kw.get("api_key"), "env": os.environ.get("OPENAI_API_KEY"),
                     "model_args": kw.get("model_args", "")})
        return _fake_results()
    return _fake


def test_the_key_reaches_the_harness_call_and_never_the_process_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    before = dict(os.environ)
    seen: list = []
    monkeypatch.setattr(harness, "run_harness", _recording(seen))
    out = _run_with(_served(api_key_env="EP_KEY"), secrets={"EP_KEY": "sk-tenant-a"})
    assert out.checked is True
    assert seen[0]["api_key"] == "sk-tenant-a"
    assert seen[0]["env"] is None, "the key was written to the shared process environment"
    assert "sk-tenant-a" not in seen[0]["model_args"]
    assert dict(os.environ) == before


def test_two_runs_with_different_keys_do_not_leak_into_each_other(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen: list = []
    monkeypatch.setattr(harness, "run_harness", _recording(seen))
    _run_with(_served(api_key_env="EP_KEY"), secrets={"EP_KEY": "sk-tenant-a"})
    _run_with(_served(api_key_env="EP_KEY"), secrets={"EP_KEY": "sk-tenant-b"})
    assert [s["api_key"] for s in seen] == ["sk-tenant-a", "sk-tenant-b"]
    assert [s["env"] for s in seen] == [None, None]
    assert "OPENAI_API_KEY" not in os.environ


def test_a_deployment_key_in_the_environment_is_not_the_tenants_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-deployment")
    seen: list = []
    monkeypatch.setattr(harness, "run_harness", _recording(seen))
    _run_with(_served(api_key_env="EP_KEY"), secrets={"EP_KEY": "sk-tenant-a"})
    assert seen[0]["api_key"] == "sk-tenant-a"
    assert os.environ["OPENAI_API_KEY"] == "sk-deployment"


def test_no_api_key_env_hands_the_harness_no_key(monkeypatch):
    seen: list = []
    monkeypatch.setattr(harness, "run_harness", _recording(seen))
    _run(_served())
    assert seen[0]["api_key"] is None


@pytest.mark.parametrize("secrets", [None, {"SOMETHING_ELSE": "x"}])
def test_a_named_key_the_run_cannot_read_is_a_refusal_not_an_unauthenticated_call(monkeypatch, secrets):
    called: list = []
    monkeypatch.setattr(harness, "run_harness", lambda **kw: called.append(kw) or _fake_results())
    out = _run_with(_served(api_key_env="EP_KEY"), secrets=secrets)
    assert out.checked is False and out.metrics == {}
    assert "EP_KEY" in out.note and "secrets" in out.note
    assert called == [], "the endpoint was called without the key the run named"


def test_the_harness_seam_builds_explicit_headers_so_lm_eval_never_reads_the_env():
    """lm-eval's local-completions sends ``Bearer $OPENAI_API_KEY`` unless the LM is
    built with an explicit, non-empty ``header`` (lm_eval 0.4.13
    models/api_models.py:322-324, models/openai_completions.py:139-140)."""
    with_key = harness.auth_kwargs("sk-a")
    assert with_key["header"]["Authorization"] == "Bearer sk-a"
    assert with_key["auth_token"] == "sk-a"
    without = harness.auth_kwargs(None)
    assert without["header"] and "Authorization" not in without["header"]
    assert "auth_token" not in without


def test_the_seam_drops_arguments_the_installed_simple_evaluate_does_not_take():
    """lm-eval 0.4.13's simple_evaluate has no decontamination argument; passing it
    raised TypeError and every default (decontaminate=true) run became a stub."""
    def simple_evaluate(model, tasks=None, limit=None, num_fewshot=None):
        return {}

    kept, dropped = harness.accepted_kwargs(
        simple_evaluate, {"limit": 4, "num_fewshot": 0, "decontamination_ngrams_path": None})
    assert kept == {"limit": 4, "num_fewshot": 0}
    assert dropped == ["decontamination_ngrams_path"]


# ── the harness runs off the event loop (ADR-0125) ──────────────────────


def test_the_event_loop_stays_responsive_while_the_harness_runs(monkeypatch):
    """lm-eval is synchronous and long, and its API backends call ``asyncio.run``
    (lm_eval 0.4.13 models/api_models.py:689, :831), which fails inside a running
    loop. The operator must hand it to a worker thread."""
    def _slow(**kw):
        time.sleep(0.5)
        return _fake_results()

    monkeypatch.setattr(harness, "run_harness", _slow)

    async def _both():
        ticks = 0

        async def _ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        t = asyncio.create_task(_ticker())
        out = await run_operator_for_test(LmEvalOp, _served())
        t.cancel()
        return out, ticks

    out, ticks = asyncio.run(_both())
    assert out.checked is True
    assert ticks >= 10, f"the loop was blocked while the harness ran ({ticks} ticks in 0.5 s)"


def test_the_harness_does_not_run_on_the_event_loop_thread(monkeypatch):
    seen: dict = {}

    def _where(**kw):
        try:
            asyncio.get_running_loop()
            seen["loop"] = True
        except RuntimeError:
            seen["loop"] = False
        return _fake_results()

    monkeypatch.setattr(harness, "run_harness", _where)
    _run(_served())
    assert seen["loop"] is False


# ── cost is counted once per case ───────────────────────────────────────


def test_a_case_with_several_metrics_is_costed_once(monkeypatch):
    def _two_metrics(**kw):
        r = _fake_results()
        r["samples"] = {"gsm8k": [
            {"doc_id": 0, "exact_match": 1.0, "acc": 1.0, "cost_usd": 0.5, "latency_ms": 10.0},
        ]}
        return r

    monkeypatch.setattr(harness, "run_harness", _two_metrics)
    out = _run(_served(max_total_cost_usd=0.6))
    assert out.cost_usd == pytest.approx(0.5)
    assert out.cases == 2  # per-case rows, one per metric


# ── the cheap no-op paths ───────────────────────────────────────────────


def test_disabled_and_dry_run_never_call_the_harness(monkeypatch):
    monkeypatch.setattr(harness, "run_harness", lambda **kw: pytest.fail("harness called"))
    assert _run(LmEvalInput(endpoint_url="http://candidate.example:8000/v1")).checked is False
    out = _run(_served(dry_run=True))
    assert out.checked is False and "dry-run" in out.note


def test_run_harness_builds_the_model_with_the_key_as_a_header(monkeypatch):
    """The seam against a stand-in ``lm_eval`` shaped like 0.4.13's API: the model is
    built through the registry with the explicit header, ``simple_evaluate`` gets the
    model object (never the key in ``model_args``), and an argument it does not take
    is dropped and reported. The stand-in is removed again, so nothing else sees an
    ``lm_eval``."""
    rec = _stand_in_lm_eval(monkeypatch)
    out = harness.run_harness(
        model="local-completions", model_args="model=m,base_url=http://x/v1", tasks=["gsm8k"],
        api_key="sk-a", num_fewshot=0, batch_size=8, limit=4, decontamination_ngrams_path=None)
    assert rec["built"][0]["header"]["Authorization"] == "Bearer sk-a"
    assert "sk-a" not in rec["built"][0]["arg_string"]
    called = rec["called"][0]
    assert called["model"].header["Authorization"] == "Bearer sk-a"
    assert "model_args" not in called
    assert called["tasks"] == ["gsm8k"] and called["limit"] == 4
    assert out[harness.DROPPED_KEY] == ["decontamination_ngrams_path"]
    assert out[harness.VERSION_KEY] == "0.4.13"
    assert os.environ.get("OPENAI_API_KEY") != "sk-a"


# ── what the installed lm-eval does not take, and a model that ignores the header ──


_ALL_ARGS = ("num_fewshot", "batch_size", "limit")


def _stand_in_lm_eval(monkeypatch, *, accepts=_ALL_ARGS, honours_header=True, version="0.4.13",
                      decontaminated=True):
    """Install a stand-in ``lm_eval`` shaped like 0.4.13's API whose ``simple_evaluate``
    takes only ``accepts`` (plus model and tasks), and whose model either keeps the
    header it is built with or, like a release that ignores it, falls back to the env
    key. Returns a record of what was built and called; ``monkeypatch`` removes it."""
    import types

    rec: dict = {"built": [], "called": []}

    class _Model:
        def __init__(self, **kw):
            self._header = kw.get("header") if honours_header else None

        @property
        def header(self):
            return self._header or {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}

        @classmethod
        def create_from_arg_string(cls, arg_string, additional_config=None):
            rec["built"].append({"arg_string": arg_string, **(additional_config or {})})
            return cls(**(additional_config or {}))

    params = ", ".join(f"{a}=None" for a in accepts)
    ns: dict = {}
    exec(f"def simple_evaluate(model=None, tasks=None, {params}):\n    return CALL(locals())", ns)
    ns["CALL"] = lambda args: rec["called"].append(args) or _fake_results(decontaminated=decontaminated)

    lm_eval = types.ModuleType("lm_eval")
    lm_eval.__version__ = version
    lm_eval.simple_evaluate = ns["simple_evaluate"]
    api = types.ModuleType("lm_eval.api")
    registry = types.ModuleType("lm_eval.api.registry")
    registry.get_model = lambda name: _Model
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.api", api)
    monkeypatch.setitem(sys.modules, "lm_eval.api.registry", registry)
    return rec


@pytest.mark.parametrize("missing", ["limit", "num_fewshot"])
def test_a_bound_the_installed_lm_eval_cannot_take_is_a_refusal(monkeypatch, missing):
    """Dropping ``limit`` would run every document of every task against the endpoint
    (smoke asks for 40); dropping ``num_fewshot`` changes the scores. Neither may be
    dropped silently: the run refuses before the model is built (which probes the
    endpoint) and sends nothing."""
    rec = _stand_in_lm_eval(monkeypatch, accepts=[a for a in _ALL_ARGS if a != missing],
                            version="9.9.9")
    out = _run(_served(decontaminate=False))
    assert out.checked is False and out.metrics == {}
    assert missing in out.note and "9.9.9" in out.note
    assert rec["built"] == [] and rec["called"] == []


def test_another_dropped_argument_is_listed_in_the_note(monkeypatch):
    rec = _stand_in_lm_eval(monkeypatch, accepts=("num_fewshot", "limit"))
    out = _run(_served(decontaminate=False))
    assert out.checked is True
    assert "batch_size" in out.note and "0.4.13" in out.note
    assert len(rec["called"]) == 1


def test_a_seam_dropped_decontamination_request_says_so(monkeypatch):
    """lm-eval 0.4.10–0.4.13 take no decontamination argument: the note must say the
    request was not passed, not blame the tasks."""
    _stand_in_lm_eval(monkeypatch, decontaminated=False)
    out = _run(_served())  # decontaminate defaults to true
    assert out.checked is True and out.decontamination_requested is True
    assert "no decontamination argument" in out.note
    assert "should_decontaminate" not in out.note


def test_a_model_that_ignores_the_header_is_refused_before_any_request(monkeypatch):
    """A release that drops ``header`` would send ``Bearer $OPENAI_API_KEY``. The seam
    compares the built model's effective header with the one it passed and refuses."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-deployment")
    rec = _stand_in_lm_eval(monkeypatch, honours_header=False)
    out = _run_with(_served(api_key_env="EP_KEY"), secrets={"EP_KEY": "sk-tenant-a"})
    assert out.checked is False and out.metrics == {}
    assert "header" in out.note
    assert "sk-tenant-a" not in out.note and "sk-deployment" not in out.note
    assert rec["called"] == [], "simple_evaluate ran with a model that ignored the header"


def test_a_model_that_ignores_the_header_is_refused_without_a_key_too(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-deployment")
    rec = _stand_in_lm_eval(monkeypatch, honours_header=False)
    out = _run(_served())
    assert out.checked is False and rec["called"] == []


def test_the_key_is_redacted_from_a_harness_failure_note(monkeypatch):
    def _boom(**kw):
        raise RuntimeError(f"401 Unauthorized (sent Bearer {kw['api_key']})")

    monkeypatch.setattr(harness, "run_harness", _boom)
    out = _run_with(_served(api_key_env="EP_KEY"), secrets={"EP_KEY": "sk-tenant-a"})
    assert out.checked is False and "harness failed" in out.note
    assert "sk-tenant-a" not in out.note and "sk-tenant-a" not in str(out.audit_metadata)


def test_the_harness_extra_is_bounded_to_the_releases_read():
    import tomllib
    from pathlib import Path

    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert project["project"]["optional-dependencies"]["harness"] == ["lm-eval>=0.4.10,<0.4.14"]
