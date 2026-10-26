"""eval/lm-eval — classic public benchmarks against a SERVED candidate (ADR-0102 D5).

The candidate is already stood up behind an OpenAI-compatible endpoint (a vLLM
server, for example). This operator points lm-evaluation-harness at THAT endpoint —
``model="local-completions"`` — instead of loading a second copy of the weights.
Three consequences worth stating, because they are the reason this is not another
flag on ``eval/benchmark``:

* **no GPU of its own.** The work is HTTP against a server someone else sized;
  ``requires_gpu = false`` is the honest declaration, and it lets a benchmark run
  on the control plane while the GPU box serves.
* **the artefact under test is the one that ships.** A merged/quantised served
  model can score differently from base+adapter loaded in-process; measuring the
  endpoint measures what users will get.
* **the tasks are DATA.** ``presets/*.toml`` under this pack, not a Python list.

Decontamination (ADR-0102 D5) is requested by default. lm-eval's 13-gram overlap
check is opt-in *per task* upstream, so a preset can ask for it and get nothing
back; the operator reports that gap rather than letting a contaminated number
pass as a clean one.

The evaluation BUDGET raises. ``EvalBudgetExceeded`` propagates out of ``run`` on
purpose — a benchmark that blew its per-case ceiling has produced numbers whose
cost the run did not agree to, and the ADR is explicit that this is a budget and
not an estimate. Every other failure (no extra, harness error) degrades to a
deterministic stub with ``checked=false``, so a missing extra never breaks a
pipeline and never emits a score a gate could pass on.
"""
from __future__ import annotations

import os
import time
from typing import Any

from cairn.core.eval.scorecard import (
    DECONTAM_SUFFIX,
    EvalBudget,
    ScorecardRow,
    assert_within_budget,
    directions_of,
    flat_metrics,
    is_decontaminated,
)
from cairn.core.operator import Operation, OperationContext, register_operator
from cairn.core.operator import ReportSpec

from operator_eval_lm import harness, presets
from operator_eval_lm.schemas.lm_eval import LmEvalConfig, LmEvalInput, LmEvalOutput

#: Metric-name suffixes the harness adds that are not gate-usable scalars.
#: Checked against the metric NAME, never the raw key: lm-eval spells the error
#: bar ``exact_match_stderr,strict-match`` — filter suffix LAST — so a check on
#: the raw key misses it and a standard error lands in the map as a metric.
_DROP_SUFFIXES = ("_stderr", "_alias")

#: ADR-0103. Stages only — a benchmark has no per-step series to stream, and the
#: three scalars are the numbers a reviewer wants on the Activity card.
LM_EVAL_REPORTS = ReportSpec(
    stages=["resolve", "evaluate", "score"],
    scalars=["tasks_run", "cases", "cost_usd"],
)


def _metric_name(raw: str) -> str:
    """``exact_match,strict-match`` → ``exact_match`` (the filter suffix is not
    part of the metric's identity)."""
    return raw.split(",", 1)[0]


def _model_args(inp: LmEvalInput) -> str:
    """lm-eval ``model_args`` for its OpenAI-compatible ``local-completions`` type."""
    parts = [f"model={inp.model}", f"base_url={inp.endpoint_url}",
             f"num_concurrent={inp.num_concurrent}"]
    if inp.tokenizer:
        parts.append(f"tokenizer={inp.tokenizer}")
    return ",".join(parts)


def _rows_from_results(results: dict, grader: str) -> list[ScorecardRow]:
    """Aggregate rows: one per (task, metric), decontaminated twins included."""
    directions: dict[tuple[str, str], str] = {}
    for task, mm in (results.get("higher_is_better") or {}).items():
        if isinstance(mm, dict):
            for name, higher in mm.items():
                if higher is not None:
                    directions[(task, name)] = "higher_better" if higher else "lower_better"

    rows: list[ScorecardRow] = []
    for task, mm in (results.get("results") or {}).items():
        if not isinstance(mm, dict):
            continue
        for raw, value in mm.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            name = _metric_name(raw)
            if any(name.endswith(s) for s in _DROP_SUFFIXES):
                continue
            decon = is_decontaminated(name)
            # A decontaminated twin inherits the base metric's direction: it is
            # the same measurement over a filtered document set.
            base = name[: -len(DECONTAM_SUFFIX)] if decon else name
            rows.append(ScorecardRow(
                metric=name, value=float(value), slice=task, grader=grader,
                direction=directions.get((task, name)) or directions.get((task, base))
                or "higher_better",
                decontaminated=decon,
            ))
    return rows


def _rows_from_samples(results: dict, grader: str) -> list[ScorecardRow]:
    """Per-case rows, when the harness logged samples. These are what the budget
    binds to — a per-case ceiling with no per-case rows would bind to nothing."""
    rows: list[ScorecardRow] = []
    for task, samples in (results.get("samples") or {}).items():
        if not isinstance(samples, list):
            continue
        for i, s in enumerate(samples):
            if not isinstance(s, dict):
                continue
            case = str(s.get("doc_id", i))
            cost = float(s.get("cost_usd", 0.0) or 0.0)
            latency = float(s.get("latency_ms", 0.0) or 0.0)
            for raw, value in s.items():
                if raw in ("doc_id", "cost_usd", "latency_ms"):
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                rows.append(ScorecardRow(
                    metric=_metric_name(raw), value=float(value), slice=task,
                    case=f"{task}:{case}", grader=grader, cost_usd=cost, latency_ms=latency,
                    decontaminated=is_decontaminated(_metric_name(raw)),
                ))
    return rows


@register_operator(name="eval/lm-eval", version="0.1.0", capabilities=["eval.lm_eval"])
class LmEvalOp(Operation[LmEvalInput, LmEvalOutput]):
    """Run an lm-evaluation-harness task preset against a served candidate."""

    input_schema = LmEvalInput
    output_schema = LmEvalOutput
    config_schema = LmEvalConfig
    long_running = True
    requires_gpu = False  # the GPU belongs to the SERVED model, not to this leg
    supports_mock = True
    reports = LM_EVAL_REPORTS
    risk = "external_write"  # drives a served endpoint and spends its capacity

    def _stub(self, inp: LmEvalInput, note: str, *, preset: str = "") -> LmEvalOutput:
        """Deterministic no-op. Never carries metrics: a benchmark that did not
        run must not emit a score a gate could pass on (the eval/benchmark rule)."""
        return LmEvalOutput(
            checked=False, note=note, preset_used=preset,
            decontamination_requested=inp.decontaminate,
            audit_metadata={"checked": False, "note": note},
        )

    def _resolve(self, inp: LmEvalInput) -> tuple[list[str], str, dict[str, Any]]:
        """(tasks, preset_name, harness kwargs) — explicit tasks beat the preset."""
        if inp.tasks:
            tasks, name = list(inp.tasks), ""
            limit = inp.limit or 0
            fewshot = inp.num_fewshot if inp.num_fewshot >= 0 else 0
            batch = inp.batch_size or 8
        else:
            preset = presets.load(inp.preset or presets.DEFAULT_PRESET)
            tasks, name = list(preset.tasks), preset.name
            limit = inp.limit or preset.limit
            fewshot = inp.num_fewshot if inp.num_fewshot >= 0 else preset.num_fewshot
            batch = inp.batch_size or preset.batch_size
        kw: dict[str, Any] = {"num_fewshot": fewshot, "batch_size": batch}
        if limit:
            kw["limit"] = limit
        if inp.decontaminate:
            # The 13-gram overlap index. Passed even when empty so the harness
            # sees the REQUEST; DESIGN.md §6 records that the exact spelling must
            # be checked against the installed release before the first live run.
            kw["decontamination_ngrams_path"] = inp.decontamination_ngrams_path or None
        return tasks, name, kw

    async def run(self, inp: LmEvalInput, ctx: OperationContext) -> LmEvalOutput:
        if not inp.enabled:
            return self._stub(inp, "lm-eval disabled (set enabled=true to run)")
        if inp.dry_run:
            return self._stub(inp, "dry-run — harness not called")
        if not inp.endpoint_url:
            return self._stub(
                inp,
                "no endpoint_url — eval/lm-eval measures a SERVED candidate; wire the serving "
                "leg's endpoint. Refusing rather than reporting a zero.",
            )

        ctx.report.stage("resolve", percent=0)
        tasks, preset_name, kw = self._resolve(inp)
        if not tasks:
            return self._stub(inp, "no tasks resolved — nothing to evaluate", preset=preset_name)
        if inp.api_key_env:
            # lm-eval's OpenAI-compatible backend reads the key from the
            # environment; the operator never puts a secret in model_args, where
            # it would land in the recorded `parameters` event.
            key = ctx.secrets.get(inp.api_key_env) if getattr(ctx, "secrets", None) else None
            if key:
                os.environ.setdefault("OPENAI_API_KEY", key)

        ctx.report.stage("evaluate", percent=10)
        started = time.monotonic()
        try:
            results = harness.run_harness(
                model="local-completions", model_args=_model_args(inp), tasks=tasks, **kw
            )
        except ImportError as e:
            return self._stub(inp, f"harness unavailable ([harness] extra): {e}", preset=preset_name)
        except Exception as e:  # noqa: BLE001 — a harness failure must not break the pipeline
            return self._stub(inp, f"harness failed: {e}", preset=preset_name)
        elapsed = round(time.monotonic() - started, 3)

        ctx.report.stage("score", percent=90)
        results = results or {}
        aggregate = _rows_from_results(results, grader="lm-eval")
        per_case = _rows_from_samples(results, grader="lm-eval")

        # The budget RAISES (ADR-0102 D5) — deliberately before the output is
        # built, so a blown ceiling cannot be read as a successful evaluation.
        assert_within_budget(per_case, EvalBudget(
            max_cost_usd_per_case=inp.max_cost_usd_per_case,
            max_latency_ms_per_case=inp.max_latency_ms_per_case,
            max_total_cost_usd=inp.max_total_cost_usd,
        ))

        rows = aggregate + per_case
        metrics = flat_metrics(rows)
        decontaminated = any(r.decontaminated for r in rows)
        cost = round(sum(r.cost_usd for r in per_case), 6)

        note = ""
        if not metrics:
            note = "harness ran but produced no scalar metrics"
        elif inp.decontaminate and not decontaminated:
            note = (
                "decontamination was requested but no `_decontaminate` metric came back — "
                f"the tasks {tasks!r} do not implement should_decontaminate, so these scores "
                "are NOT decontaminated"
            )

        ctx.report.scalars(tasks_run=len(tasks), cases=len(per_case), cost_usd=cost)
        return LmEvalOutput(
            checked=bool(metrics),
            note=note,
            preset_used=preset_name,
            tasks_run=tasks,
            scorecard=rows,
            metrics=metrics,
            directions=directions_of(rows),
            decontaminated=decontaminated,
            decontamination_requested=inp.decontaminate,
            cases=len(per_case),
            cost_usd=cost,
            total_s=elapsed,
            audit_metadata={
                "checked": bool(metrics), "preset": preset_name, "tasks": tasks,
                "metrics": metrics, "decontaminated": decontaminated,
                "decontamination_requested": inp.decontaminate,
                "cases": len(per_case), "cost_usd": cost, "note": note,
            },
        )

    def mock(self, inp: LmEvalInput, ctx: OperationContext) -> LmEvalOutput:
        return self._stub(inp, "mock — lm-eval skipped")
