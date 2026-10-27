"""eval/lm-eval — classic public benchmarks against a SERVED candidate (ADR-0102 D5).

The candidate is already stood up behind an OpenAI-compatible endpoint (a vLLM
server, for example). This operator points lm-evaluation-harness at THAT endpoint —
``model="local-completions"`` — instead of loading a second copy of the weights.
Three consequences worth stating, because they are the reason this is not another
flag on an in-process benchmark (``eval/benchmark`` in cairn's platform operators,
not public), which loads base weights + an adapter on a GPU:

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

The harness is synchronous and runs for minutes to hours, so ``run`` hands it to a
worker thread (``asyncio.to_thread``) and the event loop keeps serving other runs
(ADR-0125). Its HTTP backends also call ``asyncio.run`` themselves, which fails on
a thread that already runs a loop. A CANCEL stops the operator waiting, but a
thread cannot be killed: the harness call runs on to its end in the background,
still sending requests to the endpoint, and its result is discarded
(``supports_cancel = false`` says as much). ``to_thread`` uses the loop's default
executor, so a long or cancelled harness call holds one of its worker slots until
it finishes; enough concurrent benchmarks can fill that pool and delay everything
else in the process that uses it.

The endpoint key (``api_key_env``, read through ``ctx.secrets``) goes to the harness
call and nowhere else: never into ``os.environ``, which every run in the process
shares (ADR-0143 D3), and never into ``model_args``. A named key the run cannot
read is a refusal, not an unauthenticated call.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from cairn.core.eval.scorecard import (
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
# The parsing is shared with the on-box `eval.lm_eval` stage (cairn-extensions#42): one copy.
from operator_eval_lm.scoring import DROP_SUFFIXES as _DROP_SUFFIXES  # noqa: F401 — re-exported
from operator_eval_lm.scoring import metric_name as _metric_name
from operator_eval_lm.scoring import rows_from_results as _rows_from_results
from operator_eval_lm.schemas.lm_eval import LmEvalConfig, LmEvalInput, LmEvalOutput

#: ADR-0103. Stages only — a benchmark has no per-step series to stream, and the
#: three scalars are the numbers a reviewer wants on the Activity card.
LM_EVAL_REPORTS = ReportSpec(
    stages=["resolve", "evaluate", "score"],
    scalars=["tasks_run", "cases", "cost_usd"],
)


#: The seam's decontamination argument; lm-eval 0.4.10-0.4.13 do not take it.
_DECONTAM_ARG = "decontamination_ngrams_path"


def _redact(text: str, secret: str | None) -> str:
    """``text`` with the endpoint key replaced, before it reaches a note or the audit
    record."""
    return text.replace(secret, "[redacted]") if secret else text


def _model_args(inp: LmEvalInput) -> str:
    """lm-eval ``model_args`` for its OpenAI-compatible ``local-completions`` type."""
    parts = [f"model={inp.model}", f"base_url={inp.endpoint_url}",
             f"num_concurrent={inp.num_concurrent}"]
    if inp.tokenizer:
        parts.append(f"tokenizer={inp.tokenizer}")
    return ",".join(parts)


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
            # A case's cost is carried by its FIRST metric row only: the rows are
            # summed (cost_usd, the total ceiling), and one request scored on three
            # metrics was paid for once. Latency is a per-case maximum, so every
            # row keeps it.
            costed = False
            for raw, value in s.items():
                if raw in ("doc_id", "cost_usd", "latency_ms"):
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                rows.append(ScorecardRow(
                    metric=_metric_name(raw), value=float(value), slice=task,
                    case=f"{task}:{case}", grader=grader,
                    cost_usd=0.0 if costed else cost, latency_ms=latency,
                    decontaminated=is_decontaminated(_metric_name(raw)),
                ))
                costed = True
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
        run must not emit a score a gate could pass on."""
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
            # sees the REQUEST. A release whose simple_evaluate does not take it
            # (0.4.13 does not) gets it dropped by the seam, and the output says
            # the scores are not decontaminated (DESIGN.md §6).
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
        if not inp.tasks:
            preset = presets.load(inp.preset or presets.DEFAULT_PRESET)
            if preset.limits and not inp.limit:
                # cairn-extensions#42 — a preset that sizes tasks apart (`limits`, e.g. MMLU per
                # subject) needs one harness call per task; this operator makes one call with one
                # limit, which would score `limit` documents in every MMLU subject.
                return self._stub(
                    inp, f"preset {preset.name!r} sizes its tasks apart ({sorted(preset.limits)}); "
                         "eval/lm-eval runs one harness call with one limit. Pass `tasks` and `limit`, "
                         "or run the preset with the eval.lm_eval flow stage.", preset=preset.name)
        tasks, preset_name, kw = self._resolve(inp)
        if not tasks:
            return self._stub(inp, "no tasks resolved — nothing to evaluate", preset=preset_name)
        api_key: str | None = None
        if inp.api_key_env:
            # The key goes to the harness call only — never os.environ (shared by
            # every run in this process; ADR-0143 D3) and never model_args (logged
            # and copied into lm-eval's results). See harness.py.
            refusal = (
                f"api_key_env {inp.api_key_env!r} names a key this run cannot read; declare it "
                "in the binding's `secrets:` block (and supply it from a connection). Refusing "
                "rather than calling the endpoint without its key."
            )
            secrets = getattr(ctx, "secrets", None)
            if secrets is None:
                return self._stub(inp, refusal, preset=preset_name)
            try:
                api_key = secrets.get(inp.api_key_env)
            except KeyError as e:
                return self._stub(inp, f"{refusal} ({e})", preset=preset_name)
            if not api_key:
                return self._stub(inp, refusal, preset=preset_name)

        ctx.report.stage("evaluate", percent=10)
        started = time.monotonic()
        try:
            # Off the event loop (ADR-0125): the harness is synchronous and long.
            results = await asyncio.to_thread(
                harness.run_harness,
                model="local-completions", model_args=_model_args(inp), tasks=tasks,
                api_key=api_key, **kw,
            )
        except ImportError as e:
            return self._stub(inp, _redact(f"harness unavailable ([harness] extra): {e}", api_key),
                              preset=preset_name)
        except harness.HarnessRefusal as e:
            return self._stub(inp, _redact(f"refused: {e}", api_key), preset=preset_name)
        except Exception as e:  # noqa: BLE001 — a harness failure must not break the pipeline
            # an endpoint error can echo the request's Authorization header
            return self._stub(inp, _redact(f"harness failed: {e}", api_key), preset=preset_name)
        elapsed = round(time.monotonic() - started, 3)

        ctx.report.stage("score", percent=90)
        results = results or {}
        dropped = [a for a in (results.get(harness.DROPPED_KEY) or []) if isinstance(a, str)]
        lm_eval_version = str(results.get(harness.VERSION_KEY) or "?")
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

        notes: list[str] = []
        if not metrics:
            notes.append("harness ran but produced no scalar metrics")
        elif inp.decontaminate and not decontaminated:
            if _DECONTAM_ARG in dropped:
                notes.append(
                    f"decontamination was requested but lm-eval {lm_eval_version} takes no "
                    "decontamination argument, so it was not passed: these scores are NOT "
                    "decontaminated"
                )
            else:
                notes.append(
                    "decontamination was requested but no `_decontaminate` metric came back — "
                    f"the tasks {tasks!r} do not implement should_decontaminate, so these scores "
                    "are NOT decontaminated"
                )
        other = [a for a in dropped if a != _DECONTAM_ARG]
        if other:
            notes.append(f"not passed (lm-eval {lm_eval_version} does not take them): "
                         + ", ".join(other))
        note = "; ".join(notes)

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
