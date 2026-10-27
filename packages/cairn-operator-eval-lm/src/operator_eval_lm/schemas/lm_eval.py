"""Schemas for ``eval/lm-eval`` — classic benchmarks on a served candidate.

The shape difference from an in-process benchmark (``eval/benchmark``, one of
cairn's platform operators, not public; it loads a base model + LoRA adapter on a
GPU) is deliberate and is the reason this is a
separate operator rather than another flag on that one: here the candidate is
ALREADY SERVED — an OpenAI-compatible endpoint such as a vLLM server — so the
work is HTTP, not CUDA, the operator needs no GPU of its own, and the numbers
describe the artefact that will actually be shipped rather than a second copy of
it loaded a different way.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cairn.core.eval.scorecard import ScorecardRow

from operator_eval_lm.presets import DEFAULT_PRESET


class LmEvalInput(BaseModel):
    """One benchmark pass over a served candidate."""

    enabled: bool = Field(
        False,
        description="OFF by default, like every producer in the eval domain: a benchmark costs "
                    "endpoint time, so a pipeline opts in.",
    )
    dry_run: bool = Field(
        False, description="Skip the harness and return the deterministic stub (no scores)."
    )

    # ── the served candidate ─────────────────────────────────────────────
    endpoint_url: str = Field(
        "",
        description="OpenAI-compatible base URL of the served candidate (…/v1) — what a vLLM "
                    "server exposes. REQUIRED: with none, the operator refuses rather than reporting 0.",
    )
    model: str = Field(
        "cairn-model", description="The `model` id to request (vLLM's served_model_name)."
    )
    api_key_env: str = Field(
        "",
        description="NAME of the secret holding the endpoint's bearer key, read through "
                    "ctx.secrets (the binding must declare it in `secrets:`); not an environment "
                    "variable, and there is no inline key field. Empty = call the endpoint "
                    "without a key.",
    )
    num_concurrent: int = Field(
        4, gt=0, description="Requests in flight; the server owns the real batching."
    )
    tokenizer: str = Field(
        "",
        description="HF tokenizer id for loglikelihood tasks, when the endpoint's own tokenizer "
                    "is not discoverable. Empty = let the harness decide.",
    )

    # ── what to run ──────────────────────────────────────────────────────
    preset: str = Field(
        DEFAULT_PRESET,
        description="Task preset shipped as pack data (see presets/*.toml). Ignored when `tasks` "
                    "is given.",
    )
    tasks: list[str] = Field(
        default_factory=list,
        description="Explicit lm-eval task ids, OVERRIDING the preset. Empty = use the preset.",
    )
    limit: int = Field(
        0, ge=0,
        description="Documents per task; 0 = take the preset's. The cost bound — a run that "
                    "raises it is buying a longer evaluation on purpose.",
    )
    num_fewshot: int = Field(-1, description="Few-shot examples; -1 = take the preset's.")
    batch_size: int = Field(0, ge=0, description="Requests per batch; 0 = take the preset's.")

    # ── decontamination (ADR-0102 D5) ────────────────────────────────────
    decontaminate: bool = Field(
        True,
        description="Request lm-eval's 13-gram overlap decontamination. Only tasks that override "
                    "`should_decontaminate` produce the `_decontaminate` metric twin; when none "
                    "does, the output says so rather than implying a clean measurement.",
    )
    decontamination_ngrams_path: str = Field(
        "",
        description="Path to a 13-gram overlap index, passed to lm-eval only when the installed "
                    "release takes a decontamination argument. lm-eval 0.4.10-0.4.13 take none, so "
                    "it is not read there and the output says the scores are not decontaminated.",
    )

    # ── the budget, not an estimate (ADR-0102 D5) ────────────────────────
    max_cost_usd_per_case: float = Field(
        0.0, ge=0.0, description="Per-document cost ceiling; breaking it RAISES. 0 = unbounded."
    )
    max_latency_ms_per_case: float = Field(
        0.0, ge=0.0, description="Per-document latency ceiling; breaking it RAISES. 0 = unbounded."
    )
    max_total_cost_usd: float = Field(
        0.0, ge=0.0, description="Whole-evaluation cost ceiling; breaking it RAISES. 0 = unbounded."
    )


class LmEvalConfig(BaseModel):
    """Deployment-tunable surface (ADR-0068/0070). Read the scopes carefully.

    ``num_concurrent`` is **operational** — a workflow may push a dev endpoint
    harder. ``max_total_cost_usd`` is **governance**-scoped: it is the deployment's
    spend ceiling for one evaluation, the same call a batch-inference operator makes for
    ``max_rows``, so a run-supplied value for it is silently DROPPED and
    the profile's wins.

    The two PER-CASE ceilings are deliberately NOT here. A per-case bound is a
    per-run bound — "this evaluation may not spend more than a cent a prompt" is
    a property of the evaluation being launched, not of the environment — and a
    per-run bound declared at governance scope is discarded at run time, so the
    pack would advertise a limit it does not enforce. They live on
    ``LmEvalInput`` only, and ``test_a_per_case_ceiling_is_run_owned`` holds that
    line (it was caught here first: the ceilings started in this class and the
    runtime dropped every value the tests set).
    """

    model_config = ConfigDict(extra="forbid")

    num_concurrent: int = Field(4, title="Concurrent requests", gt=0)
    max_total_cost_usd: float = Field(
        0.0, title="Evaluation cost ceiling ($)", ge=0.0,
        description="ENV-OWNED spend ceiling for one evaluation; a run-supplied value is DROPPED.",
        json_schema_extra={"scope": "governance"},
    )


class LmEvalOutput(BaseModel):
    """Scorecard rows first; the flat map the existing gates read is derived."""

    checked: bool = Field(
        False, description="The harness ran and produced at least one scalar metric."
    )
    note: str = Field("", description="Why nothing ran, or what was odd about what did.")
    preset_used: str = Field("", description="The preset whose task list was run ('' = explicit tasks).")
    tasks_run: list[str] = Field(default_factory=list)
    scorecard: list[ScorecardRow] = Field(
        default_factory=list,
        description="Per metric / slice / case / grader. Aggregate rows (no case) are the ones "
                    "`metrics` exposes; per-case rows are the detail a budget binds to.",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Flat {task/metric: value}, DERIVED from the scorecard — the shape a gate "
                    "reads (a pipeline predicate, or a gate operator such as the platform's "
                    "model/gate or eval/delta, which are not public).",
    )
    directions: dict[str, str] = Field(
        default_factory=dict, description="{task/metric: higher_better|lower_better}."
    )
    decontaminated: bool = Field(
        False,
        description="At least one metric came back decontaminated (a `_decontaminate` twin). "
                    "False with `decontamination_requested` true means the tasks do not support "
                    "it — the note says so.",
    )
    decontamination_requested: bool = Field(False)
    cases: int = Field(0, description="Per-document rows the harness reported (0 = no per-sample log).")
    cost_usd: float = Field(0.0, description="Summed per-case cost, when the harness reported any.")
    total_s: float = Field(0.0, description="Wall-clock of the harness call.")
    audit_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Compact record for the audit sink."
    )
