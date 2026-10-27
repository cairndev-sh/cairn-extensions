"""lm-evaluation-harness results → numbers. One copy, two callers (cairn-extensions#42).

``eval/lm-eval`` (the served candidate, on the control plane) and the ``eval.lm_eval`` flow stage
(base model and base + adapter in process, on the GPU box) read the same ``simple_evaluate``
payload, so the parsing lives here once: the metric name without its filter suffix, the dropped
``_stderr`` / ``_alias`` keys, the ``_decontaminate`` twins and the metric directions.

``simple_evaluate`` returns (0.4.10-0.4.13, ``evaluator_utils.py`` 155-166 in 0.4.13)::

    {"results": {task_or_group: {"<metric>,<filter>": value, "<metric>_stderr,<filter>": se, "alias": …}},
     "higher_is_better": {task_or_group: {metric: bool}},
     "n-samples": {leaf_task: {"original": N, "effective": n}},
     "group_subtasks": {group: [child, …]}, "versions": {task: version}, …}

A group such as ``mmlu`` has its own aggregate entry in ``results`` (and a propagated direction),
and ``n-samples`` counts its leaf subtasks. ``limit`` applies per LEAF task, so ``limit=10`` on
``mmlu`` scores up to 10 documents in each of its 57 subjects.
"""
from __future__ import annotations

from dataclasses import dataclass

from cairn.core.eval.scorecard import DECONTAM_SUFFIX, ScorecardRow, is_decontaminated

#: Metric-name suffixes the harness adds that are not gate-usable scalars.
#: Checked against the metric NAME, never the raw key: lm-eval spells the error
#: bar ``exact_match_stderr,strict-match`` — filter suffix LAST — so a check on
#: the raw key misses it and a standard error lands in the map as a metric.
DROP_SUFFIXES = ("_stderr", "_alias")


def metric_name(raw: str) -> str:
    """``exact_match,strict-match`` → ``exact_match`` (the filter suffix is not
    part of the metric's identity)."""
    return raw.split(",", 1)[0]


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def rows_from_results(results: dict, grader: str) -> list[ScorecardRow]:
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
            if _number(value) is None:
                continue
            name = metric_name(raw)
            if any(name.endswith(s) for s in DROP_SUFFIXES):
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


@dataclass(frozen=True)
class TaskScore:
    """One aggregate score lm-eval reported for one requested task (a group's aggregate for a
    group): the metric, its filter, the value, its standard error (``None`` when lm-eval reported
    none, or ``"N/A"``) and its direction."""

    task: str
    metric: str
    filter: str
    value: float
    stderr: float | None
    higher_is_better: bool

    @property
    def key(self) -> str:
        """``<metric>`` for the ``none`` filter, ``<metric>_<filter>`` otherwise, so two filters
        of one metric (gsm8k's ``strict-match`` and ``flexible-extract``) never share a key."""
        return self.metric if self.filter in ("", "none") else f"{self.metric}_{self.filter}"


def task_scores(results: dict, task: str) -> list[TaskScore]:
    """The aggregate scores of ONE requested task (``task`` as it was asked for: a leaf task, or a
    group such as ``mmlu``, whose aggregate entry is the number). A group's subtasks are not
    returned. Sorted by key."""
    mm = (results.get("results") or {}).get(task)
    if not isinstance(mm, dict):
        return []
    directions = (results.get("higher_is_better") or {}).get(task) or {}
    out: list[TaskScore] = []
    for raw, value in mm.items():
        number = _number(value)
        if number is None:
            continue
        name = metric_name(raw)
        if any(name.endswith(s) for s in DROP_SUFFIXES):
            continue
        filt = raw.split(",", 1)[1] if "," in raw else ""
        stderr = _number(mm.get(f"{name}_stderr,{filt}" if filt else f"{name}_stderr"))
        higher = directions.get(name) if isinstance(directions, dict) else None
        out.append(TaskScore(task=task, metric=name, filter=filt, value=number, stderr=stderr,
                             higher_is_better=True if higher is None else bool(higher)))
    return sorted(out, key=lambda s: s.key)


def documents(results: dict, limit: int) -> tuple[int, int]:
    """``(documents this call scored, documents a call with ``limit`` would score)``, from
    lm-eval's ``n-samples``. ``limit`` applies per leaf task (0 = every document), so a group
    scores up to ``limit`` × its subtasks. Only meaningful for a call that ran ONE requested
    task: every ``n-samples`` entry then belongs to it."""
    scored = full = 0
    for counts in (results.get("n-samples") or {}).values():
        if not isinstance(counts, dict):
            continue
        original = int(counts.get("original") or 0)
        scored += int(counts.get("effective") or 0)
        full += min(limit, original) if limit else original
    return scored, full


__all__ = ["DROP_SUFFIXES", "TaskScore", "documents", "metric_name", "rows_from_results", "task_scores"]
