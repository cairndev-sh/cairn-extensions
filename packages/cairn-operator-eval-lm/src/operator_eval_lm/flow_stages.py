"""``eval.lm_eval`` — base model vs base model + adapter, on the GPU box, as a governed ml leg
(cairn-extensions#42; the design is ``docs/sft-integration.md``, option b).

A flow-stage kernel in the engine-neutral ``cairn.flow_stages`` entry-point group::

    eval_lm_eval(workdir: Path, config: dict, inputs: list[str]) -> dict   # flat scalar stats

The domain pack wires it (``cairn-pack-finetune``'s ``sft-eval`` flow) as its own ml leg, so it
stays inside every chokepoint an ml leg has: the launch door's estimate, the reservation and
settlement of the box's ``hourly_usd``, the engine's retry and cancel, and the record.

What one run does, in order:

1. **Reads** the held-out splits (``val_uri``, ``test_uri``: pinned refs, verified before use) and
   finds the adapter (``box_io``).
2. **Loads the base model once** (4-bit for ``qlora``, as it was trained).
3. **Time probe.** Scores about ``probe_items`` documents on the base model, spread over the
   preset's tasks in proportion to their limits, and a few split-loss rows. From the requests and
   seconds the MODEL spent (lm-eval's own work is counted apart) it projects the whole run: both
   passes (the candidate's priced ``CANDIDATE_FACTOR`` higher) and both split losses. If the
   projection does not fit ``time_budget_s`` (the leg's timeout) less ``time_margin``, the stage
   **raises** :class:`TimeBudgetExceeded` naming the figures: the leg fails cleanly (a clean exit
   is never retried) after minutes, instead of being killed at the timeout after hours.
4. **Base pass**, then the adapter is attached to the same loaded model and the **candidate pass**
   runs the same calls: the same tasks, limits, few-shot count, batch size and seeds. One
   ``simple_evaluate`` call per task, so MMLU's per-subject limit is its own.
5. **Split loss**, base and candidate, by one method (:data:`SPLIT_LOSS_METHOD`).
6. **Reports** per task and metric: both scores with their standard errors, the delta and its
   standard error, the documents scored; the worst delta and the number of regressions; the split
   losses. As stats (the leg's findings, ``eval_lm_eval_<key>``) and as a results file, uploaded
   beside the run's checkpoints when ``CAIRN_CHECKPOINT_BUCKET`` is set, registered as the stage's
   ``report`` artifact.

**What fails the leg and what does not.** A projection over budget, a split whose bytes do not
match their pin, an unknown preset or an adapter that cannot be found raise: they are a
configuration or a budget, not a measurement. A harness that is not installed or fails while
scoring (an import error, a CUDA out-of-memory, a task that will not load) returns
``checked=false`` with the reason and no scores, so the card says "not evaluated" and a training
run is never lost to its evaluation. A process killed on the box fails the leg and the engine
retries it (``retry:`` on the leg); the retry starts the evaluation again.

No endpoint and no secret: the ``hf`` backend loads weights in this process; no key is read, and
nothing is written to ``os.environ``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STAGE = "eval.lm_eval"
RESULTS_FILE = "lm_eval_results.json"

#: The one loss method this stage reports, for the base model and the candidate alike, by name on
#: the record (``split_loss_method``). It is not the trainer's ``eval_loss``: the trainer batches
#: and pads, so its absolute values differ (on cairn-extensions#9's validation rows, base →
#: candidate: 2.718 → 0.554 from the trainer, 2.380 → 0.453 by this method, measured by hand).
SPLIT_LOSS_METHOD = (
    "full-sequence mean token NLL: each row rendered with the chat template the adapter was trained "
    "with (its saved tokenizer's; else the trainer's rule: the base tokenizer's, else ChatML), one row "
    "per forward pass (no padding, no packing), truncated at max_seq_len, summed over every predicted "
    "token of the split and divided by their number"
)

#: How much longer a candidate pass is projected to take than the base pass it was probed on: the
#: LoRA matmuls on every adapted layer. Conservative; the first live runs will measure it.
CANDIDATE_FACTOR = 1.25
#: Split-loss rows the time probe scores.
PROBE_SPLIT_ROWS = 8


class TimeBudgetExceeded(RuntimeError):
    """The time probe projects a run past the leg's timeout. Raised before the full run."""


def _now() -> float:
    """The stage's clock (a seam: the tests drive it)."""
    return time.monotonic()


def _key(*parts: str) -> str:
    """A finding-safe key: ``arc_easy``/``acc_norm`` → ``arc_easy_acc_norm``."""
    return re.sub(r"[^0-9A-Za-z_]+", "_", "_".join(p for p in parts if p)).strip("_")


# ── rows → chat messages: the trainer's mapping, for the shapes a prepared split carries ──────

def _messages(row: dict) -> list[dict] | None:
    """One split row as the chat turns the trainer rendered it with (``cairn-operator-training``'s
    ``train_entry._to_messages``: alpaca ``instruction`` [+ ``input``] → ``output``/``response``,
    ``prompt``/``completion``, ``input``/``output``, or ``messages``). ``None`` for a row the
    trainer would refuse; such rows are counted, not scored."""
    msgs = row.get("messages")
    if isinstance(msgs, list):
        out = []
        for m in msgs:
            if not isinstance(m, dict) or not isinstance(m.get("content"), str):
                return None
            role = {"human": "user", "gpt": "assistant"}.get(str(m.get("role")).lower(), str(m.get("role")).lower())
            if role not in ("system", "user", "assistant"):
                return None
            out.append({"role": role, "content": m["content"]})
        return out if out and out[-1]["role"] == "assistant" else None
    if row.get("instruction") is not None:
        user = str(row["instruction"])
        if row.get("input"):
            user = f"{user}\n\n{row['input']}"
        assistant = row["output"] if row.get("output") is not None else row.get("response")
    elif row.get("prompt") is not None or row.get("completion") is not None:
        user, assistant = row.get("prompt"), row.get("completion")
    elif row.get("input") is not None or row.get("output") is not None:
        user, assistant = row.get("input"), row.get("output")
    else:
        return None
    user, assistant = str(user or ""), str(assistant or "")
    if not user.strip() or not assistant.strip():
        return None
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


# ── the time probe ────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskProbe:
    task: str
    limit: int
    probe_limit: int
    wall_s: float
    model_s: float
    requests: int
    documents: int
    full_documents: int

    def projected_s(self, batch_size: int, factor: float = 1.0) -> float:
        """One full pass of this task: the model's time, scaled by batches (the probe's batches are
        part-full, so this over-estimates), plus lm-eval's own work as the probe measured it."""
        overhead = max(0.0, self.wall_s - self.model_s)
        if self.documents <= 0 or self.requests <= 0:
            return overhead
        full_requests = self.full_documents * self.requests / self.documents
        per_batch = self.model_s / max(1, math.ceil(self.requests / batch_size))
        return math.ceil(full_requests / batch_size) * per_batch * factor + overhead


def probe_limits(plan: list[tuple[str, int]], probe_items: int) -> dict[str, int]:
    """Documents per task (per subtask, for a group) the probe scores: ``probe_items`` spread in
    proportion to the tasks' limits, at least one each. With a task limited to "every document"
    (0), evenly."""
    limits = [limit for _, limit in plan]
    if plan and all(limits):
        total = sum(limits)
        return {t: max(1, math.ceil(probe_items * limit / total)) for t, limit in plan}
    return {t: max(1, math.ceil(probe_items / max(1, len(plan)))) for t, _ in plan}


# ── the kernel ────────────────────────────────────────────────────────────────────────────────

def eval_lm_eval(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Benchmarks and held-out loss for the base model and the base model + adapter, with identical
    settings; see the module docstring."""
    from operator_eval_lm import box_io, inprocess, presets  # noqa: PLC0415 — off the registry-enumeration path
    from operator_eval_lm.schemas.eval_stage import EvalStageConfig  # noqa: PLC0415

    started = _now()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = EvalStageConfig.model_validate({k: v for k, v in (config or {}).items() if v is not None})
    if not cfg.stub and cfg.time_budget_s <= 0:
        raise ValueError(f"{STAGE}: time_budget_s is required (the leg's timeout in seconds): the time "
                         "probe refuses a run that would not finish inside it")
    preset = presets.load(cfg.preset)
    batch = cfg.batch_size or preset.batch_size
    plan = [(task, preset.limit_for(task)) for task in preset.tasks]

    rows = {name: box_io.read_jsonl(box_io.fetch_dataset(uri, workdir / "splits" / name))
            for name, uri in (("val", cfg.val_uri), ("test", cfg.test_uri)) if uri}
    if cfg.split_rows:
        rows = {name: r[:cfg.split_rows] for name, r in rows.items()}
    adapter, adapter_source = box_io.resolve_adapter(
        local_dir=cfg.local_adapter_dir, uri=cfg.adapter_uri, dest=workdir / "adapter", stub=cfg.stub)

    settings: dict[str, Any] = {
        "stub": cfg.stub, "preset": preset.name, "tasks": ",".join(preset.tasks), "seed": cfg.seed,
        "batch_size": batch, "num_fewshot": preset.num_fewshot, "method": cfg.method,
        "base_model": cfg.base_model, "adapter": adapter, "adapter_source": adapter_source,
        "max_seq_len": cfg.max_seq_len, "time_budget_s": cfg.time_budget_s,
        "split_loss_method": SPLIT_LOSS_METHOD,
    }
    try:
        backend = inprocess.make_backend(stub=cfg.stub)
    except ImportError as e:
        return _not_checked(settings, f"harness unavailable (install cairn-operator-eval-lm[box] on the "
                                      f"box): {e}", started)
    settings["harness_backend"] = backend.name
    try:
        m = _measure(cfg, preset, plan, batch, rows, adapter, backend, settings, started)
    except (TimeBudgetExceeded, box_io.FetchError):
        raise
    except Exception as e:  # noqa: BLE001 — a failed measurement is "not evaluated", never a score
        return _not_checked(settings, f"harness failed: {type(e).__name__}: {e}", started)
    return _report(workdir, cfg, plan, settings, m, started)


def _not_checked(settings: dict, note: str, started: float) -> dict:
    """No scores: the reason, and the settings it was asked to run with (scalars only)."""
    flat = {k: v for k, v in settings.items() if isinstance(v, (str, int, float, bool))}
    return {**flat, "checked": False, "note": note, "elapsed_s": round(_now() - started, 3)}


def _measure(cfg, preset, plan, batch: int, rows: dict, adapter: str, backend, settings: dict,
             started: float) -> dict:
    """Everything that touches the model: load, probe, both passes, both split losses. Any error
    here, but the budget refusal, is a failed measurement (the caller's ``checked=false``)."""
    from operator_eval_lm import scoring  # noqa: PLC0415

    settings["versions"] = backend.versions()
    base = backend.load(cfg.base_model, quantize_4bit=cfg.method == "qlora", seed=cfg.seed)
    settings["quantization"] = base.quantization
    template, settings["chat_template_source"] = backend.chat_template(base, adapter)
    settings["adapter_base_model"] = backend.adapter_base(adapter)
    texts: dict[str, list[str]] = {}
    skipped: dict[str, int] = {}
    for name, split in rows.items():
        msgs = [m for m in (_messages(r) for r in split) if m is not None]
        skipped[name] = len(split) - len(msgs)
        texts[name] = [backend.render(base, m, template) for m in msgs]

    # ── 3. the time probe, on the base model ────────────────────────────────────────────────
    probes: list[TaskProbe] = []
    for task, limit in plan:
        probe_limit = probe_limits(plan, cfg.probe_items)[task]
        t0 = _now()
        res, timing = backend.evaluate(base, task=task, limit=probe_limit, num_fewshot=preset.num_fewshot,
                                       batch_size=batch, seed=cfg.seed)
        scored, _ = scoring.documents(res, probe_limit)
        _, full = scoring.documents(res, limit)
        probes.append(TaskProbe(task=task, limit=limit, probe_limit=probe_limit, wall_s=_now() - t0,
                                model_s=timing.model_s, requests=timing.requests, documents=scored,
                                full_documents=full))
    split_texts = [t for name in texts for t in texts[name]]
    probe_rows = split_texts[:PROBE_SPLIT_ROWS]
    t0 = _now()
    if probe_rows:
        backend.split_loss(base, probe_rows, max_seq_len=cfg.max_seq_len)
    split_probe_s = _now() - t0
    split_s = split_probe_s * len(split_texts) / len(probe_rows) if probe_rows else 0.0

    base_pass = sum(p.projected_s(batch) for p in probes)
    candidate_pass = sum(p.projected_s(batch, CANDIDATE_FACTOR) for p in probes)
    so_far = _now() - started
    projected = so_far + base_pass + candidate_pass + split_s * (1 + CANDIDATE_FACTOR)
    allowed = cfg.time_budget_s * (1 - cfg.time_margin)
    requests = sum(p.requests for p in probes)
    probe = {"probe_documents": sum(p.documents for p in probes),
             "probe_s": round(sum(p.wall_s for p in probes) + split_probe_s, 3),
             # throughput: the model's seconds per request (lm-eval's own work excluded)
             "probe_model_s_per_request": round(sum(p.model_s for p in probes) / requests, 6) if requests else 0.0,
             "projected_s": round(projected, 1)}
    if cfg.time_budget_s and projected > allowed:
        per_task = ", ".join(f"{p.task} {p.projected_s(batch) / 60:.0f}+{p.projected_s(batch, CANDIDATE_FACTOR) / 60:.0f} min "
                             f"({p.full_documents} documents; probe {p.documents} in {p.wall_s:.0f}s)"
                             for p in probes)
        raise TimeBudgetExceeded(
            f"{STAGE} refused before the full run: projected {projected / 3600:.2f} h > {allowed / 3600:.2f} h "
            f"(the leg's {cfg.time_budget_s / 3600:g} h less a {cfg.time_margin:.0%} margin). Base + candidate "
            f"per task: {per_task}; split loss {split_s * (1 + CANDIDATE_FACTOR) / 60:.0f} min; "
            f"{so_far / 60:.0f} min spent loading and probing. Lower the preset's limits, or give the leg a "
            f"longer timeout (and budget).")

    # ── 4–5. the two passes, identical calls ────────────────────────────────────────────────
    def scored_pass(model) -> tuple[dict, dict]:
        out = {task: backend.evaluate(model, task=task, limit=limit, num_fewshot=preset.num_fewshot,
                                      batch_size=batch, seed=cfg.seed)[0] for task, limit in plan}
        losses = {name: backend.split_loss(model, t, max_seq_len=cfg.max_seq_len) for name, t in texts.items()}
        return out, losses

    base_results, base_losses = scored_pass(base)
    candidate = backend.attach(base, adapter)   # the same loaded weights, the adapter on top
    cand_results, cand_losses = scored_pass(candidate)
    return {"base": base_results, "candidate": cand_results, "base_losses": base_losses,
            "candidate_losses": cand_losses, "texts": texts, "skipped": skipped, "probes": probes,
            "probe": probe, "split_probe_s": split_probe_s}


def _report(workdir: Path, cfg, plan, settings: dict, m: dict, started: float) -> dict:
    """Step 6: the deltas, the results file and the stats, from what :func:`_measure` scored."""
    from operator_eval_lm import box_io, scoring  # noqa: PLC0415

    base_results, cand_results = m["base"], m["candidate"]
    base_losses, cand_losses, texts, skipped = m["base_losses"], m["candidate_losses"], m["texts"], m["skipped"]
    probes, probe, split_probe_s = m["probes"], m["probe"], m["split_probe_s"]
    scores: list[dict] = []
    stats: dict[str, Any] = {}
    for task, limit in plan:
        b = {s.key: s for s in scoring.task_scores(base_results[task], task)}
        c = {s.key: s for s in scoring.task_scores(cand_results[task], task)}
        n_base, _ = scoring.documents(base_results[task], limit)
        n_cand, _ = scoring.documents(cand_results[task], limit)
        stats[_key("n", task)] = n_base
        for key in sorted(b.keys() & c.keys()):
            sb, sc = b[key], c[key]
            delta = sc.value - sb.value
            se = math.sqrt(sb.stderr ** 2 + sc.stderr ** 2) if sb.stderr is not None and sc.stderr is not None else None
            scores.append({"task": task, "metric": sb.metric, "filter": sb.filter, "key": _key(task, key),
                           "higher_is_better": sb.higher_is_better, "base": sb.value, "base_stderr": sb.stderr,
                           "candidate": sc.value, "candidate_stderr": sc.stderr, "delta": delta,
                           "delta_stderr": se, "documents_base": n_base, "documents_candidate": n_cand})
    for s in scores:
        stats[f"base_{s['key']}"] = round(s["base"], 6)
        stats[f"candidate_{s['key']}"] = round(s["candidate"], 6)
        stats[f"delta_{s['key']}"] = round(s["delta"], 6)
        for who in ("base", "candidate", "delta"):
            if s[f"{who}_stderr"] is not None:
                stats[f"{who}_{s['key']}_stderr"] = round(s[f"{who}_stderr"], 6)
    # the delta in each metric's own direction: negative = the candidate is worse
    signed = [(s["delta"] if s["higher_is_better"] else -s["delta"], s) for s in scores]
    worst = min(signed, key=lambda x: x[0]) if signed else None
    regressions = [s["key"] for d, s in signed if s["delta_stderr"] is not None and d < -s["delta_stderr"]]

    split_loss: dict[str, dict] = {}
    for name in texts:
        (b_sum, b_tok), (c_sum, c_tok) = base_losses[name], cand_losses[name]
        entry = {"base": b_sum / b_tok if b_tok else None, "candidate": c_sum / c_tok if c_tok else None,
                 "rows": len(texts[name]), "rows_skipped": skipped[name], "tokens": b_tok}
        entry["delta"] = (entry["candidate"] - entry["base"]
                          if entry["base"] is not None and entry["candidate"] is not None else None)
        split_loss[name] = entry
        for who in ("base", "candidate", "delta"):
            if entry[who] is not None:
                stats[f"{who}_{name}_loss"] = round(entry[who], 6)
        stats[f"{name}_rows"] = entry["rows"]
        stats[f"{name}_tokens"] = entry["tokens"]
        if skipped[name]:
            stats[f"{name}_rows_skipped"] = skipped[name]

    checked = bool(scores) and not cfg.stub
    notes = ["stub tier: no model was loaded and nothing was measured; the numbers are placeholders"
             if cfg.stub else "" if scores else "the harness ran but no metric came back for both models"]
    trained_on = settings.get("adapter_base_model") or ""
    if trained_on and trained_on != cfg.base_model:
        # Unsloth's 4-bit loading substitutes a pre-quantised checkpoint for the named base: the
        # adapter records which. Both models here are scored on `base_model`, so the deltas compare
        # like with like; the absolute scores are of that base, not the one the adapter saw.
        notes.append(f"the adapter records it was trained on {trained_on!r}; both models were scored on "
                     f"{cfg.base_model!r} as asked")
    note = "; ".join(n for n in notes if n)
    elapsed = round(_now() - started, 3)
    report = {
        "schema": 1, "stage": STAGE, "checked": checked, "note": note,
        "settings": {k: v for k, v in settings.items() if k != "versions"}, "versions": settings["versions"],
        "tasks": [{"task": p.task, "limit": p.limit, "documents": stats.get(_key("n", p.task), 0)} for p in probes],
        "task_versions": {t: v for r in base_results.values() for t, v in (r.get("versions") or {}).items()},
        "scores": scores, "split_loss": {"method": SPLIT_LOSS_METHOD, **split_loss},
        "summary": {"metrics_compared": len(scores), "regressions": regressions,
                    "worst_delta": worst[0] if worst else None, "worst_delta_metric": worst[1]["key"] if worst else ""},
        "probe": {**probe, "tasks": [asdict(p) for p in probes], "split_probe_s": split_probe_s,
                  "candidate_factor": CANDIDATE_FACTOR},
        "elapsed_s": elapsed,
    }
    path = workdir / RESULTS_FILE
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    location, upload_error = str(path), ""
    target = box_io.results_target()
    if target is not None:
        try:
            location = box_io.upload_results(path, *target)
        except Exception as e:  # noqa: BLE001 — the scores are on the record; the file stays on the box
            upload_error = f"{type(e).__name__}: {e}"
    return {
        **{k: v for k, v in settings.items() if k != "versions"},
        "lm_eval_version": settings["versions"].get("lm_eval", "?"),
        "checked": checked, "note": note,
        **stats,
        "metrics_compared": len(scores), "regressions": len(regressions),
        "regressed": ",".join(regressions),
        **({"worst_delta": round(worst[0], 6), "worst_delta_metric": worst[1]["key"]} if worst else {}),
        **probe, "elapsed_s": elapsed,
        "results_uri": location, **({"results_upload_error": upload_error} if upload_error else {}),
        # the stage's product (the runner turns these into the registered descriptor)
        "output": location, "content_hash": digest, "kind": "report", "bytes": path.stat().st_size,
    }


# Paid GPU time, and it writes the results file to the checkpoint bucket (ADR-0081 D2: fail-closed
# risk classes; the same class as train.qlora).
eval_lm_eval.risk_class = "external_write"

__all__ = ["CANDIDATE_FACTOR", "SPLIT_LOSS_METHOD", "STAGE", "TaskProbe", "TimeBudgetExceeded",
           "eval_lm_eval", "probe_limits"]
