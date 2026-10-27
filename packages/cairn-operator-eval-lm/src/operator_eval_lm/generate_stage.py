"""``eval.generate`` — answers to held-out prompts, base model vs base model + adapter, on the GPU
box, as a governed ml leg (cairn-extensions#41).

A flow-stage kernel in the engine-neutral ``cairn.flow_stages`` entry-point group::

    eval_generate(workdir: Path, config: dict, inputs: list[str]) -> dict   # flat scalar stats

Loss says how likely the reference answers are; it does not say whether the model's own answers
are usable. This stage reads them. What one run does, in order:

1. **Reads** the test split (``test_uri``: a pinned ref, verified before use) and draws
   ``sample_n`` usable rows with ``seed`` (every usable row when there are fewer). Each row is split
   the way the trainer renders it (``flow_stages._messages``): the turns before the last are the
   prompt, the last assistant turn is the reference answer.
2. **Loads the base model once**, as ``eval.lm_eval`` does (the same backend: 4-bit for ``qlora``),
   and renders every prompt with the chat template the adapter was trained with, plus the
   generation prompt.
3. **Base answers**, greedy, in batches of ``batch_size``. After the first batch the stage projects
   the whole run (the remaining base batches, and every candidate batch priced ``CANDIDATE_FACTOR``
   higher) and **raises** :class:`~operator_eval_lm.flow_stages.TimeBudgetExceeded` when it would
   not fit ``time_budget_s`` (the leg's timeout) less ``time_margin``: a clean failure, never retried.
4. **Candidate answers**: the adapter attached to the same loaded weights, the same prompts, the
   same decoding.
5. **Scores** each answer against the reference with the baseline scorers, the same for both
   models: ``exact_match`` (equal after collapsing whitespace) and ``contains`` (the reference
   appears in the answer, whitespace collapsed). Reports each model's mean and the delta
   (candidate − base).
6. **Writes the samples** — prompt, reference, base answer, candidate answer and their scores, one
   JSON line per prompt — as the stage's ``report`` product, uploaded beside the run's checkpoints
   when ``CAIRN_CHECKPOINT_BUCKET`` is set. A validator stage on the same box (for IFC,
   ``cairn-operator-ifc-eval``'s ``eval.ifc_validate``) reads the same file.

What fails the leg and what does not: as ``eval.lm_eval``. A projection over budget, a split whose
bytes do not match their pin, no test split, or an adapter that cannot be found raise. A backend
that is not installed or fails while generating returns ``checked=false`` with the reason, no
scores and no samples. A killed attempt fails the leg and the engine retries it; the retry starts
again.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

STAGE = "eval.generate"
SAMPLES_FILE = "generation_samples.jsonl"
#: The baseline scorers, in the order they are reported.
SCORERS = ("exact_match", "contains")


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


def score(answer: str, reference: str) -> dict[str, bool]:
    """The baseline scorers for one answer."""
    a, r = _norm(answer), _norm(reference)
    return {"exact_match": a == r, "contains": bool(r) and r in a}


def draw(rows: list[dict], n: int, seed: int) -> list[tuple[int, list[dict], str]]:
    """``(row index, prompt turns, reference)`` for ``n`` usable rows drawn with ``seed``, in split
    order; every usable row when there are no more than ``n``."""
    from operator_eval_lm.flow_stages import _messages  # noqa: PLC0415 — the trainer's row mapping, one copy

    usable = []
    for i, row in enumerate(rows):
        msgs = _messages(row)
        if msgs is not None and len(msgs) >= 2:
            usable.append((i, msgs[:-1], msgs[-1]["content"]))
    if len(usable) <= n:
        return usable
    return [usable[k] for k in sorted(random.Random(seed).sample(range(len(usable)), n))]


def eval_generate(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Answers from the base model and the candidate to the same held-out prompts, scored; see the
    module docstring."""
    from operator_eval_lm import box_io, flow_stages, inprocess  # noqa: PLC0415 — off the registry path
    from operator_eval_lm.schemas.eval_stage import GenerateStageConfig  # noqa: PLC0415

    started = flow_stages._now()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = GenerateStageConfig.model_validate({k: v for k, v in (config or {}).items() if v is not None})
    if not cfg.stub and cfg.time_budget_s <= 0:
        raise ValueError(f"{STAGE}: time_budget_s is required (the leg's timeout in seconds): the first "
                         "batch's time refuses a run that would not finish inside it")
    if not cfg.test_uri:
        raise box_io.FetchError(f"{STAGE}: no test_uri: the prompts come from the test split")
    rows = box_io.read_jsonl(box_io.fetch_dataset(cfg.test_uri, workdir / "splits" / "test"))
    picked = draw(rows, cfg.sample_n, cfg.seed)
    adapter, adapter_source = box_io.resolve_adapter(
        local_dir=cfg.local_adapter_dir, uri=cfg.adapter_uri, dest=workdir / "adapter", stub=cfg.stub)
    settings: dict[str, Any] = {
        "stub": cfg.stub, "base_model": cfg.base_model, "adapter": adapter, "adapter_source": adapter_source,
        "method": cfg.method, "seed": cfg.seed, "sample_n": cfg.sample_n, "samples": len(picked),
        "split_rows": len(rows), "max_new_tokens": cfg.max_new_tokens, "batch_size": cfg.batch_size,
        "max_seq_len": cfg.max_seq_len, "time_budget_s": cfg.time_budget_s, "decoding": "greedy",
        "scorers": ",".join(SCORERS),
    }
    if not picked:
        raise box_io.FetchError(f"{STAGE}: the test split has no row with a prompt and an answer "
                                f"({len(rows)} rows read)")
    try:
        backend = inprocess.make_backend(stub=cfg.stub)
    except ImportError as e:
        return _not_checked(settings, f"backend unavailable (install cairn-operator-eval-lm[box] on the "
                                      f"box): {e}", started)
    settings["harness_backend"] = backend.name
    check = getattr(backend, "check_adapter", None)
    if check is not None:
        try:
            check(adapter)
        except Exception as e:  # noqa: BLE001 — a missing adapter fails the leg before any model loads
            raise box_io.FetchError(f"adapter {adapter!r} ({adapter_source}) can't be read: "
                                    f"{type(e).__name__}: {e}") from e
    try:
        answers = _generate(cfg, picked, adapter, backend, settings, started)
    except (flow_stages.TimeBudgetExceeded, box_io.FetchError):
        raise
    except Exception as e:  # noqa: BLE001 — a failed generation is "not evaluated", never a score
        return _not_checked(settings, f"generation failed: {type(e).__name__}: {e}", started)
    return _report(workdir, cfg, picked, answers, settings, started)


def _not_checked(settings: dict, note: str, started: float) -> dict:
    """No scores and no samples: the reason. The samples keys are present and EMPTY, never absent:
    a validator leg wires them (``$steps.generate.eval_generate_samples_path``), and the core fails
    the pipeline on a wire that resolves to nothing; an empty one tells the validator there is
    nothing to read, and it reports ``checked: false`` in turn."""
    from operator_eval_lm import flow_stages  # noqa: PLC0415

    flat = {k: v for k, v in settings.items() if isinstance(v, (str, int, float, bool))}
    return {**flat, "checked": False, "note": note, "samples_path": "", "samples_uri": "", "samples_sha256": "",
            "elapsed_s": round(flow_stages._now() - started, 3)}


def _generate(cfg, picked, adapter: str, backend, settings: dict, started: float) -> dict:
    """Everything that touches the model: load, render, both models' answers, with the projection
    after the first batch."""
    from operator_eval_lm import flow_stages  # noqa: PLC0415

    now = flow_stages._now
    settings["versions"] = backend.versions()
    base = backend.load(cfg.base_model, quantize_4bit=cfg.method == "qlora", seed=cfg.seed)
    settings["quantization"] = base.quantization
    template, settings["chat_template_source"] = backend.chat_template(base, adapter)
    settings["adapter_base_model"] = backend.adapter_base(adapter)
    prompts = [backend.render(base, turns, template, add_generation_prompt=True) for _, turns, _ in picked]
    batches = [prompts[i:i + cfg.batch_size] for i in range(0, len(prompts), cfg.batch_size)]

    def run(model, first_done: list[str] | None = None) -> tuple[list[str], float]:
        out, model_s = list(first_done or []), 0.0
        for batch in batches[1 if first_done else 0:]:
            got, timing = backend.generate(model, batch, max_new_tokens=cfg.max_new_tokens,
                                           max_seq_len=cfg.max_seq_len, seed=cfg.seed)
            if len(got) != len(batch):
                raise RuntimeError(f"the backend returned {len(got)} answers for {len(batch)} prompts")
            out.extend(got)
            model_s += timing.model_s
        return out, model_s

    t0 = now()
    first, timing = backend.generate(base, batches[0], max_new_tokens=cfg.max_new_tokens,
                                     max_seq_len=cfg.max_seq_len, seed=cfg.seed)
    if len(first) != len(batches[0]):
        raise RuntimeError(f"the backend returned {len(first)} answers for {len(batches[0])} prompts")
    per_batch = now() - t0
    so_far = now() - started
    projected = so_far + per_batch * (len(batches) - 1) + per_batch * len(batches) * flow_stages.CANDIDATE_FACTOR
    allowed = cfg.time_budget_s * (1 - cfg.time_margin)
    settings["first_batch_s"] = round(per_batch, 3)
    settings["projected_s"] = round(projected, 1)
    if cfg.time_budget_s and projected > allowed:
        # The headline goes LAST: the remote job keeps only the tail of its log as the run's error.
        raise flow_stages.TimeBudgetExceeded(
            f"First batch of {len(batches[0])} prompts: {per_batch:.0f}s; {len(batches)} batches per model, the "
            f"candidate's priced {flow_stages.CANDIDATE_FACTOR:g}× the base's; {so_far / 60:.0f} min spent "
            f"loading and on the first batch. Lower sample_n or max_new_tokens, or give the leg a longer "
            f"timeout (and budget). {STAGE} refused before the full run: projected {projected / 3600:.2f} h > "
            f"{allowed / 3600:.2f} h (the leg's {cfg.time_budget_s / 3600:g} h less a {cfg.time_margin:.0%} margin).")
    base_answers, base_s = run(base, first)
    candidate = backend.attach(base, adapter)   # the same loaded weights, the adapter on top
    cand_answers, cand_s = run(candidate)
    return {"base": base_answers, "candidate": cand_answers,
            "base_model_s": timing.model_s + base_s, "candidate_model_s": cand_s}


def _report(workdir: Path, cfg, picked, answers: dict, settings: dict, started: float) -> dict:
    """Step 5–6: the scores, the deltas and the samples file."""
    from operator_eval_lm import box_io, flow_stages  # noqa: PLC0415

    samples, totals = [], {who: dict.fromkeys(SCORERS, 0) for who in ("base", "candidate")}
    for i, ((row, turns, reference), b, c) in enumerate(zip(picked, answers["base"], answers["candidate"])):
        sb, sc = score(b, reference), score(c, reference)
        for name in SCORERS:
            totals["base"][name] += sb[name]
            totals["candidate"][name] += sc[name]
        samples.append({"index": i, "row": row, "prompt": turns, "reference": reference, "base": b,
                        "candidate": c, "scores": {name: {"base": sb[name], "candidate": sc[name]} for name in SCORERS}})
    n = len(samples)
    stats: dict[str, Any] = {}
    for name in SCORERS:
        b, c = totals["base"][name] / n, totals["candidate"][name] / n
        stats[f"base_{name}"], stats[f"candidate_{name}"] = round(b, 6), round(c, 6)
        stats[f"delta_{name}"] = round(c - b, 6)
        # a binomial standard error per model, and of the delta as if the two were independent
        # (they are paired on the same prompts, so this over-states it)
        se_b, se_c = math.sqrt(b * (1 - b) / n), math.sqrt(c * (1 - c) / n)
        stats[f"delta_{name}_stderr"] = round(math.sqrt(se_b ** 2 + se_c ** 2), 6)

    checked = not cfg.stub
    notes = ["stub tier: no model was loaded; the answers and the scores are placeholders" if cfg.stub else ""]
    trained_on = settings.get("adapter_base_model") or ""
    if trained_on and trained_on != cfg.base_model:
        notes.append(f"the adapter records it was trained on {trained_on!r}; both models answered on "
                     f"{cfg.base_model!r} as asked")
    note = "; ".join(x for x in notes if x)
    path = workdir / SAMPLES_FILE
    with path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, sort_keys=True, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    location, upload_error = str(path), ""
    target = box_io.results_target(box_io.SAMPLES_KEY)
    if target is not None:
        try:
            location = box_io.upload_results(path, *target)
        except Exception as e:  # noqa: BLE001 — the scores are on the record; the file stays on the box
            upload_error = f"{type(e).__name__}: {e}"
    versions = settings.get("versions") or {}
    return {
        **{k: v for k, v in settings.items() if k != "versions"},
        **{(k if k == "device" else f"{k}_version"): v for k, v in versions.items() if k != "lm_eval"},
        "checked": checked, "note": note, **stats,
        "base_model_s": round(answers["base_model_s"], 3), "candidate_model_s": round(answers["candidate_model_s"], 3),
        "elapsed_s": round(flow_stages._now() - started, 3),
        # the samples on this box (a validator leg on the same box reads it first) and where they were kept
        "samples_path": str(path), "samples_uri": location, "samples_sha256": digest,
        **({"samples_upload_error": upload_error} if upload_error else {}),
        # the stage's product (the runner turns these into the registered descriptor)
        "output": location, "content_hash": digest, "kind": "report", "rows": n, "bytes": path.stat().st_size,
    }


# Paid GPU time, and it writes the samples file to the checkpoint bucket (ADR-0081 D2: fail-closed
# risk classes; the same class as eval.lm_eval and train.qlora).
eval_generate.risk_class = "external_write"

__all__ = ["SAMPLES_FILE", "SCORERS", "STAGE", "draw", "eval_generate", "score"]
