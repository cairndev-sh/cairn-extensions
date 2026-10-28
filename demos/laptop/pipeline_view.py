#!/usr/bin/env python3
"""demos/laptop/pipeline_view.py — read a pipeline from the running control plane for run.sh.

    pipeline_view.py state    <pipeline-id>   # STATUS|gates waiting|failed steps|progress
    pipeline_view.py card     <pipeline-id> <step>   # what the person at that gate is shown
    pipeline_view.py versions <pipeline-id>   # train=… val=… test=… from a dataset-prep run
    pipeline_view.py summary  <pipeline-id>   # the result of an sft-local run
    pipeline_view.py trail    <pipeline-id>   # who decided each human gate, and when

Stdlib only; it shells out to `cairn remote …` (the same calls you can make by hand) and prints.
"""
from __future__ import annotations

import json
import subprocess
import sys


def _cairn(*args: str) -> dict:
    out = subprocess.run(["cairn", "remote", *args], check=True, capture_output=True, text=True).stdout
    return json.loads(out[out.index("{"):])


def _pipeline(pid: str) -> dict:
    return _cairn("pipeline", pid)


def _findings(pid: str) -> dict:
    """step id -> its findings (the leg's measured values) from `cairn remote findings`."""
    steps = _cairn("findings", pid, "--json").get("steps") or {}
    return {step: (rec.get("findings") or (rec.get("output") or {}).get("findings") or {}) for step, rec in steps.items()}


def _resolve(ref: str, record: dict, findings: dict):
    """A `show:` reference: `input.<name>` or `<step>.<key>`."""
    head, _, key = str(ref).partition(".")
    if head == "input":
        return (record.get("inputs") or {}).get(key)
    return (findings.get(head) or {}).get(key)


def _fmt(v) -> str:
    s = f"{v:.4f}" if isinstance(v, float) else str(v)
    return s if len(s) <= 100 else s[:97] + "..."


def state(pid: str) -> None:
    rec = _pipeline(pid)
    steps = rec.get("step_state") or {}
    waiting = [s for s, v in steps.items() if v.get("status") == "awaiting_approval"]
    failed = [s for s, v in steps.items() if v.get("status") in ("failed", "error", "timed_out")]
    progress = " ".join(f"{s}:{v.get('status')}" for s, v in steps.items())
    print(f"{rec.get('status')}|{' '.join(waiting)}|{' '.join(failed)}|{progress}")


def card(pid: str, step: str) -> None:
    rec = _pipeline(pid)
    findings = _findings(pid)
    plan = {s["id"]: s for s in (rec.get("plan") or {}).get("steps") or []}.get(step) or {}
    print(f"  gate {step} (approvers: {plan.get('approvers') or (rec.get('governance') or {}).get('approvers') or '-'})")
    if plan.get("injected_by"):
        print(f"  injected by policy: {plan['injected_by']}")
    shown = [(r, _resolve(r, rec, findings)) for r in plan.get("show") or []]
    shown = [(r, v) for r, v in shown if v not in (None, "", [])]
    if not shown and plan.get("ref", "").startswith("task:"):
        for k, ref in (plan.get("inputs") or {}).items():
            val = ref
            if isinstance(ref, str) and ref.startswith("$input."):
                val = (rec.get("inputs") or {}).get(ref[len("$input."):])
            elif isinstance(ref, str) and ref.startswith("$steps."):
                src, _, key = ref[len("$steps."):].partition(".")
                val = (findings.get(src) or {}).get(key)
            shown.append((k, val))
        print(f"  runs {plan['ref']}")
    width = max((len(r) for r, _ in shown), default=0)
    for ref, val in shown:
        print(f"    {ref:<{width}}  {_fmt(val)}")


def versions(pid: str) -> None:
    got = _findings(pid).get("prepare", {}).get("dataset_versions") or []
    roles = {"train": "", "val": "", "test": ""}
    for v in got:
        for role in roles:
            if v.split("@")[0].endswith(f"-{role}"):
                roles[role] = f"dataset:{v}"
    if not all(roles.values()):
        sys.exit(f"pipeline_view: the prepare run registered {got!r}, not train/val/test versions")
    for role, ref in roles.items():
        print(f"{role}={ref}")


def summary(pid: str) -> None:
    rec = _pipeline(pid)
    f = _findings(pid)
    t, e, p = f.get("train", {}), f.get("eval", {}), f.get("promote", {})
    rows = [
        ("pipeline", f"{pid}  {rec.get('status')}"),
        ("base model", (rec.get("inputs") or {}).get("base_model")),
        ("optimizer steps", t.get("train_qlora_steps")),
        ("validation loss, base model", t.get("train_qlora_baseline_eval_loss")),
        ("validation loss, candidate", t.get("train_qlora_eval_loss")),
        ("held-out val loss  base -> candidate", f"{_fmt(e.get('eval_lm_eval_base_val_loss'))} -> {_fmt(e.get('eval_lm_eval_candidate_val_loss'))}"),
        ("held-out test loss base -> candidate", f"{_fmt(e.get('eval_lm_eval_base_test_loss'))} -> {_fmt(e.get('eval_lm_eval_candidate_test_loss'))}"),
        ("eval device", e.get("eval_lm_eval_device")),
        ("promoted", f"{p.get('tag')} -> {str(p.get('artifact_id'))[:16]}…" if p.get("promoted") else "no"),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k:<{width}}  {_fmt(v)}")


def trail(pid: str) -> None:
    steps = _pipeline(pid).get("step_state") or {}
    for step, st in steps.items():
        if st.get("approver"):
            waited = (st.get("awaiting_since") or "")[:19]
            decided = (st.get("started_at") or st.get("ended_at") or "")[:19]
            print(f"  {pid}  {step:<14} approved by {st['approver']:<6} asked {waited}  decided {decided}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["--help"]
    if cmd not in {"state", "card", "versions", "summary", "trail"}:
        sys.exit(__doc__)
    {"state": state, "card": card, "versions": versions, "summary": summary, "trail": trail}[cmd](*args)
