"""ADR-0080 P4 (#1032, D6) — dataset/sample bounds eval_samples so it can never
blow the workflow-history payload limit: a hard sample cap + a per-field char clip."""
from __future__ import annotations

import asyncio
import json

from operator_dataset_checks.operators.sample import _MAX_PAIR_CHARS, _MAX_SAMPLES, SampleOp
from operator_dataset_checks.schemas.sample import SampleInput

# The ADR-0080 D6 / R5 budget for one state.findings value: it must stay well under
# Temporal's ~2 MB workflow-history payload limit, and a long list of dict rows is
# row data that belongs behind a storage URI, not in run state.
_PER_KEY_BUDGET = 524_288  # 512 KiB
_ROW_LIST_LEN = 200


class _Ctx:
    pass


def _write(tmp_path, rows):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def test_sample_caps_count(tmp_path):
    rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(5000)]
    src = _write(tmp_path, rows)
    out = asyncio.run(SampleOp().run(SampleInput(path=src, n=100000), _Ctx()))
    assert out.sampled <= _MAX_SAMPLES


def test_sample_clips_long_text(tmp_path):
    rows = [{"instruction": "x" * 20000, "output": "y" * 20000}]
    src = _write(tmp_path, rows)
    out = asyncio.run(SampleOp().run(SampleInput(path=src, n=1), _Ctx()))
    assert len(out.samples[0].input) <= _MAX_PAIR_CHARS + 16
    assert out.samples[0].input.endswith("[clipped]")


def test_capped_samples_are_payload_safe(tmp_path):
    # worst case: max pairs, each near the char cap → still under the D6 budget.
    rows = [{"instruction": "x" * 5000, "output": "y" * 5000} for _ in range(2000)]
    src = _write(tmp_path, rows)
    out = asyncio.run(SampleOp().run(SampleInput(path=src, n=100000), _Ctx()))
    eval_samples = [s.model_dump() for s in out.samples]
    size = len(json.dumps(eval_samples, default=str).encode("utf-8"))
    assert size <= _PER_KEY_BUDGET, f"eval_samples is {size:,} B > {_PER_KEY_BUDGET:,} B"
    assert len(eval_samples) <= _ROW_LIST_LEN, "eval_samples is shaped like row data"
