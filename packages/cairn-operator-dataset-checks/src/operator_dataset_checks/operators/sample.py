"""dataset/sample — deterministic (input, output) sampling for eval.

Loads the dataset, extracts (instruction, response) pairs across the common
formats, and returns an evenly-spaced sample (deterministic: same dataset +
n → same pairs). This is the deterministic feeder for the non-deterministic
eval/judge soft signal (ADR-0031) — kept in dataset_checks because the
format-aware extraction is dataset knowledge; the judge itself is generic.
"""
from __future__ import annotations

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks._extract import extract_io
from operator_dataset_checks._io import load_rows
from operator_dataset_checks.schemas.sample import (
    SampleConfig,
    SampleInput,
    SampleOutput,
    SamplePair,
)

# ADR-0080 P4 (#1032, D6): eval_samples is the ONE readiness finding that carries
# row TEXT into state.findings — a soft-signal feeder for eval/judge. Bound it so
# it can never blow the ~2 MB workflow-history payload limit: a hard sample cap and
# a per-field character cap keep it small and deterministic. (The judge only needs
# a representative sample; the full pairs live in the dataset, not in run state.)
# bounded so worst-case eval_samples (_MAX_SAMPLES × 2 fields × _MAX_PAIR_CHARS ≈
# 240 KB) stays under the D6 per-key budget (512 KiB) and the row-list-shape
# threshold (200) — proven by tests/test_sample_payload_cap.py.
_MAX_SAMPLES = 150
_MAX_PAIR_CHARS = 800


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_PAIR_CHARS else text[:_MAX_PAIR_CHARS] + "…[clipped]"


@register_operator(
    name="dataset/sample",
    version="0.1.0",
    capabilities=["dataset.sample"],
    implements=["contract://dataset.sample/1.0.0"],  # #2242 V1-B — cairn-contracts-dataset
)
class SampleOp(Operation[SampleInput, SampleOutput]):
    """Evenly-spaced, deterministic (input, output) sample for judging."""

    input_schema = SampleInput
    output_schema = SampleOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    config_schema = SampleConfig

    async def run(self, inp: SampleInput, ctx: OperationContext) -> SampleOutput:
        rows = load_rows(inp.path)
        pairs = [p for p in (extract_io(r) for r in rows) if p is not None]

        # bound n so a large eval_sample_size can't dump thousands of pairs into
        # state.findings (D6 payload safety, #1032).
        n = min(max(0, inp.n), _MAX_SAMPLES)
        if n and len(pairs) > n:
            stride = len(pairs) / n  # evenly spaced across the dataset
            picked = [pairs[int(k * stride)] for k in range(n)]
        else:
            picked = pairs[:_MAX_SAMPLES]

        samples = [SamplePair(input=_clip(i), output=_clip(o)) for i, o in picked]
        return SampleOutput(
            samples=samples,
            total_rows=len(rows),
            extractable=len(pairs),
            sampled=len(samples),
        )
