"""train/stub — deterministic fine-tuning provider (capability train.run).

The default + test/dev backend: no GPU, no spend, no network. Returns a
plausible TrainResult so the durable pipeline (dataset → train → promote) runs
and is testable end-to-end without any backend. For a real run use train/local
(CPU/MPS LoRA) or the sft-train flow's train.qlora stage on a GPU target.
"""
from __future__ import annotations

import hashlib
import json

from cairn.core.operator import Operation, OperationContext, register_operator

from operator_training.reporting import TRAIN_REPORTS
from operator_training.schemas.train import TrainResult, TrainSpec


@register_operator(name="train/stub", version="0.1.0", capabilities=["train.run"],
                   implements=["contract://train.run/1.0.0"])  # #210
class StubTrainOp(Operation[TrainSpec, TrainResult]):
    input_schema = TrainSpec
    reports = TRAIN_REPORTS  # ADR-0103 / #2025
    output_schema = TrainResult
    supports_mock = True

    @staticmethod
    def _adapter_uri(inp: TrainSpec) -> str:
        # Reflect the artifact target so the downstream promote stage gets a
        # realistic URI (the stub doesn't actually push anything).
        if inp.artifact_target == "hf":
            return f"hf://{inp.hf_repo or inp.output_name}"
        return f"stub://adapters/{inp.output_name}"

    @staticmethod
    def _real(inp: TrainSpec) -> "TrainResult | None":
        # mock-real (#126 e2e): if the caller supplies a real adapter + metrics
        # via `extra`, return them verbatim — lets a governed run exercise the
        # whole downstream (HF/S3/registry/lineage) on a GENUINE artifact with no
        # live GPU. The GPU is the only thing skipped.
        uri = inp.extra.get("mock_adapter_uri")
        if not uri:
            return None
        return TrainResult(
            status="succeeded", backend="stub", adapter_uri=str(uri),
            job_id=f"stub-{inp.output_name}",
            metrics={**(inp.extra.get("mock_metrics") or {}), "mock_real": True},
        )

    @staticmethod
    def _curve(inp: TrainSpec) -> list[float]:
        """The eval-loss curve this stub reports — #2029's ``stub_curve``.

        A ranking round needs trials that DIFFER, and a stub that returns 0.0
        for every configuration makes the halving unfalsifiable: every trial
        ties, so the count of pruned trials says nothing. Two forms:

        * ``extra.stub_curve: [0.9, 0.5, 0.3]`` — the exact curve, for a test
          that wants to name the number;
        * ``extra.stub_curve: "auto"`` — a curve DERIVED from this spec's own
          knobs, so two configurations differ and the same configuration
          reproduces its own curve exactly. It is a hash, not a model: the stub
          claims nothing about which hyperparameters are actually better.

        Anything else keeps the historical flat-zero behaviour.
        """
        raw = inp.extra.get("stub_curve")
        if isinstance(raw, (list, tuple)):
            return [float(v) for v in raw if isinstance(v, (int, float))]
        if not raw:
            return []
        knobs = json.dumps(
            {k: getattr(inp, k, None) for k in
             ("learning_rate", "epochs", "lora_r", "lora_alpha", "max_seq_len", "method")},
            sort_keys=True, default=str)
        seed = int(hashlib.sha256(knobs.encode("utf-8")).hexdigest()[:8], 16)
        base = 0.4 + (seed % 1000) / 1000.0            # 0.400 … 1.399, stable per config
        return [round(base * (1.0 + 1.0 / (i + 1)), 4) for i in range(3)]

    async def run(self, inp: TrainSpec, ctx: OperationContext) -> TrainResult:
        real = self._real(inp)
        if real is not None:
            return real
        curve = self._curve(inp)
        self._report_curve(inp, ctx, curve)
        return TrainResult(
            status="succeeded",
            backend="stub",
            adapter_uri=self._adapter_uri(inp),
            job_id=f"stub-{inp.output_name}",
            metrics={
                "train_loss": curve[-1] if curve else 0.0,
                "eval_loss": curve[-1] if curve else 0.0,
                # [[step, value], …] — T15's `curve.eval_losses` shape, so a
                # caller that CAN carry a list (a task leg, a test) reports the
                # whole curve to the pruner rather than only its last point.
                "eval_curve": [[i + 1, v] for i, v in enumerate(curve)],
                "steps": inp.max_steps or 0,
                "method": inp.method,
                "base_model": inp.base_model,
                "stub": True,
            },
        )

    @staticmethod
    def _report_curve(inp: TrainSpec, ctx: OperationContext, curve: "list[float] | None" = None) -> None:
        """#2025: the stub speaks the same declared vocabulary as a real train —
        stages, a handful of `loss` points, a checkpoint, the final scalars — so
        an e2e pipeline (stub=true) shows a live curve and the Metrics tab
        renders from real events, not fixtures. Never fails the stub."""
        try:
            steps = max(1, min(int(inp.max_steps or 0) or 5, 5))
            rep = ctx.report
            rep.stage("download")
            rep.stage("tokenise")
            rep.stage("train", percent=0.0)
            for i in range(1, steps + 1):
                rep.series(i, loss=round(2.0 / i, 4), learning_rate=inp.learning_rate)
                rep.stage("train", percent=100.0 * i / steps)
            # #2029 — when a `stub_curve` is set, the eval_loss series is the
            # curve the pruner will be shown, emitted as real `metrics` events
            # so T15's curve summary and the console's Series overlay see the
            # same numbers the ranking round did.
            for i, value in enumerate(curve or (), start=1):
                rep.series(i, eval_loss=value)
            rep.stage("eval", percent=100.0)
            best = min(curve) if curve else 0.0
            rep.checkpoint(f"stub://checkpoints/{inp.output_name}/checkpoint-{steps}", steps, metric=best)
            rep.stage("export")
            rep.scalars(final_loss=(curve[-1] if curve else round(2.0 / steps, 4)),
                        best_eval_loss=best, steps=steps, tokens_seen=0)
        except Exception:  # noqa: BLE001 — a report never fails the stub
            pass

    def mock(self, inp: TrainSpec, ctx: OperationContext) -> TrainResult:
        return TrainResult(
            status="succeeded", backend="stub",
            adapter_uri=self._adapter_uri(inp), metrics={"stub": True},
        )
