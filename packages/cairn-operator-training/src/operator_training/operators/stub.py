"""train/stub — deterministic fine-tuning provider (capability train.run).

The default + test/dev backend: no GPU, no spend, no network. Returns a
plausible TrainResult so the durable pipeline (dataset → train → promote) runs
and is testable end-to-end without any backend. For a real run use train/local
(CPU/MPS LoRA) or the sft-train flow's train.qlora stage on a GPU target.
"""
from __future__ import annotations

import hashlib
import json
import os

from cairn.core.operator import Operation, OperationContext, register_operator

from operator_training import checkpoint_sync
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
        resume = self._checkpointed_steps(inp, ctx)
        self._report_curve(inp, ctx, curve, first_step=int(resume.get("resumed_from_step", 0)) + 1)
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
                # cairn-extensions#7 — the curve's first point stands in for the base model.
                **({"baseline_eval_loss": curve[0] if curve else 0.0} if inp.baseline_eval else {}),
                # cairn-extensions#28 — the same keys train_entry reports, with a checkpoint bucket
                **{k: v for k, v in resume.items() if k in ("resumed_from_checkpoint", "checkpoint_uri")},
            },
        )

    @staticmethod
    def _checkpointed_steps(inp: TrainSpec, ctx: OperationContext) -> dict:
        """cairn-extensions#28: with ``CAIRN_CHECKPOINT_BUCKET`` set, the stub keeps checkpoints
        the way the real trainer does, so a retry of the same run can be proven to resume.

        It pulls the run's checkpoints (:func:`checkpoint_sync.pull_checkpoints_if_any`, keyed
        by the run), continues from the newest one's ``global_step``, and writes
        ``checkpoint-<step>/trainer_state.json`` every ``save_steps`` up to ``max_steps``, each
        mirrored to ``s3://<bucket>/<prefix>/<run>/checkpoint-<step>/`` before the next step.
        A no-op (``{}``) without a bucket, a workdir, ``max_steps`` or ``save_steps``: the stub
        then behaves exactly as before. It trains nothing; the files hold only the step."""
        workdir = getattr(ctx, "workdir", None)
        steps, every = int(inp.max_steps or 0), int(inp.save_steps or 0)
        target = checkpoint_sync.s3_target_from_env(output_name=inp.output_name)
        if target is None or not workdir or steps <= 0 or every <= 0:
            return {}
        bucket, prefix, run_key = target
        out_dir = checkpoint_sync.trainer_output_dir(inp.output_name, workdir)
        os.makedirs(out_dir, exist_ok=True)
        resumed = checkpoint_sync.pull_checkpoints_if_any(out_dir)
        start = _global_step(resumed) if resumed else 0
        for step in range(start + 1, steps + 1):
            if step % every and step != steps:
                continue
            ckpt = os.path.join(out_dir, f"checkpoint-{step}")
            os.makedirs(ckpt, exist_ok=True)
            with open(os.path.join(ckpt, "trainer_state.json"), "w", encoding="utf-8") as fh:
                json.dump({"global_step": step, "max_steps": steps, "stub": True}, fh)
            checkpoint_sync.upload_dir(ckpt, bucket, f"{prefix}/{run_key}/checkpoint-{step}")
        out = {"resumed_from_step": start,
               "checkpoint_uri": checkpoint_sync.checkpoint_uri_for(bucket, prefix, run_key)}
        if resumed:
            out["resumed_from_checkpoint"] = str(resumed)
        return out

    @staticmethod
    def _report_curve(inp: TrainSpec, ctx: OperationContext, curve: "list[float] | None" = None,
                      first_step: int = 1) -> None:
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
            for i in range(min(max(1, first_step), steps), steps + 1):  # a resumed run continues
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


def _global_step(checkpoint_dir: str) -> int:
    """The step a checkpoint was saved at: its ``trainer_state.json`` ``global_step`` (what the
    real trainer's checkpoints carry), else the ``checkpoint-<step>`` directory name."""
    try:
        with open(os.path.join(checkpoint_dir, "trainer_state.json"), encoding="utf-8") as fh:
            return int(json.load(fh)["global_step"])
    except (OSError, ValueError, KeyError, TypeError):
        try:
            return int(os.path.basename(checkpoint_dir.rstrip("/")).removeprefix("checkpoint-"))
        except ValueError:
            return 0
