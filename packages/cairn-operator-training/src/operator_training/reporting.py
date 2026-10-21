"""What the training operators REPORT (ADR-0103 / #2025): one declaration, one
``TrainerCallback`` that emits exactly that vocabulary — for the in-process
``train/local`` (through ``ctx.report``, the control plane's typed reporter)
and for a trainer on a remote box (through :class:`operator_training.live.LiveReporter`,
the worker-side twin with the same surface).

Events are generic, not training-shaped (ADR-0102 D7): ``progress {stage,
percent, eta_s, cost_usd}`` · ``metrics {step, series}`` · ``scalars`` ·
``checkpoint`` · ``divergence``. The Activity page and the Metrics tab render
whatever an operator declared; training simply declares the most.
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Callable, Optional

from cairn.core.operator import ReportSpec, Series

from operator_training import live

_log = logging.getLogger("operator_training.reporting")

#: The stages a training run passes through, in order.
STAGES = ("download", "tokenise", "train", "eval", "export")

#: Loss/optimiser series at every ``logging_steps``, plus the `system` series the
#: box samples every 15 s. Units are display hints for the console.
SERIES = (
    Series("loss", "nats"), Series("eval_loss", "nats"), Series("learning_rate", ""),
    Series("grad_norm", ""), Series("epoch", ""), Series("tokens_per_s", "tok/s"),
    Series("gpu_util_pct", "%"), Series("gpu_mem_gb", "GiB"), Series("gpu_mem_total_gb", "GiB"),
    Series("gpu_temp_c", "°C"), Series("gpu_power_w", "W"), Series("cpu_pct", "%"), Series("ram_gb", "GB"),
)

SCALARS = ("final_loss", "best_eval_loss", "tokens_seen", "steps", "train_runtime_s")

#: The declaration every ``train/*`` operator carries (mirrored in pack.toml,
#: held honest by the manifest-sync + drift gates).
TRAIN_REPORTS = ReportSpec(stages=STAGES, series=SERIES, scalars=SCALARS)


#: The divergence guard (#2030): a non-finite loss, or a loss above
#: ``DIVERGENCE_SPIKE_FACTOR`` × the median of the last ``DIVERGENCE_WINDOW``
#: logged losses once that many exist, stops the trainer and reports
#: ``divergence``. Stop only — the retry is a PROPOSAL a human approves.
DIVERGENCE_SPIKE_FACTOR = 4.0
DIVERGENCE_WINDOW = 5


def divergence_check(loss: float, recent: list[float], *, spike_factor: float = DIVERGENCE_SPIKE_FACTOR,
                     window: int = DIVERGENCE_WINDOW) -> Optional[str]:
    """``"nan"`` for a non-finite loss, ``"spike"`` when ``loss`` exceeds the
    spike factor × the median of the last ``window`` losses (needs ``window``
    of them), else ``None``."""
    if not math.isfinite(loss):
        return "nan"
    if len(recent) >= window:
        tail = sorted(recent[-window:])
        med = tail[len(tail) // 2]
        if med > 0 and loss > spike_factor * med:
            return "spike"
    return None


def make_report_callback(reporter: Any, *, system_every_s: float = 15.0,
                         clock: Callable[[], float] = time.monotonic,
                         sample_system: Callable[[], dict[str, float]] = live.system_sample,
                         spike_factor: float = DIVERGENCE_SPIKE_FACTOR) -> Any:
    """A ``transformers.TrainerCallback`` that speaks :data:`TRAIN_REPORTS`.

    ``reporter`` is either the control plane's ``ctx.report`` (in-process) or a
    :class:`~operator_training.live.LiveReporter` (remote) — same method names.
    Control comes back through ``reporter.control()`` when it has one (the live
    reporter does; the in-process reporter reads the run's control log through
    the same helper on the control plane, see :func:`control_from_store`):

    * ``abort`` → ``control.should_training_stop``;
    * ``checkpoint-and-stop`` → ``control.should_save`` THEN
      ``should_training_stop``, so the checkpoint the trainer writes carries
      the last logged step — the acceptance the ticket names.
    """
    from transformers import TrainerCallback  # noqa: PLC0415 — optional [local]/GPU extra; only a trainer imports this

    class _Report(TrainerCallback):
        def __init__(self) -> None:
            self._t0 = clock()
            self._last_sys = -1e9
            self._last_step = 0
            self._last_tokens = 0
            self._last_tokens_t = clock()
            self._best_eval: Optional[float] = None
            self._final_loss: Optional[float] = None
            self._recent: list[float] = []
            self.stopped_by: Optional[str] = None
            self.divergence: Optional[dict[str, Any]] = None

        # ── stages ──
        def on_train_begin(self, args, state, control, **kw):  # noqa: ANN001
            _safe(reporter.stage, "train", percent=0.0)
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kw):  # noqa: ANN001
            el = (metrics or {}).get("eval_loss")
            if isinstance(el, (int, float)):
                self._best_eval = float(el) if self._best_eval is None else min(self._best_eval, float(el))
                _safe(reporter.series, int(getattr(state, "global_step", 0) or 0), eval_loss=float(el))
            _safe(reporter.stage, "eval", percent=self._percent(state))
            return control

        def on_train_end(self, args, state, control, **kw):  # noqa: ANN001
            _safe(reporter.stage, "train", percent=100.0)
            _safe(reporter.scalars, **self._final(state))
            flush = getattr(reporter, "flush", None)
            if callable(flush):
                _safe(flush, force=True)
            return control

        # ── series at every logging step ──
        def on_log(self, args, state, control, logs=None, **kw):  # noqa: ANN001
            step = int(getattr(state, "global_step", 0) or 0)
            vals: dict[str, float] = {}
            for k in ("loss", "eval_loss", "learning_rate", "grad_norm", "epoch"):
                v = (logs or {}).get(k)
                if isinstance(v, (int, float)):
                    vals[k] = float(v)
            raw_loss = (logs or {}).get("loss")
            if isinstance(raw_loss, (int, float)):
                kind = divergence_check(float(raw_loss), self._recent, spike_factor=spike_factor)
                if kind is not None and self.stopped_by is None:
                    # #2030: STOP, report, and never restart — the retry is a proposal
                    self.divergence = {"kind": kind, "step": step, "value": float(raw_loss)}
                    _safe(reporter.divergence, kind, step, value=(None if kind == "nan" else float(raw_loss)))
                    control.should_training_stop = True
                    self.stopped_by = "divergence"
                    vals.pop("loss", None)   # a NaN is not a series point
                elif math.isfinite(float(raw_loss)):
                    self._recent.append(float(raw_loss))
            if "loss" in vals:
                self._final_loss = vals["loss"]
            seen = getattr(state, "num_input_tokens_seen", None)
            now = clock()
            if isinstance(seen, (int, float)) and seen > self._last_tokens and now > self._last_tokens_t:
                vals["tokens_per_s"] = (float(seen) - self._last_tokens) / (now - self._last_tokens_t)
                self._last_tokens, self._last_tokens_t = int(seen), now
            if vals:
                _safe(reporter.series, step, **vals)
            self._last_step = step
            pct = self._percent(state)
            if pct is not None:
                _safe(reporter.stage, "train", percent=pct, eta_s=self._eta(state, pct))
            self._system(now, step)
            return control

        def on_step_end(self, args, state, control, **kw):  # noqa: ANN001
            self._last_step = int(getattr(state, "global_step", 0) or 0)
            self._system(clock(), self._last_step)
            action = _control(reporter)
            if action == live.CHECKPOINT_AND_STOP:
                control.should_save = True
                control.should_training_stop = True
                self.stopped_by = action
            elif action == live.ABORT:
                control.should_training_stop = True
                self.stopped_by = action
            return control

        def on_save(self, args, state, control, **kw):  # noqa: ANN001
            step = int(getattr(state, "global_step", 0) or 0)
            out = getattr(args, "output_dir", "") or ""
            _safe(reporter.checkpoint, f"{out}/checkpoint-{step}" if out else f"checkpoint-{step}", step,
                  metric=self._best_eval)
            return control

        # ── helpers ──
        def _percent(self, state) -> Optional[float]:  # noqa: ANN001
            mx = int(getattr(state, "max_steps", 0) or 0)
            st = int(getattr(state, "global_step", 0) or 0)
            return (100.0 * st / mx) if mx > 0 else None

        def _eta(self, state, pct: float) -> Optional[float]:  # noqa: ANN001
            if pct <= 0:
                return None
            elapsed = clock() - self._t0
            return elapsed * (100.0 - pct) / pct

        def _system(self, now: float, step: int) -> None:
            if now - self._last_sys < system_every_s:
                return
            self._last_sys = now
            try:
                sample = sample_system()
            except Exception:  # noqa: BLE001 — a failed probe is a skipped sample
                sample = {}
            if sample:
                _safe(reporter.series, step, **sample)

        def _final(self, state) -> dict[str, Any]:  # noqa: ANN001
            out: dict[str, Any] = {"steps": int(getattr(state, "global_step", 0) or 0),
                                   "train_runtime_s": round(clock() - self._t0, 3)}
            if self._final_loss is not None:
                out["final_loss"] = self._final_loss
            if self._best_eval is not None:
                out["best_eval_loss"] = self._best_eval
            seen = getattr(state, "num_input_tokens_seen", None)
            if isinstance(seen, (int, float)):
                out["tokens_seen"] = int(seen)
            return out

    return _Report()


def _safe(fn: Callable[..., Any], *a: Any, **kw: Any) -> None:
    try:
        fn(*a, **kw)
    except Exception:  # noqa: BLE001 — a report never fails the train
        _log.debug("report call failed", exc_info=True)


def _control(reporter: Any) -> Optional[str]:
    ctl = getattr(reporter, "control", None)
    if not callable(ctl):
        return None
    try:
        return ctl()
    except Exception:  # noqa: BLE001 — same
        return None


def reporter_for(spec_or_env: Any = None) -> Any:
    """The reporter for a container entrypoint: the ``_live_env`` a submit tucked
    into the spec, or the process environment."""
    env = None
    if isinstance(spec_or_env, dict):
        env = spec_or_env.get("_live_env")
    if isinstance(env, dict):
        return live.reporter_from_env({str(k): str(v) for k, v in env.items()})
    return live.reporter_from_env(os.environ)


__all__ = ["SCALARS", "SERIES", "STAGES", "TRAIN_REPORTS", "make_report_callback", "reporter_for"]
