"""train/local — a REAL LoRA SFT trainer that runs on CPU / Apple-MPS (no GPU).

The honest "GPU mock": instead of returning a canned adapter, it actually trains —
loads the model, attaches LoRA, runs real forward/backward, emits a real loss curve,
and saves a real ``adapter_model.safetensors`` — just small enough (tiny model +
a few steps) to run anywhere in seconds. Right for e2e / CI: it exercises the real
``transformers + peft + trl`` training code path and produces genuine metrics.

NOT unsloth (CUDA-only for training) and NO bitsandbytes 4-bit (CUDA-only) — plain
LoRA in fp32 so it runs on CPU/MPS. The stack is the optional ``[local]`` extra.
Device: MPS (Apple GPU) if available, else CPU. The adapter lands at file://<dir>.
"""
from __future__ import annotations

from typing import Any

import json
import os

from cairn.core.operator import Operation, OperationContext, register_operator

from operator_training import checkpoint_sync, dataset_fetch
from operator_training.env_fingerprint import trainer_env
from operator_training.reporting import TRAIN_REPORTS
from operator_training.schemas.train import TrainResult, TrainSpec


def _base_metrics(stats, device: str, steps: int) -> dict:
    """The plain-LoRA path's metrics, pure over the trainer's ``TrainOutput``.

    Kept out of ``_train_local`` so the metrics contract — in particular the
    ``trainer_env`` fingerprint (#1622), which is what makes a CPU/MPS run's
    numbers reproducible — is testable without the ``[local]`` torch stack.
    """
    return {
        "train_loss": float(getattr(stats, "training_loss", 0.0) or 0.0),
        "steps": int(getattr(stats, "global_step", steps) or steps),
        "device": device, "local": True,
        "trainer_env": trainer_env(),
    }


def _require_local():
    # optional dependency: the [local] extra (torch/transformers/peft/trl). Lazy so
    # `import operator_training` works without the heavy training stack installed.
    try:
        import datasets  # noqa: F401, PLC0415
        import peft  # noqa: F401, PLC0415
        import torch  # noqa: F401, PLC0415
        import transformers  # noqa: F401, PLC0415
        import trl  # noqa: F401, PLC0415
    except ImportError as e:  # pragma: no cover - exercised by the no-deps test
        raise RuntimeError(
            "train/local needs the local training stack: "
            "pip install 'cairn-operator-training[local]' "
            "(torch transformers peft trl datasets accelerate)"
        ) from e


def _device() -> str:
    import torch  # noqa: PLC0415 - optional dep (see _require_local)

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# A ~1M-param random Llama that loads + trains in seconds anywhere — used in
# validation mode so the pipeline runs without downloading/loading a real base.
_VALIDATE_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _validating(spec: TrainSpec) -> bool:
    """Pipeline-validation mode (#536): explicit ``spec.validate_pipeline`` OR the
    ``CAIRN_TRAIN_VALIDATE`` env (set on a CPU box → every run validates, no
    request change)."""
    if getattr(spec, "validate_pipeline", False):
        return True
    return os.environ.get("CAIRN_TRAIN_VALIDATE", "").strip().lower() in ("1", "true", "yes", "on")


def _validate_params(spec: TrainSpec) -> tuple[str, int, float]:
    """Resolve ``(model_id, steps, throttle_seconds)`` for a run — pure, so the
    validation policy is unit-testable without the torch stack.

    Real run → the requested model + its step budget, no throttle. Validation →
    a TINY model + a modest step budget + a per-step throttle so
    ``CAIRN_TRAIN_VALIDATE_SECONDS`` (e.g. 600) stretches the tiny-model job to a
    real 10–20 min wall-time — long enough for the remote-job worker/poll,
    cancel, resume, and the MLflow metric stream to be exercised on the box."""
    if not _validating(spec):
        return spec.base_model, (spec.max_steps or 5), 0.0
    steps = spec.max_steps or 30
    target = float(os.environ.get("CAIRN_TRAIN_VALIDATE_SECONDS", "0") or 0)
    return _VALIDATE_MODEL, steps, max(0.0, target / max(1, steps))


def _to_text_fn(tok):
    """Map the common instruct schema (instruction/input/output) → a single text
    field. Uses the tokenizer's chat template when present; falls back to a plain
    concat for tiny test models that ship no chat template."""

    def to_text(row):
        instr = (row.get("instruction") or "").strip()
        extra = (row.get("input") or "").strip()
        user = (instr + "\n\n" + extra).strip() if extra else (instr or row.get("prompt", ""))
        out = row.get("output") or row.get("response") or row.get("completion") or ""
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": out}]
        try:
            return {"text": tok.apply_chat_template(msgs, tokenize=False)}
        except Exception:  # noqa: BLE001 - tiny models often have no chat template
            return {"text": f"{user}\n{out}".strip()}

    return to_text


def _use_gpu_unsloth(spec: TrainSpec) -> bool:
    """Route a REAL run on a CUDA GPU to the unsloth QLoRA engine (train_entry) — the
    same entry a trainer image runs. Validation mode + non-GPU boxes fall
    through to the CPU/MPS plain-LoRA path below. Guarded on an ACTUAL CUDA device so
    hermetic CI (no GPU) always takes the CPU path."""
    if _validating(spec):
        return False
    try:
        import torch  # noqa: PLC0415 — base dep; a cheap cuda probe
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — no torch / probe failure ⇒ CPU path
        return False


def _train_gpu_unsloth(spec: TrainSpec, reporter: Any = None) -> TrainResult:
    """Hand a real GPU run to the unsloth engine (``train_entry.main``) and wrap its
    result as a ``TrainResult``. ``adapter_uri`` is forced to the LOCAL dir
    ``train_entry`` always writes (``save_pretrained`` → ``/tmp/<output_name>``) so the
    SSH governed path can SFTP-fetch the adapter back regardless of any HF/S3 push the
    spec also requested. unsloth is a GPU-only optional dep, imported lazily inside
    ``train_entry`` — a GPU box without it fails loud ('No module named unsloth'),
    which is the honest signal to install it, not a silent CPU fallback.

    NOTE (#553 gap): this path publishes metrics only at completion — it does not
    stream a live loss curve into metrics.json the way the CPU path does (train_entry
    owns its own trainer). The poll still tracks liveness via the pid; live-progress
    streaming on the GPU path is a follow-up."""
    from operator_training.train_entry import main as _gpu_main  # noqa: PLC0415 — GPU/unsloth engine

    # #732 — force artifact_target=local so train_entry does NOT push to HF/S3 here;
    # on the SSH path the single publish point is local_entry (which honors the real
    # artifact_target after training), so this avoids a double-push. train_entry run as its
    # own entrypoint still pushes with the real target.
    result = _gpu_main({**spec.model_dump(), "artifact_target": "local"}, reporter=reporter)
    local_dir = f"/tmp/{spec.output_name}"  # train_entry.main's save_pretrained target
    return TrainResult(
        status="succeeded",
        backend="unsloth-gpu",
        adapter_uri=local_dir,
        metrics=result.get("metrics", {}) or {},
    )


def _stage(reporter: Any, name: str) -> None:
    """Best-effort stage mark on whichever reporter this run has."""
    try:
        if reporter is not None:
            reporter.stage(name)
    except Exception:  # noqa: BLE001 — a report never fails the train
        pass


async def _train_local(spec: TrainSpec, *, progress_path: str | None = None,
                       reporter: Any = None) -> TrainResult:
    """Run a real LoRA SFT and return real metrics + adapter. On a CUDA GPU this hands
    off to the unsloth 4-bit QLoRA engine (#6); on CPU/MPS (incl. validation mode) it
    runs the hermetic plain-LoRA path below.

    ``reporter`` (#2025) is the run's typed reporter — ``ctx.report`` in-process,
    a :class:`operator_training.live.LiveReporter` on a box — and receives the
    stages, the loss/eval curve, checkpoints and the final scalars through
    :func:`operator_training.reporting.make_report_callback`."""
    if _use_gpu_unsloth(spec):
        return _train_gpu_unsloth(spec, reporter)
    _require_local()
    import inspect  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    device = _device()
    out_dir = f"/tmp/{spec.output_name}"
    _stage(reporter, "download")
    os.makedirs(out_dir, exist_ok=True)
    # ADR-0064 §1/§2: pull any durable S3 checkpoint down BEFORE training starts,
    # so a relaunch on a fresh box resumes via resume_from_checkpoint. No-op if no
    # checkpoint bucket is configured or out_dir already has local checkpoints.
    checkpoint_sync.pull_checkpoints_if_any(out_dir)
    validating = _validating(spec)
    # Validation mode swaps in a tiny model + throttle (see _validate_params): it
    # proves the WHOLE pipeline (real dataset parsing + prompt formatting + trainer
    # + metrics + gates) with no GPU. The requested base_model is IGNORED on
    # purpose — the point is the plumbing, not training the real model on a CPU.
    model_id, steps, throttle_s = _validate_params(spec)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=spec.lora_r, lora_alpha=spec.lora_alpha,
        lora_dropout=0.05 if spec.lora_dropout is None else spec.lora_dropout, task_type="CAUSAL_LM",
    ))

    def _load(uri):
        # #701 — pull an s3:// prepared dataset down (or normalize file://) so the
        # remote worker trains on it; HF ids / local paths pass through unchanged.
        uri = dataset_fetch.resolve_dataset_uri(uri)
        if os.path.exists(uri) or uri.endswith((".json", ".jsonl")):
            return load_dataset("json", data_files=uri, split="train")
        return load_dataset(uri, split="train")

    # cairn-extensions#4: train on `train_uri` when set (else `dataset_uri`), evaluate on a
    # governed `eval_uri` when set (else a seeded hold-out); `test_uri` is never opened here.
    from operator_training import splits  # noqa: PLC0415 - stdlib helper

    def _bounded(d):
        # keep it small + bounded for CPU/MPS — enough rows to cover the step budget.
        return d.select(range(min(len(d), max(16, steps * 8))))

    train_ds, eval_ds, _eval_source = splits.train_and_eval(
        spec, lambda uri: _load(uri).map(_to_text_fn(tok)), seed=spec.seed,
        can_split=lambda d: len(d) > 4, prepare_train=_bounded)
    _batch = splits.batch_args(spec, per_device=1, accumulation=1)

    # Trackers (mlflow/wandb) — honor the spec so a validation run streams real
    # metrics to MLflow over its duration (creds bridged by #532). Empty (the
    # default) keeps the historical no-tracking behavior for plain local runs.
    report_to = list(spec.report_to or [])
    tracking_error = ""
    if "mlflow" in report_to:
        # #1709 — same guard as the GPU entry: an unreachable sink is dropped
        # (reason kept as a soft metric), never allowed to fail the run at the
        # callback's on_train_begin.
        from operator_training.train_entry import _preflight_mlflow  # noqa: PLC0415 - lazy: keep local tier import-light

        tracking_error = _preflight_mlflow()
        if tracking_error:
            report_to = [t for t in report_to if t != "mlflow"]
    # Checkpoints — in validation mode we save periodically so the remote-job
    # cancel/resume path has real checkpoints to resume_from.
    save_kwargs: dict = {"save_strategy": "no"}
    if validating and steps > 1:
        save_kwargs = {"save_strategy": "steps", "save_steps": max(1, min(spec.save_steps or 10, steps // 2))}

    params = set(inspect.signature(SFTConfig.__init__).parameters)
    cfg = dict(
        output_dir=out_dir, seed=spec.seed, max_steps=steps,
        learning_rate=spec.learning_rate,
        **_batch,
        logging_steps=1, report_to=report_to,
        use_cpu=(device == "cpu"),
        packing=spec.packing,
        save_total_limit=spec.save_total_limit,
        **save_kwargs,
    )
    if "dataset_text_field" in params:
        cfg["dataset_text_field"] = "text"
    if spec.gradient_checkpointing is not None:
        cfg["gradient_checkpointing"] = bool(spec.gradient_checkpointing)
    for k in ("max_seq_length", "max_length"):  # trl renamed this across versions
        if k in params:
            cfg[k] = 128 if validating else min(spec.max_seq_len, 512)
            break
    # #553: evaluate periodically when there's a held-out split, so the progress
    # callback below builds an eval_losses trajectory WHILE training — that's what
    # the learning-gate early-abort (ADR-0039) reads mid-run over remote/exec.
    if eval_ds is not None:
        es = max(1, min(spec.eval_steps or max(1, steps // 2), steps))
        for k in ("eval_strategy", "evaluation_strategy"):  # transformers renamed this
            if k in params:
                cfg[k] = "steps"
                cfg["eval_steps"] = es
                break
    args = SFTConfig(**{k: v for k, v in cfg.items() if k in params})

    # Throttle: stretch a tiny-model job to a real wall-time so the box's worker
    # poll, cancel, and resume paths have time to act (validation mode only).
    callbacks = []
    if throttle_s > 0:
        import time  # noqa: PLC0415

        class _Throttle(TrainerCallback):
            def on_step_end(self, args, state, control, **kw):  # noqa: ANN001
                time.sleep(throttle_s)

        callbacks.append(_Throttle())
    # #553: write metrics.json incrementally into the (remote) workdir so
    # remote/exec's poll surfaces live progress + the eval_losses trajectory WHILE
    # the job runs — the console shows it and the learning-gate early-abort reads
    # it. Only when the caller wires progress_path (cairn-train-local does; the
    # in-process train/local operator leaves it None). Best-effort: a write error
    # never fails the train.
    if progress_path:
        _eval_losses: list[float] = []

        def _dump(state, logs=None):  # noqa: ANN001
            payload: dict = {
                "step": int(getattr(state, "global_step", 0) or 0),
                "max_steps": int(getattr(state, "max_steps", 0) or 0),
            }
            if _eval_losses:
                payload["eval_losses"] = list(_eval_losses)
            for k in ("loss", "eval_loss", "learning_rate"):
                v = (logs or {}).get(k)
                if isinstance(v, (int, float)):
                    payload[k] = float(v)
            try:
                with open(progress_path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
            except Exception:  # noqa: BLE001 — progress is best-effort
                pass

        class _Progress(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kw):  # noqa: ANN001
                el = (metrics or {}).get("eval_loss")
                if el is not None:
                    _eval_losses.append(float(el))
                _dump(state, metrics)

            def on_log(self, args, state, control, logs=None, **kw):  # noqa: ANN001
                _dump(state, logs)

        callbacks.append(_Progress())
    # ADR-0064 §1: mirror each just-written checkpoint to S3 (no-op without a
    # configured bucket) so a relaunched job can resume past a dead box.
    callbacks.append(checkpoint_sync.make_checkpoint_callback(progress_path))
    # #2025 (ADR-0102 D4): the live curve — stages, series, checkpoints, scalars.
    if reporter is not None:
        from operator_training.reporting import make_report_callback  # noqa: PLC0415 — needs transformers, imported above under the same guard

        _stage(reporter, "tokenise")
        callbacks.append(make_report_callback(reporter))

    def _mk(**extra):
        return SFTTrainer(model=model, args=args, train_dataset=train_ds,
                          eval_dataset=eval_ds, callbacks=callbacks, **extra)
    try:
        trainer = _mk(processing_class=tok)
    except TypeError:  # older trl
        trainer = _mk(tokenizer=tok)
    _baseline = splits.baseline(spec, trainer.evaluate, has_eval=eval_ds is not None, resumed=False)
    stats = trainer.train()
    _stage(reporter, "export")
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    metrics = _base_metrics(stats, device, steps)
    metrics.update(splits.recorded(spec, _batch, _eval_source))
    metrics.update(_baseline)
    if tracking_error:
        metrics["tracking_error"] = tracking_error  # soft, like publish_error (#1709)
    if validating:
        # Flag it loud so a validation run is never mistaken for a real train,
        # and downstream gates/report can note it. requested_model preserves what
        # the REAL run would train, for provenance.
        metrics.update({"validation": True, "requested_model": spec.base_model})
    if eval_ds is not None:
        try:
            metrics["eval_loss"] = float(trainer.evaluate().get("eval_loss", 0.0))
        except Exception as e:  # noqa: BLE001
            metrics["eval_error"] = str(e)[:200]
    # ADR-0064 §2: surface the durable checkpoint location (if any checkpoint was
    # written AND a checkpoint bucket is configured) so the control loop can
    # relaunch a dead job onto it. Best-effort — never fails the train.
    checkpoint_target = checkpoint_sync.s3_target_from_env()
    if checkpoint_target is not None and checkpoint_sync.latest_checkpoint_dir(out_dir) is not None:
        bucket, prefix, run_key = checkpoint_target
        metrics["checkpoint_uri"] = checkpoint_sync.checkpoint_uri_for(bucket, prefix, run_key)
    return TrainResult(
        status="succeeded", backend="local", adapter_uri=f"file://{out_dir}",
        job_id=f"local-{spec.output_name}", metrics=metrics, run_url="",
    )


@register_operator(name="train/local", version="0.1.0", capabilities=["train.run"],
                   implements=["contract://train.run/1.0.0"])  # #210
class LocalTrainOp(Operation[TrainSpec, TrainResult]):
    input_schema = TrainSpec
    output_schema = TrainResult
    supports_mock = True
    reports = TRAIN_REPORTS  # ADR-0103 / #2025

    async def run(self, inp: TrainSpec, ctx: OperationContext) -> TrainResult:
        return await _train_local(inp, reporter=ctx.report)

    def mock(self, inp: TrainSpec, ctx: OperationContext) -> TrainResult:
        return TrainResult(
            status="succeeded", backend="local",
            adapter_uri=f"file:///tmp/{inp.output_name}", metrics={"local_mock": True},
        )
