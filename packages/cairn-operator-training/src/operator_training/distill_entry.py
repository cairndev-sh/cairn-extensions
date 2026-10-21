"""Knowledge-distillation entrypoint (#1726, revives #467) — runs ON the GPU
backend, NOT in cairn.

A small STUDENT learns from a larger TEACHER via trl's Generalized Knowledge
Distillation. #467 parked this as "a full training run underneath (teacher +
student + training loop + GPU orchestration)"; trl ships the loop, so what is
left is the same wrapper shape ``dpo_entry`` has around ``DPOTrainer``.

Two things worth knowing before editing:

* ``GKDConfig.beta`` interpolates the generalized Jensen-Shannon divergence
  (0.0 ≈ forward KL, 1.0 ≈ reverse KL). ``TrainSpec.beta`` — a different spec —
  means DPO preference strength. They are not interchangeable, which is why the
  spec here calls it ``jsd_beta`` and maps it explicitly.
* ``lmbda > 0`` makes the STUDENT GENERATE during training. That is the point of
  GKD (on-policy distillation fixes the train/inference mismatch) and it is also
  a materially heavier GPU profile than plain SFT.

Helpers are imported from ``train_entry``, not duplicated — same rule
``dpo_entry`` follows.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

from operator_training import checkpoint_sync, dataset_fetch
from operator_training.train_entry import (
    _base_metrics,
    _mlflow_log_and_register,
    _mlflow_training_run,
    _save_adapter,
    _tracking_setup,
)


def _to_messages(row: dict[str, Any]) -> dict[str, Any]:
    """cairn datasets carry ``{input, output}``; GKD wants ``messages``.

    Documented rather than silent because it is a real format boundary: a
    dataset that already has ``messages`` passes through untouched.
    """
    if isinstance(row.get("messages"), list):
        return {"messages": row["messages"]}
    user = str(row.get("input") or row.get("prompt") or "")
    assistant = str(row.get("output") or row.get("completion") or "")
    msgs = [{"role": "user", "content": user}]
    if assistant:
        msgs.append({"role": "assistant", "content": assistant})
    return {"messages": msgs}


def main(spec: dict[str, Any]) -> dict[str, Any]:
    # unsloth MUST be imported before trl (the half-patched-trainer trap found
    # live on the C2 SFT run, 2026-08-19 — same ordering dpo_entry documents).
    from unsloth import FastLanguageModel  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415
    from trl import GKDConfig, GKDTrainer  # noqa: PLC0415 — top-level in trl 0.24, NOT trl.experimental.gkd

    seed = int(spec.get("seed", 42))
    out_dir = f"/tmp/{spec.get('output_name', 'student-adapter')}"
    checkpoint_sync.pull_checkpoints_if_any(out_dir)

    student, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec["base_model"],
        max_seq_length=spec.get("max_seq_len", 1024),
        load_in_4bit=spec.get("method", "qlora") == "qlora",
    )
    # The student is the one that learns; PEFT adapters go on it.
    student = FastLanguageModel.get_peft_model(
        student,
        r=int(spec.get("lora_r", 16)),
        lora_alpha=int(spec.get("lora_alpha", 16)),
        random_state=seed,
    )

    # #1714 — strip AND verify the `#sha256=` pin, fetch s3:// via boto3, THEN
    # load. The pipeline declares `dataset_uri: {type: Dataset}`, so the launch
    # resolver hands this a pinned `s3://…#sha256=` URI; feeding that straight
    # to the loader is the FileNotFoundError train_entry/dpo_entry already fixed
    # (the fragment lands in the object key, and HF datasets cannot read s3).
    ds = load_dataset("json", data_files=dataset_fetch.resolve_dataset_uri(
        str(spec["dataset_uri"])), split="train")
    ds = ds.map(_to_messages, remove_columns=[c for c in ds.column_names if c != "messages"])

    import torch  # noqa: PLC0415 — GPU-backend-only dep, same tier as unsloth/trl above
    _bf16_ok = bool(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)
    _report_to, _tracking_error = _tracking_setup(spec)

    # Version-tolerant config build (the dpo_entry pattern): filter to the
    # parameters THIS trl actually accepts, so a pinned-version bump that adds
    # or renames a knob degrades instead of raising TypeError mid-run.
    params = set(inspect.signature(GKDConfig.__init__).parameters)
    cfg = dict(
        output_dir=out_dir, seed=seed,
        max_steps=spec.get("max_steps", 0) or -1,
        num_train_epochs=spec.get("epochs", 1.0),
        learning_rate=spec.get("learning_rate", 2e-5),
        # the deliberate rename: jsd_beta (ours) → beta (trl's JSD interpolation)
        beta=spec.get("jsd_beta", 0.5),
        lmbda=spec.get("lmbda", 0.5),
        seq_kd=bool(spec.get("seq_kd", False)),
        temperature=spec.get("temperature", 0.9),
        max_new_tokens=int(spec.get("max_new_tokens", 128)),
        max_length=spec.get("max_seq_len", 1024),
        save_strategy="steps", save_steps=int(spec.get("save_steps", 50) or 50),
        save_total_limit=spec.get("save_total_limit", 2),
        report_to=_report_to,
        run_name=spec.get("tracking_run_name") or spec.get("output_name") or None,
        bf16=_bf16_ok, fp16=not _bf16_ok,
    )
    args = GKDConfig(**{k: v for k, v in cfg.items() if k in params})

    callbacks = [checkpoint_sync.make_checkpoint_callback()]
    tok_kw = ("processing_class"
              if "processing_class" in inspect.signature(GKDTrainer.__init__).parameters
              else "tokenizer")
    trainer = GKDTrainer(
        model=student,
        # A string is instantiated by the trainer itself; the teacher is frozen
        # by construction — GKD never updates it.
        teacher_model=spec["teacher_model"],
        args=args, train_dataset=ds, callbacks=callbacks, **{tok_kw: tokenizer},
    )

    resume = os.path.isdir(out_dir) and any(
        p.startswith("checkpoint-") for p in os.listdir(out_dir))
    stats = trainer.train(resume_from_checkpoint=resume)

    student.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    adapter_uri = _save_adapter(spec, student, tokenizer, out_dir)

    metrics = _base_metrics(stats)   # train_loss + speed + the #1622 env fingerprint
    metrics["teacher_model"] = spec["teacher_model"]
    metrics["lmbda"] = spec.get("lmbda", 0.5)
    metrics["jsd_beta"] = spec.get("jsd_beta", 0.5)
    if _tracking_error:
        metrics["tracking_error"] = _tracking_error   # soft, like publish_error (#1709)
    mlrun = _mlflow_training_run(spec)
    with mlrun:
        reg = _mlflow_log_and_register(spec, out_dir, mlrun)
        if reg:
            metrics.update(reg)
    target = checkpoint_sync.s3_target_from_env()
    if target is not None and checkpoint_sync.latest_checkpoint_dir(out_dir) is not None:
        bucket, prefix, run_key = target
        metrics["checkpoint_uri"] = checkpoint_sync.checkpoint_uri_for(bucket, prefix, run_key)
    return {"adapter_uri": adapter_uri, "metrics": metrics, "run_url": ""}


def _cli() -> None:
    """Same in.json / CAIRN_TRAIN_SPEC contract as ``train_entry._cli``."""
    in_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if in_path:
        spec = json.loads(Path(in_path).read_text(encoding="utf-8"))
    else:
        raw = os.environ.get("CAIRN_TRAIN_SPEC", "").strip()
        if not raw:
            raise SystemExit("distill_entry: pass in.json or set CAIRN_TRAIN_SPEC")
        spec = json.loads(raw)
    print(json.dumps(main(spec)))


if __name__ == "__main__":
    _cli()
