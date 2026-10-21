"""DPO alignment entrypoint (#1463 round 4) — runs ON the GPU backend, NOT in cairn.

Continues a previously trained adapter (``init_adapter_uri`` is REQUIRED — DPO
aligns an existing policy, it never starts from a fresh adapter) on preference
pairs ``{prompt, chosen, rejected}`` (built by ``dataset.dpo_pairs`` from the
closed-book measure answers). trl's ``DPOTrainer`` with ``ref_model=None`` on a
PEFT model uses the ADAPTER-DISABLED base as the frozen reference — so the KL
anchor is the base model's calibrated behavior (76% abstain on this campaign),
while the loss pushes probability from each fabricated answer to the preferred
one. Same unsloth/QLoRA stack, checkpoints, MLflow logging + registry epilogue
as ``train_entry`` (helpers imported from there, not duplicated).
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


def main(spec: dict[str, Any]) -> dict[str, Any]:
    # unsloth MUST be imported before trl (same half-patched-trainer trap found
    # live on the C2 SFT run, 2026-08-19).
    from unsloth import FastLanguageModel  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415
    from trl import DPOConfig, DPOTrainer  # noqa: PLC0415

    init_uri = str(spec.get("init_adapter_uri") or "").strip()
    if not init_uri:
        raise ValueError("train.dpo requires init_adapter_uri — DPO aligns an "
                         "existing adapter (e.g. the knowledge-stage output), "
                         "it never starts from random LoRA weights")

    seed = int(spec.get("seed", 42))
    out_dir = f"/tmp/{spec.get('output_name', 'dpo-adapter')}"
    checkpoint_sync.pull_checkpoints_if_any(out_dir)

    init_dir = checkpoint_sync.fetch_uri_dir(init_uri, "/tmp/init-adapter-dpo")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=init_dir,
        max_seq_length=spec.get("max_seq_len", 2048),
        load_in_4bit=spec.get("method", "qlora") == "qlora",
    )
    for n, p in model.named_parameters():  # PEFT loads checkpoints frozen
        if "lora_" in n:
            p.requires_grad_(True)

    # Pairs → DPO's text format: prompt rendered with the chat template (so it
    # matches how the policy was trained AND how it is served), completions raw.
    # #1714 — same resolver as the SFT entry: strip + verify the `#sha256=`
    # pin, fetch s3:// via boto3, then load.
    ds = load_dataset("json", data_files=dataset_fetch.resolve_dataset_uri(
        str(spec["dataset_uri"])), split="train")
    def _render(r: dict[str, Any]) -> dict[str, Any]:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt,
                "chosen": str(r["chosen"]) + tokenizer.eos_token,
                "rejected": str(r["rejected"]) + tokenizer.eos_token}
    ds = ds.map(_render, remove_columns=[c for c in ds.column_names
                                         if c not in ("prompt", "chosen", "rejected")])

    import torch  # noqa: PLC0415 — GPU-backend-only dep, same tier as unsloth/trl above
    _bf16_ok = bool(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)
    save_steps = int(spec.get("save_steps", 50) or 50)
    # #1709 — an unreachable tracker is dropped with the reason kept, never fatal.
    _report_to, _tracking_error = _tracking_setup(spec)
    params = set(inspect.signature(DPOConfig.__init__).parameters)
    cfg = dict(
        output_dir=out_dir, seed=seed,
        max_steps=spec.get("max_steps", 0) or -1,
        num_train_epochs=spec.get("epochs", 1.0),
        learning_rate=spec.get("learning_rate", 5e-6),
        beta=spec.get("beta", 0.1),
        save_strategy="steps", save_steps=save_steps,
        save_total_limit=spec.get("save_total_limit", 2),
        report_to=_report_to,
        run_name=spec.get("tracking_run_name") or spec.get("output_name") or None,
        bf16=_bf16_ok, fp16=not _bf16_ok,
    )
    args = DPOConfig(**{k: v for k, v in cfg.items() if k in params})
    callbacks = [checkpoint_sync.make_checkpoint_callback()]
    tok_kw = "processing_class" if "processing_class" in inspect.signature(DPOTrainer.__init__).parameters else "tokenizer"
    trainer = DPOTrainer(
        model=model, ref_model=None,   # PEFT: reference = adapter-disabled base
        args=args, train_dataset=ds, callbacks=callbacks, **{tok_kw: tokenizer},
    )

    resume = os.path.isdir(out_dir) and any(
        p.startswith("checkpoint-") for p in os.listdir(out_dir))
    stats = trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    adapter_uri = _save_adapter(spec, model, tokenizer, out_dir)

    metrics = _base_metrics(stats)  # train_loss + speed metrics + the env fingerprint (#1622)
    if _tracking_error:
        metrics["tracking_error"] = _tracking_error  # soft, like publish_error (#1709)
    mlrun = _mlflow_training_run(spec)
    with mlrun:
        reg = _mlflow_log_and_register(spec, out_dir, mlrun)
        if reg:
            metrics.update(reg)
    checkpoint_target = checkpoint_sync.s3_target_from_env()
    if checkpoint_target is not None and checkpoint_sync.latest_checkpoint_dir(out_dir) is not None:
        bucket, prefix, run_key = checkpoint_target
        metrics["checkpoint_uri"] = checkpoint_sync.checkpoint_uri_for(bucket, prefix, run_key)
    return {"adapter_uri": adapter_uri, "metrics": metrics, "run_url": ""}


def _cli() -> None:
    """Same in.json/CAIRN_TRAIN_SPEC contract as ``train_entry._cli``."""
    in_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if in_path:
        spec = json.loads(Path(in_path).read_text(encoding="utf-8"))
    else:
        raw = os.environ.get("CAIRN_TRAIN_SPEC", "").strip()
        if not raw:
            raise SystemExit("cairn-dpo: no TrainSpec — set CAIRN_TRAIN_SPEC or pass an in.json path")
        spec = json.loads(raw)
    result = main(spec)
    payload = json.dumps(result)
    out_path = sys.argv[2] if len(sys.argv) > 2 else ""
    if out_path:
        Path(out_path).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    _cli()
