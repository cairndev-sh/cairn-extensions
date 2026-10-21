"""cairn-train-local — run a real CPU/MPS LoRA SFT *on a remote box*.

The on-box entrypoint for the **train/remote** path: `remote/exec` SSH-dispatches
this CLI to any box you own (#70 Axis-2), and it runs the SAME `_train_local`
trainer as the `train/local` operator (plain LoRA, no CUDA-only deps) — real
forward/backward, a real loss curve, a real adapter. Lets a workflow running on a
torch-less box (e.g. the Intel-Mac/py3.13 wall) train on a Linux box where torch
installs cleanly, while cairn stays the orchestrator.

Spec resolution (first that yields a value wins):
1. ``CAIRN_TRAIN_SPEC`` — a full TrainSpec JSON (how a caller can hand in one blob);
2. ``argv[1]`` — a path to a spec JSON file (e.g. an uploaded ``spec.json``);
3. individual ``CAIRN_<FIELD>`` env vars (how the ``run-train-remote`` binding
   maps ``$input.*`` → env, since a YAML binding maps scalars, not a nested blob).

Writes ``metrics.json`` into the cwd (the remote workdir) so ``remote/exec``'s
``poll`` picks up real metrics (loss → the model/learning gates), ``result.json``
(the full TrainResult), and copies the adapter dir into the cwd so ``remote/exec``
can SFTP-fetch it back.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

from operator_training.live import reporter_from_env
from operator_training.operators.local import _train_local
from operator_training.schemas.train import TrainSpec

# CAIRN_<ENV> → TrainSpec field, with the caster for the scalar the binding exports.
_ENV_FIELDS: dict[str, tuple[str, type]] = {
    "CAIRN_BASE_MODEL": ("base_model", str),
    "CAIRN_DATASET_URI": ("dataset_uri", str),
    "CAIRN_METHOD": ("method", str),
    "CAIRN_OUTPUT_NAME": ("output_name", str),
    "CAIRN_MAX_STEPS": ("max_steps", int),
    "CAIRN_SAVE_STEPS": ("save_steps", int),
    "CAIRN_EPOCHS": ("epochs", float),
    "CAIRN_EVAL_SPLIT": ("eval_split", float),
    "CAIRN_LEARNING_RATE": ("learning_rate", float),
    "CAIRN_LORA_R": ("lora_r", int),
    "CAIRN_LORA_ALPHA": ("lora_alpha", int),
    "CAIRN_MAX_SEQ_LEN": ("max_seq_len", int),
    "CAIRN_SEED": ("seed", int),
    "CAIRN_ARTIFACT_TARGET": ("artifact_target", str),
    "CAIRN_HF_REPO": ("hf_repo", str),
    # Experiment tracking (#6) — threaded from run-train-remote-gov. report_to is a
    # LIST, parsed separately in _spec_from_env (env values are scalar strings).
    "CAIRN_TRACKING_PROJECT": ("tracking_project", str),
    "CAIRN_TRACKING_RUN_NAME": ("tracking_run_name", str),
}


def _parse_report_to(raw: str) -> list[str]:
    """Parse CAIRN_REPORT_TO into a list. remote/exec exports a list as its Python
    repr (``export CAIRN_REPORT_TO="['mlflow']"``), so accept JSON, Python-repr, or a
    comma/space-separated string. Only the known trackers survive."""
    import ast  # noqa: PLC0415 — stdlib, one caller

    s = (raw or "").strip()
    if not s or s in ("[]", "()"):
        return []
    vals: list = []
    try:
        parsed = json.loads(s)
        vals = parsed if isinstance(parsed, list) else [parsed]
    except (ValueError, TypeError):
        try:
            parsed = ast.literal_eval(s)
            vals = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
        except (ValueError, SyntaxError):
            vals = [p for p in s.replace(",", " ").split() if p]
    return [str(v).strip() for v in vals if str(v).strip() in ("mlflow", "wandb")]


def _spec_from_env() -> dict:
    """Assemble a TrainSpec dict from individual ``CAIRN_<FIELD>`` env vars (the
    shape the ``run-train-remote`` binding hands in)."""
    out: dict = {}
    report_to = _parse_report_to(os.environ.get("CAIRN_REPORT_TO", ""))
    if report_to:
        out["report_to"] = report_to
    for env, (field, cast) in _ENV_FIELDS.items():
        raw = os.environ.get(env)
        if raw not in (None, ""):
            out[field] = cast(raw)
    return out


def load_spec(argv: list[str]) -> TrainSpec:
    blob = os.environ.get("CAIRN_TRAIN_SPEC", "").strip()
    if blob:
        return TrainSpec.model_validate_json(blob)
    if len(argv) > 1 and argv[1]:
        return TrainSpec.model_validate_json(Path(argv[1]).read_text(encoding="utf-8"))
    env_spec = _spec_from_env()
    if env_spec.get("dataset_uri"):
        return TrainSpec.model_validate(env_spec)
    raise SystemExit(
        "cairn-train-local: no TrainSpec — set CAIRN_TRAIN_SPEC, pass a spec.json "
        "path, or set CAIRN_DATASET_URI (+ optional CAIRN_BASE_MODEL etc.)"
    )


def _publish_outputs(spec: TrainSpec, result, workdir: str) -> None:
    """Write metrics.json + result.json into the workdir and copy the adapter dir
    in, so ``remote/exec`` poll reads metrics + SFTP-fetches the adapter."""
    (Path(workdir) / "metrics.json").write_text(json.dumps(result.metrics or {}), encoding="utf-8")
    (Path(workdir) / "result.json").write_text(result.model_dump_json(), encoding="utf-8")
    src = (result.adapter_uri or "").removeprefix("file://")
    if src and os.path.isdir(src):
        dst = os.path.join(workdir, spec.output_name)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copytree(src, dst, dirs_exist_ok=True)


def _apply_tracking_env() -> None:
    """Promote CAIRN_MLFLOW_TRACKING_URI → MLFLOW_TRACKING_URI so HF's MLflowCallback
    logs to the run's server (#6). Uses a CAIRN_-prefixed var (not MLFLOW_TRACKING_URI
    directly) so an EMPTY value from the binding never clobbers a URI already set in
    the box's own environment — only a non-empty value overrides."""
    uri = os.environ.get("CAIRN_MLFLOW_TRACKING_URI", "").strip()
    if uri:
        os.environ["MLFLOW_TRACKING_URI"] = uri


def _publish_adapter(spec: TrainSpec, result) -> None:
    """#732 — publish the trained adapter to HF Hub or S3 FROM THE BOX when
    ``artifact_target`` is hf|s3, recording the published URI in
    ``metrics.published_uri`` (→ findings.train_metrics.published_uri on the
    control plane).

    The adapter is STILL SFTP-fetched back regardless, so this is additive: a
    publish failure records ``publish_error`` but never fails the run, because
    the fetched adapter remains the source of truth. #1715 moved the publish
    itself into ``publish.publish_adapter_dir`` (shared with the SSH-flow path,
    which has no fetch-back and therefore lets the failure raise); the soft-fail
    policy here is unchanged.

    Needs HF_TOKEN (hf) / AWS creds + CAIRN_CHECKPOINT_BUCKET (s3) on the box."""
    from operator_training.publish import (  # noqa: PLC0415 - optional deps (hf/boto3) live behind the publish targets
        AdapterPublishError, publish_adapter_dir,
    )

    target = (getattr(spec, "artifact_target", "") or "").lower()
    if target not in ("hf", "s3"):
        return
    if not isinstance(result.metrics, dict):
        result.metrics = {}
    try:
        uri = publish_adapter_dir(
            target, result.adapter_uri or "",
            output_name=getattr(spec, "output_name", "") or "",
            hf_repo=getattr(spec, "hf_repo", "") or "",
        )
    except AdapterPublishError as e:
        result.metrics["publish_error"] = str(e)
        return
    if uri:
        result.metrics["published_uri"] = uri


def _cli() -> None:
    _apply_tracking_env()
    spec = load_spec(sys.argv)
    # #553: write live progress into the workdir's metrics.json DURING training so
    # remote/exec's poll surfaces it (the final _publish_outputs overwrites it with
    # the complete result at the end).
    progress_path = os.path.join(os.getcwd(), "metrics.json")
    reporter = reporter_from_env()  # #2025: CAIRN_RUN_ID + a transport, or a no-op
    try:
        result = asyncio.run(_train_local(spec, progress_path=progress_path, reporter=reporter))
    finally:
        reporter.close()
    _publish_adapter(spec, result)  # #732 — HF/S3 publish from the box (opt-in via artifact_target)
    _publish_outputs(spec, result, os.getcwd())
    print(result.model_dump_json())


if __name__ == "__main__":
    _cli()
