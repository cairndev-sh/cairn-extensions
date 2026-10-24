"""Durable checkpoint sync (ADR-0064 §1): the trainer owns checkpoint MECHANISM
(HF/TRL's ``save_steps`` + ``resume_from_checkpoint``); this module makes those
checkpoints survive the box by mirroring them to S3, and pulls them back down so
a relaunch on a fresh box resumes where the last one died.

Target resolution reads env exported by the ``run-train-remote`` binding (same
pattern as the tracking creds, #532): ``CAIRN_CHECKPOINT_BUCKET``,
``CAIRN_CHECKPOINT_PREFIX`` (default ``cairn-checkpoints``), plus the standard
``AWS_*`` creds/region and an optional ``AWS_ENDPOINT_URL`` (MinIO/R2). Absent a
bucket, every helper here is a no-op — today's local-only behavior, unchanged.

boto3 is imported only when an S3 client is built (:func:`s3_client`): it is not a
base dependency of this pack. The ``[gpu]`` extra carries it, because the GPU box
is where ``s3://`` data, checkpoint mirroring and the ``s3`` adapter target run;
a run with none of those never imports it. torch/transformers/trl stay lazy too
(the ``[local]``/``[gpu]`` extras), per the convention in ``operators/local.py``.

:func:`trainer_output_dir` is where a trainer writes its checkpoints and adapter.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "cairn-checkpoints"

#: The last-resort run key. It is SHARED by construction, so it is safe to PUSH
#: under (you only clobber your own) and never safe to PULL from — see
#: :func:`is_run_scoped` and #1749.
GENERIC_RUN_KEY = "adapter"


def is_run_scoped(run_key: str) -> bool:
    """Does this key identify ONE run, or is it the shared fallback?

    The distinction exists because pull and push are not symmetric (#1749):
    pushing under a shared key clobbers only your own checkpoints, while
    PULLING from one silently seeds a fresh run with a stranger's weights.
    """
    return bool(run_key) and run_key != GENERIC_RUN_KEY


def s3_target_from_env(output_name: str = "") -> tuple[str, str, str] | None:
    """Resolve ``(bucket, prefix, run_key)`` from env, or ``None`` if no bucket is
    configured — callers must no-op in that case (local-only, no regression).

    ``output_name`` is an identity of LAST resort supplied by the caller (the
    entries pass the basename of their output dir, which is exactly the spec's
    ``output_name``). It exists because the env vars below are set by the
    control plane and are absent when an entry is driven directly — the case
    that produced #1749."""
    bucket = os.environ.get("CAIRN_CHECKPOINT_BUCKET", "").strip()
    if not bucket:
        return None
    prefix = os.environ.get("CAIRN_CHECKPOINT_PREFIX", DEFAULT_PREFIX).strip("/") or DEFAULT_PREFIX
    # Per-RUN key, stable across retries of the same run (H2 resume) and distinct
    # between runs: CAIRN_RUN_ID (train-remote binding) → CAIRN_PARENT_RUN (the
    # dataflow/SSH path stamps the cairn run id here, #989) → output_name → the
    # literal "adapter". Found live 2026-08-19: on the SSH path neither RUN_ID
    # nor OUTPUT_NAME was set, so EVERY run shared one key and clobbered the rest.
    run_key = (
        os.environ.get("CAIRN_RUN_ID", "").strip()
        or os.environ.get("CAIRN_PARENT_RUN", "").strip()
        or os.environ.get("CAIRN_OUTPUT_NAME", "").strip()
        or (output_name or "").strip()
        or GENERIC_RUN_KEY
    )
    return bucket, prefix, run_key


#: The directory a stage's trainer writes under, inside the job's workdir.
TRAIN_OUTPUT_DIR = "train-output"


def trainer_output_dir(output_name: str, workdir: str | os.PathLike | None = None) -> str:
    """Where a trainer writes its ``checkpoint-*`` dirs and the saved adapter.

    With the job's ``workdir`` (a flow stage always has one): ``<workdir>/train-output/
    <output_name>``. It lives and dies with the job, so two runs that share an
    ``output_name`` on one box never see each other's checkpoints; a retry of the
    same run resumes from its own S3 checkpoints (:func:`pull_checkpoints_if_any`,
    keyed by the run, not the directory). Without one (a trainer driven directly,
    e.g. the image's ``train_entry`` entrypoint): ``/tmp/<output_name>``, as before."""
    name = (output_name or "").strip() or GENERIC_RUN_KEY
    if workdir:
        return os.path.join(os.fspath(workdir), TRAIN_OUTPUT_DIR, name)
    return f"/tmp/{name}"


def s3_client():
    """A boto3 S3 client honoring ``AWS_ENDPOINT_URL``/``AWS_DEFAULT_REGION`` (set
    by the operator env injection), for MinIO/R2-compatible stores too. Public so
    the dataset fetcher (#701) resolves the same endpoint/region as checkpoints.

    boto3 is imported here, not at module top: a run that touches no S3 needs no
    boto3 (see the module docstring)."""
    try:
        import boto3  # noqa: PLC0415 - optional: only the S3 paths need it
    except ImportError as e:
        raise RuntimeError(
            "this run reads or writes S3 (an s3:// dataset, CAIRN_CHECKPOINT_BUCKET or "
            "artifact_target: s3) and boto3 is not installed: pip install "
            "'cairn-operator-training[gpu]' (or pip install boto3)") from e
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or None
    region = os.environ.get("AWS_DEFAULT_REGION") or None
    return boto3.client("s3", endpoint_url=endpoint, region_name=region)


def _client():
    return s3_client()


def upload_dir(local_dir: str, bucket: str, key_prefix: str) -> None:
    """Upload every file under ``local_dir`` to ``s3://bucket/key_prefix/...``,
    preserving relative paths. Best-effort: logs and returns on any error, never
    raises — a sync failure must never fail the train."""
    try:
        if not os.path.isdir(local_dir):
            return
        client = _client()
        prefix = key_prefix.strip("/")
        for root, _dirs, files in os.walk(local_dir):
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, local_dir)
                key = f"{prefix}/{rel}".replace(os.sep, "/")
                client.upload_file(full, bucket, key)
    except Exception:  # noqa: BLE001 - checkpoint sync is best-effort
        logger.warning("checkpoint_sync.upload_dir failed for %s", local_dir, exc_info=True)


def upload_files(local_dir: str, names: list[str], bucket: str, key_prefix: str) -> None:
    """Upload the named files of ``local_dir`` (top level only) to
    ``s3://bucket/key_prefix/<name>``. Best-effort, like :func:`upload_dir`."""
    try:
        client = _client()
        prefix = key_prefix.strip("/")
        for name in names:
            client.upload_file(os.path.join(local_dir, name), bucket, f"{prefix}/{name}")
    except Exception:  # noqa: BLE001 - best-effort; the publish verifies what landed
        logger.warning("checkpoint_sync.upload_files failed for %s", local_dir, exc_info=True)


def download_prefix(bucket: str, key_prefix: str, local_dir: str) -> None:
    """Download every object under ``s3://bucket/key_prefix/`` into ``local_dir``,
    recreating relative paths. Best-effort: logs and returns on any error."""
    try:
        client = _client()
        prefix = key_prefix.strip("/") + "/"
        os.makedirs(local_dir, exist_ok=True)
        paginator = client.get_paginator("list_objects_v2")
        found = False
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel = key[len(prefix):]
                if not rel:
                    continue
                found = True
                dest = os.path.join(local_dir, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                client.download_file(bucket, key, dest)
        if not found:
            logger.info("checkpoint_sync.download_prefix: nothing under s3://%s/%s", bucket, prefix)
    except Exception:  # noqa: BLE001 - checkpoint sync is best-effort
        logger.warning(
            "checkpoint_sync.download_prefix failed for s3://%s/%s", bucket, key_prefix, exc_info=True
        )


def fetch_uri_dir(uri: str, dest: str) -> str:
    """Materialize a directory URI locally: ``s3://bucket/key/…`` is downloaded
    into ``dest`` (via :func:`download_prefix`), anything else is treated as an
    already-local path and returned as-is. Used by ``init_adapter_uri`` (#1463)
    to continue training from a previously saved adapter. Unlike the checkpoint
    helpers this RAISES on an empty download — silently training a fresh adapter
    when the caller asked to continue one would corrupt the experiment."""
    if not uri.startswith("s3://"):
        return uri
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    if not bucket or not key.strip("/"):
        raise ValueError(f"init adapter URI has no bucket/key: {uri}")
    download_prefix(bucket, key.strip("/"), dest)
    if not os.path.isfile(os.path.join(dest, "adapter_config.json")):
        raise FileNotFoundError(f"no adapter_config.json under {uri} (fetched to {dest})")
    return dest


def latest_checkpoint_dir(output_dir: str) -> str | None:
    """The newest ``checkpoint-*`` dir under ``output_dir`` (highest step number),
    or ``None`` if there isn't one yet. Public: callers (e.g. ``operators/local``)
    use it to decide whether a checkpoint exists to surface in metrics."""
    try:
        candidates = [
            d for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
    except OSError:
        return None
    if not candidates:
        return None

    def _step(name: str) -> int:
        try:
            return int(name.removeprefix("checkpoint-"))
        except ValueError:
            return -1

    best = max(candidates, key=_step)
    return os.path.join(output_dir, best)


def checkpoint_uri_for(bucket: str, prefix: str, run_key: str) -> str:
    """The durable-checkpoint S3 location surfaced to the control loop (in
    ``JobStatus.metrics['checkpoint_uri']``) once at least one checkpoint has
    been uploaded — see ADR-0064 §2 (relaunch-on-resumable-death)."""
    return f"s3://{bucket}/{prefix}/{run_key}/"


def _note_checkpoint_uri(progress_path: str | None, uri: str) -> None:
    """Best-effort: merge ``checkpoint_uri`` into the live progress JSON (the
    same file ``cairn-train-local`` writes as ``metrics.json``) so the
    control-plane poll picks it up without waiting for the run to finish."""
    if not progress_path:
        return
    try:
        payload: dict = {}
        if os.path.exists(progress_path):
            with open(progress_path, encoding="utf-8") as fh:
                payload = json.load(fh) or {}
        payload["checkpoint_uri"] = uri
        with open(progress_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception:  # noqa: BLE001 - progress is best-effort
        logger.warning("checkpoint_sync: failed to note checkpoint_uri", exc_info=True)


def make_checkpoint_callback(progress_path: str | None = None):
    """A ``transformers.TrainerCallback`` whose ``on_save`` mirrors the newest
    ``checkpoint-*`` dir under the trainer's ``output_dir`` to
    ``s3://bucket/prefix/run_key/``. No-op (returns a callback that does nothing)
    if no checkpoint bucket is configured. Never fails the train — every upload
    is wrapped by :func:`upload_dir`'s own best-effort guard, and ``on_save``
    itself is wrapped defensively too.

    On a successful upload, also stamps ``checkpoint_uri`` into ``progress_path``
    (when given) so the control loop's poll of ``JobStatus.metrics`` learns a
    durable checkpoint exists (ADR-0064 §2) — set once the FIRST checkpoint
    lands, kept afterward."""
    from transformers import TrainerCallback  # noqa: PLC0415 - optional [local] extra

    target = s3_target_from_env()

    class _CheckpointSyncCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
            if target is None:
                return control
            bucket, prefix, run_key = target
            try:
                output_dir = getattr(args, "output_dir", None)
                if not output_dir:
                    return control
                ckpt_dir = latest_checkpoint_dir(output_dir)
                if not ckpt_dir:
                    return control
                # Keep the ``checkpoint-N/`` directory in the key so the pull
                # side (download_prefix → out_dir/checkpoint-N/) and HF's
                # ``resume_from_checkpoint`` find it. A flat upload (the bug
                # found live 2026-08-19) could never be resumed from.
                key_prefix = f"{prefix}/{run_key}/{os.path.basename(ckpt_dir.rstrip('/'))}"
                upload_dir(ckpt_dir, bucket, key_prefix)
                _note_checkpoint_uri(progress_path, checkpoint_uri_for(bucket, prefix, run_key))
            except Exception:  # noqa: BLE001 - checkpoint sync is best-effort
                logger.warning("checkpoint sync on_save failed", exc_info=True)
            return control

    return _CheckpointSyncCallback()


def pull_checkpoints_if_any(out_dir: str) -> str | None:
    """If a checkpoint bucket is configured AND ``out_dir`` has no local
    ``checkpoint-*`` dir yet, pull ``s3://bucket/prefix/run_key/`` down into
    ``out_dir`` so ``resume_from_checkpoint`` finds them. Call BEFORE training
    starts. Best-effort: never raises. Returns the checkpoint dir a resume will
    start from, or ``None``.

    **Refuses to pull under a non-run-scoped key (#1749).** The run key falls
    back through env vars to :data:`GENERIC_RUN_KEY`, which every direct
    invocation shares. Pulling from it seeded a BRAND-NEW output dir with
    another run's step-400 checkpoint: the run "trained" 400 steps in 13
    seconds with ``train_loss`` 0.0003, exited 0, wrote a valid adapter and
    plausible metrics, and nothing warned. Only an implausible runtime caught
    it. A continual-learning campaign is a chain of runs that all look alike to
    a shared key, so this is the failure mode that silently fabricates an
    experiment — refusing the pull costs a re-run, taking it costs the result.

    The basename of ``out_dir`` is offered as an identity, since it is the
    spec's ``output_name`` and is available even when the control plane's env
    vars are not.
    """
    if latest_checkpoint_dir(out_dir) is not None:
        return latest_checkpoint_dir(out_dir)  # local checkpoints — nothing to pull
    target = s3_target_from_env(output_name=os.path.basename(str(out_dir).rstrip("/")))
    if target is None:
        return None
    bucket, prefix, run_key = target
    if not is_run_scoped(run_key):
        logger.warning(
            "checkpoint pull SKIPPED: run key %r is the shared fallback, so a pull could "
            "resume this run from a DIFFERENT run's checkpoints (#1749). Set CAIRN_RUN_ID "
            "or give the output dir a distinct name to enable resume.", run_key)
        return None
    download_prefix(bucket, f"{prefix}/{run_key}", out_dir)
    resumed = latest_checkpoint_dir(out_dir)
    if resumed is not None:
        # Loud on purpose: a resumed run and a fresh run are NOT the same
        # experiment, and the difference was previously invisible.
        logger.warning("RESUMING from pulled checkpoint %s (s3://%s/%s/%s) — this run "
                       "continues prior training, it does not start fresh",
                       resumed, bucket, prefix, run_key)
    return resumed
