"""Publish a trained adapter DIRECTORY to its declared ``artifact_target`` (#1715).

Extracted from ``local_entry._publish_adapter`` so the SSH-flow path can reach
the same code. Both paths train on the box and produce a local adapter dir; only
the direct path had a publish step, so a flow-path run declaring
``artifact_target: s3`` silently ended with ``adapter_uri = /tmp/<name>`` — a
path on a box the pipeline's own ``release`` finalizer then stops (#1715).

This helper is deliberately STRICT: it raises on any failure and returns the
published URI otherwise. The two callers need opposite failure policies, and
encoding either one here would be wrong for the other:

- the DIRECT path (``local_entry``) is additive — the adapter is SFTP-fetched
  back regardless, so a publish failure is recorded as ``publish_error`` and the
  run stands;
- the FLOW path has NO fetch-back (only manifests come home), so a swallowed
  publish failure means the adapter is destroyed when the box stops. There the
  exception must travel.

``checkpoint_sync.upload_dir`` is best-effort by contract — it logs and returns
on error, and never raises — so "it did not throw" proves nothing about whether
bytes landed. That is why publishing here VERIFIES the upload before reporting
success rather than trusting the call.
"""
from __future__ import annotations

import os

#: Any one of these means "a model/adapter was saved here". Deliberately broad:
#: the point is to catch an EMPTY or wrong directory, not to police the exact
#: save format (PEFT writes adapter_config.json + adapter_model.safetensors; a
#: full save writes config.json + model.safetensors; older runs write .bin).
_ADAPTER_MARKERS = (
    "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
    "config.json", "model.safetensors", "pytorch_model.bin",
)


class AdapterPublishError(RuntimeError):
    """The adapter could not be published to its declared artifact_target."""


def publish_adapter_dir(
    target: str,
    local_dir: str,
    *,
    output_name: str = "",
    hf_repo: str = "",
) -> str:
    """Publish ``local_dir`` to ``target`` (``hf``/``s3``) and return the URI.

    Returns "" when ``target`` is neither hf nor s3 (nothing to do — ``local``
    is a legitimate choice, not a failure). Raises :class:`AdapterPublishError`
    for every other unhappy path.
    """
    tgt = (target or "").strip().lower()
    if tgt not in ("hf", "s3"):
        return ""

    # Already remote (hf://, s3://, and the stub tier's stub://): the producer
    # published it, so there is nothing to move and re-publishing would be wrong.
    # Only file:// and bare paths name a directory on this box.
    raw = (local_dir or "").strip()
    if "://" in raw and not raw.startswith("file://"):
        return ""

    src = raw.removeprefix("file://")
    if not src or not os.path.isdir(src):
        raise AdapterPublishError(
            f"artifact_target={tgt} but there is no local adapter dir to publish "
            f"(adapter_uri={local_dir!r})"
        )
    if not any(os.path.exists(os.path.join(src, m)) for m in _ADAPTER_MARKERS):
        # A dir with no model config is not an adapter; publishing it would put a
        # URI on the record that nothing can load back.
        raise AdapterPublishError(
            f"{src} does not look like an adapter dir (none of {_ADAPTER_MARKERS} present)"
        )

    if tgt == "hf":
        return _publish_hf(src, hf_repo=hf_repo, output_name=output_name)
    return _publish_s3(src)


def _publish_hf(src: str, *, hf_repo: str, output_name: str) -> str:
    from huggingface_hub import HfApi  # noqa: PLC0415 — optional (transformers dep)

    repo = (hf_repo or output_name or "").strip()
    if not repo:
        raise AdapterPublishError("artifact_target=hf but neither hf_repo nor output_name is set")
    try:
        api = HfApi()
        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=src, repo_id=repo, repo_type="model")
    except Exception as e:  # noqa: BLE001 — re-raised as the domain error
        raise AdapterPublishError(f"hf publish failed: {e}") from e
    return f"hf://{repo}"


def _publish_s3(src: str) -> str:
    from operator_training import checkpoint_sync  # noqa: PLC0415 — optional (S3)

    tgt = checkpoint_sync.s3_target_from_env()
    if tgt is None:
        raise AdapterPublishError("artifact_target=s3 but CAIRN_CHECKPOINT_BUCKET is unset")
    bucket, prefix, run_key = tgt
    key_prefix = f"{prefix}/{run_key}/adapter"

    expected = sum(len(files) for _r, _d, files in os.walk(src))
    checkpoint_sync.upload_dir(src, bucket, key_prefix)

    # upload_dir swallows its errors by contract, so VERIFY rather than assume.
    landed = _count_objects(bucket, key_prefix)
    if landed < expected:
        raise AdapterPublishError(
            f"s3 publish incomplete: {landed}/{expected} objects under "
            f"s3://{bucket}/{key_prefix}/ (see checkpoint_sync warnings for the cause)"
        )
    return f"s3://{bucket}/{key_prefix}/"


def _count_objects(bucket: str, key_prefix: str) -> int:
    from operator_training import checkpoint_sync  # noqa: PLC0415 — optional (S3)

    try:
        client = checkpoint_sync._client()  # noqa: SLF001 — same package, one boto3 config
        paginator = client.get_paginator("list_objects_v2")
        n = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix.strip("/") + "/"):
            n += sum(1 for o in page.get("Contents", []) or [] if not o["Key"].endswith("/"))
        return n
    except Exception as e:  # noqa: BLE001 — cannot verify == cannot claim success
        raise AdapterPublishError(f"could not verify s3 publish: {e}") from e


__all__ = ["AdapterPublishError", "publish_adapter_dir"]
