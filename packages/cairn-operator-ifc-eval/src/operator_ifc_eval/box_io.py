"""What the on-box ``eval.ifc_validate`` stage reads and writes.

It reads the generation samples file (a path on this box, ``file://`` or ``s3://``), verifies its
bytes against the hash the generate leg reported, and uploads its results file beside the run's
checkpoints when ``CAIRN_CHECKPOINT_BUCKET`` is set:
``s3://<bucket>/<CAIRN_CHECKPOINT_PREFIX>/<run>/eval/ifc_validate_results.json``, the location
rule ``cairn-operator-training``'s checkpoint mirror and ``cairn-operator-eval-lm``'s stages use.

A small copy of that rule rather than an import: an operator pack never imports another
(ADR-0128 D5). boto3 is imported only on an ``s3://`` path; no credential is read or written here
(boto3 uses the box's own chain).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

DEFAULT_PREFIX = "cairn-checkpoints"
RESULTS_KEY = "eval/ifc_validate_results.json"


class FetchError(RuntimeError):
    """The samples file could not be read as asked: the stage fails the leg."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def s3_client():
    try:
        import boto3  # noqa: PLC0415 — only an s3:// path needs it
    except ImportError as e:
        raise FetchError("this stage reads or writes S3 and boto3 is not installed: pip install "
                         "'cairn-operator-ifc-eval[box]'") from e
    return boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None,
                        region_name=os.environ.get("AWS_DEFAULT_REGION") or None)


def fetch_samples(*, local_path: str, uri: str, dest_dir: Path, sha256: str = "") -> Path:
    """The samples file as a local path: ``local_path`` when it is a file on this box, else ``uri``
    (``s3://`` downloaded into ``dest_dir``, ``file://`` or a path). Verified against ``sha256``
    when given; a mismatch or nothing to read raises :class:`FetchError`."""
    if local_path and Path(local_path).is_file():
        path = Path(local_path)
    else:
        ref = str(uri or "").strip()
        if not ref:
            raise FetchError("no samples file: set samples_path or samples_uri (the generate leg's output)")
        if ref.startswith("s3://"):
            parsed = urlparse(ref)
            bucket, key = parsed.netloc, parsed.path.lstrip("/")
            if not bucket or not key:
                raise FetchError(f"malformed s3 uri: {ref!r}")
            dest_dir.mkdir(parents=True, exist_ok=True)
            path = dest_dir / (Path(key).name or "samples.jsonl")
            s3_client().download_file(bucket, key, str(path))
        else:
            path = Path(unquote(urlparse(ref).path) if ref.startswith("file://") else ref)
        if not path.is_file():
            raise FetchError(f"samples {ref!r} is not a file on this box")
    if sha256:
        got = _sha256(path)
        if got != sha256:
            raise FetchError(f"samples {path} hash {got[:12]}… but the generate leg reported {sha256[:12]}…; "
                             "refusing to score a file that changed")
    return path


def results_target() -> tuple[str, str] | None:
    """``(bucket, key)`` for the results file, or ``None`` without ``CAIRN_CHECKPOINT_BUCKET`` or a run
    to key it by (``CAIRN_RUN_ID``, else ``CAIRN_PARENT_RUN``)."""
    bucket = os.environ.get("CAIRN_CHECKPOINT_BUCKET", "").strip()
    run = os.environ.get("CAIRN_RUN_ID", "").strip() or os.environ.get("CAIRN_PARENT_RUN", "").strip()
    if not bucket or not run:
        return None
    prefix = os.environ.get("CAIRN_CHECKPOINT_PREFIX", DEFAULT_PREFIX).strip("/") or DEFAULT_PREFIX
    return bucket, f"{prefix}/{run}/{RESULTS_KEY}"


def upload(path: Path, bucket: str, key: str) -> str:
    """Upload ``path`` and check it landed (size), or raise. Returns its ``s3://`` URI."""
    client = s3_client()
    client.upload_file(str(path), bucket, key)
    head = client.head_object(Bucket=bucket, Key=key)
    if int(head.get("ContentLength") or -1) != path.stat().st_size:
        raise FetchError(f"s3://{bucket}/{key} does not hold the {path.stat().st_size} bytes written")
    return f"s3://{bucket}/{key}"


__all__ = ["DEFAULT_PREFIX", "RESULTS_KEY", "FetchError", "fetch_samples", "results_target", "s3_client", "upload"]
