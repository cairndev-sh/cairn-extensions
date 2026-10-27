"""What the on-box ``eval.lm_eval`` stage reads and writes (cairn-extensions#42).

The stage runs in ``cairn-dataflow-run`` on the GPU box, with no database and no control plane in
reach. It reads three things and writes one:

* the held-out splits: ``dataset:`` refs the control plane resolved at launch to
  ``<location>#sha256=<hash>`` (#1466's retrieval rule). The location is fetched (a local path,
  ``file://`` or ``s3://``) and the bytes are verified against the pinned hash before a row is
  read; a mismatch fails the leg, never scores on unverified data;
* the adapter: a directory on this box (the ``train`` leg's ``local_adapter_dir``, when it ran on
  the same box), an ``s3://…/`` prefix (``artifact_target: s3``), ``hf://<repo>`` or a Hub id;
* the results file, uploaded beside the run's checkpoints when ``CAIRN_CHECKPOINT_BUCKET`` is set:
  ``s3://<bucket>/<CAIRN_CHECKPOINT_PREFIX>/<run>/eval/lm_eval_results.json``, the same location
  rule the trainer's checkpoint mirror uses (``cairn-operator-training``'s ``checkpoint_sync``).

Why a second small copy of the trainer's fetch and mirror helpers instead of an import: an operator
pack never imports another operator pack (ADR-0128 D5), and the few lines here are the location
and verification rules, not a library. ``tests/test_eval_stage.py`` holds the location rule.

boto3 is imported only on an ``s3://`` path (the ``[box]`` extra carries it; on the GPU box
``cairn-operator-training[gpu]`` already does). No credential is read or written here: boto3 uses
the box's own chain (an instance role, or the box's ``AWS_*``), as the trainer does.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

#: The checkpoint mirror's default prefix (CAIRN_CHECKPOINT_PREFIX unset) — the trainer's.
DEFAULT_PREFIX = "cairn-checkpoints"
#: Where the results file goes under the run's key.
RESULTS_KEY = "eval/lm_eval_results.json"


class FetchError(RuntimeError):
    """A split or an adapter could not be read as asked. The stage fails the leg: it never scores
    on data it could not verify, or on an adapter it did not load."""


def split_pin(uri: str) -> tuple[str, str]:
    """``s3://…/x.jsonl#sha256=<hash>`` → ``(location, hash)``; ``(uri, "")`` without a pin."""
    location, _, frag = str(uri or "").partition("#")
    return location, (frag[len("sha256="):] if frag.startswith("sha256=") else "")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def s3_client():
    """A boto3 S3 client honouring ``AWS_ENDPOINT_URL`` / ``AWS_DEFAULT_REGION`` (MinIO, R2), as
    the trainer's is built."""
    try:
        import boto3  # noqa: PLC0415 — only an s3:// path needs it
    except ImportError as e:
        raise FetchError("this evaluation reads or writes S3 and boto3 is not installed: pip install "
                         "'cairn-operator-eval-lm[box]'") from e
    return boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None,
                        region_name=os.environ.get("AWS_DEFAULT_REGION") or None)


def _split_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    if not bucket or not key:
        raise FetchError(f"malformed s3 uri: {uri!r}")
    return bucket, key


def fetch_dataset(uri: str, dest_dir: Path) -> Path:
    """The split at ``uri`` as a local file, its bytes verified against the pinned hash.

    A pinned ref whose bytes differ, a location that is not a file, or an unpinned ``dataset:`` ref
    (the control plane did not resolve it) raises :class:`FetchError`."""
    raw = str(uri or "").strip()
    if raw.startswith("dataset:"):
        raise FetchError(f"{raw!r} reached the box unresolved; the control plane resolves a dataset: "
                         "ref at launch (wire a registered version as the leg's input)")
    location, expected = split_pin(raw)
    if location.startswith("s3://"):
        bucket, key = _split_s3(location)
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / (Path(key).name or "split.jsonl")
        s3_client().download_file(bucket, key, str(path))
    elif location.startswith("file://"):
        path = Path(unquote(urlparse(location).path))
    else:
        path = Path(location)
    if not path.is_file():
        raise FetchError(f"split {location!r} is not a file on this box")
    if expected:
        got = _sha256(path)
        if got != expected:
            raise FetchError(f"split {location!r} hashes {got[:12]}… but the registry pinned "
                             f"{expected[:12]}…; the object changed since it was registered. Refusing "
                             "to score on unverified data (#1466).")
    return path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except ValueError as e:
                    raise FetchError(f"{path.name} line {i} is not JSON: {e}") from e
                if isinstance(row, dict):
                    rows.append(row)
    return rows


#: ``<name>@<tag>``: a promoted tag. The box cannot resolve one (no registry here), and the core
#: does not resolve artifact tags in a leg's config yet; refused with that reason.
_TAG = re.compile(r"^[A-Za-z0-9][\w.-]*@[\w.-]+$")


def resolve_adapter(*, local_dir: str, uri: str, dest: Path, stub: bool = False) -> tuple[str, str]:
    """``(what to load, how it was found)`` for the adapter.

    In order: ``local_dir`` when it is a directory on this box (the ``train`` leg's
    ``local_adapter_dir``, on the same box); ``uri`` as an ``s3://`` prefix (downloaded into
    ``dest``), ``hf://<repo>`` or ``file://`` / a path; else ``uri`` as a Hub id. A promoted tag
    (``<name>@<tag>``) is refused. The stub tier accepts the stub trainer's ``stub://`` URI."""
    if local_dir and Path(local_dir).is_dir():
        return str(local_dir), "local_adapter_dir"
    ref = str(uri or "").strip()
    if not ref:
        raise FetchError("no adapter: set adapter_uri (s3://…/, hf://<repo>, a Hub id or a path on the box)")
    if ref.startswith("stub://"):
        if stub:
            return ref, "stub"
        raise FetchError(f"{ref!r} is the stub trainer's adapter: nothing to load (run with stub: true)")
    if ref.startswith("s3://"):
        bucket, key = _split_s3(ref)
        client = s3_client()
        prefix = key.rstrip("/") + "/"
        found = False
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                rel = obj["Key"][len(prefix):]
                if not rel or rel.endswith("/"):
                    continue
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, obj["Key"], str(target))
                found = True
        if not found or not (dest / "adapter_config.json").is_file():
            raise FetchError(f"no adapter_config.json under {ref}")
        return str(dest), "s3"
    if ref.startswith("hf://"):
        return ref[len("hf://"):].strip("/"), "hf"
    if ref.startswith("file://"):
        ref = unquote(urlparse(ref).path)
    if Path(ref).is_dir():
        return ref, "path"
    if _TAG.match(ref):
        raise FetchError(
            f"{ref!r} looks like a promoted tag (<name>@<tag>). The box cannot resolve a tag, and cairn "
            "does not resolve artifact tags in a leg's inputs yet: pass the adapter's URI (the promote "
            "step's `uri`, an s3://…/ prefix, hf://<repo>, or a Hub id)")
    if ref.startswith(("/", ".")) or "://" in ref:
        raise FetchError(f"adapter {ref!r} is not a directory on this box")
    return ref, "hub"


def run_key() -> str:
    """The run the results belong to, as the trainer's mirror keys it: ``CAIRN_RUN_ID``, else
    ``CAIRN_PARENT_RUN`` (the engines stamp the leg's run there), else ``""``."""
    return (os.environ.get("CAIRN_RUN_ID", "").strip()
            or os.environ.get("CAIRN_PARENT_RUN", "").strip())


def results_target() -> tuple[str, str] | None:
    """``(bucket, key)`` for the results file, or ``None`` without ``CAIRN_CHECKPOINT_BUCKET`` or
    without a run to key it by (a shared key would let two runs overwrite each other's results)."""
    bucket = os.environ.get("CAIRN_CHECKPOINT_BUCKET", "").strip()
    key = run_key()
    if not bucket or not key:
        return None
    prefix = os.environ.get("CAIRN_CHECKPOINT_PREFIX", DEFAULT_PREFIX).strip("/") or DEFAULT_PREFIX
    return bucket, f"{prefix}/{key}/{RESULTS_KEY}"


def upload_results(path: Path, bucket: str, key: str) -> str:
    """Upload the results file and check it landed (size), or raise. Returns its ``s3://`` URI."""
    client = s3_client()
    client.upload_file(str(path), bucket, key)
    head = client.head_object(Bucket=bucket, Key=key)
    if int(head.get("ContentLength") or -1) != path.stat().st_size:
        raise FetchError(f"s3://{bucket}/{key} does not hold the {path.stat().st_size} bytes written")
    return f"s3://{bucket}/{key}"


__all__ = ["DEFAULT_PREFIX", "RESULTS_KEY", "FetchError", "fetch_dataset", "read_jsonl", "resolve_adapter",
           "results_target", "run_key", "s3_client", "split_pin", "upload_results"]
