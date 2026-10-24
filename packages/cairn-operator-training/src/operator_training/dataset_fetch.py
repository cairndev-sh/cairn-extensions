"""Resolve a training ``dataset_uri`` to something the trainer's loader can read.

The remote worker receives a ``dataset_uri`` that may be an ``s3://`` object (a
prepared dataset a pipeline exported, #701), a ``file://`` URI, a bare
Hugging Face Hub id, an ``http(s)://`` URL, or a plain local path. HF ``datasets``
cannot read ``s3://`` without ``s3fs`` (not a dependency here), so we fetch
``s3://`` down via boto3 (the ``[gpu]`` extra, imported only for an ``s3://`` URI;
the client is ``checkpoint_sync``'s, so the MinIO/R2 endpoint resolves identically) and hand the
trainer a local file. Everything else is passed through untouched for
``datasets`` to resolve.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from urllib.parse import unquote, urlparse

from operator_training import checkpoint_sync


def _local_from_file_uri(uri: str) -> str:
    """``file:///abs`` / ``file://localhost/abs`` → the decoded local path."""
    return unquote(urlparse(uri).path)


def _fetch_s3(uri: str) -> str:
    """Download an ``s3://bucket/key`` object to a deterministic temp path and
    return it. The temp dir is keyed by a hash of the URI so a re-resolve within
    the same box reuses the download instead of re-fetching."""
    parsed = urlparse(uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"malformed s3 uri: {uri!r}")
    digest = hashlib.sha256(uri.encode()).hexdigest()[:16]
    dest_dir = os.path.join(tempfile.gettempdir(), f"cairn-ds-{digest}")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(key) or "dataset")
    checkpoint_sync.s3_client().download_file(bucket, key, dest)
    return dest


def split_pin(uri: str) -> tuple[str, str]:
    """Split ``s3://…/x.jsonl#sha256=<hash>`` into ``(location, expected_hash)``.

    #1466's retrieval rule: a launch-resolved ``dataset:<name>@<v>`` ref carries
    the registry's content hash as a URI FRAGMENT, so a NAME guarantees BYTES.
    The pin is for whoever fetches; every reader needs the bare location first.

    Without this the fragment reached HF ``datasets``/fsspec as part of the
    object key and the run died with ``FileNotFoundError: Unable to find
    's3://…/train-ready.jsonl#sha256=951b…'`` — #1714, found on the first
    attempt to feed a train leg a governed ref (2026-08-27). A non-pinned URI
    returns ``(uri, "")``.
    """
    location, _, frag = str(uri or "").partition("#")
    return location, (frag[len("sha256="):] if frag.startswith("sha256=") else "")


def verify_pin(path: str, expected_hash: str, *, origin: str = "") -> None:
    """Fail closed when fetched bytes don't match the registry's pinned hash.

    The other half of the retrieval rule, and the reason stripping the pin is
    not merely "drop the fragment": ``ingest.normalize`` has verified this way
    since #1466, and a trainer that silently trains on unverified bytes is the
    gap that made the pin decorative on this path. No pin (or no local file, e.g.
    a passed-through HF id) → nothing to verify.
    """
    if not expected_hash or not os.path.isfile(path):
        return
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected_hash:
        raise RuntimeError(
            f"dataset verification FAILED for {origin or path}: fetched bytes hash "
            f"{got[:12]}… but the registry pinned {expected_hash[:12]}… — the object "
            "changed since registration (overwrite/corruption); refusing to train on "
            "unverified data (#1466)")


def resolve_dataset_uri(uri: str) -> str:
    """Return a path (or pass-through ref) the trainer's loader can read.

    - a ``#sha256=`` pin is stripped first, and VERIFIED against the fetched
      bytes when the result is a local file (#1466/#1714);
    - ``file://…`` → the decoded local path.
    - ``s3://bucket/key`` → download via boto3, return the local file path.
    - anything else (HF Hub id, local path, ``http(s)://``) → returned unchanged;
      ``datasets`` handles those directly.

    Idempotent for non-``s3`` inputs; ``s3://`` downloads once per box+uri.
    """
    if not uri:
        return uri
    uri, expected_hash = split_pin(uri)
    origin = uri
    if uri.startswith("file://"):
        uri = _local_from_file_uri(uri)
    elif uri.startswith("s3://"):
        uri = _fetch_s3(uri)
    verify_pin(uri, expected_hash, origin=origin)
    return uri

