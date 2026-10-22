"""s3:// jsonl routing in ``_io`` + ``dataset/replay``'s s3 out_path.

Found live on a production replay leg (2026-08-28):

* an ``s3://…jsonl`` SOURCE routed through the Arrow builder, whose type
  inference turned ISO date strings into ``datetime`` objects — and
  ``canonical_hash`` (dedup) crashed on ``json.dumps``. The fix routes s3
  jsonl through the same verbatim stdlib line reader local jsonl uses.
* an ``s3://…`` OUT_PATH went into ``JsonlWriter`` as ``Path("s3://…")`` — a
  LOCAL ghost file in a directory literally named ``s3:/`` — so the mix never
  reached S3 and a train leg on another box 404'd.

Hermetic: boto3 is stubbed at the seam (`_s3_client`); no network, no AWS.
"""
from __future__ import annotations

import json
from pathlib import Path

from cairn.core.operator import OperationContext
from operator_dataset_checks import _io
from operator_dataset_checks._io import iter_rows, load_rows
from operator_dataset_checks.operators.replay import ReplayOp
from operator_dataset_checks.schemas.replay import ReplayInput


class _FakeS3Client:
    """download_file/upload_file backed by a local directory, keyed bucket/key."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.uploads: list[tuple[str, str]] = []

    def _path(self, bucket: str, key: str) -> Path:
        return self.root / bucket / key

    def download_file(self, bucket: str, key: str, dest: str) -> None:
        Path(dest).write_bytes(self._path(bucket, key).read_bytes())

    def upload_file(self, src: str, bucket: str, key: str) -> None:
        p = self._path(bucket, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(Path(src).read_bytes())
        self.uploads.append((bucket, key))


def _fake_s3(tmp_path, monkeypatch) -> _FakeS3Client:
    client = _FakeS3Client(tmp_path / "fake-s3")
    monkeypatch.setattr(_io, "_s3_client", lambda: client)
    return client


def _put(client: _FakeS3Client, bucket: str, key: str, rows) -> str:
    p = client._path(bucket, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return f"s3://{bucket}/{key}"


# The exact shape that broke: a date-like string field. Arrow infers it into a
# datetime; the verbatim reader must hand it back as the string it is.
_DATED = [{"question": f"q{i}", "answer": f"a{i}", "as_of": "2026-05-03"} for i in range(4)]


def test_load_rows_s3_jsonl_keeps_strings_verbatim(tmp_path, monkeypatch):
    client = _fake_s3(tmp_path, monkeypatch)
    uri = _put(client, "bkt", "data/facts.jsonl", _DATED)

    rows = load_rows(uri)

    assert rows == _DATED
    assert all(isinstance(r["as_of"], str) for r in rows)
    # the crash path: every row must survive the canonical-hash json.dumps
    for r in rows:
        json.dumps(r, sort_keys=True)


def test_iter_rows_s3_jsonl_streams_the_same_rows(tmp_path, monkeypatch):
    client = _fake_s3(tmp_path, monkeypatch)
    uri = _put(client, "bkt", "data/facts.jsonl", _DATED)

    assert list(iter_rows(uri)) == _DATED


def test_non_jsonl_s3_paths_do_not_take_the_verbatim_path():
    # .parquet / bare prefixes keep their existing Arrow routing untouched.
    assert not _io._is_s3_jsonl("s3://bkt/data/facts.parquet")
    assert not _io._is_s3_jsonl("s3://bkt/data")
    assert _io._is_s3_jsonl("s3://bkt/data/facts.jsonl")
    # a resolved-ref pin is stripped BEFORE _io sees the path (replay._strip_pin),
    # but the predicate must not be fooled by suffix casing.
    assert _io._is_s3_jsonl("s3://bkt/DATA/FACTS.JSONL")


async def test_replay_s3_out_path_uploads_the_mix(tmp_path, monkeypatch):
    client = _fake_s3(tmp_path, monkeypatch)
    new = _put(client, "bkt", "w2/new.jsonl", [{"q": f"n{i}", "as_of": "2026-07-01"} for i in range(8)])
    old = _put(client, "bkt", "w1/old.jsonl", [{"q": f"o{i}", "as_of": "2026-05-01"} for i in range(50)])
    dest = "s3://bkt/mix/replay-mix.jsonl"

    res = await ReplayOp().run(
        ReplayInput(new_rows_path=new, sources=[old], out_path=dest, ratio=0.2, seed=7),
        OperationContext(pipeline_id="t", node_name="replay"),
    )

    # the declared out_path is the output's truth AND the object really exists
    assert res.out_path == dest
    assert ("bkt", "mix/replay-mix.jsonl") in client.uploads
    written = [
        json.loads(l)
        for l in client._path("bkt", "mix/replay-mix.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(written) == res.total_rows == res.new_rows + res.replay_rows
    assert res.new_rows == 8 and res.replay_rows == 2  # 20% of the OUTPUT


async def test_replay_local_out_path_still_writes_locally(tmp_path, monkeypatch):
    client = _fake_s3(tmp_path, monkeypatch)
    new = _put(client, "bkt", "w2/new.jsonl", [{"q": f"n{i}"} for i in range(4)])
    out = tmp_path / "local-mix.jsonl"

    res = await ReplayOp().run(
        ReplayInput(new_rows_path=new, sources=[], out_path=str(out), ratio=0.0, seed=1),
        OperationContext(pipeline_id="t", node_name="replay"),
    )

    assert res.out_path == str(out)
    assert out.exists() and len(out.read_text().splitlines()) == 4
    assert client.uploads == []  # no s3 write for a local out_path
