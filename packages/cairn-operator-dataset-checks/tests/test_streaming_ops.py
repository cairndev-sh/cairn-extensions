"""#891 — streaming/sharded dataset I/O: the prep ops consume iter_rows (bounded
memory) instead of load_rows (whole file → list), and split writes sharded output.

Proves: (1) the streaming reservoir matches the list version byte-for-byte, (2) the
per-row split decision matches hash_split, (3) anonymize streams a byte-identical
redacted file, (4) pii_scan full-scans without materializing + caps its findings,
(5) split shards its output and the shards reassemble to the full split, leakage-free.
"""
from __future__ import annotations

import json
import math

from cairn.core.operator import OperationContext
from operator_dataset_checks._io import (
    ShardedJsonlWriter,
    _reservoir_sample,
    reservoir_sample_stream,
)
from operator_dataset_checks._split import hash_split, row_split_name
from operator_dataset_checks.operators.anonymize import AnonymizeOp
from operator_dataset_checks.operators.pii_scan import PiiScanOp
from operator_dataset_checks.operators.split import SplitOp
from operator_dataset_checks.schemas.anonymize import AnonymizeInput
from operator_dataset_checks.schemas.pii_scan import PiiInput
from operator_dataset_checks.schemas.split import SplitInput


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="stream")


def _write(tmp_path, rows, name="ds.jsonl") -> str:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def _read(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ── primitives ──────────────────────────────────────────────────────


def test_reservoir_stream_matches_list_version():
    rows = [{"i": i} for i in range(1000)]
    for k in (0, 1, 50, 999, 1000, 2000):
        listed = _reservoir_sample(rows, k)
        streamed = reservoir_sample_stream(iter(rows), k)
        assert streamed == listed, f"k={k}: streaming reservoir diverged from the list version"


def test_row_split_name_matches_hash_split():
    rows = [{"prompt": f"q{i}", "doc": f"d{i % 7}"} for i in range(300)]
    ratios = {"train": 0.8, "val": 0.2}
    for group_by in (None, "doc"):
        asg = hash_split(rows, ratios=ratios, seed=13, group_by=group_by)
        by_row = {i: name for name, idxs in asg.indices.items() for i in idxs}
        for i, row in enumerate(rows):
            assert row_split_name(row, ratios=ratios, seed=13, group_by=group_by) == by_row[i]


def test_sharded_writer_reassembles_and_names(tmp_path):
    w = ShardedJsonlWriter(tmp_path, "train", shard_size=3)
    rows = [{"i": i} for i in range(7)]
    with w:
        for r in rows:
            w.write(r)
    assert len(w.paths) == 3  # 3 + 3 + 1
    assert w.paths[0].endswith("train.jsonl") and w.paths[1].endswith("train.00001.jsonl")
    assert w.count == 7
    merged = [r for p in w.paths for r in _read(p)]
    assert merged == rows


# ── anonymize (streamed, byte-identical) ────────────────────────────


async def test_anonymize_streams_a_redacted_file(tmp_path):
    rows = [
        {"prompt": "email me at alice@example.com", "completion": "ok"},
        {"prompt": "no pii here", "completion": "fine"},
    ]
    src = _write(tmp_path, rows)
    out = str(tmp_path / "clean.jsonl")
    res = await AnonymizeOp().run(
        AnonymizeInput(path=src, output_path=out, backend="regex"), _ctx()
    )
    assert res.total == 2
    assert res.rows_redacted == 1
    written = _read(out)
    assert len(written) == 2
    assert "alice@example.com" not in json.dumps(written[0])  # redacted
    assert written[1] == rows[1]  # untouched row preserved verbatim
    # byte-shape: no trailing newline (matches the old "\n".join write)
    assert not open(out, encoding="utf-8").read().endswith("\n")


# ── pii_scan (full-scan streamed + bounded findings) ────────────────


async def test_pii_full_scan_streams_and_counts_exactly(tmp_path):
    rows = [{"t": f"contact bob{i}@x.com"} for i in range(50)] + [{"t": "clean"} for _ in range(50)]
    src = _write(tmp_path, rows)
    res = await PiiScanOp().run(PiiInput(path=src, backend="regex", sample_size=None), _ctx())
    assert res.total == 100
    assert res.rows_with_pii == 50  # exact, full scan
    assert res.sampled is False
    assert res.findings_truncated is False


async def test_pii_findings_are_capped_but_counts_stay_exact(tmp_path):
    rows = [{"t": f"m{i}@x.com"} for i in range(40)]
    src = _write(tmp_path, rows)
    res = await PiiScanOp().run(
        PiiInput(path=src, backend="regex", sample_size=None, max_findings=10), _ctx()
    )
    assert res.rows_with_pii == 40  # counts exact
    assert len(res.findings) == 10  # detail capped
    assert res.findings_truncated is True


# ── split (sharded, leakage-free, reassembles) ──────────────────────


async def test_split_shards_output_and_reassembles(tmp_path):
    rows = [{"prompt": f"q{i}", "completion": f"a{i}"} for i in range(40)]
    src = _write(tmp_path, rows)
    out_dir = str(tmp_path / "split")
    res = await SplitOp().run(
        SplitInput(
            path=src, output_dir=out_dir, backend="hash",
            train_ratio=0.75, val_ratio=0.25, seed=13, shard_size=7,
        ),
        _ctx(),
    )
    assert res.backend_used == "hash"
    assert res.total == 40
    assert res.no_leakage is True
    assert sum(res.counts.values()) == 40
    seen_rows = []
    for split, paths in res.shard_paths.items():
        # first shard is the primary path (back-compat)
        assert res.output_paths[split] == paths[0]
        assert len(paths) == math.ceil(res.counts[split] / 7) or res.counts[split] == 0
        reassembled = [r for p in paths for r in _read(p)]
        assert len(reassembled) == res.counts[split]
        seen_rows.extend(reassembled)
    # every input row lands in exactly one split's shards
    assert sorted(r["prompt"] for r in seen_rows) == sorted(r["prompt"] for r in rows)


async def test_split_unsharded_default_is_one_file_per_split(tmp_path):
    rows = [{"prompt": f"q{i}"} for i in range(20)]
    src = _write(tmp_path, rows)
    res = await SplitOp().run(
        SplitInput(path=src, output_dir=str(tmp_path / "s"), backend="hash"),
        _ctx(),
    )
    for split, paths in res.shard_paths.items():
        assert paths == [res.output_paths[split]]  # single file, back-compat shape
        assert res.output_paths[split].endswith(f"{split}.jsonl")


# ── the real prep chain: streamed anonymize → streamed FULL-scan (#889/#891) ──


async def test_streamed_anonymize_then_full_scan_certifies_clean(tmp_path):
    """The prep pipeline's OOM-critical path: scrub PII (streamed), then re-scan the
    SCRUBBED output with sample_size=null (the #889 governed FULL scan, streamed) —
    both bounded-memory, and the scrubbed bytes verify PII-free end to end."""
    rows = [{"prompt": f"reach me at user{i}@corp.com now", "completion": "sure"} for i in range(30)]
    src = _write(tmp_path, rows)
    scrubbed = str(tmp_path / "scrubbed.jsonl")

    a = await AnonymizeOp().run(
        AnonymizeInput(path=src, output_path=scrubbed, backend="regex"), _ctx()
    )
    assert a.total == 30 and a.rows_redacted == 30

    # governed re-scan: FULL (not sampled), streamed — must certify the scrub clean.
    scan = await PiiScanOp().run(
        PiiInput(path=scrubbed, backend="regex", sample_size=None), _ctx()
    )
    assert scan.total == 30
    assert scan.sampled is False
    assert scan.rows_with_pii == 0  # every email was scrubbed; the full scan proves it
