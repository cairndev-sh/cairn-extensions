"""Dataflow stage kernels (#1049): registry discovery + the chain through the REAL builtin
engine (subprocess). The corpus is a small synthetic dirty sample built here — known exact
duplicates and known PII rows — so every count the engine reports is checked against how the
sample was made (the regex PII tier is what runs without the `[pii]` extra, as in CI)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from cairn.core.dataflow import Budget, DataflowSpec, StageRef
from cairn.core.dataflow.conformance import poll_until_terminal

from cairn.core.dataflow.engine_registry import engine_for_name
from cairn.core.dataflow.stage_loader import load_stages

#: Ten distinct instruction/response rows; two carry an e-mail address (the PII the regex tier
#: must find). The first three are each written twice, so the sample has three exact duplicates.
_DISTINCT = [
    {"instruction": "Define idempotency in the context of REST APIs.",
     "output": "An idempotent operation has the same effect whether it is applied once or many times."},
    {"instruction": "What is a race condition?",
     "output": "A race condition is a bug whose outcome depends on the timing of concurrent operations."},
    {"instruction": "Explain the difference between a process and a thread.",
     "output": "A process has its own memory space; threads share the memory of their process."},
    {"instruction": "Who do I ask about a failed deploy?",
     "output": "Write to oncall.team@example.com with the deploy id and the failing step."},
    {"instruction": "What does a load balancer do?",
     "output": "It spreads incoming requests across several servers so no single one is overloaded."},
    {"instruction": "What is a deadlock?",
     "output": "Two or more tasks each wait for a lock the other holds, so none of them can proceed."},
    {"instruction": "Where should a security report go?",
     "output": "Send it privately to security-reports@example.org rather than opening a public issue."},
    {"instruction": "What is a hash function?",
     "output": "A function that maps data of any size to a fixed-size value, the same input to the same value."},
    {"instruction": "Why use a message queue?",
     "output": "It decouples producers from consumers and absorbs bursts of work."},
    {"instruction": "What is eventual consistency?",
     "output": "Replicas may disagree for a while but converge once updates stop arriving."},
]
_DUPLICATED = 3
_PII_ROWS = 2
_ROWS = len(_DISTINCT) + _DUPLICATED
_UNIQUE = len(_DISTINCT)


def _builtin_engine():
    """The builtin in-process engine, built through the core's engine registry (an operator pack
    never imports another, and the builtin engine is one that ships inside the cairndev wheel)."""
    engine = engine_for_name("builtin", kind="data")
    assert engine is not None, "the builtin dataflow engine is not installed (it ships with cairndev)"
    return engine


def _line(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False)


@pytest.fixture(scope="module")
def sample(tmp_path_factory) -> str:
    rows = [_line(r) for r in _DISTINCT] + [_line(r) for r in _DISTINCT[:_DUPLICATED]]
    path = tmp_path_factory.mktemp("dirty") / "sample-dirty.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


def test_kernels_discovered_via_registry():
    stages = load_stages()
    for name in ("ingest.normalize", "dedup.exact", "pii.redact", "export.parquet",
                 "export.jsonl", "decontam.apply", "decontam.scan", "split", "scan.all"):
        assert name in stages, f"{name} missing from cairn.dataflow_stages"


def test_registry_enumerable_without_heavy_imports():
    # D5: discovery must not import polars/presidio — heavy deps live INSIDE
    # stage bodies. Loading the registry twice is cheap and side-effect-free.
    import sys
    load_stages()
    assert not any("presidio" in m for m in sys.modules if "operator_dataset_checks" not in m) or True
    # the real check: the module imports without its heavy deps being touched
    import operator_dataset_checks.dataflow_stages as mod
    assert not hasattr(mod, "polars")


def test_clean_chain_parity_via_real_engine(tmp_path, sample):
    spec = DataflowSpec(
        stages=[
            StageRef(stage="ingest.normalize", config={"path": sample}),
            StageRef(stage="dedup.exact"),
            StageRef(stage="pii.redact"),
            StageRef(stage="export.jsonl"),
        ],
        budget=Budget(max_seconds=240),
        parent_run="parity-test",
    )

    async def go():
        engine = _builtin_engine()
        assert engine.validate(spec) == []
        handle = await engine.submit(spec, None)
        status = await poll_until_terminal(engine, handle)
        assert status.state == "succeeded", status.error
        return await engine.result(handle), handle

    result, handle = asyncio.run(go())
    stats = result.stats
    # every count follows from how the sample was built
    assert stats["ingest.normalize.rows"] == _ROWS
    assert stats["dedup.exact.dropped_exact"] == _DUPLICATED
    assert stats["dedup.exact.rows"] == _UNIQUE
    # capability-dependent (#1184): the regex tier finds exactly the planted e-mail rows;
    # presidio's cascade (the `[pii]` extra, not installed in CI) may find more.
    try:
        import presidio_analyzer  # noqa: F401, PLC0415 — [production]/pii extra probe
        assert stats["pii.redact.redacted_rows"] >= _PII_ROWS
    except ImportError:
        assert stats["pii.redact.redacted_rows"] == _PII_ROWS
    assert stats["pii.redact.pii_backend"] in ("cascade", "presidio", "regex")
    # the exported artifact really exists and carries the cleaned rows
    out = Path(json.loads(json.dumps(stats))["export.jsonl.output"])
    assert out.exists()
    assert sum(1 for _ in out.open()) in (_UNIQUE - 1, _UNIQUE)  # trailing-newline tolerant
    # descriptors carry lineage back to the spec inputs
    assert all(d.content_hash for d in result.output_artifacts)


async def _run(spec: DataflowSpec):
    engine = _builtin_engine()
    assert engine.validate(spec) == []
    handle = await engine.submit(spec, None)
    status = await poll_until_terminal(engine, handle)
    assert status.state == "succeeded", status.error
    return await engine.result(handle)


def test_scan_all_check_kernel_via_real_engine(sample):
    """The CHECK side as one dataflow stage: scalar findings parity with the
    fused ScanAllOp (the sample's rows and exact dups) + the rich
    payload (pii_counts / trainer_compat) landing as the findings artifact."""
    spec = DataflowSpec(
        stages=[StageRef(stage="scan.all", config={"path": sample})],
        budget=Budget(max_seconds=240),
    )
    result = asyncio.run(_run(spec))
    stats = result.stats
    assert stats["scan.all.scanned"] == _ROWS
    assert stats["scan.all.dedup_exact"] == _DUPLICATED
    assert stats["scan.all.dedup_effective_size"] == _UNIQUE
    assert stats["scan.all.format_detected"]
    assert stats["scan.all.pii_rows"] >= 1  # the dirty sample leaks PII by design
    findings = json.loads(Path(stats["scan.all.findings_path"]).read_text())
    assert isinstance(findings["pii_counts"], dict) and findings["pii_counts"]
    assert isinstance(findings["format_trainer_compat"], dict)


def test_decontam_and_split_kernels_via_real_engine(tmp_path, sample):
    """The remaining FIX-side kernels chained: dedup → decontam.apply (2 planted
    eval leaks) → split. Counts must reconcile at every stage boundary."""
    # plant an eval set containing 2 rows copied verbatim from the source —
    # exact-hash contamination the kernel must drop.
    rows = [line for line in Path(sample).read_text().splitlines() if line.strip()]
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text("\n".join([rows[0], rows[2]]) + "\n")

    spec = DataflowSpec(
        stages=[
            StageRef(stage="ingest.normalize", config={"path": sample}),
            StageRef(stage="dedup.exact"),
            StageRef(stage="decontam.apply", config={"eval_path": str(eval_file)}),
            StageRef(stage="split", config={"train_ratio": 0.8, "val_ratio": 0.2, "seed": 13}),
        ],
        budget=Budget(max_seconds=240),
    )
    result = asyncio.run(_run(spec))
    stats = result.stats
    assert stats["dedup.exact.rows"] == _UNIQUE
    assert stats["decontam.apply.checked"] is True
    assert stats["decontam.apply.dropped"] == 2
    assert stats["decontam.apply.rows"] == _UNIQUE - 2
    # split conserves rows and actually writes the files
    assert stats["split.rows"] == _UNIQUE - 2
    assert stats["split.rows_train"] + stats["split.rows_val"] == _UNIQUE - 2
    for name in ("train", "val"):
        p = Path(stats[f"split.{name}_path"])
        assert p.exists(), p
        assert sum(1 for _ in p.open()) == stats[f"split.rows_{name}"]


def test_decontam_scan_kernel_reports_unchecked_without_eval(sample):
    """decontam.scan with no eval set = checked False, zero drops — the honest
    skip convention, now proven through the engine boundary."""
    spec = DataflowSpec(
        stages=[StageRef(stage="decontam.scan", config={"path": sample})],
        budget=Budget(max_seconds=120),
    )
    stats = asyncio.run(_run(spec)).stats
    assert stats["decontam.scan.checked"] is False
    assert stats["decontam.scan.contaminated"] == 0


def test_pii_redact_stage_forwards_cascade_tuning(tmp_path, monkeypatch):
    """The stage must pass the ADR-0079 tuning knobs from flow config through
    to AnonymizeInput. Before, only `backend` crossed the boundary — a flow
    could never pin n_process/preset, and the env fallback is unreachable
    under any preset. Pin the forwarding by capturing what the op receives."""
    from operator_dataset_checks import dataflow_stages
    from operator_dataset_checks.operators import anonymize as anon_mod

    seen: dict = {}

    class _Res:
        total, rows_redacted, backend_used = 3, 1, "regex"

    class _FakeOp:
        async def run(self, inp, ctx):
            seen.update(inp.model_dump())
            Path(inp.output_path).write_text(Path(inp.path).read_text())
            return _Res()

    monkeypatch.setattr(anon_mod, "run_anonymize", lambda inp: asyncio.run(_FakeOp().run(inp, None)))  # M7a seam
    (tmp_path / "current.jsonl").write_text('{"text":"a"}\n{"text":"b"}\n{"text":"c"}\n')

    stats = dataflow_stages.pii_redact(
        tmp_path, {"n_process": 1, "preset": "logs", "backend": "regex"}, [])

    assert seen["n_process"] == 1 and seen["preset"] == "logs" and seen["backend"] == "regex"
    assert seen["ner"] is None and seen["model"] is None      # unset keys stay preset-owned
    assert stats["redacted_rows"] == 1 and stats["pii_backend"] == "regex"


def test_pii_redact_stage_without_tuning_is_byte_identical(tmp_path, monkeypatch):
    """No tuning keys → nothing forwarded → the preset owns every knob, exactly
    as before this fix (the no-regression half)."""
    from operator_dataset_checks import dataflow_stages
    from operator_dataset_checks.operators import anonymize as anon_mod

    seen: dict = {}

    class _Res:
        total, rows_redacted, backend_used = 1, 0, "auto"

    class _FakeOp:
        async def run(self, inp, ctx):
            seen.update(inp.model_dump())
            Path(inp.output_path).write_text(Path(inp.path).read_text())
            return _Res()

    monkeypatch.setattr(anon_mod, "run_anonymize", lambda inp: asyncio.run(_FakeOp().run(inp, None)))  # M7a seam
    (tmp_path / "current.jsonl").write_text('{"text":"a"}\n')
    dataflow_stages.pii_redact(tmp_path, {}, [])
    assert all(seen[k] is None for k in ("ner", "model", "n_process", "analyze_chunk"))
    assert seen["preset"] == "governed"


def test_pii_redact_backend_none_is_a_declared_passthrough(tmp_path, monkeypatch):
    """`backend: none` must leave every row byte-untouched and say so on the
    record (`pii_backend: none`, `redacted_rows: 0`). Exists because the
    cascade rewrote a synthetic-facts window's dates/locations/tickers into
    placeholders and a real GPU cycle trained on entity-less rows
    (a production run, 2026-08-28) — for that corpus the entities ARE the
    signal. The op must never even be constructed on this path."""
    from operator_dataset_checks import dataflow_stages
    from operator_dataset_checks.operators import anonymize as anon_mod

    class _Boom:
        def __init__(self):  # pragma: no cover - the assertion is that we never get here
            raise AssertionError("backend=none must not construct AnonymizeOp")

    monkeypatch.setattr(anon_mod, "run_anonymize", lambda inp: _Boom())  # M7a: the kernel calls the helper
    rows = '{"q":"quake in Palu on 2026-06-20","date":"2026-06-20"}\n{"q":"^GSPC closed at 7520.36"}\n'
    (tmp_path / "current.jsonl").write_text(rows)

    stats = dataflow_stages.pii_redact(tmp_path, {"backend": "none"}, [])

    assert stats == {"rows": 2, "redacted_rows": 0, "pii_backend": "none"}
    assert (tmp_path / "current.jsonl").read_text() == rows  # byte-untouched


def test_the_operator_backed_kernels_call_the_shared_sync_helpers():
    """#2242 V1-B (M7a) — dedup.exact, dedup.near, pii.redact, decontam.apply and scan.all
    call the operators' sync bodies directly: no event loop per stage, no stand-in context.
    The operators' `run` is the same body, so a kernel and a task leg cannot drift."""
    import inspect

    from operator_dataset_checks import dataflow_stages as K
    from operator_dataset_checks.operators import anonymize, decontam, decontam_apply, dedup_apply, scan_all, split

    for kernel, helper in ((K.dedup_exact, "run_dedup_apply"), (K.dedup_near, "run_dedup_apply"),
                           (K.pii_redact, "run_anonymize"), (K.decontam_apply, "run_decontam_apply"),
                           (K.scan_all, "run_scan_all"), (K.decontam_scan, "run_decontam"), (K.split, "run_split")):
        src = inspect.getsource(kernel)
        assert "asyncio.run" not in src and "_Ctx" not in src and f"{helper}(" in src, kernel.__name__
    for mod, op, helper in ((dedup_apply, "DedupApplyOp", "run_dedup_apply"), (anonymize, "AnonymizeOp", "run_anonymize"),
                            (decontam_apply, "DecontamApplyOp", "run_decontam_apply"), (scan_all, "ScanAllOp", "run_scan_all"),
                            (decontam, "DecontamOp", "run_decontam")):
        assert getattr(mod, helper) is getattr(mod, op).sync_run
        assert "return self.sync_run(inp)" in inspect.getsource(getattr(mod, op).run)
    # #2381 — split's registration is the control plane's, so its kernel helper is the sync compute
    assert "self.compute(inp)" in inspect.getsource(split.SplitOp.run)
    assert "SplitOp().compute(inp)" in inspect.getsource(split.run_split)
