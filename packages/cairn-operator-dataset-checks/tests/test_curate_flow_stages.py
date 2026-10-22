"""C1b (#1439) — the curation leg's stages: ``sample.rows`` (repeatable reservoir /
stride) and ``curate.score`` (CurateOp through the governed LLM chokepoint,
FLAT scalar stats + a per-row scores artifact). The judge is stubbed here; the
live check ran deepseek/deepseek-v4-flash on 40 real rows (40/40 scored)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from operator_dataset_checks import dataflow_stages as S


def _row(i: int) -> dict:
    return {"messages": [{"role": "user", "content": f"question number {i} about things"},
                         {"role": "assistant", "content": f"answer {i} " * (i % 5 + 1)}]}


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    src = tmp_path / "rows.jsonl"
    src.write_text("".join(json.dumps(_row(i)) + "\n" for i in range(200)), encoding="utf-8")
    wd = tmp_path / "wd"
    wd.mkdir()
    S.ingest_normalize(wd, {"path": str(src)}, [])
    return wd


def _lines(wd: Path) -> list[dict]:
    return [json.loads(l) for l in S._current(wd).read_text().splitlines()]


def test_sample_rows_reservoir_is_repeatable_and_bounded(workdir: Path, tmp_path: Path):
    pytest.importorskip("duckdb")
    import shutil
    keep = tmp_path / "keep.jsonl"
    shutil.copy(S._current(workdir), keep)
    r1 = S.sample_rows(workdir, {"n": 25, "seed": 7}, [])
    rows1 = _lines(workdir)
    shutil.copy(keep, S._current(workdir))
    S.sample_rows(workdir, {"n": 25, "seed": 7}, [])
    rows2 = _lines(workdir)
    # #2029 — the stats now also carry the SEED and an order-independent
    # row-hash digest of what was kept, so "did these two runs see the same
    # rows?" is answered from the record instead of by re-reading the corpus.
    assert r1 == {"rows": 25, "total": 200, "sampled": True, "method": "reservoir", "n": 25,
                  "seed": 7, "digest": r1["digest"]}
    assert r1["digest"].startswith("sha256:")
    assert rows1 == rows2 and len(rows1) == 25          # REPEATABLE (seed)
    assert all(r in [_row(i) for i in range(200)] for r in rows1)


def test_sample_rows_stride_and_passthrough(workdir: Path):
    r = S.sample_rows(workdir, {"n": 50, "method": "stride"}, [])
    assert r["rows"] == 50 and r["sampled"] and r["method"] == "stride"
    r = S.sample_rows(workdir, {"n": 500}, [])           # corpus ≤ n → untouched
    # `n` is what was ASKED FOR, never clamped to the corpus (#2029): "500
    # wanted, 50 present, not sampled" is the honest reading.
    assert r == {"rows": 50, "total": 50, "sampled": False, "method": "reservoir", "n": 500,
                 "seed": 13, "digest": r["digest"]}


def test_sample_rows_fraction_and_hash_method_are_seed_reproducible(workdir: Path, tmp_path: Path):
    """#2029 — the proxy-run front door: `fraction` sizes the sample against the
    corpus, and `method: hash` keeps the rows with the smallest H(seed ‖ row
    hash), which depends on the corpus and the seed and on NOTHING else — not
    row order, not duckdb's version. That is what makes two proxy runs of
    different configurations comparable rather than two experiments."""
    import shutil
    keep = tmp_path / "keep.jsonl"
    shutil.copy(S._current(workdir), keep)
    one = S.sample_rows(workdir, {"fraction": 0.05, "method": "hash", "seed": 7}, [])
    rows1 = _lines(workdir)
    shutil.copy(keep, S._current(workdir))
    two = S.sample_rows(workdir, {"fraction": 0.05, "method": "hash", "seed": 7}, [])
    assert one["n"] == 10 and one["rows"] == 10          # 5 % of 200
    assert one["digest"] == two["digest"] and rows1 == _lines(workdir)
    shutil.copy(keep, S._current(workdir))
    other = S.sample_rows(workdir, {"fraction": 0.05, "method": "hash", "seed": 8}, [])
    assert other["digest"] != one["digest"], "a different seed must keep different rows"


def test_curate_score_stage_flat_stats_and_scores_artifact(workdir: Path, monkeypatch):
    """Judge stubbed: score = 5 for even rows, 2 for odd. keep_fraction=0.5
    keeps the top half; stats are scalars (boundary rule); per-row scores land
    in curate_scores.json."""
    from operator_dataset_checks.operators import curate as C

    async def fake_llm(messages, model, temperature=0.0, **kw):
        text = messages[-1]["content"]
        n = int(text.split("question number ")[1].split(" ")[0])
        return {"choices": [{"message": {"content": "5" if n % 2 == 0 else "2"}}]}
    monkeypatch.setattr(C, "llm_complete", fake_llm)

    S.sample_rows(workdir, {"n": 40, "method": "stride"}, [])
    r = S.curate_score(workdir, {"method": "alpagasus", "model": "stub/judge",
                                 "keep_fraction": 0.5, "max_samples": 40}, [])
    assert r["scored"] == 40 and r["kept"] == 20 and r["dropped"] == 20
    assert r["parse_failures"] == 0 and r["mean_score"] == 3.5
    assert r["model"] == "stub/judge" and r["criterion"] == "keep_fraction=0.5"
    assert r["bucket_5"] == 20 and r["bucket_2"] == 20
    assert all(not isinstance(v, (dict, list)) for v in r.values())
    from cairn.core.dataflow.contract import ArtifactDescriptor
    ArtifactDescriptor(uri="file:///x", kind="dataset", content_hash="0" * 64, stats=r)
    kept = _lines(workdir)
    assert len(kept) == 20 and all(
        int(k["messages"][0]["content"].split("question number ")[1].split(" ")[0]) % 2 == 0 for k in kept)
    scores = json.loads((workdir / "curate_scores.json").read_text())
    assert len(scores["scores"]) == 40 and scores["score_distribution"] == {"5": 20, "2": 20}


def test_curate_score_stage_is_a_registered_kernel():
    from importlib.metadata import entry_points
    names = {e.name for e in entry_points(group="cairn.dataflow_stages")}
    assert {"sample.rows", "curate.score"} <= names
    assert S.sample_rows.risk_class == "external_write" and S.curate_score.risk_class == "external_write"


def test_curate_score_spend_reaches_the_stage_cost_through_the_runner(workdir: Path, monkeypatch):
    """#1441 — the curation leg's live symptom was `cost_usd: 0.0` after 100 paid
    judge calls. Through the kernel runner the stage now reports them: calls,
    tokens and `cost_usd`. Only the provider call is stubbed."""
    import contextvars
    import os

    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm

    from cairn.core.dataflow import DataflowSpec, StageRef
    from cairn.core.dataflow import runner as R

    async def fake_acompletion(**params):
        return litellm.ModelResponse(
            model=params["model"], choices=[{"index": 0, "message": {"role": "assistant", "content": "4"}}],
            usage=litellm.Usage(prompt_tokens=200, completion_tokens=5, total_tokens=205))
    monkeypatch.setattr("cairn.core.llm.client.acompletion", fake_acompletion)
    S.sample_rows(workdir, {"n": 10, "method": "stride"}, [])
    monkeypatch.setattr(R, "load_stages", lambda: {"curate.score": S.curate_score})
    spec = DataflowSpec(dataflow_ref="t/curate", stages=[StageRef(
        stage="curate.score", config={"method": "alpagasus", "model": "gpt-4o-mini",
                                      "keep_fraction": 0.5, "max_samples": 10})])
    res = contextvars.Context().run(R.run_dataflow, spec, workdir)
    st = res.stats
    assert st["curate.score.scored"] == 10 and st["curate.score.llm_calls"] == 10
    assert st["curate.score.llm_tokens_in"] == 2000 and st["curate.score.llm_unpriced_calls"] == 0
    assert st["curate.score.cost_usd"] > 0 and res.cost_usd == st["curate.score.cost_usd"]
