"""Both dedup passes report the share of their input they dropped (cairn-extensions#39).

`dataset-prep/prepare` pauses for a person when a dedup pass drops more than a limit. A pipeline
expression is plain JMESPath and cannot divide, so the stages report `dropped_fraction`
(`dropped` / the rows the stage read, 4 places, 0.0 for an empty input) and the gate compares it.
Every backend: `dedup.exact` python and duckdb, `dedup.near` with MinHash and its honest skip.
The output is checked against the 1.1.0 stage contract the kernels now implement.
"""
from __future__ import annotations

import json

import pytest

from operator_dataset_checks import dataflow_stages as stages

#: Ten rows: one row four times (three exact duplicates dropped) and six distinct rows.
EXACT = [{"prompt": "capital of France", "completion": "Paris"}] * 4 + [
    {"prompt": f"question {i}", "completion": f"answer {i}"} for i in range(6)]

_WORDS = ("river stone bridge tower garden window market harbor engine signal winter summer copper "
          "silver forest valley meadow castle library lantern compass ladder anchor feather "
          "thunder candle mirror basket saddle orchard quarry glacier canyon prairie").split()


def _prose(seed: int, n: int = 40) -> str:
    """`n` words that share few 5-word shingles with another seed's."""
    import random

    rng = random.Random(seed)
    return " ".join(rng.choice(_WORDS) for _ in range(n))


#: Four paraphrases of one 40-word answer (only the last word differs) and four distinct rows.
NEAR = [{"prompt": "describe the scene", "completion": f"{_prose(0)} {w}"}
        for w in ("ago", "back", "past", "before")] + [
    {"prompt": f"describe scene {i}", "completion": _prose(i)} for i in range(1, 5)]


def _seed(workdir, rows) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "current.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _output_schema(kernel: str) -> dict:
    """The 1.1.0 output schema, as the installed contract wheel ships it."""
    from cairn.core.pack.contracts import installed_contract_bodies

    uri = f"contract://dataset.stage.{kernel}.output/1.1.0"
    bodies = installed_contract_bodies()
    assert uri in bodies, f"{uri} is not installed (cairn-contracts-dataset-stages 1.5.0)"
    return bodies[uri][1]


def _conforms(out: dict, kernel: str) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(out, _output_schema(kernel))
    assert getattr(stages, kernel).implements == (f"contract://dataset.stage.{kernel}/1.1.0",)


def test_exact_dedup_reports_the_share_it_dropped(tmp_path):
    _seed(tmp_path, EXACT)
    out = stages.dedup_exact(tmp_path, {}, [])
    assert (out["rows"], out["dropped"], out["dropped_fraction"]) == (7, 3, 0.3)
    _conforms(out, "dedup_exact")


def test_exact_dedup_on_duckdb_reports_the_same_share(tmp_path):
    pytest.importorskip("duckdb")
    _seed(tmp_path, EXACT)
    out = stages.dedup_exact(tmp_path, {"backend": "duckdb"}, [])
    assert (out["rows"], out["dropped"], out["dropped_fraction"]) == (7, 3, 0.3)
    _conforms(out, "dedup_exact")


def test_the_share_is_rounded_to_four_places(tmp_path):
    _seed(tmp_path, EXACT[:3] + EXACT[4:])    # 9 rows, 2 duplicates: 2/9 = 0.2222…
    assert stages.dedup_exact(tmp_path, {}, [])["dropped_fraction"] == 0.2222


def test_an_empty_input_drops_nothing_and_divides_by_nothing(tmp_path, monkeypatch):
    _seed(tmp_path, [])
    exact = stages.dedup_exact(tmp_path, {}, [])
    assert (exact["rows"], exact["dropped"], exact["dropped_fraction"]) == (0, 0, 0.0)
    _conforms(exact, "dedup_exact")
    monkeypatch.setattr("operator_dataset_checks._minhash.minhash_available", lambda: False)
    skipped = stages.dedup_near(tmp_path, {}, [])
    assert skipped["dropped_fraction"] == 0.0


def test_near_dedup_with_minhash_reports_the_share_of_what_it_read(tmp_path):
    pytest.importorskip("datasketch")
    _seed(tmp_path, NEAR)
    out = stages.dedup_near(tmp_path, {"threshold": 0.8, "ngram": 5}, [])
    assert out["checked"] is True and out["dropped_near"] == out["dropped"] == 3
    assert out["dropped_fraction"] == round(3 / 8, 4) == 0.375
    _conforms(out, "dedup_near")


def test_near_dedup_on_an_empty_input_with_minhash(tmp_path):
    pytest.importorskip("datasketch")
    _seed(tmp_path, [])
    out = stages.dedup_near(tmp_path, {}, [])
    assert (out["rows"], out["dropped"], out["dropped_fraction"]) == (0, 0, 0.0)


def test_a_skipped_near_pass_reports_zero(tmp_path, monkeypatch):
    """No [minhash] extra: nothing dropped, so 0.0 — the gate then reads only the exact pass."""
    monkeypatch.setattr("operator_dataset_checks._minhash.minhash_available", lambda: False)
    _seed(tmp_path, NEAR)
    out = stages.dedup_near(tmp_path, {}, [])
    assert (out["checked"], out["dropped"], out["dropped_fraction"]) == (False, 0, 0.0)
    _conforms(out, "dedup_near")
