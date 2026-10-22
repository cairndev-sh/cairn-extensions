"""The two kernels #2021 added to the admission chain, and the multi-probe
decontamination default (ADR-0102 D1).

These pin the STAGE side of the intake verdict: the stat keys
:func:`cairn.core.intake.verdict_from_flow_stats` reads are produced here, so a
rename on either side fails a test rather than silently emptying a report.

``dedup.near`` needs datasketch (the ``[minhash]`` extra) to do its work; the
tests that need it skip without it, exactly as ``test_dedup_minhash.py`` does.
The HONEST-SKIP behaviour — which is what a lean install actually runs — is
tested unconditionally, because that path is the one the default chain takes on
most deployments and it must never fail an import.
"""
from __future__ import annotations

import json

import pytest

from operator_dataset_checks import dataflow_stages as stages
from operator_dataset_checks._minhash import minhash_available

EXACT_DUP = {"prompt": "what is the capital of France", "completion": "Paris"}
#: Two paraphrase PAIRS — each differs from its twin by one word near the end,
#: which is exactly the leak an exact hash cannot see and MinHash can.
NEAR_A1 = {"prompt": "who wrote the Odyssey",
           "completion": "The poet Homer wrote the Odyssey in ancient Greece a very long time ago"}
NEAR_A2 = {"prompt": "who wrote the Odyssey",
           "completion": "The poet Homer wrote the Odyssey in ancient Greece a very long time back"}
NEAR_B1 = {"prompt": "where is Machu Picchu",
           "completion": "Machu Picchu sits high in the Andes mountains of southern Peru near Cusco"}
NEAR_B2 = {"prompt": "where is Machu Picchu",
           "completion": "Machu Picchu sits high in the Andes mountains of southern Peru near Lima"}

#: 3 exact duplicates (the same row four times = three dropped) and 2 near
#: duplicates (two paraphrase pairs = two dropped) — the acceptance fixture.
CORPUS = [
    EXACT_DUP, EXACT_DUP, EXACT_DUP, EXACT_DUP,
    NEAR_A1, NEAR_A2, NEAR_B1, NEAR_B2,
    {"prompt": "capital of Japan", "completion": "Tokyo"},
]


def _seed(workdir, rows) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "current.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _rows(workdir) -> list[dict]:
    return [json.loads(line) for line in
            (workdir / "current.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


# ─── dedup.near ──────────────────────────────────────────────────────────────


def test_the_chain_reports_exact_and_near_duplicate_counts_separately(tmp_path):
    """ACCEPTANCE: 3 exact + 2 near-duplicate rows, BOTH counts on the record.

    Two passes, two numbers: merging them would hide which pass found what, and
    "your data has 3 copy-paste duplicates" is a different message from "your
    data has 2 paraphrases of the same example"."""
    pytest.importorskip("datasketch")
    _seed(tmp_path, CORPUS)
    exact = stages.dedup_exact(tmp_path, {}, [])
    assert exact["dropped_exact"] == 3

    near = stages.dedup_near(tmp_path, {"threshold": 0.8}, [])
    assert near["checked"] is True
    assert near["dropped_near"] == 2
    assert near["rows"] == exact["rows"] - 2


def test_three_near_duplicates_report_dropped_near_three(tmp_path):
    """#2188 (T03) acceptance fixture: three paraphrase pairs → `dropped_near: 3`,
    `checked: True` — the stage side of the check map's `dedup_near: ran`."""
    pytest.importorskip("datasketch")
    near_c1 = {"prompt": "how tall is Everest",
               "completion": "Mount Everest rises about 8849 metres above sea level on the Nepal China border"}
    near_c2 = {"prompt": "how tall is Everest",
               "completion": "Mount Everest rises about 8849 metres above sea level on the Nepal Tibet border"}
    _seed(tmp_path, [NEAR_A1, NEAR_A2, NEAR_B1, NEAR_B2, near_c1, near_c2,
                     {"prompt": "capital of Japan", "completion": "Tokyo"}])
    near = stages.dedup_near(tmp_path, {"threshold": 0.8}, [])
    assert near["checked"] is True
    assert near["dropped_near"] == 3
    assert near["rows"] == 4


def test_without_the_minhash_extra_the_near_pass_skips_honestly(tmp_path, monkeypatch):
    """The default chain must still import a dataset on a lean install: the
    kernel reports ``checked: false`` with a note and leaves the corpus alone.

    (``dedup.exact``'s opted-in duckdb backend takes the opposite line and
    fails loudly — an explicit opt-in that downgrades silently would fake the
    evidence a gate reads. A default that refuses to run is a different bug.)"""
    monkeypatch.setattr(
        "operator_dataset_checks._minhash.minhash_available", lambda: False)
    _seed(tmp_path, CORPUS)
    out = stages.dedup_near(tmp_path, {}, [])
    assert out["checked"] is False
    assert out["dropped_near"] == 0
    assert "[minhash] extra" in out["note"]
    assert len(_rows(tmp_path)) == len(CORPUS)     # nothing was touched


# ─── profile.tokens ──────────────────────────────────────────────────────────


ROWS = [
    {"prompt": "a", "completion": "one two three"},
    {"prompt": "b", "completion": "one two three four five six seven eight"},
    {"prompt": "c", "completion": "word " * 40},
]


def test_the_profile_reports_the_total_p50_p95_and_a_histogram(tmp_path):
    _seed(tmp_path, ROWS)
    out = stages.profile_tokens(tmp_path, {}, [])
    assert out["token_backend"] == "whitespace"
    assert "whitespace WORD counts" in out["token_backend_note"]
    assert out["tokens_total"] == 3 + 8 + 40
    assert out["tokens_p50"] == 8
    assert out["tokens_max"] == 40
    # Power-of-two buckets, flat scalars (only scalars cross the boundary).
    assert out["bucket_0_128"] == 3
    assert out["bucket_8192_plus"] == 0


def test_over_length_rows_are_counted_and_kept_by_default(tmp_path):
    _seed(tmp_path, ROWS)
    out = stages.profile_tokens(tmp_path, {"max_seq_len": 10}, [])
    assert out["over_length_rows"] == 1
    assert out["over_length_policy"] == "count"
    assert len(_rows(tmp_path)) == 3              # kept, never truncated


def test_the_drop_policy_removes_them_and_still_reports_the_count(tmp_path):
    _seed(tmp_path, ROWS)
    out = stages.profile_tokens(tmp_path, {"max_seq_len": 10, "over_length": "drop"}, [])
    assert out["over_length_rows"] == 1
    assert out["rows"] == 2
    kept = _rows(tmp_path)
    assert len(kept) == 2
    # The long row is GONE, not shortened — nothing in the file was rewritten.
    assert all(len(r["completion"].split()) <= 10 for r in kept)


def test_the_refuse_policy_leaves_the_rows_alone_for_the_verdict_to_refuse(tmp_path):
    """``refuse`` is a verdict decision, not a data edit: the kernel reports,
    and ``cairn.core.intake`` turns the report into ``blocked`` so the launch
    door refuses the version. Truncating here would destroy the evidence."""
    _seed(tmp_path, ROWS)
    out = stages.profile_tokens(tmp_path, {"max_seq_len": 10, "over_length": "refuse"}, [])
    assert out["over_length_rows"] == 1
    assert out["over_length_policy"] == "refuse"
    assert len(_rows(tmp_path)) == 3


def test_truncate_is_not_a_policy(tmp_path):
    _seed(tmp_path, ROWS)
    with pytest.raises(ValueError, match="over_length must be one of"):
        stages.profile_tokens(tmp_path, {"over_length": "truncate"}, [])


def test_an_unloadable_tokenizer_falls_back_to_whitespace_and_says_so(tmp_path):
    _seed(tmp_path, ROWS)
    out = stages.profile_tokens(
        tmp_path, {"base_model": "definitely/not-a-real-model-2021"}, [])
    assert out["token_backend"] == "whitespace"
    assert "definitely/not-a-real-model-2021" in out["token_backend_note"]
    assert out["tokens_total"] == 51


# ─── decontam.apply against every holdout ────────────────────────────────────


def test_the_probe_is_the_union_of_every_eval_set_it_was_given(tmp_path):
    """One index over the union — the same per-row decision as running the op
    once per set, at one pass instead of N."""
    _seed(tmp_path, [{"q": "alpha"}, {"q": "beta"}, {"q": "gamma"}])
    (tmp_path / "golden.jsonl").write_text(json.dumps({"q": "alpha"}), encoding="utf-8")
    (tmp_path / "test.jsonl").write_text(json.dumps({"q": "gamma"}), encoding="utf-8")

    out = stages.decontam_apply(tmp_path, {
        "eval_paths": [str(tmp_path / "golden.jsonl"), str(tmp_path / "test.jsonl")]}, [])
    assert out["checked"] is True
    assert out["dropped"] == 2
    assert _rows(tmp_path) == [{"q": "beta"}]
    assert out["eval_sets"].count(",") == 1       # both, as a flat scalar


def test_no_probe_at_all_is_an_honest_skip(tmp_path):
    _seed(tmp_path, [{"q": "alpha"}])
    out = stages.decontam_apply(tmp_path, {}, [])
    assert out["checked"] is False
    assert out["eval_sets"] == ""
    assert _rows(tmp_path) == [{"q": "alpha"}]


def test_an_explicit_eval_path_still_wins_over_the_default_list(tmp_path):
    _seed(tmp_path, [{"q": "alpha"}, {"q": "beta"}])
    (tmp_path / "named.jsonl").write_text(json.dumps({"q": "beta"}), encoding="utf-8")
    (tmp_path / "other.jsonl").write_text(json.dumps({"q": "alpha"}), encoding="utf-8")
    out = stages.decontam_apply(tmp_path, {
        "eval_path": str(tmp_path / "named.jsonl"),
        "eval_paths": [str(tmp_path / "other.jsonl")]}, [])
    assert _rows(tmp_path) == [{"q": "alpha"}]    # only the named set was used
    assert out["eval_sets"].endswith("named.jsonl")


def test_a_probe_that_cannot_be_read_is_left_out_of_eval_sets(tmp_path):
    """``checked`` has to mean "compared against these", so a set that would
    not read must not appear in the list the record shows."""
    _seed(tmp_path, [{"q": "alpha"}])
    out = stages.decontam_apply(tmp_path, {
        "eval_paths": [str(tmp_path / "missing.jsonl")]}, [])
    assert out["checked"] is False
    assert out["eval_sets"] == ""


@pytest.mark.skipif(not minhash_available(), reason="needs the [minhash] extra")
def test_the_near_pass_is_deterministic(tmp_path):
    _seed(tmp_path, CORPUS)
    stages.dedup_exact(tmp_path, {}, [])
    first = stages.dedup_near(tmp_path, {}, [])
    _seed(tmp_path, CORPUS)
    stages.dedup_exact(tmp_path, {}, [])
    assert stages.dedup_near(tmp_path, {}, [])["dropped_near"] == first["dropped_near"]
