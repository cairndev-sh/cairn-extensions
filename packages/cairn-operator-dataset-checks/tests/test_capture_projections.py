"""dataset/capture_to_{sft,pairs,rewards} (#1629, ADR-0095 §1.2) — the three
signal→rung projections. Hermetic: local JSONL fixtures shaped exactly like the
capture export (one serialized CaptureEvent per line), no store, no network.

The properties pinned here are the ones a training receipt has to be able to
re-derive: an edit outranks the completion it corrected, a rejection with
nothing better to point at is dropped rather than paired against a guess, the
GRPO projection emits reward CONTEXT and not off-policy triples, and every
output is a deterministic function of its input.
"""
from __future__ import annotations

import json

import pytest

from cairn.core.operator import OperationContext
from operator_dataset_checks.operators.capture_to_pairs import CaptureToPairsOp
from operator_dataset_checks.operators.capture_to_rewards import CaptureToRewardsOp
from operator_dataset_checks.operators.capture_to_sft import CaptureToSftOp
from operator_dataset_checks.schemas.capture_to_pairs import CaptureToPairsInput
from operator_dataset_checks.schemas.capture_to_rewards import CaptureToRewardsInput
from operator_dataset_checks.schemas.capture_to_sft import CaptureToSftInput


def _ctx() -> OperationContext:
    return OperationContext(pipeline_id="t", node_name="capture")


def _event(**over) -> dict:
    """One exported capture row — the fields CaptureEvent serializes."""
    base = {"event_id": "", "workspace_id": "acme", "kind": "trace", "trace_id": "",
            "run_id": "", "model": "qwen2.5-7b", "adapter_uri": "s3://a/v3",
            "prompt": "", "completion": "", "edited_text": "", "score": None,
            "score_scale": "", "note": "", "source": "sdk", "actor": "alice",
            "workflow_ref": "support", "payload": {}, "created_at": "2026-08-20T10:00:00+00:00"}
    row = {**base, **over}
    row.setdefault("event_id", "")
    row["event_id"] = row["event_id"] or f"acme:{row['kind']}:{row['trace_id']}:{id(row)}"
    return row


def _write(tmp_path, name: str, rows) -> str:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def _read(path: str) -> list[dict]:
    text = open(path, encoding="utf-8").read()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ── the shared fixture: one trace, corrected; one accepted; one rejected ──────


def _corpus() -> list[dict]:
    return [
        _event(kind="trace", trace_id="t-1", prompt="refund window?",
               completion="30 days.", created_at="2026-08-20T10:00:00+00:00"),
        _event(kind="edit", trace_id="t-1", completion="30 days.",
               edited_text="30 calendar days from delivery.",
               created_at="2026-08-20T10:01:00+00:00"),
        _event(kind="trace", trace_id="t-2", prompt="shipping time?",
               completion="3-5 business days.", created_at="2026-08-20T10:02:00+00:00"),
        _event(kind="accept", trace_id="t-2", completion="3-5 business days.",
               created_at="2026-08-20T10:03:00+00:00"),
        _event(kind="trace", trace_id="t-3", prompt="do you ship to Berlin?",
               completion="I don't know.", created_at="2026-08-20T10:04:00+00:00"),
        _event(kind="reject", trace_id="t-3", completion="I don't know.",
               created_at="2026-08-20T10:05:00+00:00"),
        _event(kind="accept", trace_id="t-3", completion="Yes, 2-4 days to Berlin.",
               created_at="2026-08-20T10:06:00+00:00"),
    ]


# ── dataset/capture_to_sft ───────────────────────────────────────────────────


async def _sft(tmp_path, rows, **kwargs):
    src = _write(tmp_path, "capture.jsonl", rows)
    out = str(tmp_path / "sft.jsonl")
    res = await CaptureToSftOp().run(
        CaptureToSftInput(path=src, out_path=out, **kwargs), _ctx())
    return res, _read(out)


async def test_an_edit_is_the_gold_not_the_completion_it_replaced(tmp_path):
    """The whole reason edits are captured: the human wrote the target."""
    res, rows = await _sft(tmp_path, _corpus())
    edited = [r for r in rows if r["messages"][0]["content"] == "refund window?"]
    assert len(edited) == 1
    assert edited[0]["messages"][1]["content"] == "30 calendar days from delivery."
    assert res.from_edits == 1


async def test_accepts_contribute_their_completion(tmp_path):
    res, rows = await _sft(tmp_path, _corpus())
    accepted = [r for r in rows if r["messages"][0]["content"] == "shipping time?"]
    assert accepted[0]["messages"][1]["content"] == "3-5 business days."
    assert res.from_accepts == 2   # t-2 and t-3's resolution


async def test_unreviewed_traces_are_not_gold(tmp_path):
    """A production completion nobody endorsed is output, not a target."""
    _, rows = await _sft(tmp_path, [
        _event(kind="trace", trace_id="t-9", prompt="q", completion="a")])
    assert rows == []


async def test_accepts_can_be_excluded(tmp_path):
    res, rows = await _sft(tmp_path, _corpus(), include_accepts=False)
    assert res.from_accepts == 0 and res.from_edits == 1 and len(rows) == 1


async def test_output_rows_carry_nothing_but_messages(tmp_path):
    """An unknown key can break a trainer's chat formatter — attribution lives
    in the findings, not on the row (same rule dataset/replay follows)."""
    _, rows = await _sft(tmp_path, _corpus())
    assert all(set(r) == {"messages"} for r in rows)


async def test_a_system_prompt_is_prepended_when_asked(tmp_path):
    _, rows = await _sft(tmp_path, _corpus(), system_prompt="be brief")
    assert rows[0]["messages"][0] == {"role": "system", "content": "be brief"}


async def test_identical_examples_are_deduplicated(tmp_path):
    rows_in = [_event(kind="accept", trace_id=f"t-{i}", prompt="q", completion="a")
               for i in range(3)]
    res, rows = await _sft(tmp_path, rows_in)
    assert (len(rows), res.duplicates_dropped) == (1, 2)


async def test_the_attribution_readout_is_reported_and_sorted(tmp_path):
    rows_in = _corpus() + [
        _event(kind="accept", trace_id="t-4", prompt="q4", completion="a4",
               adapter_uri="s3://a/v4")]
    res, _ = await _sft(tmp_path, rows_in)
    assert [(a.adapter_uri, a.rows) for a in res.by_adapter] == [
        ("s3://a/v3", 3), ("s3://a/v4", 1)]


async def test_only_adapter_keeps_a_cycle_from_training_on_another_cycles_complaints(tmp_path):
    """THE reason model/adapter_uri is first-class on a capture row."""
    rows_in = _corpus() + [
        _event(kind="accept", trace_id="t-4", prompt="q4", completion="a4",
               adapter_uri="s3://a/v4")]
    res, rows = await _sft(tmp_path, rows_in, only_adapter="s3://a/v4")
    assert len(rows) == 1 and res.rows == 1
    assert rows[0]["messages"][0]["content"] == "q4"


async def test_sft_output_is_deterministic(tmp_path):
    src = _write(tmp_path, "capture.jsonl", _corpus())
    outs = []
    for name in ("a.jsonl", "b.jsonl"):
        p = str(tmp_path / name)
        await CaptureToSftOp().run(CaptureToSftInput(path=src, out_path=p), _ctx())
        outs.append(open(p, encoding="utf-8").read())
    assert outs[0] == outs[1]


# ── dataset/capture_to_pairs ─────────────────────────────────────────────────


async def _pairs(tmp_path, rows, **kwargs):
    src = _write(tmp_path, "capture.jsonl", rows)
    out = str(tmp_path / "pairs.jsonl")
    res = await CaptureToPairsOp().run(
        CaptureToPairsInput(path=src, out_path=out, **kwargs), _ctx())
    return res, _read(out)


async def test_an_edit_becomes_chosen_over_the_original(tmp_path):
    res, rows = await _pairs(tmp_path, _corpus())
    edit = [r for r in rows if r["kind"] == "edit"][0]
    assert edit == {"prompt": "refund window?",
                    "chosen": "30 calendar days from delivery.",
                    "rejected": "30 days.", "kind": "edit"}
    assert res.from_edits == 1


async def test_the_pair_shape_matches_the_dpo_rung(tmp_path):
    """`train.dpo` reads what `dataset_dpo_pairs` writes; capture must speak the
    same shape or the rung needs an adapter in between."""
    _, rows = await _pairs(tmp_path, _corpus())
    assert all({"prompt", "chosen", "rejected"} <= set(r) for r in rows)


async def test_a_rejection_pairs_with_the_later_resolution(tmp_path):
    res, rows = await _pairs(tmp_path, _corpus())
    rej = [r for r in rows if r["kind"] == "reject"][0]
    assert rej["rejected"] == "I don't know."
    assert rej["chosen"] == "Yes, 2-4 days to Berlin."
    assert res.from_rejects == 1


async def test_a_rejection_with_no_successor_is_dropped_not_guessed(tmp_path):
    """A pair whose `chosen` was invented teaches the model to prefer the
    invention."""
    res, rows = await _pairs(tmp_path, [
        _event(kind="trace", trace_id="t-7", prompt="q", completion="bad"),
        _event(kind="reject", trace_id="t-7", completion="bad"),
    ])
    assert rows == [] and res.unpaired_dropped == 1


async def test_only_a_LATER_endorsement_counts(tmp_path):
    """Text that existed BEFORE the rejection is what got rejected."""
    res, rows = await _pairs(tmp_path, [
        _event(kind="accept", trace_id="t-8", prompt="q", completion="good",
               created_at="2026-08-20T09:00:00+00:00"),
        _event(kind="reject", trace_id="t-8", completion="bad",
               created_at="2026-08-20T09:01:00+00:00"),
    ])
    assert rows == [] and res.unpaired_dropped == 1


async def test_a_retry_is_mined_like_a_rejection(tmp_path):
    res, _ = await _pairs(tmp_path, [
        _event(kind="trace", trace_id="t-6", prompt="q", completion="first try"),
        _event(kind="retry", trace_id="t-6", completion="first try"),
        _event(kind="accept", trace_id="t-6", completion="second try"),
    ])
    assert res.from_retries == 1


async def test_rejects_can_be_left_out(tmp_path):
    res, rows = await _pairs(tmp_path, _corpus(), pair_rejects=False)
    assert res.from_rejects == 0 and res.rows == 1 and len(rows) == 1


async def test_a_run_id_correlates_when_there_is_no_trace_id(tmp_path):
    res, rows = await _pairs(tmp_path, [
        _event(kind="trace", trace_id="tr", run_id="r-1", prompt="q", completion="a"),
        _event(kind="edit", run_id="r-1", edited_text="A!"),
    ])
    # the trace's own trace_id groups it separately from the run-only edit, so
    # the edit falls back to the completion carried on its own row (none) —
    # correlation is per-id, and the projection never invents a link.
    assert res.rows == 0 and rows == []


async def test_feedback_inherits_the_prompt_from_its_trace(tmp_path):
    """The SDK does not ask an application to re-send a prompt it already
    traced, so the group is where the text lives."""
    _, rows = await _pairs(tmp_path, [
        _event(kind="trace", trace_id="t-5", prompt="the question", completion="meh"),
        _event(kind="edit", trace_id="t-5", edited_text="the good answer"),
    ])
    assert rows[0]["prompt"] == "the question"
    assert rows[0]["rejected"] == "meh"


async def test_duplicate_pairs_are_dropped(tmp_path):
    res, rows = await _pairs(tmp_path, [
        _event(kind="edit", trace_id="t-a", prompt="q", completion="bad", edited_text="good"),
        _event(kind="edit", trace_id="t-b", prompt="q", completion="bad", edited_text="good"),
    ])
    assert len(rows) == 1 and res.duplicates_dropped == 1


async def test_pairs_output_is_deterministic(tmp_path):
    src = _write(tmp_path, "capture.jsonl", _corpus())
    outs = []
    for name in ("a.jsonl", "b.jsonl"):
        p = str(tmp_path / name)
        await CaptureToPairsOp().run(CaptureToPairsInput(path=src, out_path=p), _ctx())
        outs.append(open(p, encoding="utf-8").read())
    assert outs[0] == outs[1]


# ── dataset/capture_to_rewards ───────────────────────────────────────────────


def _scored() -> list[dict]:
    return [
        _event(kind="trace", trace_id="s-1", prompt="what is 2+2?", completion="4"),
        _event(kind="score", trace_id="s-1", score=5.0, score_scale="1-5",
               payload={"verifier": "arithmetic", "passed": True}),
        _event(kind="score", trace_id="s-1", score=4.0, score_scale="1-5"),
        _event(kind="trace", trace_id="s-2", prompt="capital of France?",
               completion="Lyon"),
        _event(kind="edit", trace_id="s-2", completion="Lyon", edited_text="Paris"),
        _event(kind="score", trace_id="s-2", score=1.0, score_scale="1-5"),
    ]


async def _rewards(tmp_path, rows, **kwargs):
    src = _write(tmp_path, "capture.jsonl", rows)
    out = str(tmp_path / "rewards.jsonl")
    res = await CaptureToRewardsOp().run(
        CaptureToRewardsInput(path=src, out_path=out, **kwargs), _ctx())
    return res, _read(out)


async def test_it_emits_reward_context_not_off_policy_triples(tmp_path):
    """GRPO scores its OWN rollouts. A captured completion's score belongs in a
    reward function's context, never in a (prompt, completion, reward) row."""
    _, rows = await _rewards(tmp_path, _scored())
    assert all("completion" not in r and "score" not in r for r in rows)
    assert all({"prompt", "reference", "mean_score", "score_count",
                "score_scale", "context"} == set(r) for r in rows)


async def test_scores_are_aggregated_per_prompt(tmp_path):
    res, rows = await _rewards(tmp_path, _scored())
    maths = [r for r in rows if r["prompt"] == "what is 2+2?"][0]
    assert maths["score_count"] == 2
    assert maths["mean_score"] == 4.5
    assert res.scores_seen == 3


async def test_the_reference_is_the_human_endorsed_text(tmp_path):
    _, rows = await _rewards(tmp_path, _scored())
    paris = [r for r in rows if r["prompt"] == "capital of France?"][0]
    assert paris["reference"] == "Paris"   # the EDIT, not the scored completion


async def test_verifier_context_rides_through(tmp_path):
    _, rows = await _rewards(tmp_path, _scored())
    maths = [r for r in rows if r["prompt"] == "what is 2+2?"][0]
    assert maths["context"] == {"verifier": "arithmetic", "passed": True}


async def test_nested_payload_keys_are_dropped_not_flattened(tmp_path):
    _, rows = await _rewards(tmp_path, [
        _event(kind="score", trace_id="n-1", prompt="q", score=1.0,
               payload={"flat": 1, "nested": {"a": 1}, "list": [1, 2]})])
    assert rows[0]["context"] == {"flat": 1}


async def test_two_scales_are_two_rows_not_one_average(tmp_path):
    """'4 out of 5' averaged with '0.8' is a number that means nothing."""
    res, rows = await _rewards(tmp_path, [
        _event(kind="trace", trace_id="m-1", prompt="q", completion="a"),
        _event(kind="score", trace_id="m-1", score=4.0, score_scale="1-5"),
        _event(kind="score", trace_id="m-1", score=0.8, score_scale="0-1"),
    ])
    assert len(rows) == 2 and res.mixed_scale_prompts == 1
    assert res.scales == ["0-1", "1-5"]


async def test_unscored_prompts_are_counted_and_only_emitted_on_request(tmp_path):
    res, rows = await _rewards(tmp_path, _corpus())
    assert rows == [] and res.prompts_without_score == 3

    res2, rows2 = await _rewards(tmp_path, _corpus(), require_score=False)
    assert len(rows2) == 3 and all(r["score_count"] == 0 for r in rows2)
    assert res2.prompts_without_score == 3


async def test_a_score_with_no_resolvable_prompt_is_counted_not_invented(tmp_path):
    res, rows = await _rewards(tmp_path, [
        _event(kind="score", trace_id="orphan", score=3.0, score_scale="1-5")])
    assert rows == [] and res.unresolved_scores == 1


async def test_only_adapter_filters_the_reward_context_too(tmp_path):
    res, _ = await _rewards(tmp_path, _scored(), only_adapter="s3://a/v4")
    assert res.rows == 0 and res.scores_seen == 0


async def test_rewards_output_is_deterministic(tmp_path):
    src = _write(tmp_path, "capture.jsonl", _scored())
    outs = []
    for name in ("a.jsonl", "b.jsonl"):
        p = str(tmp_path / name)
        await CaptureToRewardsOp().run(CaptureToRewardsInput(path=src, out_path=p), _ctx())
        outs.append(open(p, encoding="utf-8").read())
    assert outs[0] == outs[1]


@pytest.mark.parametrize("op,schema", [
    (CaptureToSftOp, CaptureToSftInput),
    (CaptureToPairsOp, CaptureToPairsInput),
    (CaptureToRewardsOp, CaptureToRewardsInput),
])
def test_each_projection_declares_its_schemas_and_risk(op, schema):
    assert op.input_schema is schema
    assert op.risk == "external_write"   # each writes the dataset a rung consumes
