"""#1458 — the SFT trainer maps the corpus shapes it accepts to chat messages with the
data plane's own rules, and never trains on an empty turn.

``train_entry._to_messages`` read only ``input``/``output``, so a corpus already in
chat form (``{"messages": [...]}``: smoltalk, anything ``dataset/format`` calls
``openai_chat``) became two EMPTY turns and the trainer reported a loss over
nothing. Now: ``messages`` passes through checked as ``dataset/format`` checks it
(roles ``system``/``user``/``assistant``, ``human``/``gpt`` normalised, no empty
non-system turn, an assistant turn present); alpaca, cairn's ``input``/``output``
and ``prompt``/``completion`` map to one user + one assistant turn, told apart by
the values a row carries; anything else fails loud, naming the row.
"""
from __future__ import annotations

import pytest

from operator_training.train_entry import _row_messages, _to_messages


def test_a_chat_corpus_passes_through_instead_of_becoming_empty_turns():
    msgs = [{"role": "system", "content": "be brief"},
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"}]
    assert _to_messages({"messages": msgs}) == {"messages": msgs}


def test_the_data_planes_roles_are_normalised_and_turns_keep_role_and_content_only():
    out = _to_messages({"messages": [{"role": "Human", "content": "hi", "name": None},
                                     {"role": "GPT", "content": "hello", "tool_calls": None}]})
    assert out["messages"] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


@pytest.mark.parametrize("row, user, assistant", [
    ({"input": "hi", "output": "hello"}, "hi", "hello"),
    ({"prompt": "hi", "completion": "hello"}, "hi", "hello"),
    ({"instruction": "Translate", "output": "Bonjour"}, "Translate", "Bonjour"),
    ({"instruction": "Translate", "input": "Hello", "output": "Bonjour"}, "Translate\n\nHello", "Bonjour"),
    ({"instruction": "Translate", "response": "Bonjour"}, "Translate", "Bonjour"),
])
def test_each_pair_shape_maps_to_one_user_and_one_assistant_turn(row, user, assistant):
    assert _to_messages(row) == {"messages": [{"role": "user", "content": user},
                                              {"role": "assistant", "content": assistant}]}


@pytest.mark.parametrize("row, match", [
    ({"text": "a raw document"}, "no trainable shape"),
    ({"conversations": [{"from": "human", "value": "hi"}]}, "no trainable shape"),
    ({"input": "hi"}, "empty assistant turn"),
    ({"input": "", "output": "hello"}, "empty user turn"),
    ({"messages": [{"role": "user", "content": "hi"}]}, "does not end with the assistant"),
    ({"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "x"}]}, "empty user turn"),
    ({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                   {"role": "user", "content": "c"}, {"role": "assistant", "content": "  "}]}, "empty assistant turn"),
    ({"messages": [{"role": "tool", "content": "{}"}, {"role": "assistant", "content": "x"}]}, "role 'tool'"),
    ({"messages": [{"role": "user", "content": None}, {"role": "assistant", "content": "x"}]}, "not a string"),
    ({"messages": "not a list"}, "no trainable shape"),
    ({"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"},
                   {"role": "assistant", "content": "c"}]}, "roles must alternate"),
    ({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                   {"role": "user", "content": "c"}]}, "does not end with the assistant"),
    ({"messages": [{"role": "system", "content": "s"}]}, "only system turns"),
    ({"instruction": "  ", "input": "ctx", "output": "b"}, "empty instruction"),
])
def test_a_row_that_would_train_on_nothing_fails_loud(row, match):
    with pytest.raises(ValueError, match=match):
        _to_messages(row)


def test_a_refusal_names_the_row_so_a_large_corpus_can_be_fixed():
    assert _row_messages({"input": "hi", "output": "x"}, 0)["messages"][1]["content"] == "x"
    with pytest.raises(ValueError, match=r"dataset row 41: row would train on an empty assistant turn"):
        _row_messages({"instruction": "say", "output": ""}, 40)


# ── through the real `datasets` map the trainer runs ─────────────────────────
# `load_dataset("json")` gives every row the union of the file's columns, absent ones
# as None — which is why the mapping routes on values, not keys.

datasets = pytest.importorskip("datasets")


def _mapped(rows: list[dict], tmp_path) -> list[list[dict]]:
    """The trainer's own load: a JSONL file through ``load_dataset("json")`` (which
    unions the columns across rows — ``Dataset.from_list`` would not)."""
    import json

    f = tmp_path / "corpus.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ds = datasets.load_dataset("json", data_files=str(f), split="train",
                               cache_dir=str(tmp_path / "cache")).map(_row_messages, with_indices=True)
    return [r["messages"] for r in ds]


def test_a_file_mixing_input_output_and_prompt_completion_rows_maps_both(tmp_path):
    out = _mapped(tmp_path=tmp_path, rows=[{"input": "a", "output": "b"}, {"prompt": "c", "completion": "d"}])
    assert [[m["content"] for m in turns] for turns in out] == [["a", "b"], ["c", "d"]]


def test_a_file_mixing_alpaca_output_and_response_rows_maps_both(tmp_path):
    out = _mapped(tmp_path=tmp_path, rows=[{"instruction": "a", "output": "b"}, {"instruction": "c", "response": "d"}])
    assert [[m["content"] for m in turns] for turns in out] == [["a", "b"], ["c", "d"]]


def test_a_file_mixing_alpaca_and_input_output_rows_does_not_prefix_the_plain_rows(tmp_path):
    out = _mapped(tmp_path=tmp_path, rows=[{"instruction": "a", "input": "ctx", "output": "b"}, {"input": "c", "output": "d"}])
    assert [[m["content"] for m in turns] for turns in out] == [["a\n\nctx", "b"], ["c", "d"]]


def test_a_bad_row_in_a_real_map_names_its_row_number(tmp_path):
    with pytest.raises(ValueError, match=r"dataset row 2: row would train on an empty assistant turn"):
        _mapped(tmp_path=tmp_path, rows=[{"input": "a", "output": "b"}, {"input": "c", "output": ""}])


def test_only_the_messages_column_reaches_the_trainer(tmp_path):
    """The original columns are dropped, as distill_entry does: a raw ``prompt``
    column would steer trl's SFTTrainer off the rendered ``text`` path."""
    import json

    f = tmp_path / "corpus.jsonl"
    f.write_text(json.dumps({"prompt": "a", "completion": "b"}) + "\n", encoding="utf-8")
    ds = datasets.load_dataset("json", data_files=str(f), split="train", cache_dir=str(tmp_path / "cache"))
    ds = ds.map(_row_messages, with_indices=True, remove_columns=ds.column_names)
    assert ds.column_names == ["messages"]
    assert ds[0]["messages"] == [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
