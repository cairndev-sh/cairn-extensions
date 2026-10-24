"""The governed splits and the batch knobs (training.stage.train_qlora/1.1.0, cairn-extensions#4).

The two trainers call ``splits`` for which data trains and which evaluates, and for the batch
arguments; these tests drive it with a fake dataset, so no training stack is needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from operator_training import splits
from operator_training.flow_stages import _spec_from_config, train_qlora


class _DS:
    def __init__(self, name: str) -> None:
        self.name = name

    def train_test_split(self, *, test_size, seed):
        return {"train": _DS(f"{self.name}[train,{test_size},{seed}]"), "test": _DS(f"{self.name}[test]")}


def _loader():
    opened: list[str] = []

    def load(uri: str) -> _DS:
        opened.append(uri)
        return _DS(uri)

    return load, opened


def test_a_governed_eval_split_is_used_and_the_test_split_is_never_opened():
    load, opened = _loader()
    spec = {"dataset_uri": "d.jsonl", "train_uri": "train.jsonl", "eval_uri": "val.jsonl",
            "test_uri": "test.jsonl", "eval_split": 0.2}
    train, evaluate, source = splits.train_and_eval(spec, load, seed=7)
    assert (train.name, evaluate.name, source) == ("train.jsonl", "val.jsonl", "eval_uri")
    assert opened == ["train.jsonl", "val.jsonl"]          # no dataset_uri, no test_uri, no random split


def test_without_an_eval_split_the_seeded_hold_out_is_kept():
    load, opened = _loader()
    train, evaluate, source = splits.train_and_eval({"dataset_uri": "d.jsonl", "eval_split": 0.2}, load, seed=7)
    assert (train.name, evaluate.name, source) == ("d.jsonl[train,0.2,7]", "d.jsonl[test]", "eval_split")
    assert opened == ["d.jsonl"]


def test_a_raw_spec_without_eval_split_keeps_the_trainspec_default():
    load, _ = _loader()
    assert splits.train_and_eval({"dataset_uri": "d.jsonl"}, load, seed=1)[0].name == "d.jsonl[train,0.1,1]"


def test_no_eval_when_the_split_is_off_or_the_data_is_too_small():
    load, _ = _loader()
    assert splits.train_and_eval({"dataset_uri": "d", "eval_split": 0}, load, seed=1)[1:] == (None, "none")
    assert splits.train_and_eval({"dataset_uri": "d", "eval_split": 0.2}, load, seed=1,
                                 can_split=lambda ds: False)[1:] == (None, "none")


def test_the_batch_knobs_override_the_defaults_and_the_effective_batch_is_recorded():
    assert splits.batch_args({}, per_device=8, accumulation=1) == {
        "per_device_train_batch_size": 8, "gradient_accumulation_steps": 1}
    batch = splits.batch_args({"per_device_batch_size": 1, "gradient_accumulation_steps": 4}, per_device=8, accumulation=1)
    assert batch == {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 4}
    assert splits.recorded({"test_uri": "test.jsonl"}, batch, "eval_uri") == {
        "effective_batch_size": 4, "eval_source": "eval_uri", "test_uri": "test.jsonl"}
    assert "test_uri" not in splits.recorded({}, batch, "none")


def test_the_stage_trains_on_the_governed_split(tmp_path):
    spec = _spec_from_config({"train_uri": "s3://b/train.jsonl", "eval_uri": "s3://b/val.jsonl",
                              "test_uri": "s3://b/test.jsonl", "base_model": "m",
                              "per_device_batch_size": 2, "gradient_accumulation_steps": 4, "lora_dropout": 0.0},
                             [], tmp_path)
    assert spec.dataset_uri == "s3://b/train.jsonl" and spec.train_uri == "s3://b/train.jsonl"
    assert (spec.eval_uri, spec.test_uri) == ("s3://b/val.jsonl", "s3://b/test.jsonl")
    assert (spec.per_device_batch_size, spec.gradient_accumulation_steps, spec.lora_dropout) == (2, 4, 0.0)


def test_a_fidelity_subsample_replaces_the_governed_train_split(tmp_path):
    from operator_training.flow_stages import sample_fidelity

    corpus = tmp_path / "train.jsonl"
    corpus.write_text("".join(json.dumps({"input": f"q{i}", "output": f"a{i}"}) + "\n" for i in range(40)))
    sample_fidelity(tmp_path, {"dataset_uri": str(corpus), "fidelity_rows": 5}, [])
    spec = _spec_from_config({"train_uri": str(corpus), "base_model": "m"}, [], tmp_path)
    assert spec.train_uri.endswith("fidelity_sample.jsonl") and spec.extra["fidelity_n"] == 5


def test_train_qlora_implements_the_1_2_0_stage_contract():
    # 1.2.0 (cairn-extensions#7) keeps every 1.1.0 knob and adds baseline_eval.
    assert train_qlora.implements == ("contract://training.stage.train_qlora/1.2.0",)


def test_the_1_1_0_configuration_admits_the_knobs_and_refuses_extra():
    jsonschema = pytest.importorskip("jsonschema")
    root = Path(__file__).resolve().parents[2] / "cairn-contracts-training-stages" / "definitions"
    schema = json.loads((root / "schemas" / "training.stage.train_qlora.configuration.1.1.0.json").read_text())
    ok = {"base_model": "m", "train_uri": "t", "eval_uri": "v", "test_uri": "x", "per_device_batch_size": 1,
          "gradient_accumulation_steps": 4, "gradient_checkpointing": True, "lora_dropout": 0.0, "stub": True}
    jsonschema.validate(ok, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**ok, "extra": {}}, schema)
    stage = json.loads((root / "stages" / "training.stage.train_qlora.1.1.0.json").read_text())
    assert stage["configuration"] == "contract://training.stage.train_qlora.configuration/1.1.0"
    assert stage["output"] == "contract://training.stage.train_qlora.output/1.0.0"
