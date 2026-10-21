"""dataset.dpo_pairs + train.dpo kernels (#1463 round 4) — hermetic tier.

The pairs kernel is pure row-reshaping (no trainer deps); train.dpo is
exercised through its stub path (the real body needs the GPU backend's
unsloth/trl, same convention as train.qlora's stub tier).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from operator_training.flow_stages import dataset_dpo_pairs, train_dpo
from operator_training.schemas.train import TrainSpec


def _write_answers(workdir: Path, rows: list[dict]) -> None:
    with (workdir / "current.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


ANSWERS = [
    # held-out fabrication → abstain pair
    {"question": "Q-heldout", "answer": "gold-h", "split": "eval",
     "label": "hallucinated", "model_answer": "made-up", "fact_id": "f1"},
    # trained-fact fabrication → correct pair (chosen = gold)
    {"question": "Q-trained-wrong", "answer": "gold-t", "split": "paraphrase_eval",
     "label": "hallucinated", "model_answer": "wrong", "fact_id": "f2"},
    # trained fact it got right → recall pair (chosen = its own answer)
    {"question": "Q-trained-right", "answer": "gold-r", "split": "paraphrase_eval",
     "label": "correct", "model_answer": "gold-r", "fact_id": "f3"},
    # control fabrication → correct pair
    {"question": "Q-control", "answer": "gold-c", "split": "control",
     "label": "hallucinated", "model_answer": "nonsense", "fact_id": "f4"},
    # rows that must NOT become pairs
    {"question": "Q-abstained", "answer": "g", "split": "eval",
     "label": "abstain", "model_answer": "I don't know.", "fact_id": "f5"},
    {"question": "Q-error", "answer": "g", "split": "eval",
     "label": "error", "model_answer": "", "fact_id": "f6"},
]


def test_dpo_pairs_mixed_kinds(tmp_path: Path) -> None:
    _write_answers(tmp_path, ANSWERS)
    stats = dataset_dpo_pairs(tmp_path, {"seed": 7}, [])
    assert stats == {"rows": 4, "abstain_pairs": 1, "correct_pairs": 2, "recall_pairs": 1}
    pairs = {json.loads(l)["prompt"]: json.loads(l)
             for l in (tmp_path / "current.jsonl").open(encoding="utf-8")}
    assert pairs["Q-heldout"]["chosen"] == "I don't know."
    assert pairs["Q-heldout"]["rejected"] == "made-up"
    assert pairs["Q-trained-wrong"]["chosen"] == "gold-t"
    assert pairs["Q-trained-right"]["chosen"] == "gold-r"
    assert pairs["Q-trained-right"]["rejected"] == "I don't know."
    assert pairs["Q-control"]["chosen"] == "gold-c"
    assert "Q-abstained" not in pairs and "Q-error" not in pairs


def test_dpo_pairs_cap_and_determinism(tmp_path: Path) -> None:
    rows = [dict(ANSWERS[0], question=f"Q{i}", fact_id=f"f{i}") for i in range(10)]
    _write_answers(tmp_path, rows)
    stats = dataset_dpo_pairs(tmp_path, {"max_per_kind": 3, "seed": 1}, [])
    assert stats["rows"] == 3 and stats["abstain_pairs"] == 3
    first = (tmp_path / "current.jsonl").read_text(encoding="utf-8")
    _write_answers(tmp_path, rows)
    dataset_dpo_pairs(tmp_path, {"max_per_kind": 3, "seed": 1}, [])
    assert (tmp_path / "current.jsonl").read_text(encoding="utf-8") == first


def test_train_dpo_stub_writes_result(tmp_path: Path) -> None:
    stats = train_dpo(tmp_path, {
        "stub": True, "dataset_uri": "pairs.jsonl", "output_name": "dpo-x",
        "init_adapter_uri": "s3://b/ck/checkpoint-400/", "beta": 0.2,
    }, [])
    assert stats["status"] == "succeeded"
    assert stats["init_adapter_uri"] == "s3://b/ck/checkpoint-400/"
    assert stats["beta"] == pytest.approx(0.2)
    result = json.loads((tmp_path / "train_result.json").read_text(encoding="utf-8"))
    assert result["backend"] == "unsloth-dpo"


def test_trainspec_beta_field() -> None:
    s = TrainSpec(dataset_uri="x.jsonl")
    assert s.beta == pytest.approx(0.1)
    with pytest.raises(ValueError):
        TrainSpec(dataset_uri="x.jsonl", beta=0)
