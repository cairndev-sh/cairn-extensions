"""cairn-extensions#16 — the kernels the dataset-prep `prepare` pipeline needs: a topic label and
a template family for a dataset with neither (`fields.derive`), their distribution as counts
that carry no row content (`profile.fields`), and prompt + response length (`profile.tokens`
1.1.0). Rows are IFC/BIM-shaped Alpaca examples built here; whitespace token counts, so no
tokenizer download."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from cairn.core.dataflow.conformance_check import bundle, problems
from cairn.core.pack.contracts import _installed

from operator_dataset_checks.dataflow_stages import fields_derive, profile_fields, profile_tokens

_INSTRUCTION = "You are an IFC and BIM assistant. Answer with IFC STEP lines."
_ROWS = [
    {"instruction": _INSTRUCTION, "input": "Create a shear wall 100mm thick and 2400mm high.",
     "output": "#1=IFCPROJECT('a',$);\n#9=IFCWALLSTANDARDCASE('b',$);"},
    {"instruction": _INSTRUCTION, "input": "Create a shear wall 200mm thick and 3000mm high.",
     "output": "#1=IFCPROJECT('c',$);\n#7=IFCWALL('d',$);"},
    {"instruction": _INSTRUCTION, "input": "Create a slab 'Level 1' of 250mm.",
     "output": "#1=IFCPROJECT('e',$);\n#4=IFCSLAB('f',$);"},
    {"instruction": _INSTRUCTION, "input": "Explain what IFC is.", "output": "A schema for building data."},
]
#: The rules the IFC preset carries: the first building element in the answer, and the request
#: with its numbers and quoted names replaced.
_RULES = [
    {"field": "topic", "source": "output", "mode": "extract",
     "pattern": r"=\s*(IFC(?:WALL|SLAB|BEAM|COLUMN|DOOR|WINDOW|ROOF|STAIR)\w*)\s*\("},
    {"field": "family", "source": "input", "mode": "normalize",
     "pattern": r"\d+(?:\.\d+)?|'[^']*'", "replace": "#", "case": "lower"},
]


def _workdir(tmp_path: Path, rows=_ROWS) -> Path:
    (tmp_path / "current.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return tmp_path


def _rows(workdir: Path) -> list[dict]:
    return [json.loads(line) for line in (workdir / "current.jsonl").read_text(encoding="utf-8").splitlines()]


def _contract(stage: str, revision: str, part: str) -> dict:
    _resolved, bodies = _installed()
    uri = bodies[f"contract://dataset.stage.{stage}/{revision}"][1][part]
    return bundle({u: body for u, (_d, body) in bodies.items()}, uri)


def test_derive_labels_the_topic_and_the_template_family(tmp_path):
    wd = _workdir(tmp_path)
    stats = fields_derive(wd, {"rules": _RULES}, [])
    rows = _rows(wd)
    assert [r["topic"] for r in rows] == ["IFCWALLSTANDARDCASE", "IFCWALL", "IFCSLAB", "other"]
    assert rows[0]["family"] == rows[1]["family"] == "create a shear wall #mm thick and #mm high."
    assert rows[2]["family"] == "create a slab # of #mm."
    assert {k: rows[0][k] for k in ("instruction", "input", "output")} == _ROWS[0]  # the data is untouched
    assert stats == {"rows": 4, "fields": ["topic", "family"], "topic_matched": 3, "family_matched": 3}
    assert problems(_contract("fields_derive", "1.0.0", "configuration"), {"rules": _RULES}, "c") == []
    assert problems(_contract("fields_derive", "1.0.0", "output"), stats, "o") == []


@pytest.mark.parametrize("rules, message", [
    ([{"field": "output", "source": "input", "pattern": "x"}], "never overwrites data"),
    ([{"field": "t", "source": "input", "pattern": "x"}, {"field": "t", "source": "output", "pattern": "y"}], "two rules write"),
    ([{"field": "t", "source": "input", "pattern": "x", "mode": "rewrite"}], "mode must be one of"),
])
def test_derive_refuses_a_rule_that_would_change_or_confuse_the_data(tmp_path, rules, message):
    with pytest.raises(ValueError, match=message):
        fields_derive(_workdir(tmp_path), {"rules": rules}, [])


def test_profile_counts_labels_and_reports_groups_by_size_only(tmp_path):
    wd = _workdir(tmp_path)
    fields_derive(wd, {"rules": _RULES}, [])
    stats = profile_fields(wd, {"count": ["topic"], "groups": ["family"], "top": 2}, [])
    assert stats["topic_values"] == ["IFCWALLSTANDARDCASE", "IFCWALL"] and stats["topic_counts"] == [1, 1]
    assert (stats["topic_other"], stats["topic_distinct"], stats["topic_missing"]) == (2, 4, 0)
    assert (stats["family_groups"], stats["family_largest"], stats["family_singletons"]) == (3, 2, 2)
    assert "shear wall" not in json.dumps(stats)  # a template family is prompt text; it never reaches the record
    assert problems(_contract("profile_fields", "1.0.0", "output"), stats, "o") == []


def test_unconfigured_derive_and_profile_pass_through(tmp_path):
    """A pipeline that binds no rules and no fields (a dataset with its own columns) still runs."""
    wd = _workdir(tmp_path)
    before = (wd / "current.jsonl").read_bytes()
    assert fields_derive(wd, {}, []) == {"rows": 4, "fields": []}
    assert (wd / "current.jsonl").read_bytes() == before
    assert profile_fields(wd, {}, []) == {"rows": 4}


def test_tokens_measure_the_whole_example_on_request(tmp_path):
    wd = _workdir(tmp_path)
    response = profile_tokens(wd, {}, [])
    example = profile_tokens(wd, {"measure": "example", "max_seq_len": 20}, [])
    assert response["measure"] == "response" and "prompt_tokens_p50" not in response
    assert example["tokens_total"] > response["tokens_total"]
    assert example["tokens_total"] == sum(len(f"{r['instruction']}\n{r['input']}".split()) + len(r["output"].split())
                                          for r in _ROWS)
    assert example["response_tokens_max"] == response["tokens_max"]
    assert example["over_length_rows"] == 4  # every example is over 20 words once the prompt counts
    assert problems(_contract("profile_tokens", "1.1.0", "output"), example, "o") == []
    with pytest.raises(ValueError, match="measure must be"):
        profile_tokens(wd, {"measure": "prompt"}, [])
