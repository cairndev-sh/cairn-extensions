"""demos/laptop: the policies it ships do what their comments say, and its scripts are well formed.

The end-to-end run (a real LoRA trainer, the control plane, the gates) is `demos/laptop/run.sh`,
which CI runs on macOS; these tests are the part that needs no torch and no server.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos" / "laptop"
PACK = ROOT / "packs" / "cairn-pack-finetune"


def _plan(name: str = "sft-local"):
    from cairn.core.engine.pipeline import parse_pipeline_def, plan_pipeline

    return plan_pipeline(parse_pipeline_def((PACK / "pipelines" / f"{name}.yaml").read_text(encoding="utf-8"),
                                            pack="finetune"))


def _engine():
    from cairn.core.policy.engine import PolicyEngine

    return PolicyEngine(DEMO / "policies")


def test_both_policies_load_and_are_require_hitl():
    names = {p.name: p for p in _engine().policies}
    assert set(names) == {"laptop-training-needs-approval", "laptop-promote-needs-approval"}
    for p in names.values():
        assert [r.effect for r in p.rules] == ["require_hitl"]
        assert p.rules[0].approval_role == "ml-owners"


def test_the_training_policy_adds_nothing_to_sft_local_which_already_asks_first():
    """`approve-run` is upstream of both fine-tune legs, so the policy does not double the gate."""
    from cairn.core.policy.pipeline_gates import inject_policy_gates

    plan = _plan()
    assert inject_policy_gates(plan, engine=_engine()) == []
    assert [s.id for s in plan.steps][0] == "approve-run"


def test_the_training_policy_keeps_the_rule_for_a_copy_that_dropped_its_gate():
    """What the policy is for: a person is asked before each sft-train leg whatever the pipeline says."""
    from cairn.core.policy.pipeline_gates import inject_policy_gates

    plan = _plan()
    plan.steps = [s for s in plan.steps if s.id != "approve-run"]
    next(s for s in plan.steps if s.id == "smoke").depends_on = []
    injected = inject_policy_gates(plan, engine=_engine())
    assert injected == ["policy-gate-laptop-training-needs-approval-smoke"]   # `train` is behind it via smoke-ok
    gate = next(s for s in plan.steps if s.id == injected[0])
    assert gate.approval and gate.injected_by == "laptop-training-needs-approval"
    assert next(s for s in plan.steps if s.id == "smoke").depends_on == [gate.id]


def test_the_training_policy_names_no_operator_so_dispatch_never_matches_it():
    """A `require_hitl` policy on `pipeline_legs` alone also meets every operator call at dispatch
    in cairndev 0.3.1; the placeholder `action_types` keeps it to the plan-time rule."""
    spec = next(p for p in _engine().policies if p.name == "laptop-training-needs-approval")
    assert spec.applies_to.pipeline_legs == ["finetune/sft-train"]
    assert spec.applies_to.action_types == ["plan-time-only"]


def test_the_promote_policy_is_keyed_on_the_capability_the_promote_operator_provides():
    from cairn.core.operator import default_registry, load_all_operators

    load_all_operators()
    spec = next(p for p in _engine().policies if p.name == "laptop-promote-needs-approval")
    assert spec.applies_to.capabilities == ["artifact.promote"]
    assert "artifact.promote" in default_registry.get_metadata("artifact/promote").provides_capabilities


@pytest.mark.parametrize("script", ["run.sh", "install.sh", "env.sh"])
def test_the_shell_scripts_parse(script):
    subprocess.run(["bash", "-n", str(DEMO / script)], check=True)


def test_run_sh_help_needs_nothing_installed():
    out = subprocess.run(["bash", str(DEMO / "run.sh"), "--help"], check=True, capture_output=True, text=True).stdout
    for flag in ("--auto-approve", "--base-model", "--tiny", "--input"):
        assert flag in out


def test_the_profile_the_demo_selects_is_the_one_in_the_pack():
    env = (DEMO / "env.sh").read_text(encoding="utf-8")
    assert "packs/cairn-pack-finetune" in env and "finetune-local.toml" in env
    assert (PACK / "profiles" / "finetune-local.toml").is_file()


def test_the_tiny_model_builds_offline_and_loads(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    import importlib.util

    spec = importlib.util.spec_from_file_location("tiny_model", DEMO / "tiny_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.build(tmp_path / "m", mod.DEFAULT_DATA)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok, model = AutoTokenizer.from_pretrained(out), AutoModelForCausalLM.from_pretrained(out)
    assert len(tok) == model.config.vocab_size and tok.pad_token and sum(p.numel() for p in model.parameters()) < 500_000
