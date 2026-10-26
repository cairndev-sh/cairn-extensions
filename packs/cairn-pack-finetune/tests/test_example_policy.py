"""The example admission policy in `docs/policies/` (cairn-extensions#30): it validates the way
`cairn validate` validates it, it names only checks the launch door knows, the core's admission
decision admits versions whose verdict shows both dedup checks ran and refuses one where near dedup
was degraded, and the pack never loads it."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PACK = Path(__file__).resolve().parents[1]
POLICY = PACK / "docs" / "policies" / "require-dedup.yaml"
_INPUTS = {"train_version": "dataset:demo-train@1", "val_version": "dataset:demo-val@1",
           "test_version": "dataset:demo-test@1"}


def test_it_validates_as_a_policy():
    from cairn.core.validate import validate_path

    report = validate_path(POLICY)
    assert report.ok and report.artifact_type == "policy" and not report.warnings, report.issues


def test_it_names_only_checks_the_launch_door_knows():
    from cairn.core.intake import CHECKS

    required = yaml.safe_load(POLICY.read_text(encoding="utf-8"))["admission"]["required_checks"]
    assert required and set(required) <= set(CHECKS), set(required) - set(CHECKS)


class _Versions:
    """Three admitted versions whose verdict report gives `dedup_near` the state under test."""

    def __init__(self, near: str):
        self.near = near

    async def get_dataset_version_meta(self, name, version=None, **kw):
        role = {"demo-train": "training", "demo-val": "validation", "demo-test": "test"}[name]
        return {"role": role, "verdict": "admitted", "artifact_id": name,
                "verdict_report": {"checks": {"dedup_exact": {"state": "ran"}, "dedup_near": {"state": self.near}}}}


@pytest.mark.parametrize("near,verdict", [("ran", "admitted"), ("degraded", "refused")])
async def test_the_launch_door_applies_it(near, verdict, monkeypatch):
    import cairn.core.holdout as holdout
    from cairn.core.admission_door import decide_admission
    from cairn.core.engine.schema import PolicySpec

    class _Clean:
        ok, checked = True, True

        def as_record(self):
            return {"ok": True}

    async def no_leak(*a, **k):
        return _Clean()

    monkeypatch.setattr(holdout, "check_holdout_for_inputs", no_leak)  # the holdout door is not under test
    policy = PolicySpec.model_validate(yaml.safe_load(POLICY.read_text(encoding="utf-8")))
    decision = await decide_admission(_INPUTS, workspace_id="default", environment="production",
                                      store=_Versions(near), policies=[policy])
    assert decision.verdict == verdict, decision.refusing_reasons
    if verdict == "refused":
        assert all("requires check 'dedup_near' to have run — it is degraded" in r
                   for r in decision.refusing_reasons), decision.refusing_reasons


def test_the_pack_does_not_load_it(monkeypatch):
    from cairn.core.pack import registry
    from cairn.core.pack.loader import load_all_packs
    from cairn.core.pack.registry import all_policy_dirs

    monkeypatch.setenv("CAIRN_PACK_PATH", str(PACK))
    registry.clear()
    load_all_packs()
    try:
        # neither this checkout nor the installed wheel (which carries docs/) contributes it
        assert not [d for d in all_policy_dirs() if any(Path(d).rglob(POLICY.name))]
    finally:
        registry.clear()
