"""dataset/gate — the deterministic readiness verdict.

Turns the six checks' findings into a pass / warn / fail verdict with cited
blocking + warning reasons. This is where the *engine gates* (the §4
determinism rule): the LLM synthesis step only narrates the verdict this
operator computes — it never decides go/no-go. Pure function of its inputs:
no I/O, no LLM, fully replayable.
"""
from __future__ import annotations

from typing import Any

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks.schemas.gate import GateConfig, GateInput, GateOutput

# PII kinds that BLOCK (leaked credentials must never reach a training run).
# The rest (email / ipv4) are non-secret → warn only.
_HARD_SECRET_PII = (
    "private_key", "aws_access_key", "openai_key", "github_token", "jwt",
    "slack_token", "stripe_key",
)


@register_operator(
    name="dataset/gate",
    version="0.1.0",
    capabilities=["dataset.gate"],
    implements=["contract://dataset.gate/1.0.0"],  # #2242 V1-B — cairn-contracts-dataset
)
class GateOp(Operation[GateInput, GateOutput]):
    """Compute the cited go/no-go verdict from deterministic check findings."""

    input_schema = GateInput
    output_schema = GateOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    config_schema = GateConfig  # ADR-0068 (#742) — deployment-tunable thresholds

    async def run(self, inp: GateInput, ctx: OperationContext) -> GateOutput:
        t = inp.thresholds
        blocking: list[str] = []
        warnings: list[str] = []
        passed: list[str] = []
        cited: set[str] = set()

        # ── format ────────────────────────────────────────────────────
        if inp.format_conformant_pct is not None:
            cited.add("dataset/format")
            if inp.format_conformant_pct < t.format_min_conformant_pct:
                blocking.append(
                    f"format conformance {inp.format_conformant_pct}% "
                    f"< {t.format_min_conformant_pct}% — `dataset/format`"
                )
            else:
                passed.append(
                    f"format {inp.format_conformant_pct}% conformant — `dataset/format`"
                )
        if inp.format_trainer_compat:
            cited.add("dataset/format")
            if not any(inp.format_trainer_compat.values()):
                trainers = ", ".join(sorted(inp.format_trainer_compat))
                blocking.append(
                    f"no target trainer accepts the detected format ({trainers}) "
                    "— `dataset/format`"
                )

        # ── decontam (only when an eval set was provided) ──────────────
        if inp.decontam_checked and inp.decontam_overlap_pct is not None:
            cited.add("dataset/decontam")
            if inp.decontam_overlap_pct > t.decontam_max_overlap_pct:
                blocking.append(
                    f"train/eval contamination {inp.decontam_overlap_pct}% "
                    f"> {t.decontam_max_overlap_pct}% — `dataset/decontam`"
                )
            else:
                passed.append(
                    f"contamination {inp.decontam_overlap_pct}% within limit "
                    "— `dataset/decontam`"
                )

        # ── pii / secrets ──────────────────────────────────────────────
        if inp.pii_counts:
            cited.add("dataset/pii_scan")
            secrets = {
                k: v for k, v in inp.pii_counts.items() if k in _HARD_SECRET_PII and v
            }
            if secrets:
                kinds = ", ".join(f"{v} {k}" for k, v in sorted(secrets.items()))
                blocking.append(f"hard secrets present ({kinds}) — `dataset/pii_scan`")
            nonsecret = {
                k: v
                for k, v in inp.pii_counts.items()
                if k not in _HARD_SECRET_PII and v
            }
            if nonsecret:
                kinds = ", ".join(f"{v} {k}" for k, v in sorted(nonsecret.items()))
                warnings.append(f"PII present ({kinds}) — `dataset/pii_scan`")
            if not secrets and not nonsecret:
                passed.append("no PII detected — `dataset/pii_scan`")

        # ── dedup ──────────────────────────────────────────────────────
        if inp.dedup_pct is not None:
            cited.add("dataset/dedup")
            if inp.dedup_pct > 0:
                warnings.append(f"{inp.dedup_pct}% exact duplicates — `dataset/dedup`")
            else:
                passed.append("no exact duplicates — `dataset/dedup`")

        # ── profile (refusal ratio + short responses) ──────────────────
        if inp.profile_refusal_pct is not None:
            cited.add("dataset/profile")
            if inp.profile_refusal_pct > t.refusal_warn_pct:
                warnings.append(
                    f"high refusal ratio {inp.profile_refusal_pct}% "
                    f"> {t.refusal_warn_pct}% — `dataset/profile`"
                )
        if inp.profile_short is not None and inp.profile_total:
            cited.add("dataset/profile")
            frac = inp.profile_short / inp.profile_total
            if frac > t.short_warn_frac:
                warnings.append(
                    f"{inp.profile_short}/{inp.profile_total} responses too short "
                    "— `dataset/profile`"
                )

        # ── provenance ─────────────────────────────────────────────────
        if inp.prov_ok is not None:
            cited.add("dataset/provenance")
            if not inp.prov_ok:
                miss = (
                    f" (missing: {', '.join(inp.prov_missing)})"
                    if inp.prov_missing
                    else ""
                )
                warnings.append(f"incomplete provenance{miss} — `dataset/provenance`")
            else:
                passed.append("provenance complete — `dataset/provenance`")

        verdict = "fail" if blocking else ("warn" if warnings else "pass")
        checks_cited = sorted(cited)

        # ADR-0075 — publish a normalized run-level OUTCOME so every surface
        # (pilot page, Playground hero/tiles, CLI) reads one honest verdict off
        # the standard output.outcome channel. Deterministic: it mirrors the
        # verdict this operator just computed — the LLM synthesis never touches it.
        # `severity` matches this gate's own block/warn logic above, so consumers
        # count blockers vs warnings honestly (never "3 blockers" for 1 block +
        # 2 warnings). Only dimensions whose check actually ran are included.
        def _sev(is_block: bool, is_warn: bool) -> str:
            return "block" if is_block else ("warn" if is_warn else "ok")

        outcome: list[dict[str, Any]] = [
            {"name": "readiness", "value": verdict, "pass": verdict == "pass",
             "target": "pass", "severity": _sev(verdict == "fail", verdict == "warn")},
        ]
        if inp.pii_rows is not None:
            has_secret = any(k in _HARD_SECRET_PII and v for k, v in (inp.pii_counts or {}).items())
            has_nonsecret = any(k not in _HARD_SECRET_PII and v for k, v in (inp.pii_counts or {}).items())
            outcome.append({"name": "PII rows", "value": inp.pii_rows, "pass": inp.pii_rows == 0,
                            "target": 0, "severity": _sev(has_secret, has_nonsecret)})
        if inp.format_conformant_pct is not None:
            below = inp.format_conformant_pct < t.format_min_conformant_pct
            outcome.append({"name": "format conformant", "value": inp.format_conformant_pct,
                            "pass": not below, "target": t.format_min_conformant_pct,
                            "severity": _sev(below, False)})
        if inp.dedup_pct is not None:
            dup = inp.dedup_pct > 0
            outcome.append({"name": "duplicates", "value": inp.dedup_pct, "pass": not dup,
                            "target": 0, "severity": _sev(False, dup)})
        if inp.decontam_checked and inp.decontam_overlap_pct is not None:
            over = inp.decontam_overlap_pct > t.decontam_max_overlap_pct
            outcome.append({"name": "train/eval overlap", "value": inp.decontam_overlap_pct,
                            "pass": not over, "target": t.decontam_max_overlap_pct,
                            "severity": _sev(over, False)})
        if inp.prov_ok is not None:
            outcome.append({"name": "provenance", "value": "complete" if inp.prov_ok else "incomplete",
                            "pass": bool(inp.prov_ok), "target": "complete",
                            "severity": _sev(False, not inp.prov_ok)})

        return GateOutput(
            verdict=verdict,
            dataset_uri=inp.dataset_uri or "",  # #579 — emit the validated dataset ref
            blocking=blocking,
            warnings=warnings,
            passed=passed,
            checks_cited=checks_cited,
            outcome=outcome,
            audit_metadata={
                "verdict": verdict,
                "blocking": blocking,
                "warnings": warnings,
                "checks_cited": checks_cited,
                "thresholds": t.model_dump(),
            },
        )
