"""dataset/prep_gate — the config-gated post-scrub PII gate (#882, ADR-0075).

prep re-scans its OWN scrubbed output (run-prep-verify); this operator turns that
residual-PII count + the governance ``mode`` knob into a verdict, so prep gates on
the artifact it produced. Deterministic — a pure function of its inputs (no I/O,
no LLM), mirroring dataset/gate: it emits a verdict + a cited reason + an
outcome-channel item; a downstream guard (`prep_scrub_gate_passed` /
`prep_scrub_needs_review`) routes on ``action``.
"""
from __future__ import annotations

from cairn.core.operator import Operation, OperationContext, register_operator
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks.schemas.prep_gate import (
    _CERTIFYING_BACKENDS,
    PrepGateConfig,
    PrepGateInput,
    PrepGateOutput,
)


@register_operator(
    name="dataset/prep_gate",
    version="0.1.0",
    capabilities=["dataset.prep_gate"],
    implements=["contract://dataset.prep_gate/1.0.0"],  # #2242 V1-D — cairn-contracts-dataset
)
class PrepGateOp(Operation[PrepGateInput, PrepGateOutput]):
    """Turn prep's post-scrub PII re-scan into a config-gated verdict."""

    input_schema = PrepGateInput
    output_schema = PrepGateOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    config_schema = PrepGateConfig  # ADR-0070 — the env-owned `mode` knob
    supports_mock = False  # deterministic; nothing to stub

    async def run(self, inp: PrepGateInput, ctx: OperationContext) -> PrepGateOutput:
        rows = int(inp.post_scrub_pii_rows or 0)
        clean = rows == 0
        sampled = bool(inp.post_scrub_pii_sampled)
        backend = (inp.post_scrub_pii_backend or "").lower()
        # #889 — "clean" is only a GUARANTEE when the scan was FULL (not sampled) AND
        # on a certifying NER backend. A sampled scan can't prove 0 PII; regex/regexset
        # miss names/phone/SSN/location, so their "0 PII" isn't certifiable either.
        certifying = backend in _CERTIFYING_BACKENDS
        verified = clean and not sampled and certifying

        why = []
        if sampled:
            why.append("sampled scan (can't prove 0 rows)")
        if backend and not certifying:
            why.append(f"'{backend}' backend has no NER (misses names/phone/SSN)")
        uncert = "; ".join(why)

        if not clean:
            # dirty — a real finding; enforce per mode (unchanged from #882/#883).
            kinds = ", ".join(f"{v} {k}" for k, v in sorted(inp.post_scrub_pii_counts.items()) if v)
            detail = f" ({kinds})" if kinds else ""
            if inp.mode == "block":
                action, verdict = "block", "fail"
                blocking = [f"post-scrub PII: {rows} row(s) still carry PII after anonymize{detail} "
                            f"— dataset BLOCKED (mode=block); widen the redaction backend/kinds and re-run prep."]
            elif inp.mode == "hitl":
                action, verdict = "review", "fail"
                blocking = [f"post-scrub PII: {rows} row(s) still carry PII{detail} — HELD for human review (mode=hitl)."]
            else:  # off — surface only
                action, verdict, blocking = "pass", "pass", []
        elif verified:
            action, verdict, blocking = "pass", "pass", []
        else:
            # #889 — 0 PII found but the scan CAN'T CERTIFY clean → at most WARN, never a
            # clean PASS. A consumer/pipeline gates on `verified`, so this won't train.
            verdict = "warn"
            action = "review" if inp.mode == "hitl" else "pass"
            blocking = [f"post-scrub PII: 0 found but the scan can't CERTIFY clean — {uncert}. "
                        f"A governed run needs a full presidio scan (`[pii]` extra) to guarantee it."]

        severity = "block" if verdict == "fail" else ("warn" if verdict == "warn" else "ok")
        # #890 — the outcome carries its own provenance so a reader never mistakes an
        # estimate for a guarantee.
        outcome = [{
            "name": "post-scrub PII" if verified else "post-scrub PII (unverified)",
            "value": rows, "target": 0, "pass": clean,
            "severity": severity,
            "verified": verified, "sampled": sampled, "backend": backend or "n/a",
        }]
        return PrepGateOutput(action=action, verdict=verdict, clean=clean, verified=verified,
                              blocking=blocking, outcome=outcome)
