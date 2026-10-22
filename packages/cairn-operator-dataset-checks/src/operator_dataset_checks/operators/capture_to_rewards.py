"""dataset/capture_to_rewards — captured scores → GRPO reward CONTEXT.

ADR-0095 §1.2 / §2.1 (#1629). **Why this is not a table of
``(prompt, completion, score)`` triples**, which is what "scores → rewards"
sounds like it should be:

GRPO is on-policy. At every step it generates its own group of rollouts for a
prompt and scores THOSE with ``reward_funcs``; there is no place in the
algorithm that consumes a reward attached to somebody else's completion. A score
captured in production is attached to a completion the policy has already moved
away from — feeding it in as a training reward is off-policy noise wearing the
costume of supervision, and it gets worse every cycle as the adapter drifts from
the version that produced the text.

What production scores genuinely supply is the reward function's GROUND
CONTEXT: which prompts are worth training on, what a human was willing to ship
for each (a reference a grader can compare a fresh rollout against), how the
prompt has historically scored, and whatever the production verifier recorded
alongside its number. One row per (prompt, scale), so "4 out of 5" is never
averaged with "0.8".

D4: reads the EXPORTED file, never the capture store.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks._capture import (
    attribution,
    correlation,
    gold_of,
    iter_capture,
    text,
)
from operator_dataset_checks._io import JsonlWriter
from operator_dataset_checks.operators.capture_to_sft import adapter_counts
from operator_dataset_checks.schemas.capture_to_rewards import (
    CaptureToRewardsInput,
    CaptureToRewardsOutput,
)

# D6 payload budget: counts, scales and adapter URIs in the findings — the
# prompts and references live in the written dataset, never in run state.

#: Kinds whose text a human endorsed, best first — the `reference` a grader
#: compares a fresh rollout against.
_REFERENCE_KINDS = ("edit", "accept")


def _scalars(payload: Any) -> dict[str, Any]:
    """The verifier context worth carrying: scalar-valued keys only.

    Nested structures are dropped rather than flattened — a reward function that
    needs a whole object should read the capture export itself, and passing one
    through here would quietly put arbitrary depth into a training file.
    """
    if not isinstance(payload, dict):
        return {}
    return {k: v for k, v in payload.items() if isinstance(v, (str, int, float, bool))}


@register_operator(
    name="dataset/capture_to_rewards",
    version="0.1.0",
    capabilities=["dataset.capture_to_rewards"],
)
class CaptureToRewardsOp(Operation[CaptureToRewardsInput, CaptureToRewardsOutput]):
    """Aggregate captured scores into per-prompt reward context for GRPO."""

    input_schema = CaptureToRewardsInput
    output_schema = CaptureToRewardsOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    # #1539 — governance declarations; the [operator.*] table is generated from these
    risk = 'external_write'

    async def run(self, inp: CaptureToRewardsInput, ctx: OperationContext) -> CaptureToRewardsOutput:
        rows = [r for r in iter_capture(inp.path)
                if not inp.only_adapter
                or attribution(r)["adapter_uri"] == inp.only_adapter]

        # A score event usually carries only a correlation id, so the prompt (and
        # the reference text) come from the trace/edit rows it points at.
        prompt_by_corr: dict[str, str] = {}
        reference_by_corr: dict[str, str] = {}
        for r in rows:
            corr = correlation(r)
            if not corr:
                continue
            prompt_by_corr.setdefault(corr, "")
            if not prompt_by_corr[corr] and text(r, "prompt"):
                prompt_by_corr[corr] = text(r, "prompt")
            # an edit beats an accept beats nothing; later endorsements win, so a
            # second correction supersedes the first.
            if r.get("kind") in _REFERENCE_KINDS and gold_of(r):
                reference_by_corr[corr] = gold_of(r)

        # group in first-seen order → deterministic output without a sort.
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        scales_by_prompt: dict[str, set[str]] = {}
        scores_seen = unresolved = 0
        tally: dict[tuple[str, str], int] = {}

        for r in rows:
            if r.get("kind") != "score" or r.get("score") is None:
                continue
            corr = correlation(r)
            prompt = text(r, "prompt") or prompt_by_corr.get(corr, "")
            if not prompt:
                unresolved += 1
                continue
            scale = text(r, "score_scale")
            key = (prompt, scale)
            g = groups.setdefault(key, {"scores": [], "context": {},
                                        "reference": reference_by_corr.get(corr, "")})
            g["scores"].append(float(r["score"]))
            # later events win on a key collision — the freshest verifier reading
            g["context"].update(_scalars(r.get("payload")))
            if not g["reference"]:
                g["reference"] = reference_by_corr.get(corr, "")
            scales_by_prompt.setdefault(prompt, set()).add(scale)
            scores_seen += 1
            attr = attribution(r)
            tkey = (attr["adapter_uri"], attr["model"])
            tally[tkey] = tally.get(tkey, 0) + 1

        # prompts with no score at all — always COUNTED (a readiness pipeline wants to
        # know how much captured traffic nobody rated), emitted only when asked.
        unscored_prompts = {p for p in prompt_by_corr.values() if p and p not in scales_by_prompt}
        if not inp.require_score:
            for corr, prompt in prompt_by_corr.items():
                if prompt in unscored_prompts:
                    groups.setdefault((prompt, ""), {
                        "scores": [], "context": {},
                        "reference": reference_by_corr.get(corr, "")})
        unscored = len(unscored_prompts)

        with JsonlWriter(Path(inp.out_path)) as writer:
            for (prompt, scale), g in groups.items():
                scores = g["scores"]
                writer.write({
                    "prompt": prompt,
                    "reference": g["reference"],
                    "mean_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
                    "score_count": len(scores),
                    "score_scale": scale,
                    "context": g["context"],
                })

        return CaptureToRewardsOutput(
            out_path=inp.out_path, rows=len(groups), scores_seen=scores_seen,
            prompts_without_score=unscored,
            mixed_scale_prompts=sum(1 for s in scales_by_prompt.values() if len(s) > 1),
            scales=sorted({s for scales in scales_by_prompt.values() for s in scales}),
            unresolved_scores=unresolved, by_adapter=adapter_counts(tally),
        )
