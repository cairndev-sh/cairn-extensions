"""Schemas for dataset/capture_to_rewards — captured scores → GRPO reward CONTEXT.

ADR-0095 §1.2 / §2.1 (#1629). Deliberately NOT ``(prompt, completion, score)``
triples: GRPO is on-policy — it generates its own rollouts each step and scores
THOSE with ``reward_funcs``. A score captured in production is attached to a
completion the policy no longer produces, so feeding it as a training reward
would be off-policy noise dressed as supervision.

What captured scores ARE good for is the reward function's ground context: the
prompt set worth training on, the reference text a grader can compare against,
and whatever the production verifier recorded in ``payload``. That is what this
projection emits — one row per (prompt, score scale), with the aggregate a
reward function can read.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from operator_dataset_checks.schemas.capture_to_sft import AdapterCount


class CaptureToRewardsInput(BaseModel):
    path: str = Field(
        ..., description="The exported capture JSONL (one CaptureEvent per line).",
    )
    out_path: str = Field(..., description="Where to write the reward-context JSONL.")
    only_adapter: str = Field(
        "", description="Restrict to signal about ONE adapter_uri (attribution).",
    )
    require_score: bool = Field(
        True, description="Emit only prompts that carry at least one `score` event. "
                          "Turning this off also emits scored-nothing prompts with "
                          "score_count=0, which is useful when the reward function "
                          "is a verifier that needs no human number.",
    )


class CaptureToRewardsOutput(BaseModel):
    out_path: str = Field(..., description="Path the reward-context dataset was written to.")
    rows: int = Field(0, description="Prompt-scale groups written.")
    scores_seen: int = Field(0, description="`score` events aggregated.")
    prompts_without_score: int = Field(
        0, description="Prompts carrying no score (emitted only when "
                       "require_score is false).",
    )
    mixed_scale_prompts: int = Field(
        0, description="Prompts scored on MORE than one scale — each scale becomes "
                       "its own row rather than averaging '4 out of 5' with '0.8'.",
    )
    scales: list[str] = Field(
        default_factory=list, description="The score scales encountered, sorted.",
    )
    unresolved_scores: int = Field(
        0, description="`score` events whose prompt could not be resolved (no "
                       "prompt on the row and no correlated trace carrying one).",
    )
    by_adapter: list[AdapterCount] = Field(
        default_factory=list, description="Attribution readout, sorted.",
    )


class RewardContextRow(BaseModel):
    """One emitted row. Documented as a schema even though the operator writes
    plain dicts, so a reward function has something to read against."""

    prompt: str = Field(..., description="The production prompt.")
    reference: str = Field(
        "", description="The best human-endorsed text for this prompt (an edit "
                        "beats an accept), for graders that compare against gold.",
    )
    mean_score: float = Field(0.0, description="Mean of the captured scores.")
    score_count: int = Field(0, description="How many humans/verifiers scored it.")
    score_scale: str = Field("", description="The scale those numbers are on.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Merged scalar keys from the score events' `payload` — the "
                    "production verifier's own context, passed through verbatim.",
    )
