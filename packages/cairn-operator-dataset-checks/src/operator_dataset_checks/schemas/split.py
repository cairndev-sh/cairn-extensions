"""Schemas for dataset/split — deterministic, group-aware (+ optionally
stratified) train/val[/test] split (#383).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StratumSummary(BaseModel):
    label: str = Field(..., description="Stratify-field value (or '_all_' when not stratifying).")
    counts: dict[str, int] = Field(
        ..., description="Rows landing in each split for this stratum.",
    )


class SplitInput(BaseModel):
    path: str = Field(..., description="Path to the source JSONL dataset.")
    output_dir: str = Field(
        ..., description="Directory to write train.jsonl/val.jsonl[/test.jsonl] into.",
    )
    seed: int = Field(
        13, description="Seed for the deterministic hash/shuffle assignment. Same "
        "dataset + seed always reproduces the identical split.",
    )
    train_ratio: float = Field(0.9, description="Fraction of rows/groups assigned to train.")
    val_ratio: float = Field(0.1, description="Fraction of rows/groups assigned to val.")
    test_ratio: float = Field(
        0.0, description="Fraction assigned to test. 0 (default) omits test.jsonl "
        "entirely — SFT convention is train + a small held-out val for loss "
        "monitoring; a dedicated benchmark eval-gate covers what a classic "
        "test split would, so test is opt-in, not the classic 70/15/15 default.",
    )
    group_by: Optional[str] = Field(
        None, description="Row field (e.g. a doc/source id) whose value must never "
        "span two splits — every row sharing this field's value is assigned to the "
        "SAME split. None means every row is its own group (no grouping).",
    )
    stratify_by: Optional[str] = Field(
        None, description="Row field whose value distribution should be balanced "
        "across splits (e.g. source/task/language).",
    )
    backend: str = Field(
        "auto", description="'auto' (sklearn if installed and useful, else hash) | "
        "'hash' (pure-stdlib, always available) | 'sklearn' (StratifiedShuffleSplit/"
        "GroupShuffleSplit/StratifiedGroupKFold, needs the [split] extra).",
    )
    shard_size: int = Field(
        0, description="Rows per output shard (#891). 0 (default) writes ONE "
        "<split>.jsonl per split — the back-compatible shape. >0 rolls to "
        "<split>.00001.jsonl … every shard_size rows so a >>RAM split lands as many "
        "bounded files a downstream reader can stream/parallelize. Only the streaming "
        "hash backend shards; the sklearn (materialized) path always writes one file.",
    )

    register_as: str = Field(
        "", description="#2020 (ADR-0102 D1) — register each split as a FROZEN CHILD "
        "VERSION of this dataset name: '<register_as>-train' (role training), "
        "'-val' (validation) and '-test' (test), each with its own content hash, "
        "lineage back to the parent version, and the declared split policy on the "
        "record. Empty (default) writes files only, exactly as before. Items never "
        "move: a re-split under a different policy mints NEW children and leaves the "
        "old ones byte-identical at their old version numbers.",
    )
    parent_ref: str = Field(
        "", description="The parent's 'dataset:<name>@<version>' ref, recorded as the "
        "children's parent_version and the source of their inherited schema / column "
        "mapping. Empty resolves the head version of `register_as` when one exists.",
    )

    @field_validator("train_ratio", "val_ratio", "test_ratio")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("ratios must be >= 0")
        return v

    @model_validator(mode="after")
    def _ratios_are_sane(self) -> "SplitInput":
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if total <= 0:
            raise ValueError("at least one of train_ratio/val_ratio/test_ratio must be > 0")
        return self


class SplitConfig(BaseModel):
    """Deployment-tunable knobs of ``dataset/split`` (ADR-0070) — the split ratios
    + seed. **Operational** scope (the default): a workflow/run may tune these
    freely. A subset of ``SplitInput``; the safe defaults live on ``SplitInput``."""

    model_config = ConfigDict(extra="forbid")

    train_ratio: float = Field(
        0.9, title="Train ratio", description="Fraction of rows/groups assigned to train.",
    )
    val_ratio: float = Field(
        0.1, title="Val ratio", description="Fraction of rows/groups assigned to val.",
    )
    test_ratio: float = Field(
        0.0, title="Test ratio", description="Fraction assigned to test (0 omits test.jsonl).",
    )
    seed: int = Field(
        13, title="Split seed",
        description="Seed for the deterministic hash/shuffle assignment (reproducible split).",
    )
    shard_size: int = Field(
        0, title="Output shard size",
        description="Rows per output shard (#891); 0 writes one file per split (default).",
    )


class SplitOutput(BaseModel):
    backend_used: str = Field(..., description="'hash' or 'sklearn'.")
    total: int = Field(..., description="Total rows read from the source dataset.")
    counts: dict[str, int] = Field(
        ..., description="Rows written to each split (train/val/[test]).",
    )
    ratios_achieved: dict[str, float] = Field(
        ..., description="counts[name] / total for each split, rounded.",
    )
    ratios_requested: dict[str, float] = Field(
        ..., description="The requested train/val/[test] ratios, normalized to sum to 1.",
    )
    no_leakage: bool = Field(
        ..., description="True iff zero group (per group_by) spans more than one "
        "split — the leakage-check evidence the readiness report cites.",
    )
    group_by: Optional[str] = Field(None, description="The group field used, if any.")
    group_count: Optional[int] = Field(
        None, description="Distinct group keys seen, when group_by is set.",
    )
    stratify_by: Optional[str] = Field(None, description="The stratify field used, if any.")
    stratification_summary: list[StratumSummary] = Field(
        default_factory=list, description="Per-stratum split counts, when stratify_by is set.",
    )
    output_paths: dict[str, str] = Field(
        ..., description="Split name -> its PRIMARY JSONL file path (the first/only "
        "shard). Unchanged when unsharded; the first shard when shard_size > 0.",
    )
    shard_paths: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Split name -> every shard file written for it (#891). A "
        "single-element list per split when unsharded; the full ordered shard list "
        "when shard_size > 0 (read all of them for the complete split).",
    )
    child_versions: dict[str, str] = Field(
        default_factory=dict,
        description="#2020 — split name -> the frozen child version it registered, "
        "as 'dataset:<name>@<version>'. Empty when register_as was not set.",
    )
    child_hashes: dict[str, str] = Field(
        default_factory=dict,
        description="#2020 — split name -> the child version's content hash. Same "
        "input + same policy reproduces the same hash (and therefore the same "
        "version); a different policy mints a new one.",
    )
    split_policy: dict = Field(
        default_factory=dict,
        description="#2020 — the DECLARED policy recorded on every child version: "
        "ratios, stratify_by, group_by, seed, backend. A split whose policy is not "
        "on the record is a random tail nobody can reproduce.",
    )
