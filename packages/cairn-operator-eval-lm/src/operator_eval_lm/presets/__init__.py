"""Task presets — shipped as pack DATA (ADR-0102 D5).

Every ``*.toml`` beside this module is one preset: a named task list with the
few-shot count, the per-task document limit and the decontamination switch. The
loader is here; the *content* is data, so changing what a pipeline benchmarks is
a data edit reviewed like a config change, not a code change — the property that
made lm-evaluation-harness the pick (its tasks are YAML too).

Adding a preset: drop ``<name>.toml`` in this directory. It is discovered, and
:func:`load` validates it — an unreadable or task-less preset fails at load with
the file named, never at the harness call ten minutes into a GPU job.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

#: The preset a pipeline gets when it names none: small enough to run on EVERY
#: candidate. A default that takes an hour is a default nobody uses.
DEFAULT_PRESET = "smoke"

_DIR = Path(__file__).resolve().parent


class TaskPreset(BaseModel):
    """One preset, as read from its TOML file."""

    name: str
    description: str = ""
    tasks: list[str] = Field(default_factory=list)
    num_fewshot: int = 0
    limit: int = Field(0, ge=0, description="Documents per task; 0 = the whole task (expensive).")
    batch_size: int = Field(8, gt=0)
    decontaminate: bool = Field(
        True, description="Request lm-eval's 13-gram overlap decontamination (ADR-0102 D5)."
    )
    limits: dict[str, int] = Field(
        default_factory=dict,
        description="Per-task document limits that override `limit` (cairn-extensions#42). lm-eval "
                    "applies a limit per LEAF task, so a group's limit (mmlu: 57 subjects) is per "
                    "subject. Read by the on-box `eval.lm_eval` stage, which runs each task in its "
                    "own harness call; the served-candidate operator runs one call with `limit`.",
    )

    def limit_for(self, task: str) -> int:
        """The document limit for ``task``: its entry in ``limits``, else ``limit`` (0 = all)."""
        return int(self.limits.get(task, self.limit))


def available() -> list[str]:
    """Preset names shipped by this pack, sorted."""
    return sorted(p.stem for p in _DIR.glob("*.toml"))


def preset_path(name: str) -> Path:
    return _DIR / f"{name}.toml"


def load(name: str) -> TaskPreset:
    """Read one preset, or fail naming what exists.

    A typo'd preset name must not silently fall back to the default: the run
    would report benchmark numbers for tasks nobody asked for.
    """
    path = preset_path(name)
    if not path.is_file():
        raise ValueError(
            f"unknown lm-eval task preset {name!r}; this pack ships {available()!r} "
            f"(default: {DEFAULT_PRESET!r})"
        )
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    preset = TaskPreset.model_validate({"name": name, **data})
    if not preset.tasks:
        raise ValueError(f"task preset {name!r} ({path}) names no tasks")
    return preset


__all__ = ["DEFAULT_PRESET", "TaskPreset", "available", "load", "preset_path"]
