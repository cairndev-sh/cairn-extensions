"""cairn-operator-eval-lm — lm-evaluation-harness against a SERVED candidate.

Importing this package must NOT import ``lm_eval``: the harness is the optional
``[harness]`` extra and lives behind :mod:`operator_eval_lm.harness`.
"""
from __future__ import annotations

__all__ = ["harness", "presets"]
