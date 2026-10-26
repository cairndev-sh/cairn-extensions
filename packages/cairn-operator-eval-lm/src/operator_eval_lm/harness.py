"""The lm-evaluation-harness SEAM — one function, one lazy import.

Everything the pack knows about ``lm_eval`` is in :func:`run_harness`. The
operator never imports the library, so:

* the pack imports, registers and its whole test suite runs with the optional
  ``[harness]`` extra ABSENT (CI stays hermetic — no 2 GB of torch to test a
  scorecard shape);
* a test replaces exactly one attribute to drive the operator end to end;
* the day lm-eval changes its call signature, one function moves.

``simple_evaluate`` returns ``{results: {task: {metric: value}}, higher_is_better:
{task: {metric: bool}}, samples: {task: [...]}}``. The kwargs below are the
stable ones; the exact spelling of the decontamination argument must be checked
against the installed release before the first live run (see DESIGN.md §6).
"""
from __future__ import annotations

from typing import Any


def run_harness(*, model: str, model_args: str, tasks: list[str], **kw: Any) -> dict:
    """Call ``lm_eval.simple_evaluate``. Raises ``ImportError`` without the extra."""
    import lm_eval  # noqa: PLC0415 — optional [harness] extra, imported only when a run happens

    return lm_eval.simple_evaluate(model=model, model_args=model_args, tasks=tasks, **kw)


__all__ = ["run_harness"]
