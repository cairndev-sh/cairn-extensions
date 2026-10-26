"""The lm-evaluation-harness SEAM — one function, one lazy import.

Everything the pack knows about ``lm_eval`` is in :func:`run_harness`. The
operator never imports the library, so:

* the pack imports, registers and its whole test suite runs with the optional
  ``[harness]`` extra ABSENT (CI stays hermetic — no 2 GB of torch to test a
  scorecard shape);
* a test replaces exactly one attribute to drive the operator end to end;
* the day lm-eval changes its call signature, one function moves.

``simple_evaluate`` returns ``{results: {task: {metric: value}}, higher_is_better:
{task: {metric: bool}}, samples: {task: [...]}}``.

**The endpoint key.** lm-eval's OpenAI-compatible ``local-completions`` model sends
``Authorization: Bearer $OPENAI_API_KEY`` from the PROCESS ENVIRONMENT unless the
model is built with an explicit, non-empty ``header`` (lm-eval 0.4.13:
``models/api_models.py`` 141/159 take ``header``, 322-324 fall back to
``self.api_key``; ``models/openai_completions.py`` 139-140 read
``OPENAI_API_KEY``). The process environment is shared by every run, so the seam
never writes it and never lets lm-eval read it: it builds the model object itself
with an explicit header (the key when the run has one, none otherwise) and hands
that object to ``simple_evaluate``. The key is never in ``model_args``, which
lm-eval logs and copies into its results (``evaluator.py`` 252-254, 396).

**Arguments the installed release does not take are dropped**, not passed:
0.4.13's ``simple_evaluate`` has no decontamination argument (``evaluator.py``
55-87), and passing one raised ``TypeError``, so every default run failed. A
dropped decontamination request then shows up as the operator's "decontamination
was requested but no ``_decontaminate`` metric came back" note.
"""
from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def auth_kwargs(api_key: str | None) -> dict[str, Any]:
    """Constructor arguments that fix the request headers, so lm-eval never falls back
    to ``OPENAI_API_KEY`` from the environment. The header is never empty: an empty
    dict is falsy and lm-eval would fall back."""
    header = {"Content-Type": "application/json"}
    if not api_key:
        return {"header": header}
    header["Authorization"] = f"Bearer {api_key}"
    # auth_token: the remote-tokenizer probe at construction (utils.check_remote_tokenizer_support)
    return {"header": header, "auth_token": api_key}


def accepted_kwargs(fn: Callable[..., Any], kw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``(the kwargs fn accepts, the names it does not)``. Everything passes when fn
    takes ``**kwargs``."""
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kw), []
    kept = {k: v for k, v in kw.items() if k in params}
    return kept, sorted(set(kw) - set(kept))


def run_harness(*, model: str, model_args: str, tasks: list[str], api_key: str | None = None,
                **kw: Any) -> dict:
    """Build the lm-eval model with explicit headers and call ``lm_eval.simple_evaluate``.

    Synchronous and long; the operator runs it in a worker thread. Raises
    ``ImportError`` without the extra."""
    import lm_eval  # noqa: PLC0415 — optional [harness] extra, imported only when a run happens
    from lm_eval.api.registry import get_model  # noqa: PLC0415 — same extra

    lm = get_model(model).create_from_arg_string(
        model_args, {"batch_size": kw.get("batch_size"), **auth_kwargs(api_key)})
    call_kw, dropped = accepted_kwargs(lm_eval.simple_evaluate, kw)
    if dropped:
        logger.warning("lm-eval %s does not take %s; not passed",
                       getattr(lm_eval, "__version__", "?"), ", ".join(dropped))
    return lm_eval.simple_evaluate(model=lm, tasks=tasks, **call_kw)


__all__ = ["accepted_kwargs", "auth_kwargs", "run_harness"]
