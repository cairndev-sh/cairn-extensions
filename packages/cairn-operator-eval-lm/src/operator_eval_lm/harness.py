"""The lm-evaluation-harness SEAM — one function, one lazy import.

Everything the pack knows about ``lm_eval`` is in :func:`run_harness`. The
operator never imports the library, so:

* the pack imports, registers and its whole test suite runs with the optional
  ``[harness]`` extra ABSENT (CI stays hermetic — no 2 GB of torch to test a
  scorecard shape);
* a test replaces exactly one attribute to drive the operator end to end;
* the day lm-eval changes its call signature, one function moves.

``simple_evaluate`` returns ``{results: {task: {metric: value}}, higher_is_better:
{task: {metric: bool}}, samples: {task: [...]}}``. The seam adds two keys of its
own, :data:`DROPPED_KEY` and :data:`VERSION_KEY`, so the operator can say on the
record what was not passed.

The ``[harness]`` extra is bounded to the releases this seam was read against:
``lm-eval>=0.4.10,<0.5``. Line numbers below are 0.4.13's; 0.4.10, 0.4.11 and 0.4.12
have the same code at nearby lines. ``header``/``auth_token`` first appear in
0.4.9.2, and 0.4.9 has neither.

**The endpoint key.** lm-eval's OpenAI-compatible ``local-completions`` model sends
``Authorization: Bearer $OPENAI_API_KEY`` from the PROCESS ENVIRONMENT unless the
model is built with an explicit, non-empty ``header``:

* ``models/openai_completions.py`` 139-140 read ``OPENAI_API_KEY``;
* ``models/api_models.py`` 137/141 take ``auth_token`` and ``header`` explicitly,
  159/192 store them, and 322-324 fall back to ``self.api_key`` when ``header`` is
  empty; requests send ``self.header`` (474, 517);
* ``TemplateAPI.__init__`` swallows unknown ``**kwargs`` (143) without a word, so a
  release that dropped ``header`` would ignore it silently.

The process environment is shared by every run, so the seam never writes it and
never lets lm-eval read it. It builds the model itself, with an explicit header
(the key when the run has one, none otherwise), through
``create_from_arg_string`` (``api/model.py`` 131-146, which logs nothing). It then
**checks the built model's effective** ``header`` **against the one it passed**
and refuses on any difference, before a single benchmark request. Only then does
it hand the model object to ``simple_evaluate`` (``evaluator.py`` 266-270, "Using
pre-initialized model"). The key is never in ``model_args``, which lm-eval logs
and copies into its results (``evaluator.py`` 252-254, 396). Construction logs
batch size, max length, concurrency and tokenizer backend, never the header or
the token (``api_models.py`` 160-200). One request does happen during
construction: ``local-completions`` probes the endpoint for a remote tokenizer
(``openai_completions.py`` 26-35), and it authenticates with ``auth_token`` only
(``utils.py`` 854-877).

**Arguments the installed release does not take.** ``limit`` and ``num_fewshot``
bound the run's cost and define its scores, so a release that does not take one
is a refusal (:class:`HarnessRefusal`) before the model is built. Any other
argument is dropped and named under :data:`DROPPED_KEY`. 0.4.10-0.4.13's
``simple_evaluate`` takes no decontamination argument (``evaluator.py`` 55-87), so
a decontamination request is always dropped on the supported range.
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


class HarnessRefusal(RuntimeError):
    """The installed lm-eval cannot run this evaluation as asked. Raised before any
    benchmark request; the operator turns it into a ``checked=false`` stub."""


#: Arguments that may never be dropped: ``limit`` bounds what the run costs (smoke
#: asks for 40 documents per task, not all of them) and ``num_fewshot`` defines the
#: scores.
REQUIRED_ARGUMENTS = ("limit", "num_fewshot")
#: Keys the seam adds to the results it returns.
DROPPED_KEY = "cairn_dropped_arguments"
VERSION_KEY = "cairn_lm_eval_version"


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
    """Build the lm-eval model with explicit headers, check them, and call
    ``lm_eval.simple_evaluate``.

    Synchronous and long; the operator runs it in a worker thread. Raises
    ``ImportError`` without the extra and :class:`HarnessRefusal` when the installed
    release cannot run the evaluation as asked. Messages never carry the key."""
    import lm_eval  # noqa: PLC0415 — optional [harness] extra, imported only when a run happens
    from lm_eval.api.registry import get_model  # noqa: PLC0415 — same extra

    version = str(getattr(lm_eval, "__version__", "?"))
    call_kw, dropped = accepted_kwargs(lm_eval.simple_evaluate, kw)
    required = [a for a in REQUIRED_ARGUMENTS if a in dropped]
    if required:
        raise HarnessRefusal(
            f"lm-eval {version}'s simple_evaluate does not take {', '.join(required)}; running "
            "without it would change what the run costs or what it scores. Install a release "
            "in the [harness] range (lm-eval>=0.4.10,<0.5)."
        )

    auth = auth_kwargs(api_key)
    lm = get_model(model).create_from_arg_string(
        model_args, {"batch_size": kw.get("batch_size"), **auth})
    if getattr(lm, "header", None) != auth["header"]:
        raise HarnessRefusal(
            f"lm-eval {version} built the model with a request header other than the one passed "
            "(it would fall back to OPENAI_API_KEY from the process environment); refusing "
            "before any benchmark request. Install a release in the [harness] range "
            "(lm-eval>=0.4.10,<0.5)."
        )
    if dropped:
        logger.warning("lm-eval %s does not take %s; not passed", version, ", ".join(dropped))
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, **call_kw)
    if isinstance(results, dict):
        results = {**results, DROPPED_KEY: dropped, VERSION_KEY: version}
    return results


__all__ = ["DROPPED_KEY", "REQUIRED_ARGUMENTS", "VERSION_KEY", "HarnessRefusal",
           "accepted_kwargs", "auth_kwargs", "run_harness"]
