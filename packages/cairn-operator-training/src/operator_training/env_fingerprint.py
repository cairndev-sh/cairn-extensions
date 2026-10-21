"""What the trainer env ACTUALLY resolved to, recorded per run (#1622).

``images.py::TRAINER_PINS`` states what the env *should* be; this states what it
*was*. Both halves are needed because only one path installs from the pins
automatically (a trainer image) — the SSH box is provisioned by hand
(``pip install 'cairn-operator-training[gpu]'``) and the ``[local]`` extra resolves a range, so "the pins say trl 0.24"
is an intention, not evidence. The fingerprint is the evidence: an auditor
reading a finished run sees the torch/trl/unsloth that produced *that* adapter,
whatever the box was actually carrying (ADR-0095 reproducibility prerequisite).

It rides the existing ``metrics`` channel (``metrics['trainer_env']``) into
``TrainResult`` → ``result.json`` / ``metrics.json`` → MLflow, so there is no new
plumbing to keep alive.

Everything here is best-effort and NEVER raises: a version probe must not kill a
paid GPU run. A failure records an ``error`` key instead.
"""
from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from operator_training.images import TRAINER_PINS

#: Recorded for a pinned package that is not installed in this env — the honest
#: answer, and a meaningful one (the CPU `[local]` stack has no unsloth).
ABSENT = "absent"


def _installed(dist: str) -> str:
    try:
        return version(dist)
    except PackageNotFoundError:
        return ABSENT
    except Exception as e:  # noqa: BLE001 - a broken dist-info must not fail the run
        return f"error: {type(e).__name__}"


def _device_facts() -> dict[str, Any]:
    """``cuda_available`` (+ ``gpu_name``) when torch is importable.

    Absent keys mean "no torch here", which is itself part of the fingerprint —
    a run that recorded no ``cuda_available`` did not train on a GPU.
    """
    try:
        import torch  # noqa: PLC0415 - optional dependency: the [local] extra / backend image
    except Exception:  # noqa: BLE001 - no torch (or a broken one) ⇒ no device facts
        return {}
    out: dict[str, Any] = {}
    try:
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            out["gpu_name"] = str(torch.cuda.get_device_name(0))
    except Exception as e:  # noqa: BLE001 - a probe failure is a note, not a stop
        out["device_probe_error"] = f"{type(e).__name__}: {e}"
    return out


def trainer_env() -> dict[str, Any]:
    """The per-run trainer-env fingerprint.

    ``{<pinned package>: version-or-'absent', python, platform, cuda_available?,
    gpu_name?}`` — every key of ``TRAINER_PINS`` is always present, so a diff
    against the pins is a plain dict comparison.
    """
    try:
        env: dict[str, Any] = {name: _installed(name) for name in TRAINER_PINS}
        env["python"] = platform.python_version()
        env["platform"] = platform.platform()
        env.update(_device_facts())
        return env
    except Exception as e:  # noqa: BLE001 - the fingerprint must never fail a paid run
        return {"error": f"{type(e).__name__}: {e}"}
