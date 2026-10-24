"""#1622 — the trainer pins are ONE set, kept in three places that can't import
each other.

``images.py::TRAINER_PINS`` is canonical. The two copies exist because neither
consumer is Python: this pack's ``pyproject.toml`` ``[local]`` extra (the CPU
subset, ranges) and ``[gpu]`` extra (the exact set a GPU box installs, ``==``)
are declarative dependency tables. That is the classic two-lists-in-two-files
rot — pinning without this guard just moves the drift from "which version" to
"which file". So the test IS half the deliverable: change a pin and it stays red
until both copies follow.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

from operator_training.images import TRAINER_PINS, _TRAINER_DEPS

_PACK_ROOT = Path(__file__).resolve().parents[1]

#: Pinned packages the CPU `[local]` extra deliberately does NOT carry: both are
#: CUDA-only, so the plain-LoRA mock in operators/local.py cannot install them.
_GPU_ONLY = {"unsloth", "unsloth_zoo", "bitsandbytes"}

_REQ = re.compile(r"^([A-Za-z0-9_.-]+)(.*)$")


def _extra(name: str) -> dict[str, str]:
    """``{package: specifier}`` from one of the pack's optional-dependency tables."""
    data = tomllib.loads((_PACK_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    out = {}
    for req in data["project"]["optional-dependencies"][name]:
        m = _REQ.match(req.strip())
        assert m, f"unparseable requirement in [{name}]: {req!r}"
        out[m.group(1)] = m.group(2)
    return out


def _local_extra() -> dict[str, str]:
    return _extra("local")


def test_trainer_deps_are_the_pins_exactly():
    # The canonical set as `==` requirements, nothing bare.
    assert _TRAINER_DEPS == [f"{n}=={v}" for n, v in TRAINER_PINS.items()]
    assert all("==" in dep for dep in _TRAINER_DEPS)


def test_pins_are_exact_versions():
    # A range here would defeat the point: "build the image twice, get the same
    # versions" is the acceptance criterion.
    for name, version in TRAINER_PINS.items():
        assert re.fullmatch(r"\d+(\.\d+)+", version), f"{name} pin is not an exact version: {version}"



def test_local_extra_is_a_pinned_subset():
    local = _local_extra()
    # Same package set, minus the CUDA-only ones the CPU mock can't have.
    assert set(local) == set(TRAINER_PINS) - _GPU_ONLY
    for name, spec in local.items():
        major_minor = ".".join(TRAINER_PINS[name].split(".")[:2])
        # Floor == the pin's major.minor: the mock may pick up patch/minor
        # releases, but it starts where the GPU trainer is.
        assert spec.startswith(f">={major_minor},"), (
            f"[local] {name}{spec} does not floor at the pin {TRAINER_PINS[name]}")
        # …and it is bounded, so it can never drift a major away from the trainer.
        assert "<" in spec, f"[local] {name}{spec} is unbounded — it can drift a major"


def test_gpu_only_pins_stay_out_of_the_cpu_extra():
    # Guards the reason the sets differ: installing either of these into the
    # CPU/MPS extra breaks the plain-LoRA path (operators/local.py docstring).
    local = _local_extra()
    for name in _GPU_ONLY:
        assert name not in local


#: What `[gpu]` carries beside the trainer pins: the S3 client for s3:// data, checkpoint
#: mirroring and `artifact_target: s3` (transport, not part of the trainer environment).
_GPU_TRANSPORT = {"boto3": ">=1.34"}


def test_gpu_extra_is_the_pins_exactly():
    # A GPU box is installed from this extra (`pip install 'cairn-operator-training[gpu]'`),
    # so it has to be the same set, version for version, or the proven ssh path proves a
    # different environment from the one the image runs.
    assert _extra("gpu") == {**{name: f"=={version}" for name, version in TRAINER_PINS.items()},
                             **_GPU_TRANSPORT}
