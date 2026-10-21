"""The canonical trainer environment, pinned exactly (#1622).

``TRAINER_PINS`` is the one version set the Unsloth QLoRA trainer runs on: the ``[gpu]`` extra
installs it exactly on a CUDA box, the CPU ``[local]`` extra floors at it, and
``env_fingerprint.trainer_env()`` records what a run actually got. See the comment block on it.
"""
from __future__ import annotations

# ── The trainer env, pinned exactly (#1622, ADR-0095 reproducibility) ─────────
# These names used to be BARE, so the image resolved whatever existed on build
# day: two runs of the same pipeline weeks apart trained on different
# trl/unsloth with different defaults, and `trl.GRPOTrainer` (what #1623 needs)
# came and went with them. cairn pins the digest of job images so an auditor can
# reconstruct what executed — the trainer env has to be pinned to the same
# standard. `env_fingerprint.trainer_env()` records what a run ACTUALLY got.
#
# HOW this set was chosen (2026-08-26): anchor on the latest stable unsloth and
# take the single resolution its own dependency metadata admits —
#   uv pip compile --python-platform linux --python-version 3.11
#     (unsloth==2026.8.21 + the seven bare names below)
# unsloth CAPS every library it monkey-patches (torch<2.12, trl<=0.24,
# transformers<=5.5 minus a blocklist, datasets<4.4, peft>=0.18, bitsandbytes
# >=0.45.5 !=0.46/0.48, accelerate>=0.34.1), so most pins here are the HIGHEST
# stable release unsloth admits, NOT the newest on PyPI. That gap is the whole
# hazard: today's newest trl (1.10) and datasets (5.0.1) are both outside
# unsloth's range, so a bare `pip install unsloth trl datasets` re-arms exactly
# the import-order / renamed-kwarg traps documented in `dpo_entry`'s header and
# in `train_entry.main`.
#   torch 2.11.0        highest under unsloth's `torch<2.12.0` (2.13.0 is current).
#                       The CUDA build still has to match the GPU box's driver.
#   unsloth 2026.8.21   latest stable; the anchor for everything above.
#   unsloth_zoo 2026.8.15  unsloth's own companion (`unsloth_zoo>=2026.8.15`); it
#                       ships on unsloth's daily cadence, so leaving it free
#                       re-floats the very patches unsloth pins.
#   trl 0.24.0          unsloth's ceiling (`trl<=0.24.0,!=0.19.0`) — trl 1.x is OUT.
#                       `GRPOTrainer`/`GRPOConfig` ARE top-level exports at 0.24.0
#                       (verified in the wheel), so #1623's GRPO slice has its
#                       trainer here. It is also the trl the C2 GPU run hit: the
#                       eos_token and `processing_class` notes in `train_entry`
#                       are 0.24 behaviours.
#   peft 0.20.0         latest stable; satisfies `peft>=0.18.0,!=0.11.0`.
#   transformers 5.5.0  unsloth's ceiling (`<=5.5.0`, plus a blocklist of known-bad
#                       4.5x/5.0/5.1 releases). Newer 5.x exists; unsloth has not
#                       cleared it.
#   datasets 4.3.0      highest under `datasets<4.4.0,!=4.0.*,!=4.1.0` — and the
#                       exact version unsloth's own `[studio]` extra pins.
#   bitsandbytes 0.50.1 latest stable; satisfies `>=0.45.5,!=0.46.0,!=0.48.0`.
#                       CUDA-only — which is why it is absent from `[local]`.
#   accelerate 1.14.0   latest stable; satisfies `>=0.34.1`.
#
# THIS DICT IS THE CANONICAL SET. Two copies exist by necessity and neither can
# import Python: the `[local]` (CPU subset) and `[gpu]` (the exact set a GPU box
# installs) extras in this pack's pyproject.toml. The drift guard is
# `tests/test_trainer_pins.py` — change a pin here and it goes red until both
# copies follow.
TRAINER_PINS: dict[str, str] = {
    "torch": "2.11.0",
    "unsloth": "2026.8.21",
    "unsloth_zoo": "2026.8.15",
    "trl": "0.24.0",
    "peft": "0.20.0",
    "transformers": "5.5.0",
    "datasets": "4.3.0",
    "bitsandbytes": "0.50.1",
    "accelerate": "1.14.0",
}

_TRAINER_DEPS = [f"{name}=={version}" for name, version in TRAINER_PINS.items()]
