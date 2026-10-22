"""dataset-prep — prepare a dataset once into registered train / val / test versions
(cairn-extensions#16). Data only; this shim builds the Pack from the ``pack.toml`` beside it."""
from __future__ import annotations

from cairn.core.pack import Pack

pack = Pack.from_manifest(__file__)

__all__ = ["pack"]
