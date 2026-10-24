"""finetune — fine-tune a model on prepared dataset versions (cairn-extensions#7). Data only;
this shim builds the Pack from the ``pack.toml`` beside it."""
from __future__ import annotations

from cairn.core.pack import Pack

pack = Pack.from_manifest(__file__)

__all__ = ["pack"]
