"""Token-count backends for dataset/profile (issue #381).

Two backends, one interface — same shape as :mod:`_pii`:

  - **whitespace** (default, zero-dep): ``str.split()`` word count. Cheap and
    deterministic, but not what a model's context window actually consumes —
    subword tokenizers routinely split one "word" into several tokens.
  - **tokenizers** (accurate, ``[tokenizers]`` extra): HF ``tokenizers`` —
    the Rust-backed library used by every HF model's fast tokenizer. Reports
    real subword token counts for a named vocab.

``backend="auto"`` uses tokenizers when importable, else whitespace — mirrors
``_pii.resolve``. Note: the ``tokenizers`` PyPI package may already be present
as a transitive dep of ``transformers`` (it is in this repo's dev venv), but it
is declared as its own optional extra here so a base install without either
package still works via the whitespace fallback.

Loading a vocab has two paths:
  - ``tokenizer_path``: a local ``tokenizer.json`` (``Tokenizer.from_file`` —
    fully offline, what the hermetic tests use).
  - ``tokenizer_name``: a HF Hub id (``Tokenizer.from_pretrained`` — fetches
    from the Hub on first use; the production path, needs network/cache).
Exactly one may be given; ``tokenizer_path`` wins if both are set.
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from typing import Optional


def tokenizers_available() -> bool:
    return bool(importlib.util.find_spec("tokenizers"))


@lru_cache(maxsize=8)
def _load_tokenizer(tokenizer_name: Optional[str], tokenizer_path: Optional[str]):
    from tokenizers import Tokenizer  # optional dependency (tokenizers extra)

    if tokenizer_path:
        return Tokenizer.from_file(tokenizer_path)
    if tokenizer_name:
        return Tokenizer.from_pretrained(tokenizer_name)
    raise ValueError("tokenizer backend requires tokenizer_name or tokenizer_path")


def resolve(backend: str) -> str:
    if backend == "auto":
        return "tokenizers" if tokenizers_available() else "whitespace"
    if backend == "tokenizers" and not tokenizers_available():
        raise RuntimeError(
            "backend='tokenizers' requires the [tokenizers] extra: "
            "pip install 'cairn-operator-dataset-checks[tokenizers]'."
        )
    return backend


def count_whitespace(text: str) -> int:
    """Zero-dep word count — the default hermetic path."""
    return len(text.split())


def count_tokenizers(
    text: str, *, tokenizer_name: Optional[str], tokenizer_path: Optional[str],
) -> int:
    """Real subword token count via a loaded HF ``tokenizers`` vocab."""
    tok = _load_tokenizer(tokenizer_name, tokenizer_path)
    return len(tok.encode(text).ids)


def count(
    text: str,
    *,
    backend: str = "auto",
    tokenizer_name: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> int:
    resolved = resolve(backend)
    if resolved == "tokenizers":
        return count_tokenizers(
            text, tokenizer_name=tokenizer_name, tokenizer_path=tokenizer_path,
        )
    return count_whitespace(text)
