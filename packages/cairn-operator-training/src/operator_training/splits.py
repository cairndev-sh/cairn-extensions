"""The governed splits and the batch knobs, shared by the GPU and the CPU trainer
(``training.stage.train_qlora/1.1.0``, cairn-extensions#4).

* **Which data trains, which evaluates.** ``train_uri`` (a registered train version) is the
  training data when set, else ``dataset_uri``. ``eval_uri`` (a registered validation version)
  is the evaluation data when set, and then ``eval_split`` is not used; without it the trainer
  keeps its seeded random hold-out of the training data. ``test_uri`` is recorded on the result
  and never opened here: the held-out test set reaches the evaluation legs after training, not
  the trainer.
* **Batch.** ``per_device_batch_size`` and ``gradient_accumulation_steps`` override the
  trainer's defaults when above 0; the effective batch (their product on one device) is recorded
  so a run's record says what it trained with.

Stdlib only and duck-typed over a ``TrainSpec`` or its ``model_dump()`` dict, so both trainers
and the tests call the same code with no training stack installed.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def _get(spec: Any, key: str, default: Any = None) -> Any:
    value = spec.get(key, default) if isinstance(spec, dict) else getattr(spec, key, default)
    return default if value is None else value


def train_source(spec: Any) -> str:
    return str(_get(spec, "train_uri", "") or _get(spec, "dataset_uri", "") or "")


def train_and_eval(spec: Any, load: Callable[[str], Any], *, seed: int,
                   can_split: Callable[[Any], bool] = lambda ds: True,
                   prepare_train: Optional[Callable[[Any], Any]] = None) -> tuple[Any, Any, str]:
    """``(train_ds, eval_ds, eval_source)``. ``load`` turns a URI into the trainer's dataset;
    ``prepare_train`` (optional) trims the training data before any split. ``eval_source`` is
    ``eval_uri``, ``eval_split`` or ``none`` — what the evaluation numbers were computed on."""
    train = load(train_source(spec))
    if prepare_train is not None:
        train = prepare_train(train)
    eval_uri = str(_get(spec, "eval_uri", "") or "")
    if eval_uri:
        return train, load(eval_uri), "eval_uri"
    # 0.1 when a raw spec dict omits it: the same default TrainSpec declares
    fraction = float(_get(spec, "eval_split", 0.1) or 0.0)
    if fraction > 0 and can_split(train):
        parts = train.train_test_split(test_size=fraction, seed=seed)
        return parts["train"], parts["test"], "eval_split"
    return train, None, "none"


def batch_args(spec: Any, *, per_device: int, accumulation: int) -> dict[str, int]:
    """The ``SFTConfig`` batch arguments: the spec's when above 0, else the trainer's defaults."""
    return {
        "per_device_train_batch_size": int(_get(spec, "per_device_batch_size", 0) or 0) or per_device,
        "gradient_accumulation_steps": int(_get(spec, "gradient_accumulation_steps", 0) or 0) or accumulation,
    }


def recorded(spec: Any, batch: dict[str, int], eval_source: str) -> dict[str, Any]:
    """The scalars a run records about its data and batch."""
    out: dict[str, Any] = {
        "effective_batch_size": batch["per_device_train_batch_size"] * batch["gradient_accumulation_steps"],
        "eval_source": eval_source,
    }
    test_uri = str(_get(spec, "test_uri", "") or "")
    if test_uri:
        out["test_uri"] = test_uri
    return out


__all__ = ["batch_args", "recorded", "train_and_eval", "train_source"]
