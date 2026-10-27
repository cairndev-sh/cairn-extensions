"""The in-process SEAM of the on-box ``eval.lm_eval`` stage (cairn-extensions#42).

Everything the stage needs from torch, transformers, peft and lm-eval is behind
:class:`Backend`, so the stage's logic (the time probe, both passes with identical settings, the
deltas, the split loss and the results file) runs and is tested without a GPU and without the
``[box]`` extra:

* :class:`HfBackend` — the real one, on the GPU box. It loads the base model ONCE (4-bit for
  ``qlora``, as it was trained), scores it with lm-eval's ``hf`` backend (``HFLM`` over the loaded
  model, one ``simple_evaluate`` call per task), then wraps the SAME loaded model with the adapter
  (``peft.PeftModel``) and scores it again with the same calls. Read against lm-eval 0.4.13
  (``models/huggingface.py`` 217-230: a pre-initialised model is taken as is) and transformers 5.5;
  not run here (no GPU): the first live run proves it (``DESIGN.md`` §7).
* :class:`StubBackend` — the stub tier (``stub: true``): no model, deterministic placeholder
  numbers, so the pipeline's every leg, gate and record run in CI. The stage marks its output
  ``stub`` and never ``checked``.

The time probe needs the time the MODEL spent, apart from lm-eval's own work (building tasks,
downloading datasets, computing metrics): :func:`_instrument` wraps the loaded ``HFLM``'s three
request methods (``evaluator.py`` 600 calls ``getattr(lm, reqtype)(requests)``) and counts their
requests and seconds.

No endpoint and no key: the ``hf`` backend loads weights from the Hugging Face Hub (or its cache)
with whatever Hub access the box has, and sends no request to any model endpoint. Nothing is
written to ``os.environ``.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from operator_eval_lm import harness

#: ChatML, the template the trainer gives a base model's tokenizer that has none
#: (``cairn-operator-training``'s ``chat_template.CHATML``; a copy, because an operator pack never
#: imports another — ``cairn-pack-finetune``'s tests hold the two equal).
CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

#: The ``simple_evaluate`` arguments a pass may never lose: ``limit`` bounds the cost,
#: ``num_fewshot`` defines the scores, and the seeds make base and candidate comparable.
REQUIRED = ("limit", "num_fewshot", "random_seed", "numpy_random_seed", "torch_random_seed",
            "fewshot_random_seed")


@dataclass
class Timing:
    """What the model spent in one ``simple_evaluate`` call: requests answered and seconds."""

    requests: int = 0
    model_s: float = 0.0


@dataclass
class Loaded:
    """A loaded model as the stage holds it. ``lm`` is the lm-eval wrapper, built once per model
    and reused across tasks (its batch arithmetic stays the same)."""

    model: Any
    tokenizer: Any
    quantization: str
    adapter: str = ""
    lm: Any = None
    extra: dict = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def versions(self) -> dict[str, str]: ...

    def load(self, base_model: str, *, quantize_4bit: bool, seed: int) -> Loaded: ...

    def chat_template(self, loaded: Loaded, adapter: str) -> tuple[str, str]: ...

    def adapter_base(self, adapter: str) -> str: ...

    def render(self, loaded: Loaded, messages: list[dict], template: str) -> str: ...

    def attach(self, loaded: Loaded, adapter: str) -> Loaded: ...

    def evaluate(self, loaded: Loaded, *, task: str, limit: int, num_fewshot: int, batch_size: int,
                 seed: int) -> tuple[dict, Timing]: ...

    def split_loss(self, loaded: Loaded, texts: list[str], *, max_seq_len: int) -> tuple[float, int]: ...


def _instrument(lm: Any, timing: Timing):
    """Count the requests and seconds of ``lm``'s request methods; returns the undo."""
    wrapped: list[str] = []
    for name in ("loglikelihood", "loglikelihood_rolling", "generate_until"):
        fn = getattr(lm, name, None)
        if fn is None:
            continue

        def timed(requests, *args, _fn=fn, **kwargs):
            started = time.monotonic()
            try:
                return _fn(requests, *args, **kwargs)
            finally:
                timing.requests += len(requests)
                timing.model_s += time.monotonic() - started

        setattr(lm, name, timed)
        wrapped.append(name)

    def undo() -> None:
        for name in wrapped:
            lm.__dict__.pop(name, None)

    return undo


class HfBackend:
    """torch + transformers + peft + lm-eval's ``hf`` backend, on the GPU box."""

    name = "hf"

    def __init__(self) -> None:
        import lm_eval  # noqa: F401, PLC0415 — the [box] extra; ImportError tells the stage it is absent
        import peft  # noqa: F401, PLC0415
        import torch  # noqa: F401, PLC0415
        import transformers  # noqa: F401, PLC0415

    def versions(self) -> dict[str, str]:
        import lm_eval  # noqa: PLC0415
        import peft  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import transformers  # noqa: PLC0415

        return {"lm_eval": str(getattr(lm_eval, "__version__", "?")), "torch": torch.__version__,
                "transformers": transformers.__version__, "peft": peft.__version__,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}

    def load(self, base_model: str, *, quantize_4bit: bool, seed: int) -> Loaded:
        import torch  # noqa: PLC0415
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        pre = getattr(AutoConfig.from_pretrained(base_model), "quantization_config", None)
        # float16, not bfloat16: a T4 (compute capability 7.5) has no bf16, the trainer's rule too
        kw: dict[str, Any] = {"dtype": torch.float16}
        if torch.cuda.is_available():
            kw["device_map"] = {"": 0}
        if pre is not None:
            method = pre.get("quant_method", "?") if isinstance(pre, dict) else getattr(pre, "quant_method", "?")
            quantization = f"pre-quantized checkpoint ({method})"
        elif quantize_4bit:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)
            quantization = "4-bit nf4 on load (double quant, float16 compute)"
        else:
            quantization = "none (float16)"
        model = AutoModelForCausalLM.from_pretrained(base_model, **kw)
        model.eval()
        return Loaded(model=model, tokenizer=tokenizer, quantization=quantization)

    def chat_template(self, loaded: Loaded, adapter: str) -> tuple[str, str]:
        """The template the adapter was trained with: the one its saved tokenizer carries (the
        trainer saves the tokenizer it rendered with), else the trainer's own rule — the base
        tokenizer's template, else ChatML (``chat_template.ensure_chat_template``)."""
        from transformers import AutoTokenizer  # noqa: PLC0415

        try:
            template = getattr(AutoTokenizer.from_pretrained(adapter), "chat_template", None)
        except Exception:  # noqa: BLE001 — an adapter without tokenizer files
            template = None
        if template:
            return template, "adapter"
        if getattr(loaded.tokenizer, "chat_template", None):
            return loaded.tokenizer.chat_template, "base model"
        return CHATML, "chatml"

    def adapter_base(self, adapter: str) -> str:
        """The base the adapter records it was trained on (``base_model_name_or_path``); with
        Unsloth's 4-bit loading that is the pre-quantised checkpoint it substituted."""
        from peft import PeftConfig  # noqa: PLC0415

        try:
            return str(PeftConfig.from_pretrained(adapter).base_model_name_or_path or "")
        except Exception:  # noqa: BLE001 — unknown is reported as unknown
            return ""

    def render(self, loaded: Loaded, messages: list[dict], template: str) -> str:
        return loaded.tokenizer.apply_chat_template(messages, tokenize=False, chat_template=template)

    def attach(self, loaded: Loaded, adapter: str) -> Loaded:
        from peft import PeftModel  # noqa: PLC0415

        model = PeftModel.from_pretrained(loaded.model, adapter)
        model.eval()
        return Loaded(model=model, tokenizer=loaded.tokenizer, quantization=loaded.quantization,
                      adapter=adapter)

    def evaluate(self, loaded: Loaded, *, task: str, limit: int, num_fewshot: int, batch_size: int,
                 seed: int) -> tuple[dict, Timing]:
        import lm_eval  # noqa: PLC0415
        from lm_eval.models.huggingface import HFLM  # noqa: PLC0415

        if loaded.lm is None:
            loaded.lm = HFLM(pretrained=loaded.model, tokenizer=loaded.tokenizer, batch_size=batch_size)
        kw = {"tasks": [task], "limit": limit or None, "num_fewshot": num_fewshot, "log_samples": False,
              "random_seed": seed, "numpy_random_seed": seed, "torch_random_seed": seed,
              "fewshot_random_seed": seed}
        call, dropped = harness.accepted_kwargs(lm_eval.simple_evaluate, kw)
        lost = [a for a in REQUIRED if a in dropped]
        if lost:
            raise harness.HarnessRefusal(
                f"lm-eval {getattr(lm_eval, '__version__', '?')}'s simple_evaluate does not take "
                f"{', '.join(lost)}; base and candidate would not be scored alike. Install a release "
                "in the [box] range (lm-eval>=0.4.10,<0.4.14).")
        timing = Timing()
        undo = _instrument(loaded.lm, timing)
        try:
            results = lm_eval.simple_evaluate(model=loaded.lm, **call)
        finally:
            undo()
        return dict(results or {}), timing

    def split_loss(self, loaded: Loaded, texts: list[str], *, max_seq_len: int) -> tuple[float, int]:
        import torch  # noqa: PLC0415

        total, tokens = 0.0, 0
        device = next(loaded.model.parameters()).device
        with torch.no_grad():
            for text in texts:
                ids = loaded.tokenizer(text, return_tensors="pt", truncation=True,
                                       max_length=max_seq_len).input_ids.to(device)
                if ids.shape[1] < 2:
                    continue
                logits = loaded.model(input_ids=ids).logits[0, :-1].float()
                total += torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum").item()
                tokens += ids.shape[1] - 1
        return total, tokens


def _unit(*parts: object) -> float:
    """A deterministic number in [0, 1) from ``parts``."""
    digest = hashlib.sha256("\0".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


class StubBackend:
    """The stub tier: no model is loaded and nothing is measured. Numbers are placeholders, the
    same for the same inputs; the candidate differs from the base by a small offset. A group task
    (``mmlu``) reports three subjects."""

    name = "stub"

    def versions(self) -> dict[str, str]:
        return {"lm_eval": "stub"}

    def load(self, base_model: str, *, quantize_4bit: bool, seed: int) -> Loaded:
        return Loaded(model=None, tokenizer=None, quantization="stub (nothing loaded)",
                      extra={"base_model": base_model, "seed": seed})

    def chat_template(self, loaded: Loaded, adapter: str) -> tuple[str, str]:
        return CHATML, "chatml"

    def adapter_base(self, adapter: str) -> str:
        return ""

    def render(self, loaded: Loaded, messages: list[dict], template: str) -> str:
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)

    def attach(self, loaded: Loaded, adapter: str) -> Loaded:
        return Loaded(model=None, tokenizer=None, quantization=loaded.quantization, adapter=adapter,
                      extra=dict(loaded.extra))

    def evaluate(self, loaded: Loaded, *, task: str, limit: int, num_fewshot: int, batch_size: int,
                 seed: int) -> tuple[dict, Timing]:
        leaves = [f"{task}_s{i}" for i in range(3)] if task == "mmlu" else [task]
        n_samples = {leaf: {"original": 1000, "effective": min(limit, 1000) if limit else 1000}
                     for leaf in leaves}
        value = 0.4 + 0.4 * _unit(loaded.extra.get("base_model"), task, seed)
        if loaded.adapter:
            value += 0.04 * (_unit(loaded.adapter, task) - 0.5)
        docs = sum(c["effective"] for c in n_samples.values())
        se = (value * (1 - value) / max(1, docs)) ** 0.5
        return ({"results": {task: {"acc,none": value, "acc_stderr,none": se, "alias": task}},
                 "higher_is_better": {task: {"acc": True}}, "n-samples": n_samples,
                 "versions": {task: "stub"}},
                Timing(requests=4 * docs, model_s=0.0))

    def split_loss(self, loaded: Loaded, texts: list[str], *, max_seq_len: int) -> tuple[float, int]:
        tokens = sum(max(1, len(t.split())) for t in texts)
        per_token = 0.5 if loaded.adapter else 2.0
        return per_token * tokens, tokens


def make_backend(*, stub: bool) -> Backend:
    """The backend for this run. Raises ImportError when the real one's libraries are absent."""
    return StubBackend() if stub else HfBackend()


__all__ = ["CHATML", "REQUIRED", "Backend", "HfBackend", "Loaded", "StubBackend", "Timing", "make_backend"]
