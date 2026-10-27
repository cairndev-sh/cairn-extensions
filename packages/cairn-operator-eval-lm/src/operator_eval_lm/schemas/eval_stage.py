"""Configuration of the ``eval.lm_eval`` flow stage (cairn-extensions#42).

A flow stage takes a flat config dict (``$config.*`` substituted, absent keys omitted); this model
validates it, so a typo'd method or a negative budget fails before a model loads. Uncontracted
(no ``contract://`` stage contract yet): the shape is this package's, versioned with it.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: The preset a leg gets when it names none.
DEFAULT_STAGE_PRESET = "forgetting"


class EvalStageConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base_model: str = Field(..., min_length=1, description="Hugging Face id (or path) of the base model.")
    adapter_uri: str = Field("", description="The adapter: s3://…/ prefix, hf://<repo>, a Hub id or a path "
                                             "on the box. A promoted tag (<name>@<tag>) is refused.")
    local_adapter_dir: str = Field("", description="The train leg's adapter directory on this box; used "
                                                   "first when it exists.")
    method: Literal["qlora", "lora"] = Field("qlora", description="qlora: the base is loaded in 4 bit, as "
                                                                  "it was trained; lora: float16.")
    preset: str = Field(DEFAULT_STAGE_PRESET, description="Task preset (this package's presets/*.toml).")
    seed: int = Field(1234, description="lm-eval's four seeds (python, numpy, torch, few-shot), the same "
                                        "for both models.")
    batch_size: int = Field(0, ge=0, description="Per-model batch size; 0 = the preset's.")
    val_uri: str = Field("", description="The validation split, as the control plane resolved it "
                                         "(<location>#sha256=<hash>).")
    test_uri: str = Field("", description="The test split, resolved the same way.")
    max_seq_len: int = Field(2048, gt=0, description="Split-loss rows are truncated at this many tokens.")
    split_rows: int = Field(0, ge=0, description="Rows of each split scored for loss; 0 = every row.")
    time_budget_s: float = Field(0.0, ge=0.0, description="The leg's timeout in seconds. The time probe "
                                                          "refuses a run projected past it. Required "
                                                          "outside the stub tier.")
    time_margin: float = Field(0.1, ge=0.0, lt=1.0, description="Fraction of the budget kept back for "
                                                                "what the probe cannot see (the "
                                                                "results upload, variance).")
    probe_items: int = Field(50, gt=0, description="Documents the time probe scores on the base model "
                                                   "before the full run, spread over the tasks in "
                                                   "proportion to their limits.")
    stub: bool = Field(False, description="The stub tier: no model is loaded; placeholder numbers, "
                                          "never `checked`.")


__all__ = ["DEFAULT_STAGE_PRESET", "EvalStageConfig"]
