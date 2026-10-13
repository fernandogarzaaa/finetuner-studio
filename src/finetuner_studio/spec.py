"""Run specification: the single source of truth for a fine-tuning run.

A spec is a YAML (or JSON) document. It is validated here, before any GPU work
starts, so that bad configs fail in milliseconds instead of twenty minutes in.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SPEC_VERSION = 1


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class DatasetFormat(str, Enum):
    messages = "messages"  # [{"role": ..., "content": ...}, ...]
    alpaca = "alpaca"  # instruction / input / output
    prompt_completion = "prompt_completion"  # prompt / completion


class DatasetSpec(_Strict):
    source: Literal["hf", "jsonl"] = "hf"
    path: str = Field(..., description="HF dataset id or path to a .jsonl file")
    config_name: str | None = None
    split: str = "train"
    eval_split: str | None = Field(
        None, description="Use this split for evaluation; otherwise hold out eval_size"
    )
    eval_size: float | int = Field(
        0.05, description="Fraction (<1) or absolute count of rows held out for eval"
    )
    format: DatasetFormat = DatasetFormat.messages
    columns: dict[str, str] = Field(
        default_factory=dict,
        description="Map canonical column names (messages, instruction, input, output, "
        "prompt, completion, system) to the dataset's column names",
    )
    system_prompt: str | None = None
    max_samples: int | None = Field(None, gt=0)
    shuffle_seed: int = 42

    @field_validator("eval_size")
    @classmethod
    def _eval_size(cls, v: float | int) -> float | int:
        if isinstance(v, float) and not 0 < v < 1:
            raise ValueError("eval_size as a fraction must be between 0 and 1")
        if isinstance(v, int) and v < 1:
            raise ValueError("eval_size as a count must be >= 1")
        return v

    @model_validator(mode="after")
    def _jsonl_path(self) -> DatasetSpec:
        if self.source == "jsonl" and not self.path.endswith((".jsonl", ".json")):
            raise ValueError("jsonl source needs a path ending in .jsonl or .json")
        return self


class ModelSpec(_Strict):
    base_model: str = Field(..., description="HF model id or local path")
    revision: str | None = None
    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    max_seq_length: int = Field(1024, ge=64, le=32768)
    chat_template: str | None = Field(
        None, description="Jinja chat template; defaults to the tokenizer's own"
    )
    trust_remote_code: bool = False
    attn_implementation: Literal["auto", "eager", "sdpa", "flash_attention_2"] = "auto"


class LoraSpec(_Strict):
    r: int = Field(16, ge=1, le=512)
    alpha: int = Field(32, ge=1)
    dropout: float = Field(0.05, ge=0, lt=1)
    target_modules: list[str] | Literal["all-linear"] = "all-linear"
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = False


class TrainingSpec(_Strict):
    epochs: float = Field(1.0, gt=0)
    max_steps: int = Field(-1, description="Overrides epochs when > 0")
    learning_rate: float = Field(2e-4, gt=0, lt=1)
    per_device_batch_size: int = Field(2, ge=1)
    gradient_accumulation_steps: int = Field(8, ge=1)
    warmup_ratio: float = Field(0.03, ge=0, lt=1)
    lr_scheduler: Literal["cosine", "linear", "constant", "constant_with_warmup"] = "cosine"
    weight_decay: float = Field(0.0, ge=0)
    gradient_checkpointing: bool = True
    packing: bool = False
    logging_steps: int = Field(10, ge=1)
    save_steps: int = Field(200, ge=1)
    seed: int = 42
    optim: str = "paged_adamw_8bit"


class Metric(str, Enum):
    perplexity = "perplexity"  # lower is better
    rouge_l = "rouge_l"
    token_f1 = "token_f1"
    exact_match = "exact_match"

    @property
    def higher_is_better(self) -> bool:
        return self is not Metric.perplexity


class EvaluationSpec(_Strict):
    metrics: list[Metric] = Field(
        default_factory=lambda: [Metric.perplexity, Metric.rouge_l, Metric.token_f1]
    )
    max_samples: int = Field(200, ge=1)
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.0, ge=0)
    batch_size: int = Field(4, ge=1)


class Guardrail(_Strict):
    metric: Metric
    max_regression: float = Field(
        ..., ge=0, description="Largest allowed relative regression vs base (0.05 = 5%)"
    )


class ShipRule(_Strict):
    primary_metric: Metric = Metric.rouge_l
    min_improvement: float = Field(
        0.02, description="Minimum relative improvement on the primary metric (0.02 = 2%)"
    )
    guardrails: list[Guardrail] = Field(
        default_factory=lambda: [Guardrail(metric=Metric.perplexity, max_regression=0.10)]
    )
    min_eval_samples: int = Field(
        30, ge=1, description="Fewer scored samples than this yields INCONCLUSIVE"
    )


class OutputSpec(_Strict):
    dir: str = "runs"
    merge_adapter: bool = False
    push_to_hub: str | None = Field(None, description="Hub repo id to push the adapter to")


class RunSpec(_Strict):
    version: int = SPEC_VERSION
    name: str = Field(..., pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    description: str | None = None
    model: ModelSpec
    dataset: DatasetSpec
    lora: LoraSpec = Field(default_factory=LoraSpec)
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    ship_rule: ShipRule = Field(default_factory=ShipRule)
    output: OutputSpec = Field(default_factory=OutputSpec)

    @field_validator("version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != SPEC_VERSION:
            raise ValueError(f"unsupported spec version {v}; this build reads version {SPEC_VERSION}")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> RunSpec:
        rule_metrics = {self.ship_rule.primary_metric, *(g.metric for g in self.ship_rule.guardrails)}
        missing = rule_metrics - set(self.evaluation.metrics)
        if missing:
            names = ", ".join(sorted(m.value for m in missing))
            raise ValueError(f"ship_rule uses metrics not listed in evaluation.metrics: {names}")
        if self.model.quantization != "none" and self.model.dtype == "float32":
            raise ValueError("quantized training needs float16/bfloat16 compute (dtype)")
        return self

    # --- IO -----------------------------------------------------------------

    @classmethod
    def from_yaml(cls, text: str) -> RunSpec:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("spec must be a YAML mapping")
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: str | Path) -> RunSpec:
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def effective_batch_size(self) -> int:
        return self.training.per_device_batch_size * self.training.gradient_accumulation_steps


def json_schema() -> dict[str, Any]:
    """JSON Schema for editors and the web UI."""
    return RunSpec.model_json_schema()
