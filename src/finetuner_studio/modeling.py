"""Model and tokenizer loading shared by training and evaluation."""

from __future__ import annotations

from typing import Any

from .hardware import resolve_dtype
from .spec import ModelSpec


def load_tokenizer(spec: ModelSpec):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        spec.base_model, revision=spec.revision, trust_remote_code=spec.trust_remote_code
    )
    if spec.chat_template:
        tok.chat_template = spec.chat_template
    if tok.chat_template is None:
        raise ValueError(
            f"{spec.base_model} has no chat template; set model.chat_template in the spec"
        )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(spec: ModelSpec, for_training: bool):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = resolve_dtype(spec.dtype)
    kwargs: dict[str, Any] = {
        "revision": spec.revision,
        "trust_remote_code": spec.trust_remote_code,
        "torch_dtype": dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if spec.attn_implementation != "auto":
        kwargs["attn_implementation"] = spec.attn_implementation

    if spec.quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif spec.quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(spec.base_model, **kwargs)
    if for_training:
        model.config.use_cache = False
    return model, dtype
