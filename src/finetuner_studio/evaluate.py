"""Base-vs-tuned evaluation.

The base model is loaded once and the adapter attached on top. Base scores are
taken with the adapter disabled, tuned scores with it enabled, so both models
share one copy of the weights in VRAM (important on a 16 GB T4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import EvalExample, split_for_eval
from .events import Event, EventSink
from .metrics import perplexity_from_loss, score_generations
from .modeling import load_model, load_tokenizer
from .spec import RunSpec


@dataclass
class EvalOutput:
    base: dict[str, float]
    tuned: dict[str, float]
    samples: int


def _completion_nll(model, tokenizer, ex: EvalExample, max_len: int) -> tuple[float, int]:
    """Summed NLL of the reference answer's tokens given the prompt."""
    import torch

    prompt_text = tokenizer.apply_chat_template(ex.prompt, add_generation_prompt=True, tokenize=False)
    full_text = tokenizer.apply_chat_template(
        [*ex.prompt, {"role": "assistant", "content": ex.reference}], tokenize=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_len]
    n_prompt = min(len(prompt_ids), len(full_ids))
    n = len(full_ids) - n_prompt
    if n < 1:
        return 0.0, 0
    input_ids = torch.tensor([full_ids], device=model.device)
    labels = input_ids.clone()
    labels[:, :n_prompt] = -100
    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=labels).loss
    # HF shifts labels by one, so n tokens are scored when the prompt is non-empty.
    return float(loss) * n, n


def _generate(model, tokenizer, examples: list[EvalExample], spec: RunSpec) -> list[str]:
    import torch

    e = spec.evaluation
    tokenizer.padding_side = "left"
    outs: list[str] = []
    for i in range(0, len(examples), e.batch_size):
        batch = examples[i : i + e.batch_size]
        prompts = [
            tokenizer.apply_chat_template(x.prompt, add_generation_prompt=True, tokenize=False)
            for x in batch
        ]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=spec.model.max_seq_length,
            add_special_tokens=False,
        ).to(model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": e.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": e.temperature > 0,
        }
        if e.temperature > 0:
            gen_kwargs["temperature"] = e.temperature
        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        outs += tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [o.strip() for o in outs]


def _score(model, tokenizer, examples, spec: RunSpec, label: str, sink: EventSink):
    names = [m.value for m in spec.evaluation.metrics]
    result: dict[str, float] = {}
    if "perplexity" in names:
        total_nll, total_tok = 0.0, 0
        for ex in examples:
            nll, n = _completion_nll(model, tokenizer, ex, spec.model.max_seq_length)
            total_nll += nll
            total_tok += n
        if total_tok:
            result["perplexity"] = perplexity_from_loss(total_nll / total_tok)
    preds: list[str] = []
    gen_metrics = [n for n in names if n != "perplexity"]
    if gen_metrics:
        preds = _generate(model, tokenizer, examples, spec)
        result.update(score_generations(preds, [x.reference for x in examples], gen_metrics))
    sink.emit(Event("eval_result", {"model": label, "metrics": result}))
    return result, preds


def evaluate(spec: RunSpec, eval_ds, adapter_dir: Path, run_dir: Path, sink: EventSink) -> EvalOutput:
    from peft import PeftModel

    n = min(spec.evaluation.max_samples, len(eval_ds))
    examples = [split_for_eval(row["messages"]) for row in eval_ds.select(range(n))]

    tokenizer = load_tokenizer(spec.model)
    base_model, _ = load_model(spec.model, for_training=False)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    with model.disable_adapter():
        base, base_preds = _score(model, tokenizer, examples, spec, "base", sink)
    tuned, tuned_preds = _score(model, tokenizer, examples, spec, "tuned", sink)

    samples_path = run_dir / "eval_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            row = {
                "prompt": ex.prompt,
                "reference": ex.reference,
                "base": base_preds[i] if i < len(base_preds) else None,
                "tuned": tuned_preds[i] if i < len(tuned_preds) else None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    sink.emit(Event("artifact", {"kind": "eval_samples", "path": str(samples_path)}))
    return EvalOutput(base=base, tuned=tuned, samples=len(examples))
