"""QLoRA / LoRA supervised fine-tuning with TRL."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

from .data import PreparedData
from .events import Event, EventSink
from .modeling import load_model, load_tokenizer
from .spec import RunSpec


def _callback(sink: EventSink):
    from transformers import TrainerCallback

    class EventCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            sink.emit(Event("progress", {"stage": "train", "current": 0, "total": state.max_steps}))

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" in logs:
                sink.emit(
                    Event(
                        "train_metrics",
                        {
                            "step": state.global_step,
                            "loss": logs.get("loss"),
                            "learning_rate": logs.get("learning_rate"),
                            "epoch": logs.get("epoch"),
                            "grad_norm": logs.get("grad_norm"),
                        },
                    )
                )

        def on_step_end(self, args, state, control, **kwargs):
            sink.emit(
                Event("progress", {"stage": "train", "current": state.global_step, "total": state.max_steps})
            )

    return EventCallback()


def _sft_config(**kwargs: Any):
    """Build an SFTConfig across TRL versions (max_seq_length was renamed max_length)."""
    from trl import SFTConfig

    names = {f.name for f in dataclasses.fields(SFTConfig)}
    if "max_seq_length" in kwargs and "max_seq_length" not in names:
        kwargs["max_length"] = kwargs.pop("max_seq_length")
    return SFTConfig(**{k: v for k, v in kwargs.items() if k in names})


def train(spec: RunSpec, data: PreparedData, run_dir: Path, sink: EventSink) -> Path:
    import torch
    from peft import LoraConfig
    from trl import SFTTrainer

    tokenizer = load_tokenizer(spec.model)
    model, dtype = load_model(spec.model, for_training=True)

    lora = LoraConfig(
        r=spec.lora.r,
        lora_alpha=spec.lora.alpha,
        lora_dropout=spec.lora.dropout,
        target_modules=spec.lora.target_modules,
        bias=spec.lora.bias,
        use_rslora=spec.lora.use_rslora,
        task_type="CAUSAL_LM",
    )

    t = spec.training
    on_gpu = torch.cuda.is_available()
    config = _sft_config(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=t.epochs,
        max_steps=t.max_steps,
        learning_rate=t.learning_rate,
        per_device_train_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler,
        weight_decay=t.weight_decay,
        gradient_checkpointing=t.gradient_checkpointing and on_gpu,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t.logging_steps,
        save_steps=t.save_steps,
        save_total_limit=2,
        seed=t.seed,
        optim=t.optim if on_gpu else "adamw_torch",
        fp16=on_gpu and dtype == torch.float16,
        bf16=on_gpu and dtype == torch.bfloat16,
        max_seq_length=spec.model.max_seq_length,
        packing=t.packing,
        report_to=[],
        disable_tqdm=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=data.train,
        processing_class=tokenizer,
        peft_config=lora,
        callbacks=[_callback(sink)],
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    sink.emit(
        Event(
            "log",
            {
                "level": "info",
                "message": f"trainable params: {trainable:,} / {total:,} "
                f"({100 * trainable / max(total, 1):.2f}%)",
            },
        )
    )

    result = trainer.train()
    loss = result.training_loss
    msg = "training done" if math.isnan(loss) else f"training done: mean loss {loss:.4f}"
    sink.emit(Event("log", {"level": "info", "message": msg}))

    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    sink.emit(Event("artifact", {"kind": "adapter", "path": str(adapter_dir)}))

    del trainer, model
    if on_gpu:
        torch.cuda.empty_cache()
    return adapter_dir
