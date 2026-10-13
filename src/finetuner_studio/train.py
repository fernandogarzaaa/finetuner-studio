"""QLoRA / LoRA supervised fine-tuning with TRL."""

from __future__ import annotations

import math
from pathlib import Path

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
            sink.emit(
                Event(
                    "progress",
                    {"stage": "train", "current": state.global_step, "total": state.max_steps},
                )
            )

    return EventCallback()


def train(spec: RunSpec, data: PreparedData, run_dir: Path, sink: EventSink) -> Path:
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer = load_tokenizer(spec.model)
    model, dtype = load_model(spec.model, for_training=True)

    import torch

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
    checkpoints = run_dir / "checkpoints"
    config = SFTConfig(
        output_dir=str(checkpoints),
        num_train_epochs=t.epochs,
        max_steps=t.max_steps,
        learning_rate=t.learning_rate,
        per_device_train_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler,
        weight_decay=t.weight_decay,
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t.logging_steps,
        save_steps=t.save_steps,
        save_total_limit=2,
        seed=t.seed,
        optim=t.optim if torch.cuda.is_available() else "adamw_torch",
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
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

    trainable, total = 0, 0
    for p in trainer.model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
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
    final_loss = result.training_loss
    sink.emit(
        Event(
            "log",
            {
                "level": "info",
                "message": f"training done: mean loss {final_loss:.4f}"
                if not math.isnan(final_loss)
                else "training done",
            },
        )
    )

    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    sink.emit(Event("artifact", {"kind": "adapter", "path": str(adapter_dir)}))

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return adapter_dir
