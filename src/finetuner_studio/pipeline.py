"""End-to-end run: load data -> train -> evaluate base vs tuned -> verdict -> report."""

from __future__ import annotations

import random
import time
import traceback
from pathlib import Path
from typing import Any

from . import report as report_mod
from .events import Event, EventSink
from .spec import RunSpec
from .verdict import decide


class _Stage:
    def __init__(self, sink: EventSink, name: str) -> None:
        self.sink, self.name = sink, name

    def __enter__(self):
        self.sink.emit(Event("stage", {"stage": self.name, "status": "start"}))
        return self

    def __exit__(self, exc_type, exc, tb):
        status = "end" if exc is None else "failed"
        self.sink.emit(Event("stage", {"stage": self.name, "status": status}))
        return False


def run(spec: RunSpec, run_dir: Path, sink: EventSink, simulate: bool = False) -> dict[str, Any]:
    """Execute a run. Returns the report dict. Raises on failure after emitting an error event."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.yaml").write_text(spec.to_yaml(), encoding="utf-8")
    try:
        if simulate:
            return _simulate(spec, run_dir, sink)
        return _real(spec, run_dir, sink)
    except Exception as exc:
        sink.emit(Event("error", {"message": str(exc), "traceback": traceback.format_exc()}))
        raise


def _real(spec: RunSpec, run_dir: Path, sink: EventSink) -> dict[str, Any]:
    from . import data as data_mod
    from .evaluate import evaluate
    from .train import train

    with _Stage(sink, "load_data"):
        prepared = data_mod.prepare(spec.dataset)
        sink.emit(
            Event(
                "log",
                {
                    "level": "info",
                    "message": f"train rows: {len(prepared.train)}, eval rows: {len(prepared.eval)}, "
                    f"skipped: {prepared.skipped_rows}",
                },
            )
        )

    with _Stage(sink, "train"):
        adapter_dir = train(spec, prepared, run_dir, sink)

    with _Stage(sink, "evaluate"):
        ev = evaluate(spec, prepared.eval, adapter_dir, run_dir, sink)

    return _finish(spec, run_dir, sink, ev.base, ev.tuned, ev.samples, {})


def _finish(spec, run_dir, sink, base, tuned, samples, extra) -> dict[str, Any]:
    with _Stage(sink, "report"):
        verdict = decide(base, tuned, spec.ship_rule, samples)
        sink.emit(Event("verdict", verdict.to_dict()))
        rep = report_mod.build(spec, verdict, base, tuned, extra)
        j, m = report_mod.write(run_dir, rep)
        sink.emit(Event("artifact", {"kind": "report_json", "path": str(j)}))
        sink.emit(Event("artifact", {"kind": "report_md", "path": str(m)}))
    return rep


def _simulate(spec: RunSpec, run_dir: Path, sink: EventSink) -> dict[str, Any]:
    """A GPU-free dry run that exercises the full event flow. Numbers are synthetic."""
    rng = random.Random(spec.training.seed)
    delay = 0.05

    with _Stage(sink, "load_data"):
        sink.emit(Event("log", {"level": "warning", "message": "SIMULATED run: no model is loaded"}))
        time.sleep(delay)

    steps = spec.training.max_steps if spec.training.max_steps > 0 else 30
    with _Stage(sink, "train"):
        loss = 2.2
        for step in range(1, steps + 1):
            loss = max(0.3, loss * (0.97 + rng.random() * 0.02))
            if step % spec.training.logging_steps == 0 or step == steps:
                sink.emit(
                    Event(
                        "train_metrics",
                        {"step": step, "loss": round(loss, 4),
                         "learning_rate": spec.training.learning_rate, "epoch": step / steps},
                    )
                )
            sink.emit(Event("progress", {"stage": "train", "current": step, "total": steps}))
            time.sleep(delay)

    names = [m.value for m in spec.evaluation.metrics]
    base_vals = {"perplexity": 9.0, "rouge_l": 0.30, "token_f1": 0.35, "exact_match": 0.05}
    lift = {"perplexity": 0.82, "rouge_l": 1.18, "token_f1": 1.15, "exact_match": 1.6}
    base = {n: base_vals[n] for n in names}
    tuned = {n: round(base_vals[n] * lift[n], 4) for n in names}
    samples = min(spec.evaluation.max_samples, 100)
    with _Stage(sink, "evaluate"):
        sink.emit(Event("eval_result", {"model": "base", "metrics": base}))
        time.sleep(delay)
        sink.emit(Event("eval_result", {"model": "tuned", "metrics": tuned}))

    return _finish(spec, run_dir, sink, base, tuned, samples, {"simulated": True})
