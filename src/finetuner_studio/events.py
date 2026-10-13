"""Structured run events.

The pipeline never talks to the database or the web layer directly. It emits
events to a sink. When a run is executed by the worker, the child process uses
`StdoutSink`, which writes one tagged JSON line per event; the worker parses
those lines (and treats every untagged line as a plain log line).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

EVENT_PREFIX = "@@FTS_EVENT "

EventType = Literal[
    "stage",  # {"stage": "load_data" | "train" | "evaluate" | "report", "status": "start"|"end"|"failed"}
    "log",  # {"level": "info"|"warning"|"error", "message": str}
    "train_metrics",  # {"step": int, "loss": float, "learning_rate": float, "epoch": float}
    "progress",  # {"stage": str, "current": int, "total": int}
    "eval_result",  # {"model": "base"|"tuned", "metrics": {...}}
    "verdict",  # verdict dict
    "artifact",  # {"kind": str, "path": str}
    "error",  # {"message": str, "traceback": str}
]

STAGES = ("load_data", "train", "evaluate", "report")


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, text: str) -> Event:
        raw = json.loads(text)
        return cls(type=raw["type"], data=raw.get("data", {}), ts=raw.get("ts", time.time()))


class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...


class StdoutSink:
    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdout

    def emit(self, event: Event) -> None:
        self.stream.write(EVENT_PREFIX + event.to_json() + "\n")
        self.stream.flush()


class ListSink:
    """In-memory sink, used by tests and the Python API."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


class ConsoleSink:
    """Human-readable output for interactive CLI use."""

    def emit(self, event: Event) -> None:
        d = event.data
        if event.type == "stage":
            print(f"==> {d.get('stage')} {d.get('status')}", flush=True)
        elif event.type == "log":
            print(f"[{d.get('level', 'info')}] {d.get('message')}", flush=True)
        elif event.type == "train_metrics":
            print(f"step {d.get('step')}: loss={d.get('loss')} lr={d.get('learning_rate')}", flush=True)
        elif event.type == "eval_result":
            print(f"eval[{d.get('model')}]: {d.get('metrics')}", flush=True)
        elif event.type == "verdict":
            print(f"VERDICT: {d.get('decision')} - {d.get('summary')}", flush=True)
        elif event.type == "error":
            print(f"ERROR: {d.get('message')}", file=sys.stderr, flush=True)


def parse_line(line: str) -> Event:
    """Turn one line of child-process output into an Event."""
    line = line.rstrip("\n")
    if line.startswith(EVENT_PREFIX):
        try:
            return Event.from_json(line[len(EVENT_PREFIX) :])
        except (ValueError, KeyError):
            pass
    return Event(type="log", data={"level": "info", "message": line})
