"""GPU job worker.

Runs one job at a time (one GPU, one tenant). Each run executes in a child
process so that CUDA memory is fully released afterwards, a crash cannot take
the worker down, and cancellation is a clean terminate.
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..events import STAGES, parse_line
from .db import Database, RunStatus, utcnow
from .settings import Settings

log = logging.getLogger("fts.worker")

_FLUSH_EVERY = 0.5  # seconds
_KILL_GRACE = 15  # seconds between SIGTERM and SIGKILL


class Worker:
    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def serve_forever(self) -> None:
        orphans = self.db.fail_orphans("worker restarted while the run was in progress")
        if orphans:
            log.warning("marked %d orphaned run(s) as failed", orphans)
        log.info("worker ready (pid %s)", os.getpid())
        while not self._stop.is_set():
            run = self.db.claim_next(os.getpid())
            if run is None:
                self._stop.wait(self.settings.poll_interval)
                continue
            try:
                self.execute(run.id)
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("run %s crashed the worker loop", run.id)
                self.db.update_run(run.id, status=RunStatus.failed.value, error=str(exc),
                                   finished_at=utcnow())

    # ------------------------------------------------------------------------

    def execute(self, run_id: str) -> None:
        run = self.db.get_run(run_id)
        assert run is not None
        run_dir = self.settings.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        spec_path = run_dir / "spec.yaml"
        spec_path.write_text(run.spec_yaml, encoding="utf-8")
        self.db.update_run(run_id, run_dir=str(run_dir))

        cmd = [sys.executable, "-u", "-m", "finetuner_studio.cli", "execute",
               str(spec_path), "--run-dir", str(run_dir)]
        if run.simulate or self.settings.force_simulate:
            cmd.append("--simulate")

        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=env, start_new_session=True,
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def _reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=_reader, daemon=True).start()

        state = {"decision": None, "summary": None, "error": None, "stage": None, "progress": None}
        buf: list = []
        last_flush = time.monotonic()
        eof = False
        cancelled = False
        term_sent_at: float | None = None

        while not eof:
            try:
                item = lines.get(timeout=0.25)
            except queue.Empty:
                item = ""
            if item is None:
                eof = True
            elif item:
                ev = parse_line(item)
                if ev.type == "log" and not ev.data.get("message", "").strip():
                    pass
                else:
                    buf.append((ev.ts, ev.type, ev.data))
                    self._track(state, ev.type, ev.data)

            now = time.monotonic()
            if buf and (now - last_flush >= _FLUSH_EVERY or eof):
                self.db.add_events(run_id, buf)
                buf = []
                last_flush = now
                self.db.update_run(run_id, stage=state["stage"], progress=state["progress"])

            if not cancelled:
                current = self.db.get_run(run_id)
                if current is not None and current.cancel_requested:
                    cancelled = True
                    term_sent_at = now
                    self._signal(proc, signal.SIGTERM)
            elif term_sent_at and now - term_sent_at > _KILL_GRACE and proc.poll() is None:
                self._signal(proc, signal.SIGKILL)
                term_sent_at = None

        code = proc.wait()
        if buf:
            self.db.add_events(run_id, buf)

        if cancelled:
            status, error = RunStatus.cancelled, None
        elif code == 0:
            status, error = RunStatus.succeeded, None
        else:
            status, error = RunStatus.failed, state["error"] or f"run exited with code {code}"
        self.db.update_run(
            run_id,
            status=status.value,
            error=error,
            decision=state["decision"],
            summary=state["summary"],
            stage=state["stage"],
            progress=state["progress"],
            finished_at=utcnow(),
        )
        log.info("run %s finished: %s", run_id, status.value)

    @staticmethod
    def _signal(proc: subprocess.Popen, sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    @staticmethod
    def _track(state: dict, etype: str, data: dict) -> None:
        if etype == "stage" and data.get("status") == "start":
            state["stage"] = data.get("stage")
            if state["stage"] in STAGES:
                state["progress"] = STAGES.index(state["stage"]) / len(STAGES)
        elif etype == "progress" and data.get("stage") == "train" and data.get("total"):
            frac = min(1.0, max(0.0, data["current"] / data["total"]))
            state["progress"] = (1 + frac) / len(STAGES)
        elif etype == "verdict":
            state["decision"] = data.get("decision")
            state["summary"] = data.get("summary")
            state["progress"] = 1.0
        elif etype == "error":
            state["error"] = data.get("message")


def run_dir_for(settings: Settings, run_id: str) -> Path:
    return settings.runs_dir / run_id
