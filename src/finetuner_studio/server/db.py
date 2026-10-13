"""Persistence: runs and their event streams."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    event,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {RunStatus.succeeded, RunStatus.failed, RunStatus.cancelled}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default=RunStatus.queued.value, index=True)
    spec_yaml: Mapped[str] = mapped_column(Text)
    base_model: Mapped[str] = mapped_column(String(256))
    simulate: Mapped[bool] = mapped_column(Boolean, default=False)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    run_dir: Mapped[str | None] = mapped_column(String(512), nullable=True)
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    progress: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "base_model": self.base_model,
            "simulate": self.simulate,
            "stage": self.stage,
            "progress": self.progress,
            "decision": self.decision,
            "summary": self.summary,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
        }


class RunEvent(Base):
    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    ts: Mapped[float] = mapped_column(Float)
    type: Mapped[str] = mapped_column(String(32))
    data: Mapped[dict[str, Any]] = mapped_column(JSON)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "ts": self.ts, "type": self.type, "data": self.data}


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


class Database:
    def __init__(self, url: str) -> None:
        sqlite = url.startswith("sqlite")
        connect_args = {"check_same_thread": False, "timeout": 30} if sqlite else {}
        self.engine: Engine = create_engine(url, connect_args=connect_args, future=True)
        if sqlite:

            @event.listens_for(self.engine, "connect")
            def _pragmas(conn, _):  # pragma: no cover - trivial
                cur = conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()

        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False, future=True)

    # --- runs ---------------------------------------------------------------

    def create_run(self, name: str, spec_yaml: str, base_model: str, simulate: bool) -> Run:
        with self.Session.begin() as s:
            run = Run(name=name, spec_yaml=spec_yaml, base_model=base_model, simulate=simulate)
            s.add(run)
        return run

    def get_run(self, run_id: str) -> Run | None:
        with self.Session() as s:
            return s.get(Run, run_id)

    def list_runs(self, limit: int = 100, offset: int = 0) -> list[Run]:
        with self.Session() as s:
            q = select(Run).order_by(Run.created_at.desc()).limit(limit).offset(offset)
            return list(s.scalars(q))

    def update_run(self, run_id: str, **fields: Any) -> None:
        with self.Session.begin() as s:
            s.execute(update(Run).where(Run.id == run_id).values(**fields))

    def claim_next(self, pid: int) -> Run | None:
        """Atomically move the oldest queued run to running."""
        with self.Session.begin() as s:
            run = s.scalars(
                select(Run).where(Run.status == RunStatus.queued.value).order_by(Run.created_at).limit(1)
            ).first()
            if run is None:
                return None
            res = s.execute(
                update(Run)
                .where(Run.id == run.id, Run.status == RunStatus.queued.value)
                .values(status=RunStatus.running.value, started_at=utcnow(), worker_pid=pid)
            )
            if res.rowcount != 1:
                return None
            run_id = run.id
        return self.get_run(run_id)

    def request_cancel(self, run_id: str) -> Run | None:
        with self.Session.begin() as s:
            run = s.get(Run, run_id)
            if run is None:
                return None
            if run.status == RunStatus.queued.value:
                run.status = RunStatus.cancelled.value
                run.finished_at = utcnow()
            elif run.status == RunStatus.running.value:
                run.cancel_requested = True
        return self.get_run(run_id)

    def fail_orphans(self, reason: str) -> int:
        """Runs left 'running' by a dead worker are marked failed on startup."""
        with self.Session.begin() as s:
            res = s.execute(
                update(Run)
                .where(Run.status == RunStatus.running.value)
                .values(status=RunStatus.failed.value, error=reason, finished_at=utcnow())
            )
            return res.rowcount or 0

    def delete_run(self, run_id: str) -> bool:
        with self.Session.begin() as s:
            run = s.get(Run, run_id)
            if run is None:
                return False
            s.execute(delete(RunEvent).where(RunEvent.run_id == run_id))
            s.delete(run)
            return True

    # --- events -------------------------------------------------------------

    def add_events(self, run_id: str, events: list[tuple[float, str, dict[str, Any]]]) -> None:
        if not events:
            return
        with self.Session.begin() as s:
            s.add_all(
                RunEvent(run_id=run_id, ts=ts, type=t, data=json.loads(json.dumps(d, default=str)))
                for ts, t, d in events
            )

    def events_after(
        self, run_id: str, after_id: int = 0, limit: int = 1000, types: list[str] | None = None
    ) -> list[RunEvent]:
        with self.Session() as s:
            q = select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.id > after_id)
            if types:
                q = q.where(RunEvent.type.in_(types))
            return list(s.scalars(q.order_by(RunEvent.id).limit(limit)))
