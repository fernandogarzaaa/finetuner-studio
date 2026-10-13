"""HTTP API (FastAPI) and static hosting for the web UI."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from .. import __version__, hardware
from ..spec import RunSpec, json_schema
from . import settings as settings_mod
from .db import Database, RunStatus
from .worker import Worker


class SpecBody(BaseModel):
    spec_yaml: str


class CreateRunBody(SpecBody):
    simulate: bool = False


def _validation_errors(exc: Exception) -> list[dict[str, Any]]:
    if isinstance(exc, ValidationError):
        return [
            {"loc": ".".join(str(p) for p in e["loc"]), "msg": e["msg"]}
            for e in exc.errors(include_url=False)
        ]
    return [{"loc": "", "msg": str(exc)}]


def _parse_spec(text: str) -> RunSpec:
    try:
        return RunSpec.from_yaml(text)
    except Exception as exc:  # YAML errors and validation errors
        raise HTTPException(status_code=422, detail={"errors": _validation_errors(exc)}) from exc


def create_app(settings: settings_mod.Settings | None = None, start_worker: bool = False) -> FastAPI:
    cfg = settings or settings_mod.get()
    db = Database(cfg.db_url)
    worker = Worker(db, cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_worker:
            threading.Thread(target=worker.serve_forever, name="fts-worker", daemon=True).start()
        yield
        worker.stop()

    app = FastAPI(title="Fine-Tuner Studio", version=__version__, lifespan=lifespan)
    app.state.db = db
    app.state.settings = cfg
    app.state.worker = worker

    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
        )

    def auth(request: Request, token: str | None = Query(None)) -> None:
        """Bearer token, or ?token= for EventSource clients that can't set headers."""
        if not cfg.api_token:
            return
        header = request.headers.get("authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else token
        if supplied != cfg.api_token:
            raise HTTPException(status_code=401, detail="invalid or missing API token")

    def get_run_or_404(run_id: str):
        run = db.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run

    api = [Depends(auth)]

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "auth": bool(cfg.api_token)}

    @app.get("/api/hardware", dependencies=api)
    def hw() -> dict[str, Any]:
        return {**hardware.summary(), "force_simulate": cfg.force_simulate}

    @app.get("/api/spec/schema", dependencies=api)
    def schema() -> dict[str, Any]:
        return json_schema()

    @app.get("/api/spec/examples", dependencies=api)
    def examples() -> list[dict[str, str]]:
        if not cfg.examples_dir:
            return []
        return [
            {"name": p.stem, "spec_yaml": p.read_text(encoding="utf-8")}
            for p in sorted(cfg.examples_dir.glob("*.yaml"))
        ]

    @app.post("/api/spec/validate", dependencies=api)
    def validate(body: SpecBody) -> JSONResponse:
        try:
            spec = RunSpec.from_yaml(body.spec_yaml)
        except Exception as exc:
            return JSONResponse({"valid": False, "errors": _validation_errors(exc)})
        return JSONResponse(
            {
                "valid": True,
                "errors": [],
                "spec": spec.model_dump(mode="json"),
                "effective_batch_size": spec.effective_batch_size(),
            }
        )

    @app.post("/api/runs", status_code=201, dependencies=api)
    def create_run(body: CreateRunBody) -> dict[str, Any]:
        spec = _parse_spec(body.spec_yaml)
        run = db.create_run(spec.name, spec.to_yaml(), spec.model.base_model, body.simulate)
        return run.to_dict()

    @app.get("/api/runs", dependencies=api)
    def list_runs(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        return [r.to_dict() for r in db.list_runs(limit, offset)]

    @app.get("/api/runs/{run_id}", dependencies=api)
    def get_run(run_id: str) -> dict[str, Any]:
        run = get_run_or_404(run_id)
        return {**run.to_dict(), "spec_yaml": run.spec_yaml}

    @app.post("/api/runs/{run_id}/cancel", dependencies=api)
    def cancel(run_id: str) -> dict[str, Any]:
        run = get_run_or_404(run_id)
        if RunStatus(run.status).terminal:
            raise HTTPException(status_code=409, detail=f"run is already {run.status}")
        updated = db.request_cancel(run_id)
        assert updated is not None
        return updated.to_dict()

    @app.delete("/api/runs/{run_id}", status_code=204, dependencies=api)
    def delete(run_id: str) -> None:
        run = get_run_or_404(run_id)
        if not RunStatus(run.status).terminal:
            raise HTTPException(status_code=409, detail="cancel the run before deleting it")
        if run.run_dir:
            shutil.rmtree(run.run_dir, ignore_errors=True)
        db.delete_run(run_id)

    @app.get("/api/runs/{run_id}/events", dependencies=api)
    def events(
        run_id: str,
        after: int = Query(0, ge=0),
        limit: int = Query(1000, ge=1, le=5000),
        types: str | None = None,
    ):
        get_run_or_404(run_id)
        type_list = [t for t in types.split(",") if t] if types else None
        return [e.to_dict() for e in db.events_after(run_id, after, limit, type_list)]

    @app.get("/api/runs/{run_id}/stream", dependencies=api)
    async def stream(run_id: str, request: Request, after: int = Query(0, ge=0)):
        get_run_or_404(run_id)
        last_id = int(request.headers.get("last-event-id") or after)

        async def gen():
            nonlocal last_id
            idle_after_terminal = 0
            while True:
                if await request.is_disconnected():
                    return
                batch = await asyncio.to_thread(db.events_after, run_id, last_id, 500)
                for e in batch:
                    last_id = e.id
                    yield f"id: {e.id}\nevent: run_event\ndata: {json.dumps(e.to_dict())}\n\n"
                run = await asyncio.to_thread(db.get_run, run_id)
                if run is None:
                    return
                if RunStatus(run.status).terminal and not batch:
                    idle_after_terminal += 1
                    if idle_after_terminal >= 2:
                        yield f"event: run_status\ndata: {json.dumps(run.to_dict())}\n\n"
                        return
                if not batch:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(0.5)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def _artifact(run_id: str, name: str) -> Path:
        run = get_run_or_404(run_id)
        if not run.run_dir:
            raise HTTPException(status_code=404, detail="run has no artifacts yet")
        path = Path(run.run_dir) / name
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"{name} not found")
        return path

    @app.get("/api/runs/{run_id}/report", dependencies=api)
    def report(run_id: str):
        return json.loads(_artifact(run_id, "report.json").read_text(encoding="utf-8"))

    @app.get("/api/runs/{run_id}/report.md", dependencies=api)
    def report_md(run_id: str):
        text = _artifact(run_id, "report.md").read_text(encoding="utf-8")
        return PlainTextResponse(text, media_type="text/markdown")

    @app.get("/api/runs/{run_id}/samples", dependencies=api)
    def samples(run_id: str, limit: int = Query(50, ge=1, le=1000)):
        path = _artifact(run_id, "eval_samples.jsonl")
        out = []
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                out.append(json.loads(line))
        return out

    @app.get("/api/runs/{run_id}/adapter.zip", dependencies=api)
    def adapter_zip(run_id: str):
        adapter = _artifact(run_id, "adapter")
        archive = adapter.parent / "adapter.zip"
        if not archive.exists():
            shutil.make_archive(str(archive.with_suffix("")), "zip", adapter)
        return FileResponse(archive, filename=f"{run_id}-adapter.zip")

    if cfg.web_dist and (cfg.web_dist / "index.html").exists():
        dist = cfg.web_dist
        if (dist / "assets").exists():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(status_code=404)
            candidate = (dist / path).resolve()
            if path and candidate.is_file() and dist.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app
