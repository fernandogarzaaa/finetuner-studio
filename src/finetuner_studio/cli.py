"""Command line interface.

    fts validate spec.yaml
    fts run spec.yaml                       # foreground run (Colab, scripts, CI)
    fts serve                               # API + web UI + in-process worker
    fts serve --no-worker  &  fts worker    # split API and GPU worker
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

from . import __version__
from .events import ConsoleSink, StdoutSink
from .spec import RunSpec, json_schema

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Fine-Tuner Studio")


def _load(spec_path: Path) -> RunSpec:
    try:
        return RunSpec.from_file(spec_path)
    except Exception as exc:
        typer.secho(f"Invalid spec: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def validate(spec_path: Path) -> None:
    """Validate a spec without touching the GPU."""
    spec = _load(spec_path)
    typer.secho(
        f"OK: {spec.name} ({spec.model.base_model}), effective batch {spec.effective_batch_size()}",
        fg=typer.colors.GREEN,
    )


@app.command()
def schema() -> None:
    """Print the spec JSON Schema."""
    typer.echo(json.dumps(json_schema(), indent=2))


@app.command()
def hardware() -> None:
    """Show detected GPUs."""
    from . import hardware as hw

    typer.echo(json.dumps(hw.summary(), indent=2))


@app.command()
def run(
    spec_path: Path,
    run_dir: Path = typer.Option(None, help="Defaults to <output.dir>/<name>"),
    simulate: bool = typer.Option(False, help="GPU-free dry run with synthetic numbers"),
) -> None:
    """Fine-tune, evaluate against the base, and print the verdict.

    Exit code: 0 = SHIP or INCONCLUSIVE, 3 = DONT_SHIP, 1 = run failed, 2 = invalid spec.
    """
    from .pipeline import run as run_pipeline

    spec = _load(spec_path)
    target = run_dir or Path(spec.output.dir) / spec.name
    try:
        report = run_pipeline(spec, target, ConsoleSink(), simulate=simulate)
    except Exception as exc:
        raise typer.Exit(1) from exc
    typer.echo(f"\nReport: {target / 'report.md'}")
    raise typer.Exit(3 if report["verdict"]["decision"] == "DONT_SHIP" else 0)


@app.command(hidden=True)
def execute(
    spec_path: Path,
    run_dir: Path = typer.Option(...),
    simulate: bool = typer.Option(False),
) -> None:
    """Internal: executed by the worker; emits machine-readable events on stdout."""
    from .pipeline import run as run_pipeline

    spec = _load(spec_path)
    try:
        run_pipeline(spec, run_dir, StdoutSink(), simulate=simulate)
    except Exception:
        sys.exit(1)


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    worker: bool = typer.Option(True, help="Run the GPU worker inside the API process"),
) -> None:
    """Start the API and web UI."""
    import uvicorn

    from .server.app import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(create_app(start_worker=worker), host=host, port=port)


@app.command(name="worker")
def worker_cmd() -> None:
    """Run the GPU worker on its own (pair with `serve --no-worker`)."""
    from .server import settings as settings_mod
    from .server.db import Database
    from .server.worker import Worker

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = settings_mod.get()
    w = Worker(Database(cfg.db_url), cfg)
    try:
        w.serve_forever()
    except KeyboardInterrupt:
        w.stop()


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
