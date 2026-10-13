from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    home: Path
    db_url: str
    api_token: str | None
    cors_origins: list[str] = field(default_factory=list)
    force_simulate: bool = False
    poll_interval: float = 1.0
    web_dist: Path | None = None
    examples_dir: Path | None = None

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"


def _bool(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}


def _path(env: str, default: Path) -> Path | None:
    raw = os.environ.get(env)
    p = Path(raw) if raw else default
    return p if p.exists() else None


def load() -> Settings:
    home = Path(os.environ.get("FTS_HOME", ".fts")).resolve()
    home.mkdir(parents=True, exist_ok=True)
    return Settings(
        home=home,
        db_url=os.environ.get("FTS_DB_URL", f"sqlite:///{home / 'fts.db'}"),
        api_token=os.environ.get("FTS_API_TOKEN") or None,
        cors_origins=[o for o in os.environ.get("FTS_CORS_ORIGINS", "").split(",") if o],
        force_simulate=_bool(os.environ.get("FTS_SIMULATE")),
        poll_interval=float(os.environ.get("FTS_POLL_INTERVAL", "1.0")),
        web_dist=_path("FTS_WEB_DIST", _REPO / "web" / "dist"),
        examples_dir=_path("FTS_EXAMPLES_DIR", _REPO / "examples" / "specs"),
    )


@lru_cache
def get() -> Settings:
    return load()
