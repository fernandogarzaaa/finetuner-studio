from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    home: Path
    db_url: str
    api_token: str | None
    cors_origins: list[str] = field(default_factory=list)
    force_simulate: bool = False
    poll_interval: float = 1.0
    web_dist: Path | None = None

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"


def _bool(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}


def load() -> Settings:
    home = Path(os.environ.get("FTS_HOME", ".fts")).resolve()
    home.mkdir(parents=True, exist_ok=True)
    web = os.environ.get("FTS_WEB_DIST")
    default_web = Path(__file__).resolve().parents[3] / "web" / "dist"
    return Settings(
        home=home,
        db_url=os.environ.get("FTS_DB_URL", f"sqlite:///{home / 'fts.db'}"),
        api_token=os.environ.get("FTS_API_TOKEN") or None,
        cors_origins=[o for o in os.environ.get("FTS_CORS_ORIGINS", "").split(",") if o],
        force_simulate=_bool(os.environ.get("FTS_SIMULATE")),
        poll_interval=float(os.environ.get("FTS_POLL_INTERVAL", "1.0")),
        web_dist=Path(web) if web else (default_web if default_web.exists() else None),
    )


@lru_cache
def get() -> Settings:
    return load()
