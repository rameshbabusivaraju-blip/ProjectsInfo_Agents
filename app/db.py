"""Database engine and schema creation for the local SQLite store."""

import os
from pathlib import Path

from sqlmodel import SQLModel, create_engine

from app import models  # noqa: F401  — import registers the tables on SQLModel.metadata

DB_PATH = Path(os.environ.get("PROJECTPULSE_DB", "projectpulse.db"))

engine = create_engine(f"sqlite:///{DB_PATH}")


def init_db() -> None:
    """Create any tables that do not already exist."""
    SQLModel.metadata.create_all(engine)