"""The schema creates cleanly and contains exactly the tables we expect."""

from sqlalchemy import inspect
from sqlmodel import SQLModel, create_engine

from app import models  # noqa: F401


def test_schema_creates_expected_tables() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    tables = set(inspect(engine).get_table_names())

    assert tables == {
        "people",
        "sprints",
        "tickets",
        "ticket_sprints",
        "work_logs",
        "commits",
        "pull_requests",
        "pr_reviews",
         "confluence_pages",
    }