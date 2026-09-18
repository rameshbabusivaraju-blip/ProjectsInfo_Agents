"""SQLModel schemas for the Jira slice of the ProjectPulse store.

Only the tables needed for catalogue sections A, B and H. GitHub, Actions,
Confluence and incidents come in later tickets.
"""

from datetime import datetime

from sqlmodel import Field, SQLModel


class Person(SQLModel, table=True):
    __tablename__ = "people"

    account_id: str = Field(primary_key=True)
    display_name: str
    github_username: str | None = None


class Sprint(SQLModel, table=True):
    __tablename__ = "sprints"

    id: int = Field(primary_key=True)
    name: str
    state: str
    goal: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    complete_date: datetime | None = None


class Ticket(SQLModel, table=True):
    __tablename__ = "tickets"

    key: str = Field(primary_key=True)
    summary: str
    issue_type: str
    hierarchy_level: int
    status: str
    status_category: str
    story_points: float | None = None
    parent_key: str | None = Field(default=None, foreign_key="tickets.key")
    assignee_id: str | None = Field(default=None, foreign_key="people.account_id")
    created: datetime
    updated: datetime
    resolved: datetime | None = None
    original_estimate_seconds: int | None = None
    time_spent_seconds: int | None = None
    description: str | None = None


class TicketSprint(SQLModel, table=True):
    """Every sprint a ticket has ever been in, in order."""

    __tablename__ = "ticket_sprints"

    ticket_key: str = Field(primary_key=True, foreign_key="tickets.key")
    sprint_id: int = Field(primary_key=True, foreign_key="sprints.id")
    position: int


class WorkLog(SQLModel, table=True):
    __tablename__ = "work_logs"

    id: str = Field(primary_key=True)
    ticket_key: str = Field(foreign_key="tickets.key")
    author_id: str = Field(foreign_key="people.account_id")
    started: datetime
    seconds: int
    comment: str | None = None