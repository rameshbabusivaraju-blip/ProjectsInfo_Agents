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

class Commit(SQLModel, table=True):
    """One commit on the default branch."""

    __tablename__ = "commits"

    sha: str = Field(primary_key=True)
    message: str
    author_login: str | None = None
    authored_at: datetime
    # Ticket id parsed out of the commit message, e.g. AGENTS-24. None means
    # the commit broke the ADR-016 convention — that is question C5.
    ticket_key: str | None = None


class PullRequest(SQLModel, table=True):
    """One pull request, open, closed or merged."""

    __tablename__ = "pull_requests"

    number: int = Field(primary_key=True)
    title: str
    author_login: str | None = None
    state: str
    created_at: datetime
    merged_at: datetime | None = None
    closed_at: datetime | None = None
    head_branch: str
    ticket_key: str | None = None


class PrReview(SQLModel, table=True):
    """One review left on a pull request."""

    __tablename__ = "pr_reviews"

    id: int = Field(primary_key=True)
    pr_number: int = Field(foreign_key="pull_requests.number")
    reviewer_login: str | None = None
    # APPROVED, CHANGES_REQUESTED or COMMENTED
    state: str
    submitted_at: datetime | None = None
    body: str | None = None

class ConfluencePage(SQLModel, table=True):
    """One Confluence page — current content only, not a history."""

    __tablename__ = "confluence_pages"

    id: str = Field(primary_key=True)
    title: str
    space_key: str
    url: str
    version: int
    version_author_id: str | None = Field(default=None, foreign_key="people.account_id")
    updated: datetime
    body_text: str | None = None