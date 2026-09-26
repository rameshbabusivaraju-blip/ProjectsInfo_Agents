"""ProjectPulse API.

Exposes the agent over HTTP. /health proves the deployment pipeline works,
from the walking-skeleton phase. /ask (AGENTS-42) answers a real question by
calling app.agent's compiled LangGraph agent -- the same graph app/agent.py's
own CLI (python -m app.agent) already calls.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from app.agent import agent

app = FastAPI(
    title="ProjectPulse API",
    description="Answers questions about the health of the ProjectPulse project.",
    version="0.1.0",
)

@app.get("/health")
def health() -> dict[str, str]:
    """Return service status and version so callers can confirm what is deployed."""
    return {"status": "ok", "version": "0.1.0"}


class AskRequest(BaseModel):
    """Body for POST /ask: the question, in plain English."""

    question: str


class AskResponse(BaseModel):
    """What POST /ask returns: the answer, plus how the question was classified.

    metric_key is included (rather than just the answer) so Swagger and any
    caller can see which path handled the question -- useful while the agent
    only covers some of the catalogue, and free: classify_intent already
    computes it.
    """

    question: str
    answer: str
    metric_key: str


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    """Run a question through the agent and return its answer.

    A thin wrapper. All the real logic already lives in app.agent -- this
    just exposes agent.invoke() over HTTP.
    """
    result = agent.invoke({"question": request.question})
    return AskResponse(
        question=request.question,
        answer=result["answer"],
        metric_key=result["metric_key"],
    )