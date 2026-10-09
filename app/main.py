"""ProjectPulse API.

Exposes the agent over HTTP. /health proves the deployment pipeline works,
from the walking-skeleton phase. /ask (AGENTS-42) answers a real question by
calling app.agent's compiled LangGraph agent -- the same graph app/agent.py's
own CLI (python -m app.agent) already calls.

/ask and /files need an X-API-Key header (AGENTS-53); /health stays open so
the host can check the service without a key.
"""
import os
from pathlib import Path
from typing import Any
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.agent import agent
from app.agent_loop import loop_result, run_loop
from app.auth import require_api_key
from app.excel_export import EXPORTS_DIR

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


class Source(BaseModel):
    """One document passage an answer drew on: the page title, its link, the text and the score."""

    title: str
    url: str | None = None
    text: str
    score: float


class AskResponse(BaseModel):
    """What POST /ask returns: the answer, the rows behind it, any sources and any file.

    metric_key is included (rather than just the answer) so Swagger and any
    caller can see which path handled the question -- useful while the agent
    only covers some of the catalogue, and free: classify_intent already
    computes it.

    rows is the data the answer was written from, so a screen can show a real
    table instead of parsing a sentence (AGENTS-65). file_url is set only when
    the agent wrote a file; it is a path on this API that downloads it.

    sources holds the document passages behind a narrative or hybrid answer
    (AGENTS-74), kept apart from rows so a screen can show the figures and the
    reasons separately. It is empty for a metric answer.
    """

    question: str
    answer: str
    metric_key: str
    rows: list[dict[str, Any]]
    sources: list[Source] = []
    file_url: str | None = None


def _run_agent(question: str) -> dict[str, Any]:
    """Answer with the fixed agent or the tool-calling loop, as the AGENT_MODE setting says.

    "fixed" (the default) is the agent in app/agent.py. "loop" is the tool-calling loop in
    app/agent_loop.py (ADR-022). Both return the same fields, so /ask answers the same way.
    """
    mode = os.environ.get("AGENT_MODE", "fixed")
    if mode == "fixed":
        return agent.invoke({"question": question})
    if mode == "loop":
        return loop_result(run_loop(question))
    raise HTTPException(status_code=503, detail="AGENT_MODE must be 'fixed' or 'loop'.")


@app.post("/ask", dependencies=[Depends(require_api_key)])
def ask(request: AskRequest) -> AskResponse:
    """Run a question through the agent and return its answer.

    A thin wrapper. All the real logic already lives in app.agent and app.agent_loop --
    this just exposes one of them over HTTP, chosen by AGENT_MODE.
    """
    result = _run_agent(request.question)
    file_path = result.get("file_path")
    return AskResponse(
        question=request.question,
        answer=result["answer"],
        metric_key=result["metric_key"],
        rows=result.get("rows", []),
        sources=result.get("sources", []),
        file_url=f"/files/{Path(file_path).name}" if file_path else None,
    )


@app.get("/files/{filename}", dependencies=[Depends(require_api_key)])
def download_file(filename: str) -> FileResponse:
    """Download a file the agent wrote, such as an Excel export (AGENTS-65).

    Only files directly inside the exports folder can be reached. The name is
    resolved to a real path and checked, so a name like ".." cannot climb out
    of that folder.
    """
    exports_dir = EXPORTS_DIR.resolve()
    path = (exports_dir / filename).resolve()
    if path.parent != exports_dir or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, filename=path.name)
