"""ProjectPulse API.

Entry point for the FastAPI application. For now it exposes a single health
endpoint, used to prove the deployment pipeline works before any real logic
exists (see the walking skeleton rationale in the plan of action).
"""

from fastapi import FastAPI

app = FastAPI(
    title="ProjectPulse API",
    description="Answers questions about the health of the ProjectPulse project.",
    version="0.1.0",
)

@app.get("/health")
def health() -> dict[str, str]:
    """Return a fixed response so a caller can confirm the service is running."""
    return {"status": "ok"}