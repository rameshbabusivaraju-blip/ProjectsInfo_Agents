"""Tests for the health endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    """GET /health responds 200 with the expected body."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}