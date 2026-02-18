"""Tests for the health route."""

from fastapi.testclient import TestClient

from psap_agent.src.routes.health import router


class TestHealthRoute:
    """Test cases for health endpoint."""

    def test_health_endpoint(self):
        """Test health endpoint returns correct response."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Template Agent"

    def test_health_endpoint_content_type(self):
        """Test health endpoint returns correct content type."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"
