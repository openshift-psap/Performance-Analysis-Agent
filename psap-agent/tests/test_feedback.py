"""Tests for the feedback route."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langfuse import Langfuse

from psap_agent.src.routes.feedback import router
from psap_agent.src.schema import FeedbackRequest

RUN_ID = "run_123"
TRACE_ID = Langfuse.create_trace_id(seed=RUN_ID)


def _test_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestFeedbackRoute:
    """Test cases for feedback endpoint."""

    @patch("psap_agent.src.routes.feedback.get_client")
    def test_feedback_uses_explicit_langfuse_trace_id(self, mock_get_client):
        client = _test_client()
        langfuse_client = Mock()
        mock_get_client.return_value = langfuse_client

        response = client.post(
            "/v1/feedback",
            json={
                "run_id": RUN_ID,
                "langfuse_trace_id": TRACE_ID,
                "key": "response_quality",
                "score": 1.0,
                "kwargs": {"comment": "Great response"},
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"
        langfuse_client.create_score.assert_called_once_with(
            trace_id=TRACE_ID,
            name="response_quality",
            value=1.0,
            comment="comment: Great response",
        )
        langfuse_client.api.trace.list.assert_not_called()

    @patch("psap_agent.src.routes.feedback.get_client")
    def test_legacy_run_id_is_resolved_to_real_trace_id(self, mock_get_client):
        client = _test_client()
        langfuse_client = Mock()
        langfuse_client.api.trace.list.return_value.data = [
            SimpleNamespace(id=TRACE_ID, metadata={"run_id": RUN_ID})
        ]
        mock_get_client.return_value = langfuse_client

        response = client.post(
            "/v1/feedback",
            json={"run_id": RUN_ID, "key": "response_quality", "score": 1.0},
        )

        assert response.status_code == 200
        langfuse_client.create_score.assert_called_once_with(
            trace_id=TRACE_ID,
            name="response_quality",
            value=1.0,
            comment=None,
        )

    @patch("psap_agent.src.routes.feedback._resolve_trace_id", return_value=None)
    @patch("psap_agent.src.routes.feedback.get_client")
    def test_unresolved_run_id_fails_without_orphan_score(
        self, mock_get_client, _mock_resolve
    ):
        client = _test_client()
        langfuse_client = Mock()
        mock_get_client.return_value = langfuse_client

        response = client.post(
            "/v1/feedback",
            json={"run_id": RUN_ID, "key": "response_quality", "score": 0.0},
        )

        assert response.status_code == 503
        langfuse_client.create_score.assert_not_called()

    @patch("psap_agent.src.routes.feedback.get_client")
    def test_invalid_explicit_trace_id_is_rejected(self, mock_get_client):
        client = _test_client()
        langfuse_client = Mock()
        mock_get_client.return_value = langfuse_client

        response = client.post(
            "/v1/feedback",
            json={
                "run_id": "run_123",
                "langfuse_trace_id": "run_123",
                "key": "response_quality",
                "score": 0.0,
            },
        )

        assert response.status_code == 422
        langfuse_client.create_score.assert_not_called()

    @patch("psap_agent.src.routes.feedback.get_client")
    def test_trace_id_must_match_run_id(self, mock_get_client):
        client = _test_client()
        langfuse_client = Mock()
        mock_get_client.return_value = langfuse_client
        mismatched_trace_id = Langfuse.create_trace_id(seed="another-run")

        response = client.post(
            "/v1/feedback",
            json={
                "run_id": RUN_ID,
                "langfuse_trace_id": mismatched_trace_id,
                "key": "response_quality",
                "score": 0.0,
            },
        )

        assert response.status_code == 422
        langfuse_client.create_score.assert_not_called()

    def test_feedback_request_model_accepts_trace_id(self):
        feedback = FeedbackRequest(
            run_id="run_123",
            langfuse_trace_id=TRACE_ID,
            key="response_quality",
            score=1.0,
        )
        assert feedback.run_id == "run_123"
        assert feedback.langfuse_trace_id == TRACE_ID
