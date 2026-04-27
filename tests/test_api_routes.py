from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_ask_service
from src.api.schemas import AskRequest
from src.llm.postprocess import PostprocessedResponse
from src.llm.types import TokenUsage
from src.retrieval.types import ContextSource


class FakeAskService:
    def __init__(self) -> None:
        self.requests: list[AskRequest] = []

    def answer(self, request: AskRequest) -> PostprocessedResponse:
        self.requests.append(request)
        return PostprocessedResponse(
            answer="Grounded answer.",
            sources=[
                ContextSource(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    rank=1,
                    score=0.9,
                    metadata={"section": "intro"},
                )
            ],
            model="fake-model",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=4, output_tokens=3, total_tokens=7),
            metadata=request.metadata,
        )


class FailingAskService:
    def answer(self, request: AskRequest) -> PostprocessedResponse:
        raise ValueError("question cannot be empty after normalization.")


class UnexpectedFailingAskService:
    def answer(self, request: AskRequest) -> PostprocessedResponse:
        raise OSError("backend socket is unavailable.")


class ApiRoutesTests(unittest.TestCase):
    def test_health_returns_ok(self) -> None:
        client = TestClient(create_app())

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_ask_returns_service_response(self) -> None:
        app = create_app()
        service = FakeAskService()
        app.dependency_overrides[get_ask_service] = lambda: service
        client = TestClient(app)

        response = client.post(
            "/ask",
            json={
                "question": "What is FastAPI?",
                "generation": {"temperature": 0.1, "max_tokens": 32},
                "metadata": {"trace_id": "abc"},
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "answer": "Grounded answer.",
            "sources": [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "rank": 1,
                    "score": 0.9,
                    "metadata": {"section": "intro"},
                }
            ],
            "warnings": [],
            "model": "fake-model",
            "finish_reason": "stop",
            "usage": {
                "input_tokens": 4,
                "output_tokens": 3,
                "total_tokens": 7,
            },
            "metadata": {"trace_id": "abc"},
        })
        self.assertEqual(service.requests[0].question, "What is FastAPI?")

    def test_ask_returns_503_without_configured_service(self) -> None:
        client = TestClient(create_app())

        with patch(
            "src.api.dependencies.build_ask_service",
            side_effect=RuntimeError("RAG answer service is not configured."),
        ):
            response = client.post("/ask", json={"question": "What is FastAPI?"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {
            "detail": (
                "RAG answer service is unavailable: "
                "RAG answer service is not configured."
            ),
        })

    def test_ask_maps_value_error_to_400(self) -> None:
        app = create_app()
        app.dependency_overrides[get_ask_service] = lambda: FailingAskService()
        client = TestClient(app)

        response = client.post("/ask", json={"question": "What is FastAPI?"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {
            "detail": "question cannot be empty after normalization.",
        })

    def test_ask_maps_unexpected_runtime_error_to_503(self) -> None:
        app = create_app()
        app.dependency_overrides[get_ask_service] = lambda: UnexpectedFailingAskService()
        client = TestClient(app)

        response = client.post("/ask", json={"question": "What is FastAPI?"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {
            "detail": "backend socket is unavailable.",
        })


if __name__ == "__main__":
    unittest.main()
