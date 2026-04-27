from __future__ import annotations

import unittest

import httpx

from src.ui.client import RAGApiClient


class UiClientTests(unittest.TestCase):
    def test_health_calls_api_health_endpoint(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.url.path, "/health")
            return httpx.Response(200, json={"status": "ok"})

        client = RAGApiClient(
            base_url="http://testserver",
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(client.health(), {"status": "ok"})

    def test_ask_posts_generation_payload(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append({
                "method": request.method,
                "path": request.url.path,
                "json": request.read(),
            })
            return httpx.Response(
                200,
                json={
                    "answer": "ok",
                    "sources": [],
                    "warnings": [],
                    "metadata": {},
                },
            )

        client = RAGApiClient(
            base_url="http://testserver",
            transport=httpx.MockTransport(handler),
        )

        response = client.ask(
            question="What is FastAPI?",
            temperature=0.1,
            max_tokens=64,
            top_p=0.9,
        )

        self.assertEqual(response["answer"], "ok")
        self.assertEqual(requests[0]["method"], "POST")
        self.assertEqual(requests[0]["path"], "/ask")
        self.assertEqual(
            requests[0]["json"],
            (
                b'{"question":"What is FastAPI?",'
                b'"generation":{"temperature":0.1,"max_tokens":64,"top_p":0.9}}'
            ),
        )


if __name__ == "__main__":
    unittest.main()
