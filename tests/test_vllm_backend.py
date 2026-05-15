from __future__ import annotations

import json
import unittest

import httpx

from src.llm.backends import VllmBackend
from src.llm.types import LLMGenerationParams, LLMRequest


class VllmBackendTests(unittest.TestCase):
    def test_generate_posts_chat_completion_and_parses_response(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers.get("Authorization")
            captured["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "model": "Qwen/Qwen3-14B-AWQ",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Grounded answer."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 4,
                        "total_tokens": 16,
                    },
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        backend = VllmBackend(
            base_url="http://localhost:8000/v1/",
            model="Qwen/Qwen3-14B-AWQ",
            api_key="local-vllm",
            max_tokens=128,
            top_p=0.9,
            client=client,
        )

        response = backend.generate(
            LLMRequest(
                query="What is FastAPI?",
                context_text="FastAPI is a Python web framework.",
                system_prompt="Use only context.",
                params=LLMGenerationParams(temperature=0.1, stop=["</s>"]),
            )
        )

        self.assertEqual(response.text, "Grounded answer.")
        self.assertEqual(response.model, "Qwen/Qwen3-14B-AWQ")
        self.assertEqual(response.finish_reason, "stop")
        self.assertIsNotNone(response.usage)
        assert response.usage is not None
        self.assertEqual(response.usage.input_tokens, 12)
        self.assertEqual(response.usage.output_tokens, 4)
        self.assertEqual(response.usage.total_tokens, 16)
        self.assertEqual(
            captured["url"],
            "http://localhost:8000/v1/chat/completions",
        )
        self.assertEqual(captured["authorization"], "Bearer local-vllm")

        payload = captured["payload"]
        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertEqual(payload["model"], "Qwen/Qwen3-14B-AWQ")
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual(payload["max_tokens"], 128)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["stop"], ["</s>"])
        self.assertEqual(payload["stream"], False)
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": "Use only context."},
                {
                    "role": "user",
                    "content": (
                        "Use the provided context to answer the question. "
                        "If context is insufficient, say so.\n\n"
                        "Context:\nFastAPI is a Python web framework.\n\n"
                        "Question:\nWhat is FastAPI?"
                    ),
                },
            ],
        )

    def test_generate_wraps_http_errors_with_response_body(self) -> None:
        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(400, text="bad request")
            )
        )
        backend = VllmBackend(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen3-14B-AWQ",
            client=client,
        )

        with self.assertRaisesRegex(RuntimeError, "400 bad request"):
            backend.generate(LLMRequest(query="hello"))

    def test_generate_rejects_streaming_until_supported(self) -> None:
        backend = VllmBackend(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen3-14B-AWQ",
            client=httpx.Client(transport=httpx.MockTransport(lambda request: None)),
        )

        with self.assertRaisesRegex(ValueError, "streaming"):
            backend.generate(
                LLMRequest(
                    query="hello",
                    params=LLMGenerationParams(stream=True),
                )
            )


if __name__ == "__main__":
    unittest.main()
