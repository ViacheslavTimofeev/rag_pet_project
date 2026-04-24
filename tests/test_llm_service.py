from __future__ import annotations

from collections.abc import Mapping
import unittest

from src.llm.service import LLMService
from src.llm.types import (
    LLMBackend,
    LLMGenerationParams,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)
from src.retrieval.types import BuiltContext


class RecordingBackend(LLMBackend):
    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    @property
    def backend_name(self) -> str:
        return "recording"

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            text="answer",
            model="fake-model",
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


class RecordingPromptBuilder:
    def __init__(self, request: LLMRequest) -> None:
        self._request = request
        self.calls: list[dict[str, object]] = []

    def build_request(
        self,
        query: str,
        built_context: BuiltContext,
        *,
        params: LLMGenerationParams | None = None,
        metadata: Mapping[str, str | int | float | bool] | None = None,
    ) -> LLMRequest:
        self.calls.append(
            {
                "query": query,
                "context_text": built_context.text,
                "params": params,
                "metadata": metadata,
            }
        )
        return self._request


class LLMServiceTests(unittest.TestCase):
    def test_generate_builds_prompt_and_calls_backend(self) -> None:
        request = LLMRequest(query="normalized")
        backend = RecordingBackend()
        prompt_builder = RecordingPromptBuilder(request)
        service = LLMService(backend=backend, prompt_builder=prompt_builder)
        params = LLMGenerationParams(temperature=0.1, max_tokens=128)
        metadata = {"trace_id": "abc"}

        response = service.generate(
            "hello",
            BuiltContext(text="ctx"),
            params=params,
            metadata=metadata,
        )

        self.assertEqual(response.text, "answer")
        self.assertEqual(response.model, "fake-model")
        self.assertEqual(prompt_builder.calls, [
            {
                "query": "hello",
                "context_text": "ctx",
                "params": params,
                "metadata": metadata,
            }
        ])
        self.assertEqual(backend.calls, [request])

    def test_generate_uses_default_prompt_builder(self) -> None:
        backend = RecordingBackend()
        service = LLMService(backend=backend)

        response = service.generate("hello", BuiltContext(text="ctx"))

        self.assertEqual(response.text, "answer")
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0].query, "hello")
        self.assertEqual(backend.calls[0].context_text, "ctx")


if __name__ == "__main__":
    unittest.main()
