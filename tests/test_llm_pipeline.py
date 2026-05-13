from __future__ import annotations

from collections.abc import Mapping
import unittest

from src.llm.pipeline import AnswerGenerationPipeline
from src.llm.service import LLMService
from src.llm.types import (
    LLMBackend,
    LLMGenerationParams,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)
from src.retrieval.types import BuiltContext, ContextSource


class RecordingBackend(LLMBackend):
    def __init__(self, response: LLMResponse) -> None:
        self._response = response
        self.calls: list[LLMRequest] = []

    @property
    def backend_name(self) -> str:
        return "recording"

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return self._response


class RecordingPromptBuilder:
    def __init__(self) -> None:
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
        return LLMRequest(
            query=query.strip(),
            context_text=built_context.text,
            sources=list(built_context.sources),
            params=params or LLMGenerationParams(),
            metadata=dict(metadata or {}),
        )


class AnswerGenerationPipelineTests(unittest.TestCase):
    def test_run_generates_and_postprocesses_response(self) -> None:
        source = ContextSource(
            chunk_id="c1",
            document_id="d1",
            rank=1,
            score=0.95,
            metadata={"section": "intro"},
        )
        context = BuiltContext(text="grounding context", sources=[source])
        response = LLMResponse(
            text="\nA grounded answer.\n",
            model="fake-model",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=12, output_tokens=4, total_tokens=16),
        )
        backend = RecordingBackend(response)
        prompt_builder = RecordingPromptBuilder()
        service = LLMService(backend=backend, prompt_builder=prompt_builder)
        pipeline = AnswerGenerationPipeline(service=service)
        params = LLMGenerationParams(temperature=0.1, max_tokens=64)
        metadata = {"trace_id": "abc"}

        result = pipeline.run(
            " question ",
            context,
            params=params,
            metadata=metadata,
        )

        self.assertEqual(result.answer, "A grounded answer.")
        self.assertEqual(result.sources, [source])
        self.assertEqual(result.model, "fake-model")
        self.assertEqual(result.usage, response.usage)
        self.assertEqual(result.metadata["trace_id"], "abc")
        self.assertIsInstance(result.metadata["llm_ms"], float)
        self.assertIsInstance(result.metadata["generation_ms"], float)
        self.assertEqual(result.warnings, [])
        self.assertEqual(prompt_builder.calls, [
            {
                "query": " question ",
                "context_text": "grounding context",
                "params": params,
                "metadata": metadata,
            }
        ])
        self.assertEqual(len(backend.calls), 1)

    def test_run_uses_context_for_missing_context_warning(self) -> None:
        backend = RecordingBackend(LLMResponse(text="ok"))
        service = LLMService(backend=backend, prompt_builder=RecordingPromptBuilder())
        pipeline = AnswerGenerationPipeline(service=service, min_answer_chars=2)

        result = pipeline.run("question", BuiltContext(text=""))

        self.assertEqual(result.warnings, ["missing_context"])

    def test_init_rejects_invalid_min_answer_chars(self) -> None:
        service = LLMService(
            backend=RecordingBackend(LLMResponse(text="answer")),
            prompt_builder=RecordingPromptBuilder(),
        )

        with self.assertRaises(ValueError):
            AnswerGenerationPipeline(service=service, min_answer_chars=0)


if __name__ == "__main__":
    unittest.main()
