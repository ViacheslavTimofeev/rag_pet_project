from __future__ import annotations

import unittest

from src.api.schemas import AskRequest, GenerationParamsRequest
from src.api.service import RagAskService
from src.llm.postprocess import PostprocessedResponse
from src.llm.types import LLMGenerationParams
from src.retrieval.types import BuiltContext


class FakeRetrievalPipeline:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, query: str) -> BuiltContext:
        self.queries.append(query)
        return BuiltContext(text="retrieved context")


class FakeGenerationPipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(
        self,
        query: str,
        built_context: BuiltContext,
        *,
        params: LLMGenerationParams | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> PostprocessedResponse:
        self.calls.append(
            {
                "query": query,
                "context_text": built_context.text,
                "params": params,
                "metadata": metadata,
            }
        )
        return PostprocessedResponse(answer="answer", metadata=dict(metadata or {}))


class RagAskServiceTests(unittest.TestCase):
    def test_answer_runs_retrieval_then_generation(self) -> None:
        retrieval = FakeRetrievalPipeline()
        generation = FakeGenerationPipeline()
        service = RagAskService(
            retrieval_pipeline=retrieval,  # type: ignore[arg-type]
            generation_pipeline=generation,  # type: ignore[arg-type]
        )

        response = service.answer(
            AskRequest(
                question=" What is FastAPI? ",
                generation=GenerationParamsRequest(
                    temperature=0.2,
                    max_tokens=64,
                    top_p=0.9,
                    stop=["END"],
                ),
                metadata={"trace_id": "abc"},
            )
        )

        self.assertEqual(response.answer, "answer")
        self.assertEqual(response.metadata, {"trace_id": "abc"})
        self.assertEqual(retrieval.queries, ["What is FastAPI?"])
        self.assertEqual(len(generation.calls), 1)
        call = generation.calls[0]
        self.assertEqual(call["query"], "What is FastAPI?")
        self.assertEqual(call["context_text"], "retrieved context")
        self.assertEqual(call["metadata"], {"trace_id": "abc"})

        params = call["params"]
        self.assertIsInstance(params, LLMGenerationParams)
        assert isinstance(params, LLMGenerationParams)
        self.assertEqual(params.temperature, 0.2)
        self.assertEqual(params.max_tokens, 64)
        self.assertEqual(params.top_p, 0.9)
        self.assertEqual(params.stop, ["END"])

    def test_answer_rejects_blank_question_after_strip(self) -> None:
        service = RagAskService(
            retrieval_pipeline=FakeRetrievalPipeline(),  # type: ignore[arg-type]
            generation_pipeline=FakeGenerationPipeline(),  # type: ignore[arg-type]
        )

        with self.assertRaises(ValueError):
            service.answer(AskRequest(question="   "))


if __name__ == "__main__":
    unittest.main()
