from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from src.eval import (
    RetrievalEvalExample,
    load_retrieval_eval_dataset,
    run_retrieval_eval,
    save_retrieval_eval_run,
)
from src.retrieval.types import RetrievedChunk, Retriever


class FakeRetriever(Retriever):
    def __init__(self, results_by_query: dict[str, list[RetrievedChunk]]) -> None:
        self._results_by_query = results_by_query
        self.calls: list[str] = []

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        self.calls.append(query)
        return list(self._results_by_query.get(query, []))


def _chunk(chunk_id: str, *, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=f"text for {chunk_id}",
        score=1.0 / rank,
        rank=rank,
        metadata={"section": "body"},
    )


class RetrievalRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_root = Path(".tmp_test") / self.id().replace(".", "_")
        if self._temp_root.exists():
            shutil.rmtree(self._temp_root)
        self._temp_root.mkdir(parents=True)

    def tearDown(self) -> None:
        if self._temp_root.exists():
            shutil.rmtree(self._temp_root)

    def test_load_retrieval_eval_dataset_reads_jsonl_examples(self) -> None:
        path = self._temp_root / "retrieval_sources_gt.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "q1",
                            "question": "What is alpha?",
                            "relevant_chunk_ids": ["c1"],
                            "relevant_document_ids": ["d1"],
                            "reference_answer": "Alpha is first.",
                        }
                    ),
                    "",
                ]
            ),
            encoding="utf-8",
        )

        examples = load_retrieval_eval_dataset(path)

        self.assertEqual(
            examples,
            [
                RetrievalEvalExample(
                    id="q1",
                    question="What is alpha?",
                    relevant_chunk_ids=("c1",),
                    relevant_document_ids=("d1",),
                    reference_answer="Alpha is first.",
                )
            ],
        )

    def test_load_retrieval_eval_dataset_accepts_document_only_examples(self) -> None:
        path = self._temp_root / "retrieval_sources_gt.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "q1",
                    "question": "What is alpha?",
                    "relevant_document_ids": ["d1"],
                    "reference_answer": "Alpha is first.",
                }
            ),
            encoding="utf-8",
        )

        examples = load_retrieval_eval_dataset(path)

        self.assertEqual(examples[0].relevant_chunk_ids, ())
        self.assertEqual(examples[0].relevant_document_ids, ("d1",))

    def test_run_retrieval_eval_captures_outputs_and_metrics(self) -> None:
        examples = [
            RetrievalEvalExample(
                id="q1",
                question="question one",
                relevant_chunk_ids=("c2",),
                relevant_document_ids=("doc-c2",),
                reference_answer="answer one",
            ),
            RetrievalEvalExample(
                id="q2",
                question="question two",
                relevant_chunk_ids=("c9",),
                relevant_document_ids=("doc-c9",),
                reference_answer="answer two",
            ),
        ]
        retriever = FakeRetriever(
            {
                "question one": [_chunk("c1", rank=1), _chunk("c2", rank=2)],
                "question two": [_chunk("c3", rank=1)],
            }
        )

        result = run_retrieval_eval(
            retriever,
            examples,
            k_values=[1, 2],
            dataset_path="data/eval/retrieval_gt.jsonl",
            run_id="test-run",
            metadata={"retriever": "fake"},
        )

        self.assertEqual(retriever.calls, ["question one", "question two"])
        self.assertEqual(result.run_id, "test-run")
        self.assertEqual(result.dataset_path, "data/eval/retrieval_gt.jsonl")
        self.assertEqual(
            result.metadata,
            {"retriever": "fake", "metric_target_level": "chunk_id"},
        )
        self.assertEqual(result.metrics.query_count, 2)
        self.assertEqual(result.metrics.recall_at_k[1], 0.0)
        self.assertEqual(result.metrics.recall_at_k[2], 0.5)
        self.assertEqual(result.metrics.hit_rate_at_k[2], 0.5)
        self.assertEqual(result.metrics.mean_reciprocal_rank_at_k[2], 0.25)
        self.assertEqual(
            [chunk.chunk_id for chunk in result.results[0].retrieved_chunks],
            ["c1", "c2"],
        )
        self.assertEqual(result.results[0].metrics["reciprocal_rank_at_k"][2], 0.5)

    def test_run_retrieval_eval_can_score_document_level_targets(self) -> None:
        examples = [
            RetrievalEvalExample(
                id="q1",
                question="question one",
                relevant_chunk_ids=(),
                relevant_document_ids=("doc-c2",),
            )
        ]
        retriever = FakeRetriever(
            {"question one": [_chunk("c1", rank=1), _chunk("c2", rank=2)]}
        )

        result = run_retrieval_eval(
            retriever,
            examples,
            k_values=[1, 2],
        )

        self.assertEqual(result.metadata["metric_target_level"], "document_id")
        self.assertEqual(result.metrics.recall_at_k[1], 0.0)
        self.assertEqual(result.metrics.recall_at_k[2], 1.0)
        self.assertEqual(result.metrics.hit_rate_at_k[2], 1.0)

    def test_save_retrieval_eval_run_writes_json_artifact(self) -> None:
        result = run_retrieval_eval(
            FakeRetriever({"question": [_chunk("c1", rank=1)]}),
            [
                RetrievalEvalExample(
                    id="q1",
                    question="question",
                    relevant_chunk_ids=("c1",),
                )
            ],
            k_values=[1],
            run_id="test-run",
        )

        output_path = self._temp_root / "runs" / "retrieval.json"
        save_retrieval_eval_run(result, output_path)
        saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["run_id"], "test-run")
        self.assertEqual(saved["metrics"]["query_count"], 1)
        self.assertEqual(saved["results"][0]["retrieved_chunks"][0]["chunk_id"], "c1")

    def test_invalid_or_empty_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_retrieval_eval(FakeRetriever({}), [], k_values=[1])

        path = self._temp_root / "bad.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "q1",
                    "question": "question",
                    "relevant_chunk_ids": [],
                    "relevant_document_ids": [],
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            load_retrieval_eval_dataset(path)


if __name__ == "__main__":
    unittest.main()
