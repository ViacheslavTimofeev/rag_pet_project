from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from scripts import run_eval
from src.retrieval.types import RetrievedChunk, Retriever


class FakeRetriever(Retriever):
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                text=f"retrieved for {query}",
                score=0.9,
                rank=1,
            )
        ]


class RunEvalScriptTests(unittest.TestCase):
    def test_load_eval_config_and_get_values(self) -> None:
        tmp_root = Path(".tmp_test") / f"run-eval-config-{uuid.uuid4().hex}"
        try:
            tmp_root.mkdir(parents=True)
            config_path = tmp_root / "eval.yaml"
            config_path.write_text(
                (
                    "eval:\n"
                    "  dataset:\n"
                    "    path: data/eval/retrieval_sources_gt.jsonl\n"
                    "  retrieval:\n"
                    "    k_values: [1, 5]\n"
                ),
                encoding="utf-8",
            )

            config = run_eval.load_eval_config(config_path)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(
            run_eval.get_dataset_path(config),
            Path("data/eval/retrieval_sources_gt.jsonl"),
        )
        self.assertEqual(run_eval.get_k_values(config), [1, 5])
        self.assertEqual(run_eval.get_retrieval_mode(config), "retriever")

    def test_retrieval_command_saves_run_artifact(self) -> None:
        tmp_root = Path(".tmp_test") / f"run-eval-command-{uuid.uuid4().hex}"
        try:
            tmp_root.mkdir(parents=True)
            dataset_path = tmp_root / "retrieval_gt.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "question": "question",
                        "relevant_chunk_ids": ["c1"],
                        "relevant_document_ids": ["d1"],
                        "reference_answer": "answer",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = tmp_root / "eval.yaml"
            config_path.write_text(
                (
                    "eval:\n"
                    "  dataset:\n"
                    f"    path: {dataset_path.as_posix()}\n"
                    "  retrieval:\n"
                    "    k_values: [1]\n"
                ),
                encoding="utf-8",
            )
            output_dir = tmp_root / "runs"
            args = run_eval.build_parser().parse_args(
                [
                    "retrieval",
                    "--eval-config",
                    str(config_path),
                    "--retrieval-config",
                    "configs/retrieval.yaml",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            with (
                patch.object(run_eval, "build_retriever", return_value=FakeRetriever()),
                patch.object(run_eval, "print_retrieval_summary"),
            ):
                exit_code = run_eval.run_retrieval_command(args)

            artifacts = list(output_dir.glob("*.json"))
            saved = json.loads(artifacts[0].read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(saved["metrics"]["query_count"], 1)
        self.assertEqual(saved["metrics"]["recall_at_k"]["1"], 1.0)
        self.assertEqual(saved["metadata"]["eval_retrieval_mode"], "retriever")
        self.assertEqual(
            saved["metadata"]["configs"]["eval"]["eval"]["dataset"]["path"],
            dataset_path.as_posix(),
        )
        self.assertIn("retrieval", saved["metadata"]["configs"]["retrieval"])
        self.assertEqual(saved["results"][0]["retrieved_chunks"][0]["chunk_id"], "c1")

    def test_retrieval_command_can_evaluate_pipeline_mode(self) -> None:
        tmp_root = Path(".tmp_test") / f"run-eval-pipeline-{uuid.uuid4().hex}"
        try:
            tmp_root.mkdir(parents=True)
            dataset_path = tmp_root / "retrieval_gt.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "question": "question",
                        "relevant_chunk_ids": ["c1"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = tmp_root / "eval.yaml"
            config_path.write_text(
                (
                    "eval:\n"
                    "  dataset:\n"
                    f"    path: {dataset_path.as_posix()}\n"
                    "  retrieval:\n"
                    "    mode: pipeline\n"
                    "    k_values: [1]\n"
                ),
                encoding="utf-8",
            )
            output_dir = tmp_root / "runs"
            args = run_eval.build_parser().parse_args(
                [
                    "retrieval",
                    "--eval-config",
                    str(config_path),
                    "--retrieval-config",
                    "configs/retrieval.yaml",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            with (
                patch.object(
                    run_eval,
                    "build_retrieval_pipeline",
                    return_value=FakeRetriever(),
                ),
                patch.object(run_eval, "build_retriever") as build_retriever,
                patch.object(run_eval, "print_retrieval_summary"),
            ):
                exit_code = run_eval.run_retrieval_command(args)

            artifacts = list(output_dir.glob("*.json"))
            saved = json.loads(artifacts[0].read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(exit_code, 0)
        build_retriever.assert_not_called()
        self.assertEqual(saved["metadata"]["eval_retrieval_mode"], "pipeline")
        self.assertEqual(saved["results"][0]["retrieved_chunks"][0]["chunk_id"], "c1")


if __name__ == "__main__":
    unittest.main()
