from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from scripts import build_ragas_dataset


class FakeTestset:
    def __init__(self) -> None:
        self.saved_path: Path | None = None

    def to_jsonl(self, path: str | Path) -> None:
        output_path = Path(path)
        self.saved_path = output_path
        output_path.write_text(
            json.dumps({"question": "What is FastAPI?", "answer": "A framework."})
            + "\n",
            encoding="utf-8",
        )


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.testset = FakeTestset()

    def generate_with_langchain_docs(
        self,
        documents: object,
        *,
        testset_size: int,
        run_config: object,
        raise_exceptions: bool,
    ) -> FakeTestset:
        self.calls.append(
            {
                "documents": documents,
                "testset_size": testset_size,
                "run_config": run_config,
                "raise_exceptions": raise_exceptions,
            }
        )
        return self.testset


class BuildRagasDatasetScriptTests(unittest.TestCase):
    def test_load_chunk_documents_reads_jsonl_chunks(self) -> None:
        tmp_root = Path(".tmp_test") / f"ragas-chunks-{uuid.uuid4().hex}"
        try:
            tmp_root.mkdir(parents=True)
            chunks_path = tmp_root / "chunks.jsonl"
            chunks_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "chunk_id": "c1",
                                "document_id": "d1",
                                "chunk_index": 0,
                                "text": "First chunk text",
                                "metadata": {"source": "index.md"},
                            }
                        ),
                        json.dumps(
                            {
                                "chunk_id": "c2",
                                "document_id": "d2",
                                "chunk_index": 1,
                                "text": "Second chunk text",
                                "metadata": {"section": "security"},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            docs = build_ragas_dataset.load_chunk_documents(chunks_path, max_docs=1)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "First chunk text")
        self.assertEqual(docs[0].metadata["chunk_id"], "c1")
        self.assertEqual(docs[0].metadata["document_id"], "d1")
        self.assertEqual(docs[0].metadata["chunk_index"], 0)
        self.assertEqual(docs[0].metadata["source"], "index.md")

    def test_load_markdown_documents_reads_non_empty_markdown_files(self) -> None:
        tmp_root = Path(".tmp_test") / f"ragas-docs-{uuid.uuid4().hex}"
        try:
            source_dir = tmp_root / "raw"
            nested_dir = source_dir / "security"
            nested_dir.mkdir(parents=True)
            (source_dir / "index.md").write_text("# Intro\nFastAPI", encoding="utf-8")
            (nested_dir / "first-steps.md").write_text(
                "# Security\nOAuth2",
                encoding="utf-8",
            )
            (source_dir / "empty.md").write_text("  ", encoding="utf-8")

            docs = build_ragas_dataset.load_markdown_documents(source_dir)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["source"], "index.md")
        self.assertEqual(docs[0].metadata["document_id"], "index-md")
        self.assertEqual(docs[1].metadata["source"], "security/first-steps.md")
        self.assertEqual(docs[1].metadata["document_id"], "security-first-steps-md")

    def test_run_command_builds_testset_from_config_and_writes_jsonl(self) -> None:
        tmp_root = Path(".tmp_test") / f"ragas-command-{uuid.uuid4().hex}"
        try:
            tmp_root.mkdir(parents=True)
            chunks_path = tmp_root / "chunks.jsonl"
            chunks_path.write_text(
                json.dumps(
                    {
                        "chunk_id": "c1",
                        "document_id": "d1",
                        "chunk_index": 0,
                        "text": "# Intro\nFastAPI",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = tmp_root / "ragas.jsonl"
            eval_config = tmp_root / "eval.yaml"
            eval_config.write_text(
                (
                    "eval:\n"
                    "  dataset:\n"
                    f"    chunk_source: {chunks_path.as_posix()}\n"
                    "  ragas:\n"
                    f"    output_path: {output_path.as_posix()}\n"
                    "    testset_size: 3\n"
                    "    max_workers: 1\n"
                    "    timeout_seconds: 30\n"
                    "    max_tokens: 2048\n"
                    "    max_doc_chars: 100\n"
                ),
                encoding="utf-8",
            )
            model_config = tmp_root / "model.yaml"
            model_config.write_text(
                (
                    "embedding:\n"
                    "  active_backend: sentence_transformer\n"
                    "  sentence_transformer:\n"
                    "    model_name: sentence-transformers/all-MiniLM-L6-v2\n"
                    "llm:\n"
                    "  active_backend: vllm\n"
                    "  vllm:\n"
                    "    base_url: http://127.0.0.1:8000/v1\n"
                    "    model: Qwen/Qwen3-14B-AWQ\n"
                    "    api_key: local-vllm\n"
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                eval_config=eval_config,
                model_config=model_config,
                source_dir=None,
                chunks=None,
                output=None,
                testset_size=None,
                max_docs=None,
                max_workers=None,
                timeout=None,
                max_tokens=None,
                max_doc_chars=None,
                keep_going=True,
            )
            generator = FakeGenerator()
            fake_run_config = object()

            with (
                patch.object(
                    build_ragas_dataset,
                    "build_testset_generator",
                    return_value=generator,
                ) as build_generator,
                patch.object(
                    build_ragas_dataset,
                    "build_run_config",
                    return_value=fake_run_config,
                ) as build_run_config,
            ):
                exit_code = build_ragas_dataset.run_command(args)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(exit_code, 0)
        build_generator.assert_called_once()
        self.assertEqual(build_generator.call_args.kwargs["max_tokens"], 2048)
        build_run_config.assert_called_once_with(timeout=30, max_workers=1)
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(generator.calls[0]["testset_size"], 3)
        self.assertEqual(generator.calls[0]["run_config"], fake_run_config)
        self.assertEqual(generator.calls[0]["raise_exceptions"], False)
        self.assertEqual(generator.testset.saved_path, output_path)


if __name__ == "__main__":
    unittest.main()
