from __future__ import annotations

from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

from src.retrieval.factory import (
    _build_llamaindex_retriever,
    build_retrieval_pipeline,
    build_retrieval_pipeline_from_config,
    build_retriever,
    build_retriever_from_config,
    load_retrieval_config,
)
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.retriever import LlamaIndexRetriever
from src.retrieval.types import BuiltContext, ContextBuilder, Reranker, RetrievedChunk


class FakeLlamaIndexBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, query: str) -> list[object]:
        self.calls.append(query)
        return []


class FakeHuggingFaceEmbedding:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeQdrantVectorStore:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeBuiltRetriever:
    def __init__(self, *, similarity_top_k: int) -> None:
        self.similarity_top_k = similarity_top_k

    def retrieve(self, query: str) -> list[object]:
        return []


class FakeVectorStoreIndex:
    last_call: dict[str, object] | None = None

    @classmethod
    def from_vector_store(cls, *, vector_store, embed_model):
        cls.last_call = {
            "vector_store": vector_store,
            "embed_model": embed_model,
        }

        class FakeIndex:
            def as_retriever(self, *, similarity_top_k: int):
                return FakeBuiltRetriever(similarity_top_k=similarity_top_k)

        return FakeIndex()


class FakeCoreModule:
    VectorStoreIndex = FakeVectorStoreIndex


class FakeReranker(Reranker):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {
                "query": query,
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
                "top_k": top_k,
            }
        )
        return list(chunks)


class FakeContextBuilder(ContextBuilder):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def build(self, chunks: list[RetrievedChunk]) -> BuiltContext:
        self.calls.append([chunk.chunk_id for chunk in chunks])
        return BuiltContext(text="built")


class RetrievalFactoryTests(unittest.TestCase):
    def test_load_retrieval_config_reads_yaml_file(self) -> None:
        tmp_root = Path(".tmp_test") / f"retrieval-config-{uuid.uuid4().hex}"
        try:
            tmp_root.mkdir(parents=True)
            config_path = tmp_root / "retrieval.yaml"
            config_path.write_text(
                (
                    "retrieval:\n"
                    "  llamaindex:\n"
                    "    top_k: 5\n"
                ),
                encoding="utf-8",
            )

            config = load_retrieval_config(config_path)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(config["retrieval"]["llamaindex"]["top_k"], 5)

    def test_build_retriever_from_config_wraps_in_llamaindex_retriever(self) -> None:
        config = {
            "retrieval": {
                "llamaindex": {
                    "top_k": 5,
                }
            }
        }
        backend = FakeLlamaIndexBackend()

        retriever = build_retriever_from_config(
            config,
            llamaindex_retriever=backend,
        )

        self.assertIsInstance(retriever, LlamaIndexRetriever)
        results = retriever.retrieve("hello")
        self.assertEqual(results, [])
        self.assertEqual(backend.calls, ["hello"])

    def test_build_retriever_loads_config_before_building(self) -> None:
        with patch("src.retrieval.factory.load_retrieval_config") as load_config:
            with patch("src.retrieval.factory.build_retriever_from_config") as build_from_config:
                config = {"retrieval": {"llamaindex": {}}}
                backend = FakeLlamaIndexBackend()
                load_config.return_value = config
                build_from_config.return_value = object()

                result = build_retriever(llamaindex_retriever=backend)

        load_config.assert_called_once()
        build_from_config.assert_called_once_with(
            config,
            llamaindex_retriever=backend,
        )
        self.assertIs(result, build_from_config.return_value)

    def test_build_retrieval_pipeline_from_config_composes_runtime_chain(self) -> None:
        config = {
            "retrieval": {
                "llamaindex": {
                    "top_k": 5,
                },
                "reranker": {
                    "active_backend": "identity",
                    "top_k": 3,
                },
                "context_builder": {
                    "max_chunks": 2,
                    "max_chars": 100,
                    "dedup_by_document": True,
                    "chunk_separator": "\n\n",
                },
            }
        }
        backend = FakeLlamaIndexBackend()
        reranker = FakeReranker()
        context_builder = FakeContextBuilder()

        pipeline = build_retrieval_pipeline_from_config(
            config,
            llamaindex_retriever=backend,
            reranker=reranker,
            context_builder=context_builder,
        )

        self.assertIsInstance(pipeline, RetrievalPipeline)
        built_context = pipeline.run("hello")
        self.assertEqual(built_context.text, "built")
        self.assertEqual(backend.calls, ["hello"])
        self.assertEqual(reranker.calls, [{"query": "hello", "chunk_ids": [], "top_k": 3}])
        self.assertEqual(context_builder.calls, [[]])

    def test_build_retrieval_pipeline_loads_config_before_building(self) -> None:
        with patch("src.retrieval.factory.load_retrieval_config") as load_config:
            with patch(
                "src.retrieval.factory.build_retrieval_pipeline_from_config"
            ) as build_from_config:
                config = {"retrieval": {"llamaindex": {}, "reranker": {"active_backend": "identity"}}}
                backend = FakeLlamaIndexBackend()
                reranker = FakeReranker()
                context_builder = FakeContextBuilder()
                load_config.return_value = config
                build_from_config.return_value = object()

                result = build_retrieval_pipeline(
                    llamaindex_retriever=backend,
                    reranker=reranker,
                    context_builder=context_builder,
                )

        load_config.assert_called_once()
        build_from_config.assert_called_once_with(
            config,
            llamaindex_retriever=backend,
            reranker=reranker,
            context_builder=context_builder,
        )
        self.assertIs(result, build_from_config.return_value)

    def test_build_llamaindex_retriever_builds_qdrant_backed_retriever(self) -> None:
        config = {
            "top_k": 7,
            "model_config_path": "configs/model.yaml",
            "qdrant": {
                "collection_name": "docs",
                "url": "http://localhost:6333",
                "api_key_env": None,
                "prefer_grpc": True,
            },
        }
        model_config = {
            "embedding": {
                "active_backend": "sentence_transformer",
                "sentence_transformer": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "device": "cpu",
                    "batch_size": 16,
                    "normalize_embeddings": False,
                },
            }
        }
        fake_client = object()

        with patch("src.retrieval.factory.load_model_config", return_value=model_config):
            with patch(
                "src.retrieval.factory._get_llamaindex_huggingface_embedding_cls",
                return_value=FakeHuggingFaceEmbedding,
            ):
                with patch(
                    "src.retrieval.factory._build_qdrant_client",
                    return_value=fake_client,
                ) as build_qdrant_client:
                    with patch(
                        "src.retrieval.factory._get_llamaindex_qdrant_vector_store_cls",
                        return_value=FakeQdrantVectorStore,
                    ):
                        with patch(
                            "src.retrieval.factory._get_llamaindex_core_module",
                            return_value=FakeCoreModule,
                        ):
                            retriever = _build_llamaindex_retriever(config)

        build_qdrant_client.assert_called_once_with(config["qdrant"])
        self.assertEqual(retriever.similarity_top_k, 7)
        last_call = FakeVectorStoreIndex.last_call
        self.assertIsNotNone(last_call)
        assert last_call is not None

        vector_store = last_call["vector_store"]
        embed_model = last_call["embed_model"]
        assert isinstance(vector_store, FakeQdrantVectorStore)
        assert isinstance(embed_model, FakeHuggingFaceEmbedding)

        self.assertIsInstance(vector_store, FakeQdrantVectorStore)
        self.assertEqual(
            vector_store.kwargs,
            {
                "client": fake_client,
                "collection_name": "docs",
            },
        )
        self.assertEqual(
            embed_model.kwargs,
            {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",
                "embed_batch_size": 16,
                "normalize": False,
            },
        )

    def test_build_llamaindex_retriever_rejects_unsupported_embedding_backend(self) -> None:
        config = {
            "qdrant": {
                "collection_name": "docs",
                "url": "http://localhost:6333",
            }
        }
        model_config = {
            "embedding": {
                "active_backend": "openai",
            }
        }

        with patch("src.retrieval.factory.load_model_config", return_value=model_config):
            with self.assertRaises(ValueError):
                _build_llamaindex_retriever(config)


if __name__ == "__main__":
    unittest.main()
