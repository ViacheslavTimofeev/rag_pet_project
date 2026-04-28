from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from src.embeddings.factory import build_embedder
from src.ingest.types import Chunk
from src.retrieval.factory import load_retrieval_config
from src.vectordb.db import QdrantVectorStore
from src.vectordb.indexing import build_vector_index


DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_RETRIEVAL_CONFIG_PATH = Path("configs/retrieval.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed processed chunks and upsert them into Qdrant."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to chunks.jsonl produced by scripts.ingest.",
    )
    parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=DEFAULT_RETRIEVAL_CONFIG_PATH,
        help="Path to retrieval YAML with Qdrant settings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of chunks to embed/upsert per batch.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    retrieval_config = load_retrieval_config(args.retrieval_config)
    qdrant_config = get_qdrant_config(retrieval_config)
    model_config_path = get_model_config_path(retrieval_config)

    chunks = load_chunks(args.chunks)
    embedder = build_embedder(model_config_path)
    vector_store = QdrantVectorStore(
        url=require_str(qdrant_config, "url"),
        api_key=get_optional_str(qdrant_config, "api_key_env"),
        prefer_grpc=get_bool(qdrant_config, "prefer_grpc", default=False),
    )
    collection_name = require_str(qdrant_config, "collection_name")

    result = build_vector_index(
        chunks,
        embedder=embedder,
        vector_store=vector_store,
        collection_name=collection_name,
        batch_size=args.batch_size,
    )

    print(f"Indexed {result.chunks_indexed} chunks into collection {result.collection_name!r}")
    print(f"Embedding model: {result.embedding_model}")
    print(f"Vector dimension: {result.vector_dimension}")
    return 0


def load_chunks(path: str | Path) -> list[Chunk]:
    chunks_path = Path(path)
    chunks: list[Chunk] = []
    with chunks_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Chunk row {line_number} must be a JSON object.")
            chunks.append(chunk_from_mapping(row, line_number=line_number))
    return chunks


def chunk_from_mapping(row: Mapping[str, Any], *, line_number: int) -> Chunk:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Chunk row {line_number} metadata must be a mapping.")

    return Chunk(
        chunk_id=require_str(row, "chunk_id"),
        document_id=require_str(row, "document_id"),
        chunk_index=require_int(row, "chunk_index"),
        text=require_str(row, "text"),
        metadata=dict(metadata),
    )


def get_qdrant_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    retrieval_config = get_mapping(config, "retrieval")
    llamaindex_config = get_mapping(retrieval_config, "llamaindex")
    return get_mapping(llamaindex_config, "qdrant")


def get_model_config_path(config: Mapping[str, Any]) -> Path:
    retrieval_config = get_mapping(config, "retrieval")
    llamaindex_config = get_mapping(retrieval_config, "llamaindex")
    value = llamaindex_config.get("model_config_path", "configs/model.yaml")
    if not isinstance(value, str) or not value:
        raise ValueError("'model_config_path' must be a non-empty string path.")
    return Path(value)


def get_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"'{key}' must be a mapping.")
    return value


def require_str(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty string.")
    return value


def require_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"'{key}' must be an integer.")
    return value


def get_optional_str(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty string or null.")
    return value


def get_bool(config: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"'{key}' must be a boolean.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
