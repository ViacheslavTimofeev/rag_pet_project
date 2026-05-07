from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from src.embeddings.factory import load_model_config
from src.runtime.device import require_cuda_device

from .context_builder import TokenBudgetContextBuilder
from .pipeline import RetrievalPipeline
from .reranker import CrossEncoderReranker, IdentityReranker
from .retriever import LlamaIndexRetriever
from .types import ContextBuilder, Reranker, Retriever
import yaml

DEFAULT_RETRIEVAL_CONFIG_PATH = Path("configs/retrieval.yaml")


def load_retrieval_config(
    config_path: str | Path = DEFAULT_RETRIEVAL_CONFIG_PATH,
) -> dict[str, Any]:
    """Load retrieval configuration from YAML."""
    
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Retrieval config must be a mapping at the top level.")

    return loaded


def build_retriever_from_config(
    config: Mapping[str, Any],
    *,
    llamaindex_retriever: Any | None = None,
) -> Retriever:
    """Build the project retriever from a parsed retrieval config mapping."""

    retrieval_config = config.get("retrieval")
    if not isinstance(retrieval_config, Mapping):
        raise ValueError("Retrieval config must contain a 'retrieval' mapping.")

    llamaindex_config = retrieval_config.get("llamaindex")
    if not isinstance(llamaindex_config, Mapping):
        raise ValueError(
            "Retrieval config must contain a 'retrieval.llamaindex' mapping."
        )

    raw_retriever = llamaindex_retriever
    if raw_retriever is None:
        raw_retriever = _build_llamaindex_retriever(llamaindex_config)

    return LlamaIndexRetriever(retriever=raw_retriever)


def build_retriever(
    config_path: str | Path = DEFAULT_RETRIEVAL_CONFIG_PATH,
    *,
    llamaindex_retriever: Any | None = None,
) -> Retriever:
    """Load config from disk and build the active project retriever."""

    return build_retriever_from_config(
        load_retrieval_config(config_path),
        llamaindex_retriever=llamaindex_retriever,
    )


def build_retrieval_pipeline_from_config(
    config: Mapping[str, Any],
    *,
    llamaindex_retriever: Any | None = None,
    reranker: Reranker | None = None,
    context_builder: ContextBuilder | None = None,
) -> RetrievalPipeline:
    """Build retrieval runtime flow: retrieve, rerank, then assemble context."""

    retrieval_config = config.get("retrieval")
    if not isinstance(retrieval_config, Mapping):
        raise ValueError("Retrieval config must contain a 'retrieval' mapping.")

    built_retriever = build_retriever_from_config(
        config,
        llamaindex_retriever=llamaindex_retriever,
    )

    reranker_config = _get_mapping(retrieval_config, "reranker")
    built_reranker = reranker or _build_reranker_from_config(reranker_config)
    rerank_top_k = _get_optional_positive_int(reranker_config, "top_k")

    context_builder_config = retrieval_config.get("context_builder", {})
    if context_builder_config is None:
        context_builder_config = {}
    if not isinstance(context_builder_config, Mapping):
        raise ValueError("'context_builder' must be a mapping or null.")
    built_context_builder = context_builder or _build_context_builder_from_config(
        context_builder_config
    )

    return RetrievalPipeline(
        retriever=built_retriever,
        reranker=built_reranker,
        context_builder=built_context_builder,
        rerank_top_k=rerank_top_k,
    )


def build_retrieval_pipeline(
    config_path: str | Path = DEFAULT_RETRIEVAL_CONFIG_PATH,
    *,
    llamaindex_retriever: Any | None = None,
    reranker: Reranker | None = None,
    context_builder: ContextBuilder | None = None,
) -> RetrievalPipeline:
    """Load config from disk and build the composed retrieval runtime flow."""

    return build_retrieval_pipeline_from_config(
        load_retrieval_config(config_path),
        llamaindex_retriever=llamaindex_retriever,
        reranker=reranker,
        context_builder=context_builder,
    )


def _build_llamaindex_retriever(config: Mapping[str, Any]) -> Any:
    """Construct the raw LlamaIndex retriever used by the project wrapper."""

    llamaindex_config = _require_mapping(config, "retrieval.llamaindex")
    model_config = load_model_config(
        _get_path(
            llamaindex_config,
            "model_config_path",
            default=Path("configs/model.yaml"),
        )
    )
    embed_model = _build_llamaindex_embedding_model(model_config)
    qdrant_client = _build_qdrant_client(_get_mapping(llamaindex_config, "qdrant"))
    vector_store = _build_llamaindex_vector_store(
        _get_mapping(llamaindex_config, "qdrant"),
        qdrant_client=qdrant_client,
    )
    index = _build_llamaindex_index(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    return index.as_retriever(similarity_top_k=_get_int(llamaindex_config, "top_k", default=5))


def _build_reranker_from_config(config: Mapping[str, Any]) -> Reranker:
    active_backend = _require_str(config, "active_backend")
    if active_backend == "identity":
        return IdentityReranker()
    if active_backend == "cross_encoder":
        cross_encoder_config = _get_mapping(config, "cross_encoder")
        return CrossEncoderReranker(
            _require_str(cross_encoder_config, "model_name"),
            batch_size=_get_int(cross_encoder_config, "batch_size", default=32),
            max_length=_get_optional_positive_int(
                cross_encoder_config,
                "max_length",
            ),
            device=_get_reranker_device(cross_encoder_config),
            local_files_only=_get_bool(
                cross_encoder_config,
                "local_files_only",
                default=False,
            ),
            use_sigmoid=_get_bool(cross_encoder_config, "use_sigmoid", default=False),
        )
    raise ValueError(
        "Unsupported reranker backend. Expected one of: 'identity', "
        "'cross_encoder'."
    )


def _build_context_builder_from_config(config: Mapping[str, Any]) -> ContextBuilder:
    return TokenBudgetContextBuilder(
        max_chunks=_get_optional_positive_int(config, "max_chunks", default=5),
        max_chars=_get_optional_positive_int(config, "max_chars", default=4000),
        dedup_by_document=_get_bool(config, "dedup_by_document", default=True),
        chunk_separator=_get_non_empty_str(config, "chunk_separator", default="\n\n"),
    )


def _build_llamaindex_embedding_model(model_config: Mapping[str, Any]) -> Any:
    embedding_config = _get_mapping(model_config, "embedding")
    active_backend = embedding_config.get("active_backend")
    if active_backend != "sentence_transformer":
        raise ValueError(
            "Only 'sentence_transformer' embedding backend is supported for "
            "LlamaIndex retrieval wiring."
        )

    backend_config = _get_mapping(embedding_config, "sentence_transformer")
    embedding_cls = _get_llamaindex_huggingface_embedding_cls()
    return embedding_cls(
        model_name=_require_str(backend_config, "model_name"),
        device=require_cuda_device(),
        embed_batch_size=_get_int(backend_config, "batch_size", default=32),
        normalize=_get_bool(backend_config, "normalize_embeddings", default=True),
        local_files_only=_get_bool(backend_config, "local_files_only", default=False),
    )


def _get_reranker_device(config: Mapping[str, Any]) -> str | None:
    device = _get_non_empty_str(config, "device", default="cuda")
    if device == "cuda":
        return require_cuda_device()
    if device in {"cpu", "mps"}:
        return device
    if device == "auto":
        return None
    raise ValueError(
        "reranker.cross_encoder.device must be 'cuda', 'cpu', 'mps', or 'auto'."
    )


def _build_qdrant_client(config: Mapping[str, Any]) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError(
            "qdrant-client is required to build the LlamaIndex retriever. "
            f"Original import error: {exc}"
        ) from exc

    return QdrantClient(
        url=_require_str(config, "url"),
        api_key=_get_env_value(config, "api_key_env"),
        prefer_grpc=_get_bool(config, "prefer_grpc", default=False),
    )


def _build_llamaindex_vector_store(
    config: Mapping[str, Any], *, qdrant_client: Any
) -> Any:
    vector_store_cls = _get_llamaindex_qdrant_vector_store_cls()
    return vector_store_cls(
        client=qdrant_client,
        collection_name=_require_str(config, "collection_name"),
    )


def _build_llamaindex_index(*, vector_store: Any, embed_model: Any) -> Any:
    core_module = _get_llamaindex_core_module()
    return core_module.VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError(f"'{key}' must be a mapping.")
    return config


def _get_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"'{key}' must be a mapping.")
    return value


def _require_str(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty string.")
    return value


def _get_optional_str(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty string or null.")
    return value


def _get_bool(config: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"'{key}' must be a boolean.")
    return value


def _get_int(config: Mapping[str, Any], key: str, *, default: int) -> int:
    value = config.get(key, default)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"'{key}' must be a positive integer.")
    return value


def _get_optional_positive_int(
    config: Mapping[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int | None:
    value = config.get(key, default)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"'{key}' must be a positive integer or null.")
    return value


def _get_non_empty_str(config: Mapping[str, Any], key: str, *, default: str) -> str:
    value = config.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty string.")
    return value


def _get_path(config: Mapping[str, Any], key: str, *, default: Path) -> Path:
    value = config.get(key, default)
    if isinstance(value, Path):
        return value
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty string path.")
    return Path(value)


def _get_env_value(config: Mapping[str, Any], key: str) -> str | None:
    env_name = _get_optional_str(config, key)
    if env_name is None:
        return None
    return os.getenv(env_name)


def _get_llamaindex_core_module() -> Any:
    try:
        from llama_index import core as core_module
    except ImportError:
        try:
            import llama_index.core as core_module
        except ImportError as exc:
            raise ImportError(
                "llama-index-core is required to build the LlamaIndex retriever."
            ) from exc
    return core_module


def _get_llamaindex_qdrant_vector_store_cls() -> Any:
    try:
        from llama_index.vector_stores.qdrant import QdrantVectorStore
    except ImportError as exc:
        raise ImportError(
            "llama-index-vector-stores-qdrant is required to build the "
            f"LlamaIndex retriever. Original import error: {exc}"
        ) from exc
    return QdrantVectorStore


def _get_llamaindex_huggingface_embedding_cls() -> Any:
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:
        raise ImportError(
            "llama-index-embeddings-huggingface is required to build the "
            f"LlamaIndex retriever. Original import error: {exc}"
        ) from exc
    return HuggingFaceEmbedding
