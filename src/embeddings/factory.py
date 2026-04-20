from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .embedder import SentenceTransformerEmbedder
from .types import EmbeddingModel

DEFAULT_MODEL_CONFIG_PATH = Path("configs/model.yaml")


def load_model_config(config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH) -> dict[str, Any]:
    """Load model configuration from YAML."""

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load configs/model.yaml.") from exc

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Model config must be a mapping at the top level.")

    return loaded


def build_embedder_from_config(config: Mapping[str, Any]) -> EmbeddingModel:
    """Build the active embedding backend from a parsed config mapping."""

    embedding_config = config.get("embedding")
    if not isinstance(embedding_config, Mapping):
        raise ValueError("Model config must contain an 'embedding' mapping.")

    active_backend = embedding_config.get("active_backend")
    if not isinstance(active_backend, str) or not active_backend:
        raise ValueError("'embedding.active_backend' must be a non-empty string.")

    if active_backend == "sentence_transformer":
        backend_config = embedding_config.get(active_backend)
        if not isinstance(backend_config, Mapping):
            raise ValueError(
                "Model config must contain an 'embedding.sentence_transformer' "
                "mapping when that backend is active."
            )

        return SentenceTransformerEmbedder(
            model_name=_require_str(backend_config, "model_name"),
            normalize_embeddings=_get_bool(
                backend_config, "normalize_embeddings", default=True
            ),
            batch_size=_get_int(backend_config, "batch_size", default=32),
            device=_get_optional_str(backend_config, "device"),
        )

    raise ValueError(f"Unsupported embedding backend: {active_backend!r}")


def build_embedder(config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH) -> EmbeddingModel:
    """Load config from disk and build the active embedding backend."""

    return build_embedder_from_config(load_model_config(config_path))


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
