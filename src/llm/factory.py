from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.embeddings.factory import load_model_config
from src.llm.backends import LlamaCppBackend
from src.llm.types import LLMBackend

DEFAULT_MODEL_CONFIG_PATH = Path("configs/model.yaml")


def build_llm_backend_from_config(config: Mapping[str, Any]) -> LLMBackend:
    """Build the active LLM backend from a parsed model config mapping."""

    llm_config = config.get("llm")
    if not isinstance(llm_config, Mapping):
        raise ValueError("Model config must contain an 'llm' mapping.")

    active_backend = llm_config.get("active_backend")
    if not isinstance(active_backend, str) or not active_backend:
        raise ValueError("'llm.active_backend' must be a non-empty string.")

    if active_backend == "llamacpp":
        backend_config = llm_config.get("llamacpp")
        if not isinstance(backend_config, Mapping):
            raise ValueError(
                "Model config must contain an 'llm.llamacpp' mapping when that "
                "backend is active."
            )
        return LlamaCppBackend(
            model_path=_require_str(backend_config, "model_path"),
            n_ctx=_get_int(backend_config, "n_ctx", default=4096),
            n_gpu_layers=_get_int_with_min(
                backend_config, "n_gpu_layers", default=0, min_value=-1
            ),
            temperature=_get_float(backend_config, "temperature", default=0.0),
            max_tokens=_get_int(backend_config, "max_tokens", default=512),
            top_p=_get_float(backend_config, "top_p", default=1.0),
            stop=_get_optional_str_list(backend_config, "stop"),
            seed=_get_optional_int(backend_config, "seed"),
            chat_format=_get_optional_str(backend_config, "chat_format"),
            verbose=_get_bool(backend_config, "verbose", default=False),
        )

    raise ValueError(f"Unsupported llm backend: {active_backend!r}")


def build_llm_backend(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
) -> LLMBackend:
    """Load model config from disk and build the active LLM backend."""

    return build_llm_backend_from_config(load_model_config(config_path))


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


def _get_int_with_min(
    config: Mapping[str, Any], key: str, *, default: int, min_value: int
) -> int:
    value = config.get(key, default)
    if not isinstance(value, int) or value < min_value:
        raise ValueError(f"'{key}' must be an integer greater than or equal to {min_value}.")
    return value


def _get_optional_int(config: Mapping[str, Any], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"'{key}' must be an integer or null.")
    return value


def _get_float(config: Mapping[str, Any], key: str, *, default: float) -> float:
    value = config.get(key, default)
    if not isinstance(value, (int, float)):
        raise ValueError(f"'{key}' must be a number.")
    return float(value)


def _get_optional_str_list(config: Mapping[str, Any], key: str) -> list[str] | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"'{key}' must be a list of non-empty strings or null.")
    return list(value)
