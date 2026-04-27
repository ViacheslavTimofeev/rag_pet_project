from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


DEFAULT_API_CONFIG_PATH = Path("configs/api.yaml")


@dataclass(frozen=True, slots=True)
class ApiServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"
    workers: int = 1


@dataclass(frozen=True, slots=True)
class ApiCorsConfig:
    enabled: bool = True
    allow_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    allow_methods: list[str] = field(default_factory=lambda: ["GET", "POST"])
    allow_headers: list[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = False


@dataclass(frozen=True, slots=True)
class ApiRequestConfig:
    timeout_seconds: int = 120
    max_question_chars: int = 4000


@dataclass(frozen=True, slots=True)
class ApiRuntimeConfig:
    retrieval_config_path: Path = Path("configs/retrieval.yaml")
    model_config_path: Path = Path("configs/model.yaml")
    eager_startup: bool = False


@dataclass(frozen=True, slots=True)
class ApiUiConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    inbrowser: bool = True
    api_base_url: str = "http://127.0.0.1:8000"


@dataclass(frozen=True, slots=True)
class ApiConfig:
    title: str = "RAG Local Assistant API"
    version: str = "0.1.0"
    description: str = "Local-first API for retrieval-augmented assistant workflows."
    server: ApiServerConfig = field(default_factory=ApiServerConfig)
    cors: ApiCorsConfig = field(default_factory=ApiCorsConfig)
    request: ApiRequestConfig = field(default_factory=ApiRequestConfig)
    runtime: ApiRuntimeConfig = field(default_factory=ApiRuntimeConfig)
    ui: ApiUiConfig = field(default_factory=ApiUiConfig)


def load_api_config(config_path: str | Path = DEFAULT_API_CONFIG_PATH) -> ApiConfig:
    """Load and validate API configuration from YAML."""

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load configs/api.yaml.") from exc

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("API config must be a mapping at the top level.")

    return build_api_config_from_mapping(loaded)


def build_api_config_from_mapping(config: Mapping[str, Any]) -> ApiConfig:
    """Build typed API settings from a parsed config mapping."""

    api_config = config.get("api")
    if not isinstance(api_config, Mapping):
        raise ValueError("API config must contain an 'api' mapping.")

    return ApiConfig(
        title=_get_non_empty_str(api_config, "title", default="RAG Local Assistant API"),
        version=_get_non_empty_str(api_config, "version", default="0.1.0"),
        description=_get_non_empty_str(
            api_config,
            "description",
            default="Local-first API for retrieval-augmented assistant workflows.",
        ),
        server=_build_server_config(_get_optional_mapping(api_config, "server")),
        cors=_build_cors_config(_get_optional_mapping(api_config, "cors")),
        request=_build_request_config(_get_optional_mapping(api_config, "request")),
        runtime=_build_runtime_config(_get_optional_mapping(api_config, "runtime")),
        ui=_build_ui_config(_get_optional_mapping(api_config, "ui")),
    )


def _build_server_config(config: Mapping[str, Any]) -> ApiServerConfig:
    return ApiServerConfig(
        host=_get_non_empty_str(config, "host", default="127.0.0.1"),
        port=_get_port(config, "port", default=8000),
        reload=_get_bool(config, "reload", default=False),
        log_level=_get_non_empty_str(config, "log_level", default="info"),
        workers=_get_positive_int(config, "workers", default=1),
    )


def _build_cors_config(config: Mapping[str, Any]) -> ApiCorsConfig:
    return ApiCorsConfig(
        enabled=_get_bool(config, "enabled", default=True),
        allow_origins=_get_str_list(
            config,
            "allow_origins",
            default=["http://localhost:3000"],
        ),
        allow_methods=_get_str_list(config, "allow_methods", default=["GET", "POST"]),
        allow_headers=_get_str_list(config, "allow_headers", default=["*"]),
        allow_credentials=_get_bool(config, "allow_credentials", default=False),
    )


def _build_request_config(config: Mapping[str, Any]) -> ApiRequestConfig:
    return ApiRequestConfig(
        timeout_seconds=_get_positive_int(config, "timeout_seconds", default=120),
        max_question_chars=_get_positive_int(config, "max_question_chars", default=4000),
    )


def _build_runtime_config(config: Mapping[str, Any]) -> ApiRuntimeConfig:
    return ApiRuntimeConfig(
        retrieval_config_path=_get_path(
            config,
            "retrieval_config_path",
            default=Path("configs/retrieval.yaml"),
        ),
        model_config_path=_get_path(
            config,
            "model_config_path",
            default=Path("configs/model.yaml"),
        ),
        eager_startup=_get_bool(config, "eager_startup", default=False),
    )


def _build_ui_config(config: Mapping[str, Any]) -> ApiUiConfig:
    return ApiUiConfig(
        enabled=_get_bool(config, "enabled", default=True),
        host=_get_non_empty_str(config, "host", default="127.0.0.1"),
        port=_get_port(config, "port", default=7860),
        share=_get_bool(config, "share", default=False),
        inbrowser=_get_bool(config, "inbrowser", default=True),
        api_base_url=_get_non_empty_str(
            config,
            "api_base_url",
            default="http://127.0.0.1:8000",
        ),
    )


def _get_optional_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"'api.{key}' must be a mapping or null.")
    return value


def _get_non_empty_str(
    config: Mapping[str, Any],
    key: str,
    *,
    default: str,
) -> str:
    value = config.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty string.")
    return value


def _get_bool(config: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"'{key}' must be a boolean.")
    return value


def _get_positive_int(config: Mapping[str, Any], key: str, *, default: int) -> int:
    value = config.get(key, default)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"'{key}' must be a positive integer.")
    return value


def _get_port(config: Mapping[str, Any], key: str, *, default: int) -> int:
    value = _get_positive_int(config, key, default=default)
    if value > 65535:
        raise ValueError(f"'{key}' must be between 1 and 65535.")
    return value


def _get_str_list(
    config: Mapping[str, Any],
    key: str,
    *,
    default: list[str],
) -> list[str]:
    value = config.get(key, default)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"'{key}' must be a list of non-empty strings.")
    return list(value)


def _get_path(config: Mapping[str, Any], key: str, *, default: Path) -> Path:
    value = config.get(key, default)
    if isinstance(value, Path):
        return value
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty string path.")
    return Path(value)
