from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.api.config import DEFAULT_API_CONFIG_PATH, load_api_config


@dataclass(frozen=True, slots=True)
class UiConfig:
    host: str
    port: int
    share: bool
    inbrowser: bool
    api_base_url: str
    request_timeout_seconds: int


def load_ui_config(config_path: str | Path = DEFAULT_API_CONFIG_PATH) -> UiConfig:
    """Load UI settings from the shared API config file."""

    api_config = load_api_config(config_path)
    return UiConfig(
        host=api_config.ui.host,
        port=api_config.ui.port,
        share=api_config.ui.share,
        inbrowser=api_config.ui.inbrowser,
        api_base_url=api_config.ui.api_base_url,
        request_timeout_seconds=api_config.request.timeout_seconds,
    )
