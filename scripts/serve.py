from __future__ import annotations

from threading import Thread
import time

import httpx

from src.api.config import load_api_config
from src.ui.app import launch_ui
from src.ui.config import UiConfig


def main() -> None:
    """Run the local API server and, when enabled, the Gradio UI."""

    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("uvicorn is required to serve the API.") from exc

    config = load_api_config()
    if not config.ui.enabled:
        uvicorn.run(
            "src.api.app:app",
            host=config.server.host,
            port=config.server.port,
            reload=config.server.reload,
            log_level=config.server.log_level,
            workers=config.server.workers,
        )
        return

    server = uvicorn.Server(
        uvicorn.Config(
            "src.api.app:app",
            host=config.server.host,
            port=config.server.port,
            reload=False,
            log_level=config.server.log_level,
            workers=1,
        )
    )
    api_thread = Thread(target=server.run, daemon=True)
    api_thread.start()

    _wait_for_api(config.ui.api_base_url)
    launch_ui(
        UiConfig(
            host=config.ui.host,
            port=config.ui.port,
            share=config.ui.share,
            inbrowser=config.ui.inbrowser,
            api_base_url=config.ui.api_base_url,
            request_timeout_seconds=config.request.timeout_seconds,
        )
    )


def _wait_for_api(base_url: str, *, attempts: int = 30, delay_seconds: float = 0.25) -> None:
    for _ in range(attempts):
        try:
            response = httpx.get(f"{base_url.rstrip('/')}/health", timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(delay_seconds)

    raise RuntimeError(f"API did not become healthy at {base_url!r}.")


if __name__ == "__main__":
    main()
