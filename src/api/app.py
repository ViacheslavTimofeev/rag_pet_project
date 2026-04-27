from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import ApiConfig, load_api_config
from src.api.dependencies import build_ask_service
from src.api.routes import router


def create_app(config: ApiConfig | None = None) -> FastAPI:
    """Create the local RAG assistant API application."""

    api_config = config or load_api_config()
    app = FastAPI(
        title=api_config.title,
        version=api_config.version,
        description=api_config.description,
    )

    if api_config.cors.enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=api_config.cors.allow_origins,
            allow_methods=api_config.cors.allow_methods,
            allow_headers=api_config.cors.allow_headers,
            allow_credentials=api_config.cors.allow_credentials,
        )

    if api_config.runtime.eager_startup:
        build_ask_service()

    app.include_router(router)

    return app


app = create_app()
