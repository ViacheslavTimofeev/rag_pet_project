from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create the local RAG assistant API application."""

    app = FastAPI(
        title="RAG Local Assistant API",
        version="0.1.0",
        description="Local-first API for retrieval-augmented assistant workflows.",
    )

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
