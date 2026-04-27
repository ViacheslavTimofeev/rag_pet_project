from __future__ import annotations

from typing import Any

import httpx

from src.ui.client import RAGApiClient
from src.ui.config import UiConfig, load_ui_config


def build_ui(config: UiConfig | None = None) -> Any:
    """Build the Gradio UI without launching it."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "gradio is required to run the UI. Install it in the 'rag' env."
        ) from exc

    ui_config = config or load_ui_config()
    client = RAGApiClient(
        base_url=ui_config.api_base_url,
        timeout_seconds=ui_config.request_timeout_seconds,
    )

    def submit(
        question: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> tuple[str, list[dict[str, Any]], list[str], dict[str, Any]]:
        try:
            response = client.ask(
                question=question,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        except httpx.HTTPStatusError as exc:
            detail = _extract_error_detail(exc.response)
            return "", [], [f"{exc.response.status_code}: {detail}"], {}
        except httpx.HTTPError as exc:
            return "", [], [str(exc)], {}

        return (
            str(response.get("answer", "")),
            _normalize_sources(response.get("sources", [])),
            list(response.get("warnings", [])),
            {
                "model": response.get("model"),
                "finish_reason": response.get("finish_reason"),
                "usage": response.get("usage"),
                "metadata": response.get("metadata"),
            },
        )

    with gr.Blocks(title="RAG Local Assistant") as demo:
        gr.Markdown("# RAG Local Assistant")
        with gr.Row():
            question = gr.Textbox(
                label="Question",
                lines=5,
                placeholder="Ask a question about the indexed documents",
            )
        with gr.Row():
            temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.0,
                step=0.1,
                label="Temperature",
            )
            max_tokens = gr.Slider(
                minimum=64,
                maximum=4096,
                value=1024,
                step=64,
                label="Max tokens",
            )
            top_p = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top p",
            )
        ask_button = gr.Button("Ask", variant="primary")
        answer = gr.Textbox(label="Answer", lines=10)
        sources = gr.JSON(label="Sources")
        warnings = gr.JSON(label="Warnings")
        metadata = gr.JSON(label="Run metadata")

        ask_button.click(
            submit,
            inputs=[question, temperature, max_tokens, top_p],
            outputs=[answer, sources, warnings, metadata],
        )

    return demo


def launch_ui(config: UiConfig | None = None) -> None:
    """Launch the Gradio UI using shared config values."""

    ui_config = config or load_ui_config()
    demo = build_ui(ui_config)
    demo.launch(
        server_name=ui_config.host,
        server_port=ui_config.port,
        share=ui_config.share,
        inbrowser=ui_config.inbrowser,
    )


def _normalize_sources(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return str(detail or response.text)
