from __future__ import annotations

import os
from typing import Any

import httpx

from src.llm.types import LLMBackend, LLMRequest, LLMResponse, TokenUsage


class VllmBackend(LLMBackend):
    """vLLM OpenAI-compatible chat completions adapter."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be a non-empty string.")
        if not model:
            raise ValueError("model must be a non-empty string.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive number.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        if top_p <= 0:
            raise ValueError("top_p must be a positive number.")
        if api_key_env is not None and not api_key_env:
            raise ValueError("api_key_env must be a non-empty string or null.")

        resolved_api_key = api_key
        if resolved_api_key is None and api_key_env is not None:
            resolved_api_key = os.environ.get(api_key_env)

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = resolved_api_key
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._stop = list(stop or [])
        self._client = client or httpx.Client(timeout=timeout_seconds)

    @property
    def backend_name(self) -> str:
        return "vllm"

    def generate(self, request: LLMRequest) -> LLMResponse:
        query = request.query.strip()
        if not query:
            raise ValueError("request.query must be a non-empty string.")
        if request.params.stream:
            raise ValueError("VllmBackend does not support streaming responses yet.")

        response = self._client.post(
            f"{self._base_url}/chat/completions",
            headers=self._build_headers(),
            json=self._build_payload(request),
            timeout=self._timeout_seconds,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "vLLM chat completion request failed: "
                f"{exc.response.status_code} {exc.response.text}"
            ) from exc

        return self._parse_response(response.json())

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._build_chat_messages(request),
            "temperature": request.params.temperature,
            "max_tokens": request.params.max_tokens,
            "top_p": request.params.top_p,
            "stream": False,
        }

        if payload["max_tokens"] is None:
            payload["max_tokens"] = self._max_tokens
        if payload["top_p"] is None:
            payload["top_p"] = self._top_p

        stop = request.params.stop or self._stop
        if stop:
            payload["stop"] = stop

        return payload

    def _build_chat_messages(self, request: LLMRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        user_prompt = self._build_prompt(request)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _build_prompt(self, request: LLMRequest) -> str:
        if request.context_text:
            return (
                "Use the provided context to answer the question. "
                "If context is insufficient, say so.\n\n"
                f"Context:\n{request.context_text}\n\n"
                f"Question:\n{request.query.strip()}"
            )
        return request.query.strip()

    def _parse_response(self, response: Any) -> LLMResponse:
        if not isinstance(response, dict):
            return LLMResponse(text=str(response), raw=response)

        choices = response.get("choices")
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        if not isinstance(first_choice, dict):
            first_choice = {}

        message = first_choice.get("message")
        text = ""
        if isinstance(message, dict):
            content = message.get("content")
            text = content if isinstance(content, str) else str(content or "")
        elif "text" in first_choice:
            candidate_text = first_choice.get("text")
            text = candidate_text if isinstance(candidate_text, str) else str(
                candidate_text or ""
            )

        finish_reason = first_choice.get("finish_reason")
        usage = self._parse_usage(response.get("usage"))

        model = response.get("model")
        model_name = model if isinstance(model, str) else None

        return LLMResponse(
            text=text,
            model=model_name,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=usage,
            raw=response,
        )

    def _parse_usage(self, usage: Any) -> TokenUsage | None:
        if not isinstance(usage, dict):
            return None

        raw_input_tokens = usage.get("prompt_tokens")
        raw_output_tokens = usage.get("completion_tokens")
        raw_total_tokens = usage.get("total_tokens")

        if not isinstance(raw_input_tokens, int):
            return None
        if not isinstance(raw_output_tokens, int):
            return None
        if not isinstance(raw_total_tokens, int):
            return None

        return TokenUsage(
            input_tokens=raw_input_tokens,
            output_tokens=raw_output_tokens,
            total_tokens=raw_total_tokens,
        )
