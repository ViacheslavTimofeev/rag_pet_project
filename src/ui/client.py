from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class RAGApiClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: int = 120
    transport: httpx.BaseTransport | None = None

    def health(self) -> dict[str, Any]:
        with httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = client.get("/health")
            response.raise_for_status()
            return response.json()

    def ask(
        self,
        *,
        question: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question}
        generation = _generation_payload(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        if generation:
            payload["generation"] = generation

        with httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = client.post("/ask", json=payload)
            response.raise_for_status()
            return response.json()


def _generation_payload(
    *,
    temperature: float | None,
    max_tokens: int | None,
    top_p: float | None,
) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p
    return payload
