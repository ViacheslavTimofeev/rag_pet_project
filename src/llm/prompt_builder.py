from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.llm.types import LLMGenerationParams, LLMRequest
from src.retrieval.types import BuiltContext


DEFAULT_SYSTEM_PROMPT = (
    "You are a grounded assistant. Use only the provided context when answering. "
    "If the context is insufficient, explicitly say you do not have enough "
    "information. Do not fabricate facts or sources."
)


@dataclass(slots=True)
class PromptBuilderConfig:
    """Configuration for deterministic retrieval-to-prompt formatting."""

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    include_source_labels: bool = True
    max_context_chars: int | None = None
    answer_language: str | None = None
    answer_style: str | None = None


class GroundedPromptBuilder:
    """Build normalized LLM requests from query and retrieved context."""

    def __init__(self, config: PromptBuilderConfig | None = None) -> None:
        self._config = config or PromptBuilderConfig()
        if (
            self._config.max_context_chars is not None
            and self._config.max_context_chars <= 0
        ):
            raise ValueError("max_context_chars must be a positive integer or null.")

    def build_request(
        self,
        query: str,
        built_context: BuiltContext,
        *,
        params: LLMGenerationParams | None = None,
        metadata: Mapping[str, str | int | float | bool] | None = None,
    ) -> LLMRequest:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must be a non-empty string.")

        return LLMRequest(
            query=normalized_query,
            context_text=self._format_context(built_context),
            system_prompt=self._build_system_prompt(),
            sources=list(built_context.sources),
            params=params or LLMGenerationParams(),
            metadata=dict(metadata or {}),
        )

    def _build_system_prompt(self) -> str:
        instructions = [self._config.system_prompt.strip()]
        if self._config.answer_language:
            instructions.append(
                f"Respond in {self._config.answer_language.strip()} unless the user requests otherwise."
            )
        if self._config.answer_style:
            instructions.append(f"Keep the response style: {self._config.answer_style.strip()}.")
        return "\n".join(instruction for instruction in instructions if instruction)

    def _format_context(self, built_context: BuiltContext) -> str:
        if self._config.include_source_labels and built_context.used_chunks:
            context_text = self._format_labeled_chunks(built_context)
        else:
            context_text = built_context.text.strip()

        if not context_text:
            return ""

        max_context_chars = self._config.max_context_chars
        if max_context_chars is None or len(context_text) <= max_context_chars:
            return context_text

        truncated = context_text[: max_context_chars - 15].rstrip()
        return f"{truncated}\n...[truncated]"

    def _format_labeled_chunks(self, built_context: BuiltContext) -> str:
        blocks: list[str] = []
        for index, chunk in enumerate(built_context.used_chunks, start=1):
            label = (
                f"[{index}] doc={chunk.document_id} chunk={chunk.chunk_id} rank={chunk.rank}"
            )
            blocks.append(f"{label}\n{chunk.text.strip()}")
        return "\n\n".join(block for block in blocks if block.strip())
