from __future__ import annotations

from .types import BuiltContext, ContextBuilder, ContextSource, RetrievedChunk


class TokenBudgetContextBuilder(ContextBuilder):
    """Build context text from ranked chunks with configurable size constraints."""

    def __init__(
        self,
        *,
        max_chunks: int | None = 5,
        max_chars: int | None = 4000,
        dedup_by_document: bool = True,
        chunk_separator: str = "\n\n",
    ) -> None:
        if max_chunks is not None and max_chunks <= 0:
            raise ValueError("max_chunks must be a positive integer or null.")
        if max_chars is not None and max_chars <= 0:
            raise ValueError("max_chars must be a positive integer or null.")
        if not chunk_separator:
            raise ValueError("chunk_separator must be a non-empty string.")

        self._max_chunks = max_chunks
        self._max_chars = max_chars
        self._dedup_by_document = dedup_by_document
        self._chunk_separator = chunk_separator

    def build(self, chunks: list[RetrievedChunk]) -> BuiltContext:
        selected_chunks: list[RetrievedChunk] = []
        seen_documents: set[str] = set()
        current_chars = 0

        for chunk in chunks:
            if self._dedup_by_document and chunk.document_id in seen_documents:
                continue

            if self._max_chunks is not None and len(selected_chunks) >= self._max_chunks:
                break

            additional_chars = len(chunk.text)
            if selected_chunks:
                additional_chars += len(self._chunk_separator)

            if (
                self._max_chars is not None
                and current_chars + additional_chars > self._max_chars
            ):
                continue

            selected_chunks.append(chunk)
            current_chars += additional_chars
            if self._dedup_by_document:
                seen_documents.add(chunk.document_id)

        return BuiltContext(
            text=self._chunk_separator.join(chunk.text for chunk in selected_chunks),
            sources=[self._to_source(chunk) for chunk in selected_chunks],
            used_chunks=list(selected_chunks),
        )

    def _to_source(self, chunk: RetrievedChunk) -> ContextSource:
        return ContextSource(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            rank=chunk.rank,
            score=chunk.score,
            metadata=dict(chunk.metadata),
        )
