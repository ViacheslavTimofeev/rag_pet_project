from __future__ import annotations

from dataclasses import dataclass

from .types import Chunk, RawDocument


@dataclass(slots=True)
class CharacterTextChunker:
    """Split normalized documents into overlapping text chunks."""

    chunk_size: int = 1000
    chunk_overlap: int = 200

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def chunk_many_documents(self, documents: list[RawDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_single_document(document))
        return chunks

    def chunk_single_document(self, document: RawDocument) -> list[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        for chunk_index, start in enumerate(self._iter_chunk_starts(text)):
            chunk_text, end = self._slice_chunk(text, start)
            chunk_id = f"{document.document_id}-chunk-{chunk_index:04d}"
            metadata: dict[str, str | int] = {
                **document.metadata,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "char_start": start,
                "char_end": end,
            }
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    metadata=metadata,
                )
            )

        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_count"] = len(chunks)
            chunk.metadata["prev_chunk_id"] = chunks[chunk_index - 1].chunk_id if chunk_index > 0 else ""
            chunk.metadata["next_chunk_id"] = (
                chunks[chunk_index + 1].chunk_id if chunk_index + 1 < len(chunks) else ""
            )

        return chunks

    def _iter_chunk_starts(self, text: str):
        start = 0
        step = self.chunk_size - self.chunk_overlap
        text_length = len(text)

        while start < text_length:
            yield start
            if start + self.chunk_size >= text_length:
                break
            next_start = start + step
            start = self._align_start(text, next_start)

    def _align_start(self, text: str, start: int) -> int:
        text_length = len(text)
        if start >= text_length:
            return text_length

        original_start = start
        if start > 0 and self._is_word_char(text[start - 1]) and self._is_word_char(text[start]):
            while start < text_length and self._is_word_char(text[start]):
                start += 1
            if start >= text_length:
                return original_start
            while start < text_length and not self._is_word_char(text[start]):
                start += 1
            if start >= text_length:
                return original_start

        while start < text_length and text[start].isspace():
            start += 1

        return start

    def _slice_chunk(self, text: str, start: int) -> tuple[str, int]:
        max_end = min(len(text), start + self.chunk_size)
        end = self._find_boundary(text, start, max_end)
        if end <= start:
            end = max_end
        return text[start:end].strip(), end

    def _find_boundary(self, text: str, start: int, max_end: int) -> int:
        if max_end == len(text):
            return max_end

        boundary_window = text[start:max_end]
        paragraph_break = boundary_window.rfind("\n\n")
        if paragraph_break > 0:
            return start + paragraph_break

        line_break = boundary_window.rfind("\n")
        if line_break > 0:
            return start + line_break

        sentence_breaks = [". ", "! ", "? "]
        for marker in sentence_breaks:
            sentence_break = boundary_window.rfind(marker)
            if sentence_break > 0:
                return start + sentence_break + 1

        word_break = boundary_window.rfind(" ")
        if word_break > 0:
            return start + word_break

        return max_end

    @staticmethod
    def _is_word_char(char: str) -> bool:
        return char.isalnum() or char == "_"
