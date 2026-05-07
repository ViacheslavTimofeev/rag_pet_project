from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Chunk, RawDocument


@dataclass(frozen=True, slots=True)
class MarkdownParentSection:
    """Markdown section used as parent context for child chunks."""

    text: str
    headers: dict[str, str]


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


@dataclass(slots=True)
class MarkdownParentChildChunker:
    """Split markdown documents into section parents and retrievable child chunks."""

    child_chunk_size: int = 800
    child_chunk_overlap: int = 120
    include_header_prefix: bool = True
    extract_symbols: bool = True

    def __post_init__(self) -> None:
        if self.child_chunk_size <= 0:
            raise ValueError("child_chunk_size must be greater than zero")
        if self.child_chunk_overlap < 0:
            raise ValueError("child_chunk_overlap cannot be negative")
        if self.child_chunk_overlap >= self.child_chunk_size:
            raise ValueError(
                "child_chunk_overlap must be smaller than child_chunk_size"
            )

    def chunk_many_documents(self, documents: list[RawDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_single_document(document))
        return chunks

    def chunk_single_document(self, document: RawDocument) -> list[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        parents = self._split_parent_sections(text)
        chunks: list[Chunk] = []
        for parent_index, parent in enumerate(parents):
            parent_id = f"{document.document_id}-parent-{parent_index:04d}"
            parent_text = parent.text
            section_path = self._section_path(parent.headers, document)
            symbols = self._extract_symbols(parent_text) if self.extract_symbols else []

            child_texts = self._split_child_chunks(parent_text)
            for child_index, child_text in enumerate(child_texts):
                chunk_index = len(chunks)
                chunk_id = f"{document.document_id}-chunk-{chunk_index:04d}"
                metadata: dict[str, str | int] = {
                    **document.metadata,
                    "chunk_id": chunk_id,
                    "document_id": document.document_id,
                    "chunk_index": chunk_index,
                    "parent_id": parent_id,
                    "parent_index": parent_index,
                    "child_index": child_index,
                    "section_path": section_path,
                    "symbols": ", ".join(symbols),
                }
                metadata.update(parent.headers)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        chunk_index=chunk_index,
                        text=self._build_child_text(
                            child_text,
                            section_path=section_path,
                            symbols=symbols,
                        ),
                        metadata=metadata,
                    )
                )

        self._attach_neighbor_metadata(chunks)
        return chunks

    def _split_parent_sections(self, text: str) -> list[MarkdownParentSection]:
        try:
            from langchain_text_splitters import MarkdownHeaderTextSplitter
        except ImportError as exc:
            raise ImportError(
                "langchain-text-splitters is required for MarkdownParentChildChunker."
            ) from exc

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
                ("####", "header_4"),
            ],
            strip_headers=False,
        )
        documents = splitter.split_text(text)
        if not documents:
            return [MarkdownParentSection(text=text, headers={})]

        parents: list[MarkdownParentSection] = []
        for split_document in documents:
            parent_text = split_document.page_content.strip()
            if not parent_text:
                continue
            headers = {
                key: str(value)
                for key, value in split_document.metadata.items()
                if key.startswith("header_") and value
            }
            parents.append(MarkdownParentSection(text=parent_text, headers=headers))

        return parents or [MarkdownParentSection(text=text, headers={})]

    def _split_child_chunks(self, text: str) -> list[str]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise ImportError(
                "langchain-text-splitters is required for MarkdownParentChildChunker."
            ) from exc

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        child_texts = [chunk.strip() for chunk in splitter.split_text(text)]
        return [chunk for chunk in child_texts if chunk]

    def _build_child_text(
        self,
        text: str,
        *,
        section_path: str,
        symbols: list[str],
    ) -> str:
        if not self.include_header_prefix:
            return text

        prefix_lines = []
        if section_path:
            prefix_lines.append(f"Section: {section_path}")
        if symbols:
            prefix_lines.append(f"Symbols: {', '.join(symbols)}")
        if not prefix_lines:
            return text
        return "\n".join(prefix_lines) + "\n\n" + text

    def _section_path(
        self,
        headers: dict[str, str],
        document: RawDocument,
    ) -> str:
        ordered_headers = [
            headers[key]
            for key in ("header_1", "header_2", "header_3", "header_4")
            if key in headers
        ]
        if ordered_headers:
            return " > ".join(ordered_headers)
        return document.metadata.get("title", document.document_id)

    def _extract_symbols(self, text: str) -> list[str]:
        candidates: list[str] = []
        for match in re.finditer(r"`([^`\n]{2,80})`", text):
            candidates.append(match.group(1).strip())
        for match in re.finditer(
            r"\b(?:[A-Z]{2,}|[A-Z][A-Za-z0-9_]*[A-Z][A-Za-z0-9_]*)\b",
            text,
        ):
            candidates.append(match.group(0))

        symbols: list[str] = []
        for candidate in candidates:
            normalized = " ".join(candidate.split())
            if not self._is_symbol_like(normalized):
                continue
            if normalized not in symbols:
                symbols.append(normalized)
        return symbols[:24]

    def _is_symbol_like(self, value: str) -> bool:
        if len(value) < 2:
            return False
        if value.startswith(("http://", "https://")):
            return False
        if " " in value:
            return False
        return any(
            marker in value
            for marker in ("_", ".", "(", ")", "[", "]", ":", "/")
        ) or value.isupper() or sum(char.isupper() for char in value) >= 2

    def _attach_neighbor_metadata(self, chunks: list[Chunk]) -> None:
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_count"] = len(chunks)
            chunk.metadata["prev_chunk_id"] = (
                chunks[chunk_index - 1].chunk_id if chunk_index > 0 else ""
            )
            chunk.metadata["next_chunk_id"] = (
                chunks[chunk_index + 1].chunk_id
                if chunk_index + 1 < len(chunks)
                else ""
            )
