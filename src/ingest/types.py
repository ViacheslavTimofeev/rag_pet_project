from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RawDocument:
    """Normalized source document ready for chunking."""

    document_id: str
    source_path: Path
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """Chunked document fragment ready for embedding and indexing."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    metadata: dict[str, str | int] = field(default_factory=dict)
