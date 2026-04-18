from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path

from .types import RawDocument

SUPPORTED_SUFFIXES = {".md", ".mdx", ".html", ".htm", ".txt"}


@dataclass(slots=True)
class FastAPITutorialIngestor:
    """Load a local export of the FastAPI tutorial into normalized documents."""

    source_dir: Path
    glob: str = "**/*"

    def load(self) -> list[RawDocument]:
        documents: list[RawDocument] = []
        for path in sorted(self.source_dir.glob(self.glob)):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue

            text = _normalize_text(path.read_text(encoding="utf-8"))
            if not text:
                continue

            relative_path = path.relative_to(self.source_dir)
            section = _section_from_relative_path(relative_path)
            document_id = _document_id_from_relative_path(relative_path)
            documents.append(
                RawDocument(
                    document_id=document_id,
                    source_path=path,
                    text=text,
                    metadata={
                        "dataset": "fastapi_tutorial",
                        "source": str(relative_path).replace("\\", "/"),
                        "section": section,
                        "title": path.stem.replace("-", " ").replace("_", " ").strip(),
                    },
                )
            )

        return documents


def _normalize_text(content: str) -> str:
    text = unescape(content)
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def _section_from_relative_path(relative_path: Path) -> str:
    parts = relative_path.parts[:-1]
    return "/".join(parts) if parts else "root"


def _document_id_from_relative_path(relative_path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", relative_path.as_posix().lower()).strip("-")

