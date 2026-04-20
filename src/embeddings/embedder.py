from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .types import EmbeddingModel, EmbeddingVector


class SentenceTransformerEmbedder(EmbeddingModel):
    """Sentence Transformers adapter that implements the embedding contract."""

    def __init__(
        self,
        model_name: str,
        *,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._normalize_embeddings = normalize_embeddings
        self._batch_size = batch_size
        self._device = device
        self._model = model if model is not None else self._load_model()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int | None:
        return self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        if not texts:
            return []

        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required to use "
                "SentenceTransformerEmbedder."
            ) from exc

        model_kwargs: dict[str, Any] = {}
        if self._device is not None:
            model_kwargs["device"] = self._device

        return SentenceTransformer(self._model_name, **model_kwargs)
