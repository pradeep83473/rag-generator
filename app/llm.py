"""LLM provider abstraction. OpenAI is the production implementation; tests
inject a deterministic fake so the whole pipeline runs offline."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from app.config import Settings


class ProviderNotConfigured(RuntimeError):
    """Raised when the LLM provider cannot be created (e.g. missing API key)."""


class LLMProvider(Protocol):
    embedding_model: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 matrix of L2-normalised embeddings."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant's reply text."""


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class OpenAIProvider:
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise ProviderNotConfigured(
                "OPENAI_API_KEY is not set. Add it to your environment or .env file."
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            timeout=settings.openai_timeout_seconds,
            max_retries=3,
        )
        self._settings = settings
        self.embedding_model = settings.openai_embedding_model

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        batch_size = self._settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self._client.embeddings.create(model=self.embedding_model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return normalize(np.asarray(vectors, dtype=np.float32))

    def chat(self, messages: list[dict[str, str]]) -> str:
        kwargs = {}
        if self._settings.llm_temperature is not None:
            kwargs["temperature"] = self._settings.llm_temperature
        response = self._client.chat.completions.create(
            model=self._settings.openai_chat_model, messages=messages, **kwargs
        )
        return (response.choices[0].message.content or "").strip()
