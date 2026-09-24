"""Orchestration: ingestion and question answering over a collection."""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

import numpy as np

from app.config import Settings
from app.generation import (
    build_answer_messages,
    build_condense_messages,
    cited_numbers,
    is_declined,
)
from app.ingestion.chunking import chunk_sections
from app.ingestion.loaders import EmptyDocument, UnsupportedFileType, load_document
from app.llm import LLMProvider, OpenAIProvider
from app.retrieval import RetrievedChunk, hybrid_search
from app.store import Collection, CollectionManager, DuplicateDocument


@dataclass
class IngestResult:
    filename: str
    status: str  # "indexed" | "duplicate" | "error"
    document_id: str | None = None
    chunks: int = 0
    detail: str | None = None


@dataclass
class Answer:
    answer: str
    search_query: str
    sources: list[dict] = field(default_factory=list)
    grounded: bool = True


class RAGService:
    def __init__(self, settings: Settings, provider: LLMProvider | None = None):
        self.settings = settings
        self.collections = CollectionManager(settings.data_dir)
        self._provider = provider

    @property
    def provider(self) -> LLMProvider:
        # Created lazily so the server (and its UI) starts even before a key is
        # configured; requests that need the LLM return a clear error instead.
        if self._provider is None:
            self._provider = OpenAIProvider(self.settings)
        return self._provider

    # ---------------- ingestion ----------------
    def ingest(self, collection: Collection, filename: str, data: bytes) -> IngestResult:
        sha256 = hashlib.sha256(data).hexdigest()
        existing = collection.find_by_hash(sha256)
        if existing:
            return self._duplicate(filename, existing)
        try:
            sections = load_document(filename, data)
        except (UnsupportedFileType, EmptyDocument) as exc:
            return IngestResult(filename, "error", detail=str(exc))
        except Exception as exc:  # corrupt/encrypted files etc.
            return IngestResult(filename, "error", detail=f"Could not read '{filename}': {exc}")

        doc_id = uuid.uuid4().hex[:12]
        chunks = chunk_sections(
            sections, doc_id, filename, self.settings.chunk_size, self.settings.chunk_overlap
        )
        # Embedding is the slow, network-bound step; it runs outside the
        # collection lock so queries keep working during ingestion.
        vectors = self.provider.embed([c.search_text for c in chunks])
        pages = sorted({s.page for s in sections if s.page})
        try:
            collection.add_document(
                {
                    "id": doc_id,
                    "filename": filename,
                    "sha256": sha256,
                    "size_bytes": len(data),
                    "pages": len(pages) or None,
                },
                chunks,
                vectors,
                self.provider.embedding_model,
            )
        except DuplicateDocument as exc:
            return self._duplicate(filename, exc.existing)
        return IngestResult(filename, "indexed", doc_id, len(chunks))

    @staticmethod
    def _duplicate(filename: str, existing: dict) -> IngestResult:
        return IngestResult(
            filename, "duplicate", existing["id"], existing["chunks"],
            f"Already indexed as '{existing['filename']}'.",
        )

    # ---------------- question answering ----------------
    def _search_query(self, question: str, history: list[dict[str, str]]) -> str:
        if not history:
            return question
        rewritten = self.provider.chat(build_condense_messages(history, question))
        return rewritten.strip().strip('"') or question

    def retrieve(self, collection: Collection, query: str, top_k: int) -> list[RetrievedChunk]:
        query_vector: np.ndarray = self.provider.embed([query])[0]
        return hybrid_search(
            collection, query, query_vector, top_k, self.settings.retrieval_candidates
        )

    def ask(
        self,
        collection: Collection,
        question: str,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> Answer:
        history = (history or [])[-self.settings.history_turns :]
        if not collection.chunks:
            return Answer(
                "This knowledge base has no documents yet. Upload some files first.",
                question, grounded=False,
            )
        search_query = self._search_query(question, history)
        results = self.retrieve(collection, search_query, top_k or self.settings.retrieval_top_k)
        messages = build_answer_messages(collection.meta, question, results, history)
        answer = self.provider.chat(messages)

        cited = cited_numbers(answer, len(results))
        sources = []
        for number, result in enumerate(results, start=1):
            chunk = result.chunk
            sources.append(
                {
                    "number": number,
                    "cited": number in cited,
                    "document_id": chunk.doc_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "text": chunk.text,
                    "similarity": round(result.similarity, 4),
                }
            )
        grounded = bool(cited) and not is_declined(answer)
        return Answer(answer, search_query, sources, grounded)
