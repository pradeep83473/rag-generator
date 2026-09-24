"""Persistent, per-collection index.

Each collection ("knowledge base") is an isolated directory:

    data/collections/<id>/meta.json        name, instructions, documents
    data/collections/<id>/chunks.json      chunk text + citation metadata
    data/collections/<id>/embeddings.npy   (n_chunks, dim) float32, L2-normalised

A plain NumPy matrix is deliberate: exact cosine search over tens of thousands
of chunks takes milliseconds, has no extra infrastructure, and is trivially
inspectable. The Collection API is small enough to swap in pgvector/Qdrant
later without touching the rest of the app.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.bm25 import BM25
from app.ingestion.chunking import Chunk


class CollectionNotFound(KeyError):
    pass


class DocumentNotFound(KeyError):
    pass


class EmbeddingModelMismatch(RuntimeError):
    pass


class DuplicateDocument(ValueError):
    def __init__(self, existing: dict):
        super().__init__(existing["id"])
        self.existing = existing


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write(tmp)
    os.replace(tmp, path)


class Collection:
    def __init__(self, path: Path, meta: dict):
        self.path = path
        self.meta = meta
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25 = BM25([])
        self.lock = threading.RLock()

    # ---------- persistence ----------
    @classmethod
    def create(cls, root: Path, name: str, description: str, instructions: str) -> "Collection":
        collection_id = uuid.uuid4().hex[:12]
        path = root / collection_id
        path.mkdir(parents=True)
        meta = {
            "id": collection_id,
            "name": name,
            "description": description,
            "instructions": instructions,
            "created_at": _now(),
            "embedding_model": None,
            "documents": [],
        }
        collection = cls(path, meta)
        collection._save()
        return collection

    @classmethod
    def load(cls, path: Path) -> "Collection":
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        collection = cls(path, meta)
        chunks_file, emb_file = path / "chunks.json", path / "embeddings.npy"
        if chunks_file.exists() and emb_file.exists():
            raw = json.loads(chunks_file.read_text(encoding="utf-8"))
            collection.chunks = [Chunk(**c) for c in raw]
            collection.embeddings = np.load(emb_file)
        collection._rebuild_lexical()
        return collection

    def _save(self) -> None:
        _atomic_write(
            self.path / "meta.json",
            lambda p: p.write_text(json.dumps(self.meta, indent=2), encoding="utf-8"),
        )
        if self.embeddings is not None:
            _atomic_write(
                self.path / "chunks.json",
                lambda p: p.write_text(
                    json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False),
                    encoding="utf-8",
                ),
            )

            def write_npy(p: Path):
                with open(p, "wb") as fh:
                    np.save(fh, self.embeddings)

            _atomic_write(self.path / "embeddings.npy", write_npy)

    def _rebuild_lexical(self) -> None:
        self.bm25 = BM25([c.search_text for c in self.chunks])

    # ---------- queries ----------
    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def documents(self) -> list[dict]:
        return self.meta["documents"]

    def find_by_hash(self, sha256: str) -> dict | None:
        return next((d for d in self.documents if d["sha256"] == sha256), None)

    def summary(self) -> dict:
        return {
            **{k: v for k, v in self.meta.items() if k != "documents"},
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
        }

    # ---------- mutations ----------
    def add_document(self, doc: dict, chunks: list[Chunk], vectors: np.ndarray, model: str) -> None:
        with self.lock:
            # Re-checked under the lock: two concurrent uploads of the same file
            # both pass the early check in the service while they are embedding.
            existing = self.find_by_hash(doc["sha256"])
            if existing:
                raise DuplicateDocument(existing)
            current = self.meta.get("embedding_model")
            if current and current != model and self.chunks:
                raise EmbeddingModelMismatch(
                    f"This knowledge base was indexed with '{current}' but the server now uses "
                    f"'{model}'. Create a new knowledge base or restore the original model."
                )
            self.meta["embedding_model"] = model
            self.chunks.extend(chunks)
            self.embeddings = (
                vectors if self.embeddings is None else np.vstack([self.embeddings, vectors])
            )
            self.documents.append({**doc, "chunks": len(chunks), "added_at": _now()})
            self._rebuild_lexical()
            self._save()

    def remove_document(self, doc_id: str) -> None:
        with self.lock:
            if not any(d["id"] == doc_id for d in self.documents):
                raise DocumentNotFound(doc_id)
            keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
            self.chunks = [self.chunks[i] for i in keep]
            if self.embeddings is not None:
                self.embeddings = self.embeddings[keep]
            self.meta["documents"] = [d for d in self.documents if d["id"] != doc_id]
            self._rebuild_lexical()
            self._save()


class CollectionManager:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir) / "collections"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._collections: dict[str, Collection] = {}
        for path in sorted(self.root.iterdir()):
            if (path / "meta.json").exists():
                collection = Collection.load(path)
                self._collections[collection.id] = collection

    def create(self, name: str, description: str = "", instructions: str = "") -> Collection:
        with self._lock:
            collection = Collection.create(self.root, name, description, instructions)
            self._collections[collection.id] = collection
            return collection

    def get(self, collection_id: str) -> Collection:
        try:
            return self._collections[collection_id]
        except KeyError:
            raise CollectionNotFound(collection_id) from None

    def list(self) -> list[Collection]:
        return sorted(self._collections.values(), key=lambda c: c.meta["created_at"])

    def delete(self, collection_id: str) -> None:
        with self._lock:
            collection = self.get(collection_id)
            del self._collections[collection_id]
            shutil.rmtree(collection.path, ignore_errors=True)
