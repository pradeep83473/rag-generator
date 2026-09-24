"""Hybrid retrieval: dense cosine + BM25, fused with Reciprocal Rank Fusion.

RRF only uses ranks, so the two retrievers' incomparable score scales never
need calibrating; a chunk that ranks well in both lists rises to the top.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ingestion.chunking import Chunk
from app.store import Collection

RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity: float  # cosine similarity to the query


def hybrid_search(
    collection: Collection,
    query_text: str,
    query_vector: np.ndarray,
    top_k: int,
    candidates: int,
) -> list[RetrievedChunk]:
    with collection.lock:
        if not collection.chunks or collection.embeddings is None:
            return []
        similarities = collection.embeddings @ query_vector.reshape(-1)
        lexical = collection.bm25.scores(query_text)
        chunks = list(collection.chunks)

    fused: dict[int, float] = {}
    dense_rank = np.argsort(-similarities)[:candidates]
    for rank, idx in enumerate(dense_rank):
        fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)

    lexical_rank = [i for i in np.argsort(-lexical)[:candidates] if lexical[i] > 0]
    for rank, idx in enumerate(lexical_rank):
        fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)

    best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [RetrievedChunk(chunks[i], float(similarities[i])) for i, _ in best]
