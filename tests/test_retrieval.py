from app.bm25 import BM25
from app.service import RAGService


def test_bm25_prefers_matching_document():
    bm25 = BM25(["the cat sat on the mat", "error code E-4471 means disk full", "dogs bark"])
    scores = bm25.scores("what is E-4471")
    assert scores.argmax() == 1


def test_hybrid_search_finds_relevant_chunk(settings, provider):
    service = RAGService(settings, provider)
    kb = service.collections.create("Docs")
    service.ingest(kb, "cooking.txt", b"Boil pasta in salted water for nine minutes.")
    service.ingest(kb, "network.txt", b"Router error code E-4471 means the firmware update failed.")
    service.ingest(kb, "garden.txt", b"Water tomatoes deeply twice a week in summer.")
    results = service.retrieve(kb, "What does error E-4471 mean?", top_k=2)
    assert results[0].chunk.filename == "network.txt"


def test_collections_persist_and_reload(settings, provider):
    service = RAGService(settings, provider)
    kb = service.collections.create("Persisted", instructions="Be brief.")
    service.ingest(kb, "a.txt", b"Alpha beta gamma.")
    reloaded = RAGService(settings, provider).collections.get(kb.id)
    assert reloaded.meta["instructions"] == "Be brief."
    assert len(reloaded.chunks) == 1
    assert reloaded.embeddings.shape[0] == 1
    assert reloaded.bm25.scores("gamma")[0] > 0
