def _create(client, name="KB", **extra):
    res = client.post("/api/collections", json={"name": name, **extra})
    assert res.status_code == 201
    return res.json()["id"]


def _upload(client, kb, *files):
    res = client.post(
        f"/api/collections/{kb}/documents",
        files=[("files", (name, content, "application/octet-stream")) for name, content in files],
    )
    assert res.status_code == 200
    return res.json()["results"]


def test_end_to_end_question_answering(client, provider):
    kb = _create(client, "Handbook", instructions="Answer for new employees.")
    results = _upload(client, kb, ("leave.md", b"# Leave\nEmployees get 24 days of annual leave."),
                      ("it.txt", b"Laptops are replaced every three years."))
    assert [r["status"] for r in results] == ["indexed", "indexed"]

    res = client.post(f"/api/collections/{kb}/query", json={"question": "How many days of annual leave?"})
    assert res.status_code == 200
    body = res.json()
    assert body["grounded"] is True
    assert body["sources"][0]["filename"] == "leave.md"
    assert body["sources"][0]["cited"] is True

    system_prompt = provider.chat_calls[-1][0]["content"]
    assert "Handbook" in system_prompt and "Answer for new employees." in system_prompt
    assert "24 days" in provider.chat_calls[-1][-1]["content"]


def test_follow_up_question_is_rewritten(client, provider):
    kb = _create(client)
    _upload(client, kb, ("a.txt", b"The warranty lasts two years."))
    history = [{"role": "user", "content": "Tell me about the warranty"},
               {"role": "assistant", "content": "It lasts two years [1]."}]
    body = client.post(f"/api/collections/{kb}/query",
                       json={"question": "Can it be extended?", "history": history}).json()
    assert body["search_query"].startswith("standalone:")
    answer_messages = provider.chat_calls[-1]
    assert answer_messages[1] == history[0]
    # Citation numbers from an earlier answer point at a different source list.
    assert answer_messages[2]["content"] == "It lasts two years."


def test_decline_with_lead_in_is_not_grounded(client, provider):
    kb = _create(client)
    _upload(client, kb, ("a.txt", b"The warranty lasts two years."))
    provider.reply = ("Sorry — I couldn’t find the answer to that in the documents in "
                      "this knowledge base. The sources only cover the warranty [1].")
    body = client.post(f"/api/collections/{kb}/query", json={"question": "Capital of France?"}).json()
    assert body["grounded"] is False


def test_concurrent_duplicate_upload_is_indexed_once(client, provider):
    kb = _create(client)
    service = client.app.state.service
    collection = service.collections.get(kb)
    embed = provider.embed

    def embed_with_concurrent_upload(texts):
        # While the first upload is embedding, the same file arrives again.
        provider.embed = embed
        nested.append(service.ingest(collection, "copy.txt", b"same content"))
        return embed(texts)

    nested = []
    provider.embed = embed_with_concurrent_upload
    first = service.ingest(collection, "a.txt", b"same content")
    assert nested[0].status == "indexed" and first.status == "duplicate"
    assert len(collection.documents) == 1 and len(collection.chunks) == 1


def test_document_sets_are_isolated(client):
    hr, eng = _create(client, "HR"), _create(client, "Engineering")
    _upload(client, hr, ("hr.txt", b"Parental leave is sixteen weeks."))
    _upload(client, eng, ("eng.txt", b"Deploys happen every Tuesday."))
    body = client.post(f"/api/collections/{eng}/query", json={"question": "parental leave"}).json()
    assert {s["filename"] for s in body["sources"]} == {"eng.txt"}


def test_duplicates_and_unsupported_files(client):
    kb = _create(client)
    first = _upload(client, kb, ("a.txt", b"same content"))
    again = _upload(client, kb, ("copy.txt", b"same content"), ("virus.exe", b"MZ"), ("blank.txt", b"  "))
    assert first[0]["status"] == "indexed"
    assert [r["status"] for r in again] == ["duplicate", "error", "error"]
    assert client.get(f"/api/collections/{kb}").json()["document_count"] == 1


def test_delete_document_removes_its_chunks(client):
    kb = _create(client)
    _upload(client, kb, ("a.txt", b"alpha"), ("b.txt", b"bravo"))
    docs = client.get(f"/api/collections/{kb}/documents").json()
    assert client.delete(f"/api/collections/{kb}/documents/{docs[0]['id']}").status_code == 204
    summary = client.get(f"/api/collections/{kb}").json()
    assert summary["document_count"] == 1 and summary["chunk_count"] == 1


def test_empty_collection_answers_without_llm(client, provider):
    kb = _create(client)
    body = client.post(f"/api/collections/{kb}/query", json={"question": "anything?"}).json()
    assert body["grounded"] is False and provider.chat_calls == []


def test_missing_collection_is_404(client):
    assert client.post("/api/collections/nope/query", json={"question": "x"}).status_code == 404
    assert client.delete("/api/collections/nope").status_code == 404


def test_missing_api_key_gives_clear_503(settings):
    from fastapi.testclient import TestClient
    from app.main import create_app

    client = TestClient(create_app(settings))  # no provider, no key
    kb = client.post("/api/collections", json={"name": "x"}).json()["id"]
    res = client.post(f"/api/collections/{kb}/documents", files=[("files", ("a.txt", b"hi", "text/plain"))])
    assert res.status_code == 503 and "OPENAI_API_KEY" in res.json()["detail"]


def test_ui_and_health(client):
    assert "RAG Generator" in client.get("/").text
    health = client.get("/api/health").json()
    assert ".pdf" in health["supported_extensions"]
