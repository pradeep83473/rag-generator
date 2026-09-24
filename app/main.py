"""HTTP API + web UI.

A "knowledge base" is one generated RAG app. Create it, upload any documents
at runtime, and it immediately answers questions over them:

    POST   /api/collections                          create a knowledge base
    GET    /api/collections                          list them
    DELETE /api/collections/{id}                     delete one
    POST   /api/collections/{id}/documents           upload files (multipart)
    GET    /api/collections/{id}/documents           list documents
    DELETE /api/collections/{id}/documents/{doc_id}  remove a document
    POST   /api/collections/{id}/query               ask a question
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.llm import LLMProvider, ProviderNotConfigured
from app.schemas import CollectionCreate, QueryRequest, QueryResponse
from app.service import RAGService
from app.store import CollectionNotFound, DocumentNotFound, EmbeddingModelMismatch

logger = logging.getLogger("rag")
STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, provider: LLMProvider | None = None) -> FastAPI:
    settings = settings or get_settings()
    service = RAGService(settings, provider)
    app = FastAPI(title="RAG Generator", version="1.0.0")
    app.state.service = service

    # ---------- error mapping ----------
    @app.exception_handler(CollectionNotFound)
    async def _collection_missing(_: Request, exc: CollectionNotFound):
        return JSONResponse({"detail": f"Knowledge base '{exc.args[0]}' not found."}, 404)

    @app.exception_handler(DocumentNotFound)
    async def _document_missing(_: Request, exc: DocumentNotFound):
        return JSONResponse({"detail": f"Document '{exc.args[0]}' not found."}, 404)

    @app.exception_handler(ProviderNotConfigured)
    async def _no_provider(_: Request, exc: ProviderNotConfigured):
        return JSONResponse({"detail": str(exc)}, 503)

    @app.exception_handler(EmbeddingModelMismatch)
    async def _model_mismatch(_: Request, exc: EmbeddingModelMismatch):
        return JSONResponse({"detail": str(exc)}, 409)

    try:
        from openai import APIError

        @app.exception_handler(APIError)
        async def _openai_error(_: Request, exc: APIError):
            logger.exception("OpenAI request failed")
            return JSONResponse({"detail": f"OpenAI request failed: {exc.message}"}, 502)
    except ImportError:  # pragma: no cover
        pass

    # ---------- routes ----------
    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "llm_configured": bool(settings.openai_api_key) or provider is not None,
            "chat_model": settings.openai_chat_model,
            "embedding_model": settings.openai_embedding_model,
            "supported_extensions": SUPPORTED_EXTENSIONS,
            "max_upload_mb": settings.max_upload_mb,
        }

    @app.post("/api/collections", status_code=201)
    def create_collection(body: CollectionCreate):
        collection = service.collections.create(
            body.name.strip(), body.description.strip(), body.instructions.strip()
        )
        return collection.summary()

    @app.get("/api/collections")
    def list_collections():
        return [c.summary() for c in service.collections.list()]

    @app.get("/api/collections/{collection_id}")
    def get_collection(collection_id: str):
        return service.collections.get(collection_id).summary()

    @app.delete("/api/collections/{collection_id}", status_code=204)
    def delete_collection(collection_id: str):
        service.collections.delete(collection_id)

    @app.post("/api/collections/{collection_id}/documents")
    def upload_documents(collection_id: str, files: list[UploadFile] = File(...)):
        collection = service.collections.get(collection_id)
        limit = settings.max_upload_mb * 1024 * 1024
        results = []
        for upload in files:
            filename = Path(upload.filename or "upload").name
            data = upload.file.read(limit + 1)
            if len(data) > limit:
                results.append({"filename": filename, "status": "error",
                                "detail": f"File exceeds {settings.max_upload_mb} MB."})
                continue
            results.append(asdict(service.ingest(collection, filename, data)))
        return {"results": results, "collection": collection.summary()}

    @app.get("/api/collections/{collection_id}/documents")
    def list_documents(collection_id: str):
        return service.collections.get(collection_id).documents

    @app.delete("/api/collections/{collection_id}/documents/{document_id}", status_code=204)
    def delete_document(collection_id: str, document_id: str):
        service.collections.get(collection_id).remove_document(document_id)

    @app.post("/api/collections/{collection_id}/query", response_model=QueryResponse)
    def query(collection_id: str, body: QueryRequest):
        collection = service.collections.get(collection_id)
        answer = service.ask(
            collection,
            body.question.strip(),
            [m.model_dump() for m in body.history],
            body.top_k,
        )
        return asdict(answer)

    # ---------- UI ----------
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
