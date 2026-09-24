# RAG Generator

Upload any set of documents at runtime and get a question-answering assistant over them, with answers grounded in, and cited to, the source passages. One deployment serves any number of independent document sets ("knowledge bases") with no code changes.

## Reading the brief

The brief has four requirements. Each one maps to a concrete design decision:

| Requirement | What it implies | How it's met |
|---|---|---|
| Accepts documents at runtime | Ingestion is an API operation, not a build step. Heterogeneous formats, bad files and duplicates must be handled gracefully. | `POST /api/collections/{id}/documents` (multipart) and drag-and-drop in the UI. Loaders for PDF, DOCX, Markdown, TXT, HTML, CSV, JSON. Per-file results (`indexed` / `duplicate` / `error` with a reason). SHA-256 de-duplication. |
| Creates a RAG application over those documents | "Generator" means each document set becomes its own self-contained app: its own index, its own identity, optionally its own answer style. | A knowledge base is created with a name, description and optional answer instructions. It gets an isolated, persisted index and its own query endpoint. |
| Grounded answers | The model must answer from the documents, show where each claim came from, and say so when the documents don't contain the answer. | Hybrid retrieval, a strict grounding prompt with numbered sources, mandatory inline citations `[n]`, an explicit "not found" answer, and a `grounded` flag plus the full source passages in every response. Document text is treated as untrusted data (prompt-injection guard). |
| Works with different document sets without code changes | Nothing domain-specific may live in code: no hard-coded prompts about a topic, no fixed schema, no fixed file list. | All behaviour is driven by uploaded content and per-knowledge-base settings. Global tunables come from environment variables. Tests verify that two knowledge bases never see each other's documents. |

## Architecture

```
            ┌────────────── ingestion ──────────────┐
 upload ──▶ │ loader (by extension) → sections      │
            │   (PDF pages keep page numbers)       │
            │ recursive chunker (1000 chars, 150    │
            │   overlap; paragraph→line→sentence)   │
            │ OpenAI embeddings (batched)           │
            └───────────────┬───────────────────────┘
                            ▼
            data/collections/<id>/  meta.json · chunks.json · embeddings.npy
                            │                 (+ in-memory BM25 index)
                            ▼
 question ─▶ follow-up? ─▶ condense to standalone query (LLM)
                            │
                            ▼
            hybrid retrieval: cosine (dense) + BM25 (lexical)
                            fused with Reciprocal Rank Fusion → top 6
                            │
                            ▼
            grounded prompt: numbered sources + rules + KB instructions
                            │
                            ▼
            OpenAI chat model → answer with [n] citations
                            │
                            ▼
            response: answer · grounded flag · cited/retrieved sources
```

```
app/
  main.py            FastAPI routes, error mapping, serves the UI
  service.py         orchestration: ingest(), retrieve(), ask()
  ingestion/
    loaders.py       bytes → text sections, one loader per format
    chunking.py      recursive structure-aware chunking with overlap
  store.py           per-collection persistent index + manager
  bm25.py            dependency-free Okapi BM25
  retrieval.py       hybrid search + Reciprocal Rank Fusion
  generation.py      prompts, source formatting, citation parsing
  llm.py             provider interface + OpenAI implementation
  cli.py             same pipeline from the command line
  evaluate.py        retrieval and answer-quality evaluation
  static/index.html  single-file web UI
tests/               28 offline tests with a deterministic fake LLM
samples/             two unrelated document sets for demoing
eval/                labelled question sets for the sample document sets
scripts/smoke_test.sh  end-to-end check against a live server
transcripts/         exported AI agent transcripts from building this
```

## Key design decisions

**Hybrid retrieval rather than embeddings alone.** Dense embeddings handle paraphrase ("time off" ↔ "annual leave") but blur exact tokens such as error codes, SKUs, clause numbers and names. BM25 catches those. Reciprocal Rank Fusion combines the two using ranks only, so no score calibration is needed between them. This matters because the brief says *any* document set: some will be prose-heavy, others identifier-heavy.

**A NumPy matrix as the vector store.** Exact cosine search over tens of thousands of chunks is a single matrix-vector product that takes milliseconds. It needs no extra service, is persisted as plain files, and is easy to inspect. The `Collection` interface is small, so swapping in pgvector, Qdrant or FAISS for larger corpora is a contained change.

**Citations are part of the contract.** Sources are numbered in the prompt, the model must cite them inline, and the API returns every retrieved passage marked `cited` or not. The UI turns each `[n]` into a clickable highlight that opens the exact passage. The `grounded` flag is false when the model cites nothing or declines.

**Follow-up questions are rewritten before retrieval.** "Can it be extended?" retrieves nothing useful on its own. When history is present, the question is condensed into a standalone query first, and the rewritten query is returned so the behaviour is transparent. Citation markers are stripped from earlier answers before they go back to the model, because their numbers referred to a different set of sources.

**Chunking keeps structure and location.** Text is split on the coarsest boundary that fits (paragraph, then line, then sentence, then word) and chunks overlap so answers straddling a boundary stay retrievable. PDFs are chunked per page so citations carry page numbers. CSV rows are rewritten as `column: value` pairs so a chunk remains meaningful without its header row. The filename is prepended to each chunk's search text because it's often a strong relevance signal.

**Per-knowledge-base instructions.** The owner can add answer guidance ("answer for new joiners in plain English"). It's appended after the grounding rules, so it shapes style without overriding grounding.

**Operational details.** Embedding happens outside the collection lock so queries keep working during ingestion. Writes are atomic (`tmp` + `os.replace`). The embedding model is recorded per knowledge base, and mixing models is refused with a clear 409, since vectors from different models aren't comparable. The server starts without an API key and returns a clear 503 on LLM-dependent calls rather than crashing. The duplicate check is repeated under the collection lock, so two simultaneous uploads of the same file index it once.

## Running it

Requires Python 3.10+ and an OpenAI API key.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your key in OPENAI_API_KEY
uvicorn app.main:app --reload
```

Open http://localhost:8000, create a knowledge base, drop in files, and ask questions. Interactive API docs are at http://localhost:8000/docs.

### Without an OpenAI key

Any OpenAI-compatible server works through `OPENAI_BASE_URL`. For example, with [Ollama](https://ollama.com) running locally:

```bash
ollama pull llama3.1 && ollama pull nomic-embed-text
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama                      # any non-empty value
OPENAI_CHAT_MODEL=llama3.1
OPENAI_EMBEDDING_MODEL=nomic-embed-text
```

Answer quality then depends on the local model; run the evaluation below to compare.

With Docker:

```bash
docker build -t rag-generator .
docker run -p 8000:8000 --env-file .env -v rag-data:/data rag-generator
```

### Try two different document sets

With the server running:

```bash
./scripts/smoke_test.sh
```

This creates one knowledge base from `samples/handbook` (HR policy) and one from `samples/product` (router FAQ), asks each a question, and asks an out-of-scope question that should be declined.

### CLI

```bash
python -m app.cli create "Product docs"            # prints the id
python -m app.cli ingest <id> ./samples/product
python -m app.cli ask <id> "What does error E-4471 mean?"
python -m app.cli chat <id>                         # multi-turn
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/collections` | Create a knowledge base `{name, description?, instructions?}` |
| `GET` | `/api/collections` | List knowledge bases |
| `GET` | `/api/collections/{id}` | Knowledge base summary |
| `DELETE` | `/api/collections/{id}` | Delete it and its index |
| `POST` | `/api/collections/{id}/documents` | Upload one or more files (`files` field, multipart) |
| `GET` | `/api/collections/{id}/documents` | List documents |
| `DELETE` | `/api/collections/{id}/documents/{doc_id}` | Remove a document and its chunks |
| `POST` | `/api/collections/{id}/query` | Ask `{question, history?, top_k?}` |
| `GET` | `/api/health` | Models in use, supported formats, key status |

Example query response:

```json
{
  "answer": "Up to 5 unused days can be carried over and must be used by 31 March [1].",
  "search_query": "How many days of annual leave can I carry over?",
  "grounded": true,
  "sources": [
    {"number": 1, "cited": true, "filename": "leave-policy.md", "page": null,
     "document_id": "b623e01ab29c", "similarity": 0.61, "text": "## Carry-over\nUp to 5 unused days..."}
  ]
}
```

## Configuration

All settings are environment variables (see `.env.example`). The main ones:

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | none | Required |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Any chat-completions model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Fixed per knowledge base once used |
| `OPENAI_BASE_URL` | none | OpenAI-compatible gateway |
| `LLM_TEMPERATURE` | `0` | Leave empty for models that only accept the default |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Characters |
| `RETRIEVAL_TOP_K` | `6` | Passages sent to the model |
| `DATA_DIR` | `data` | Where indexes are stored |

## Evaluation

`app/evaluate.py` measures quality against a labelled question set. Each file in `eval/` points at a folder of documents and lists questions with the file that should answer them and terms the answer must contain, plus questions the documents can't answer. The evaluator builds a fresh knowledge base in a temporary directory and reports:

| Metric | Passes when |
|---|---|
| retrieval | the expected file is among the retrieved passages |
| citation | the answer cites a passage from the expected file |
| answer | every expected term appears in the answer |
| grounded | the answer is marked grounded |
| declined | an out-of-scope question gets a not-grounded answer |

```bash
python -m app.evaluate eval/handbook.json eval/product.json
python -m app.evaluate eval/*.json --fail-under 0.9 --json report.json   # for CI
```

The out-of-scope cases include questions answered by the *other* sample set, which checks that knowledge bases stay isolated with real models. To evaluate your own documents, write a spec in the same format; no code changes are needed.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite runs fully offline by injecting a deterministic fake provider (hashed bag-of-words embeddings, scripted chat replies). It covers chunk sizing and overlap, every loader including PDF page numbers, BM25 and hybrid ranking, persistence and reload, the full upload → query flow, follow-up rewriting and citation stripping, isolation between knowledge bases, detecting declined answers, duplicate, concurrent-duplicate and unsupported files, document deletion, the evaluator, and the missing-key error path.

## Development transcripts

The complete Claude Code sessions used to build and review this project are in `transcripts/`, exported with `/export`.

## Limitations and next steps

Scanned PDFs have no text layer and are rejected with a message; adding OCR would fix that. Ingestion is synchronous, which is fine for typical uploads but large batches would benefit from a background job queue with progress reporting. A cross-encoder or LLM reranker over the fused candidates would improve precision on large corpora. Answers are returned in one piece; streaming would improve perceived latency. For multi-user deployment the next additions would be authentication, per-user knowledge bases, and a managed vector store. The evaluation checks answers by expected terms; an LLM judge would score faithfulness on free-form answers where exact terms don't apply.
