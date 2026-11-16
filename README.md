# RAG Agentic AI

An **agentic Retrieval-Augmented Generation (RAG)** system built with LangGraph, FastAPI, Pinecone, and HuggingFace embeddings. Ask questions about Agentic AI and get answers grounded in source document chunks — with an autonomous self-grading loop that retries when the answer quality is low.

---

## Architecture

```
User → POST /chat
         │
         ▼
   FastAPI (app.py)
         │
         ▼
   LangGraph Agent (src/graph.py)
         │
   ┌─────▼──────┐     ┌──────────────┐     ┌────────────┐
   │retrieve_node│────▶│generate_node │────▶│ grade_node │
   └─────────────┘     └──────────────┘     └─────┬──────┘
          ▲                                        │
          │         low quality / hallucination    │
          └────────────────────────────────────────┘
                                                   │ high quality
                                                   ▼
                                                 [END]
         │
         ▼
   ChatResponse (answer + sources)
```

| Layer | Technology | Role |
|-------|-----------|------|
| Vector Store | Pinecone | Stores and searches document embeddings |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Converts text → 384-dim vectors (free) |
| LLM | OpenRouter (GPT-4o-mini / Claude / Llama) | Generates answers |
| Agent Framework | LangGraph | Orchestrates the agentic retry loop |
| API | FastAPI + Uvicorn | REST interface |
| Validation | Pydantic v2 | Type-safe data contracts throughout |

---

## Project Structure

```
rag-agentic-ai/
├── app.py                  # FastAPI server with built-in Web UI & REST endpoints
├── requirements.txt        # Production dependencies
├── .env.example            # Environment template
├── Makefile                # Developer shortcuts
├── data/
│   └── Ebook-Agentic-AI.pdf # Source knowledge document
└── src/
    ├── config.py           # Application settings loaded from .env
    ├── exceptions.py       # Custom exception hierarchy
    ├── schemas.py          # Pydantic data models (single source of truth)
    ├── ingestion.py        # PDF → chunks → embeddings → Pinecone (run once)
    ├── retriever.py        # Pinecone similarity search → RetrievedChunk[]
    ├── graph.py            # LangGraph agent: retrieve → generate → grade
    └── static/
        └── index.html      # Interactive Chat Web UI
```

---

## Setup

### 1. Clone & create virtual environment

```bash
git clone <repo-url>
cd rag-agentic-ai
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# OpenRouter (get key at openrouter.ai — free tier available)
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=openai/gpt-4o-mini

# Pinecone (get key at app.pinecone.io)
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=agentic-ai-index

# Optional tuning
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
CHUNK_SIZE=800
CHUNK_OVERLAP=200
TOP_K=5
```

### 3. Create the Pinecone index

In the [Pinecone console](https://app.pinecone.io):
- **Index name:** `agentic-ai-index`
- **Dimensions:** `384`
- **Metric:** `cosine`

### 4. Ingest the PDF (run once)

```bash
python src/ingestion.py
```

This loads the PDF, splits it into chunks, embeds with HuggingFace, and stores in Pinecone.

### 5. Start the server

```bash
python app.py
# or
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

## API Usage

### Interactive Docs

Open **http://localhost:8000/docs** for the full Swagger UI.

### `POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is agentic AI?", "top_k": 5}'
```

**Response:**

```json
{
  "answer": "Agentic AI refers to AI systems that can act autonomously to achieve goals, plan multi-step tasks, use tools, and adapt based on feedback...",
  "sources": [
    {
      "content": "Agentic AI refers to systems that can autonomously...",
      "score": 0.9234,
      "page": 12,
      "source": "Ebook-Agentic-AI.pdf"
    }
  ]
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "1.0.0"}
```

---

## How the Agentic Loop Works

1. **`retrieve_node`** — Runs a similarity search against Pinecone for the user's question. Returns the top-K most relevant text chunks.

2. **`generate_node`** — Passes the chunks + question to the LLM via a carefully crafted system prompt that forbids hallucination. Produces an answer grounded in the context.

3. **`grade_node`** — A *second* LLM call that acts as a quality judge: it reads the question, context, and answer, and returns `"good"` or `"retry"`.

4. **Router** — If `"retry"`, the graph loops back to `retrieve_node` (up to `MAX_RETRIES` times). If `"good"`, the graph ends and returns the answer.

This loop is what makes the system **agentic** — it autonomously decides whether its own output is good enough before returning it to the user.

---

## Tech Stack Justification

| Choice | Why |
|--------|-----|
| **LangGraph** | Native support for cyclic agent graphs with state management. LangChain alone only supports DAGs. |
| **OpenRouter** | Access to GPT-4o-mini, Claude, Llama, DeepSeek — all from one API key. No vendor lock-in. |
| **HuggingFace embeddings** | Free, open-source, runs locally. `all-MiniLM-L6-v2` is fast and accurate for semantic search. |
| **Pinecone** | Managed vector DB with a generous free tier. No infrastructure to manage. |
| **FastAPI** | Async, Pydantic-native, auto-generates OpenAPI docs. Best-in-class Python API framework. |
| **Pydantic v2** | Single source of truth for data shapes. Prevents the "passing raw dicts" anti-pattern. |
