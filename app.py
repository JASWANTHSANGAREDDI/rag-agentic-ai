"""
app.py — Production-grade FastAPI server for the RAG Agentic AI system.

ENDPOINTS:
    GET  /           → Welcome + links
    GET  /health     → Liveness probe  (is the process up?)
    GET  /readiness  → Readiness probe (are dependencies reachable?)
    POST /chat       → Full RAG pipeline

PRODUCTION NOTES:
  - Every request gets a unique X-Request-ID header for end-to-end tracing.
  - CORS origins are configurable via settings.cors_origins — NOT hardcoded *.
  - `allow_credentials=True` is only set when origins are NOT "*"
    (browsers block credentials + wildcard — RFC 6454).
  - The lifespan context pre-warms the embedding model and Pinecone connection
    so the first real request is fast.
  - `reload=True` is NOT used in the production entry point.
  - Structured logging uses %s format for log aggregator compatibility.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from src.config import settings
from src.exceptions import GenerationError, RetrievalError
from src.graph import get_final_answer, rag_graph
from src.retriever import get_embeddings, get_vector_store
from src.schemas import AgentState, ChatRequest, ChatResponse

# ---------------------------------------------------------------------------
# Logging — structured, %s-formatted for log aggregator compatibility
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("rag.api")


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm heavy resources. Fail fast if dependencies are unreachable."""
    logger.info("Starting RAG Agentic AI API | model=%s | index=%s",
                settings.model_name, settings.pinecone_index_name)

    logger.info("Pre-loading embedding model: %s", settings.embedding_model)
    get_embeddings()

    logger.info("Connecting to Pinecone index: %s", settings.pinecone_index_name)
    get_vector_store()

    logger.info("API ready to serve requests.")
    yield
    logger.info("API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Agentic AI",
    description=(
        "An agentic Retrieval-Augmented Generation system built with LangGraph.\n\n"
        "Ask questions about Agentic AI and receive answers grounded in the source ebook."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — secure configuration
#
# Browser rule: if allow_origins=["*"] then allow_credentials MUST be False.
# (RFC 6454 + browsers will block the request otherwise.)
# Set CORS_ORIGINS=https://yourapp.com in production.
# ---------------------------------------------------------------------------

_origins = settings.cors_origins_list
_credentials = "*" not in _origins  # only send credentials to explicit origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
)


# ---------------------------------------------------------------------------
# Middleware: Request ID + latency tracking
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """
    Attach a unique request ID to every request and response.

    - Reads X-Request-ID from the incoming request (allows callers to
      correlate their own trace IDs with our logs).
    - Falls back to a generated UUID4 if none is provided.
    - Echoes the ID in the response header.
    - Logs method, path, status, and latency for every request.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{elapsed_ms:.1f}ms"

    logger.info(
        "%s %s | status=%d | %.1fms | req_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "Unhandled exception | req_id=%s | path=%s | error=%s",
        req_id, request.url.path, exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal error occurred. Please try again.",
            "request_id": req_id,
        },
    )


# ---------------------------------------------------------------------------
# UI & Service Discovery Endpoints
# ---------------------------------------------------------------------------

_STATIC_DIR = PROJECT_ROOT / "src" / "static"
_INDEX_HTML_PATH = _STATIC_DIR / "index.html"


@app.get("/", summary="Chat Web UI", response_class=HTMLResponse, tags=["UI"])
async def root(request: Request):
    """
    Serve the modern Chat Web UI by default.
    If requested with 'Accept: application/json', returns service metadata instead.
    """
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse({
            "service": "RAG Agentic AI",
            "version": app.version,
            "docs": "/docs",
            "ui": "/",
            "health": "/health",
            "readiness": "/readiness",
            "chat": "POST /chat",
        })

    if _INDEX_HTML_PATH.exists():
        return HTMLResponse(content=_INDEX_HTML_PATH.read_text(encoding="utf-8"))

    return HTMLResponse("<h2>UI template not found</h2>", status_code=404)


@app.get("/ui", summary="Dedicated Chat Web UI", response_class=HTMLResponse, tags=["UI"])
async def ui_endpoint():
    """Direct URL for the Chat UI."""
    if _INDEX_HTML_PATH.exists():
        return HTMLResponse(content=_INDEX_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>UI template not found</h2>", status_code=404)


# ---------------------------------------------------------------------------
# GET /health  — liveness probe
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness probe", tags=["Meta"])
async def health():
    """
    Returns 200 if the process is running.
    Use for Kubernetes liveness probes — does NOT check external dependencies.
    """
    return {"status": "ok", "version": app.version}


# ---------------------------------------------------------------------------
# GET /readiness  — readiness probe
# ---------------------------------------------------------------------------

@app.get("/readiness", summary="Readiness probe", tags=["Meta"])
async def readiness():
    """
    Returns 200 if the app can serve traffic (embeddings loaded, Pinecone reachable).
    Use for Kubernetes readiness probes — will block traffic until this passes.
    """
    checks: dict[str, str] = {}

    try:
        get_embeddings()
        checks["embeddings"] = "ok"
    except Exception as exc:
        checks["embeddings"] = f"error: {exc}"

    try:
        get_vector_store()
        checks["pinecone"] = "ok"
    except Exception as exc:
        checks["pinecone"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "ready" if all_ok else "not_ready", "checks": checks},
    )


# ---------------------------------------------------------------------------
# POST /chat  — main RAG endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question about Agentic AI",
    tags=["RAG"],
    responses={
        200: {"description": "Answer generated successfully"},
        422: {"description": "Invalid request (empty question, top_k out of range, etc.)"},
        500: {"description": "Pipeline error (LLM or Pinecone unavailable)"},
    },
)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    """
    Run the full RAG Agentic AI pipeline:

    1. **Retrieve** — similarity search in Pinecone for relevant chunks
    2. **Generate** — LLM produces an answer grounded in the chunks
    3. **Grade** — a second LLM call evaluates quality; loops if low quality
    4. **Return** — clean answer + source chunks

    The pipeline loops up to `MAX_RETRIES` times before accepting any answer.
    """
    req_id = getattr(http_request.state, "request_id", "unknown")
    logger.info(
        "POST /chat | req_id=%s | question=%.80r | top_k=%d",
        req_id, request.question, request.top_k,
    )

    initial_state: AgentState = {
        "question": request.question,
        "context": [],
        "answer": "",
        "grade": "",
        "retry_count": 0,
        "top_k": request.top_k,
    }

    try:
        final_state = cast(AgentState, await rag_graph.ainvoke(initial_state))
    except RetrievalError as exc:
        logger.error("POST /chat | req_id=%s | retrieval error: %s", req_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to query the knowledge base. Please try again shortly.",
        )
    except GenerationError as exc:
        logger.error("POST /chat | req_id=%s | generation error: %s", req_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to generate an answer. Please try again shortly.",
        )
    except Exception as exc:
        logger.error("POST /chat | req_id=%s | unexpected error: %s", req_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred in the AI pipeline.",
        )

    answer = get_final_answer(final_state)
    sources = final_state.get("context", [])

    logger.info(
        "POST /chat | req_id=%s | answer_len=%d | sources=%d | grade=%s",
        req_id, len(answer), len(sources), final_state.get("grade"),
    )

    return ChatResponse(answer=answer, sources=sources)


# ---------------------------------------------------------------------------
# Entry point — for direct `python app.py` execution (NOT for production)
# In production use: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.warning(
        "Running with uvicorn in single-worker mode. "
        "For production, use: uvicorn app:app --host 0.0.0.0 --port %d --workers 4",
        settings.port,
    )
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,   # Never True in production
    )
