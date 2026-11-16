"""
retriever.py — Pinecone vector-store retrieval layer.

PRODUCTION NOTES:
  - Embedding model and vector-store client are cached with @lru_cache.
    They load once per process — not on every request.
  - All Pinecone I/O is wrapped in try/except with structured logging.
    A RetrievalError is raised so callers can handle it explicitly.
  - The unused `Document` import has been removed.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import settings
from src.exceptions import RetrievalError
from src.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singletons — heavy objects that must load exactly once per process
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace sentence-transformer model into RAM.

    @lru_cache(maxsize=1) means this runs once regardless of how many
    threads call it concurrently — subsequent calls return the cached object.
    """
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def get_vector_store() -> PineconeVectorStore:
    """
    Open a connection to the Pinecone index.

    Uses the cached embeddings object so we never load the model twice.
    """
    logger.info("Connecting to Pinecone index: %s", settings.pinecone_index_name)
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=get_embeddings(),
        pinecone_api_key=settings.pinecone_api_key,
        text_key="text",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
    """
    Retrieve the most relevant document chunks from Pinecone for `query`.

    Args:
        query:  The user's question or search string.
        top_k:  Number of chunks to return. Defaults to settings.top_k.

    Returns:
        List of RetrievedChunk objects, ordered by descending similarity score.

    Raises:
        RetrievalError: If the Pinecone call fails for any reason.
    """
    k = top_k if top_k is not None else settings.top_k
    logger.debug("Retrieving top-%d chunks for query: %.80s", k, query)

    try:
        vector_store = get_vector_store()
        raw_results = vector_store.similarity_search_with_score(query, k=k)
    except Exception as exc:
        logger.error("Pinecone retrieval failed: %s", exc, exc_info=True)
        raise RetrievalError(f"Vector store retrieval failed: {exc}") from exc

    chunks: list[RetrievedChunk] = []
    for doc, score in raw_results:
        chunks.append(
            RetrievedChunk(
                content=doc.page_content,
                score=round(float(score), 4),
                page=int(doc.metadata.get("page", -1)),
                source=str(doc.metadata.get("source", "unknown")),
            )
        )

    logger.debug("Retrieved %d chunks (top score: %.4f)", len(chunks), chunks[0].score if chunks else 0)
    return chunks
